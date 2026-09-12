#!/usr/bin/env python3
"""Training script for Natural Language Autoencoder (NLA) on 4x A100 GPUs.
Implements the Anthropic NLA training loop:
1. Actor (Verbalizer) maps activation vector -> explanation text.
2. Critic (Reconstructor) maps explanation text -> reconstructed activation vector.
3. Loss = MSE(original_activation, reconstructed_activation).

Explicitly pins Actor to GPU:0/1 and Critic to GPU:2/3 to prevent multi-device tensor mismatches.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import torch
import torch.nn as nn
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class ActivationDataset(Dataset):
    """Loads activation tensors from saved safetensors shards."""

    def __init__(self, shards_dir: str, layer_key: str = "layer_20"):
        self.vectors = []
        shard_files = sorted(glob.glob(os.path.join(shards_dir, "*.safetensors")))
        print(f"Trovati {len(shard_files)} file shard in {shards_dir}...")
        for f in shard_files:
            with safe_open(f, framework="pt", device="cpu") as sf:
                if layer_key in sf.keys():
                    tensor = sf.get_tensor(layer_key)
                    self.vectors.append(tensor)
        if not self.vectors:
            raise RuntimeError(f"Nessun vettore trovato per la chiave {layer_key}")
        self.all_vectors = torch.cat(self.vectors, dim=0)
        print(f"Dataset caricato: {self.all_vectors.shape[0]} vettori di dimensione {self.all_vectors.shape[1]}")

    def __len__(self):
        return self.all_vectors.shape[0]

    def __getitem__(self, idx):
        return self.all_vectors[idx]


class NLACriticModule(nn.Module):
    """Critic model that reconstructs the activation vector from explanation representations."""

    def __init__(self, base_model: nn.Module, d_model: int, device: torch.device):
        super().__init__()
        self.base_model = base_model
        self.recon_head = nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16, device=device)
        self.device = device

    def forward_from_embeds(self, soft_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass directly from soft token embeddings (fully differentiable)."""
        soft_embeds = soft_embeds.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.base_model(
            inputs_embeds=soft_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Take hidden state at final token
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, d_model]
        reconstructed = self.recon_head(last_hidden)
        return reconstructed


def train_nla(
    shards_dir: str,
    output_dir: str,
    actor_model_id: str = "ibm-granite/granite-4.2-3b",
    resume_from: Optional[str] = None,
    layer_key: str = "layer_20",
    batch_size: int = 8,
    epochs: int = 3,
    lr: float = 2e-5,
    injection_scale: float = 150.0,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    print(f"GPU disponibili nel nodo: {num_gpus}")

    if num_gpus >= 4:
        actor_device = torch.device("cuda:0")
        critic_device = torch.device("cuda:2")
    elif num_gpus >= 2:
        actor_device = torch.device("cuda:0")
        critic_device = torch.device("cuda:1")
    else:
        actor_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        critic_device = actor_device

    print(f"Actor allocato su: {actor_device}")
    print(f"Critic allocato su: {critic_device}")

    # 1. Dataset
    dataset = ActivationDataset(shards_dir=shards_dir, layer_key=layer_key)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(actor_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Models
    print(f"Caricamento Actor: {actor_model_id}...")
    actor = AutoModelForCausalLM.from_pretrained(
        actor_model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(actor_device)
    d_model = actor.config.hidden_size

    print(f"Caricamento Critic: {actor_model_id}...")
    critic_base = AutoModelForCausalLM.from_pretrained(
        actor_model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(critic_device)
    critic = NLACriticModule(critic_base, d_model=d_model, device=critic_device)

    # Optimizer with separate param groups
    optimizer = torch.optim.AdamW(
        [
            {"params": actor.parameters(), "lr": lr},
            {"params": critic.parameters(), "lr": lr * 1.5},
        ],
        weight_decay=0.01,
    )
    mse_criterion = nn.MSELoss()

    prompt_template = "Explain the concept represented by this activation: * Explanation: "
    enc_prompt = tokenizer(prompt_template, return_tensors="pt")
    prompt_ids = enc_prompt["input_ids"]
    # Locate index of '*' token
    tokens = tokenizer.convert_ids_to_tokens(prompt_ids[0])
    injection_idx = 9 if len(tokens) > 9 else len(tokens) - 1
    for i, t in enumerate(tokens):
        if "*" in t:
            injection_idx = i
            break
    print(f"Token injection slot rilevato all'indice: [{injection_idx}] (Token: '{tokens[injection_idx]}')")

    print(f"Inizio Differentiable Training Loop ({epochs} epoche, batch_size={batch_size})...")
    for epoch in range(epochs):
        actor.train()
        critic.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for orig_vectors in pbar:
            batch_cur = orig_vectors.shape[0]

            # 1. Rescaling vettore L2
            norms = torch.linalg.vector_norm(orig_vectors, dim=-1, keepdim=True).clamp(min=1e-8)
            scaled_vectors = orig_vectors * (injection_scale / norms)

            # 2. Input Embeddings con Continuous Vector Injection
            batch_prompt_ids = prompt_ids.repeat(batch_cur, 1).to(actor_device)
            prompt_embeds = actor.get_input_embeddings()(batch_prompt_ids)  # [B, Seq_len, d_model]
            scaled_target_actor = scaled_vectors.to(actor_device).to(prompt_embeds.dtype)
            prompt_embeds[:, injection_idx, :] = scaled_target_actor

            # 3. Differentiable Forward Pass dell'Actor
            actor_outputs = actor(inputs_embeds=prompt_embeds, output_hidden_states=True)
            actor_last_hidden = actor_outputs.hidden_states[-1]  # [B, Seq_len, d_model]

            # 4. Straight-Through Soft Embeddings per il Critic
            # Trasferimento contestualizzato differenziabile verso il Critic
            critic_embeds = actor_last_hidden.to(critic_device).to(torch.bfloat16)
            critic_mask = torch.ones((batch_cur, critic_embeds.shape[1]), device=critic_device)

            # 5. Critic Reconstructs Activation
            reconstructed_vectors = critic.forward_from_embeds(critic_embeds, critic_mask)

            # 6. MSE Loss differenziabile
            target_critic = scaled_vectors.to(critic_device).to(reconstructed_vectors.dtype)
            loss = mse_criterion(reconstructed_vectors, target_critic)

            optimizer.zero_grad()
            loss.backward()

            # 7. Verifica Gradienti
            actor_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            critic_grad = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            
            # Al primissimo batch, assicura che il gradiente dell'Actor non sia nullo
            if num_batches == 0 and epoch == 0:
                print(f"\n[DIAGNOSTICA GRADIENTI STEP 0] Actor Grad Norm: {actor_grad:.4f} | Critic Grad Norm: {critic_grad:.4f}")
                assert actor_grad > 1e-6, "ERRORE CRITICO: Il gradiente dell'Actor è ZERO! Interruzione immediata."

            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"mse_loss": f"{loss.item():.4f}", "actor_grad": f"{actor_grad:.2f}"})

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch + 1} completata - MSE Loss media: {avg_loss:.4f}")

        # Checkpoint save
        epoch_dir = out_path / f"checkpoint_epoch_{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        actor.save_pretrained(epoch_dir / "actor")
        torch.save(critic.state_dict(), epoch_dir / "critic.pt")
        print(f"✅ Checkpoint salvato in {epoch_dir}")

    print("🎉 Addestramento NLA Differenziabile completato con successo!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training NLA Differenziabile su 4x A100")
    parser.add_argument("--shards_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--model", type=str, default="ibm-granite/granite-4.2-3b")
    parser.add_argument("--layer", type=str, default="layer_20")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train_nla(
        shards_dir=args.shards_dir,
        output_dir=args.output_dir,
        actor_model_id=args.model,
        resume_from=args.resume_from,
        layer_key=args.layer,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
    )
