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
    """Critic model that reconstructs the activation vector from explanation token embeddings."""

    def __init__(self, base_model: nn.Module, d_model: int, device: torch.device):
        super().__init__()
        self.base_model = base_model
        self.recon_head = nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16, device=device)
        self.device = device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.base_model(
            input_ids=input_ids,
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
    lr: float = 1e-5,
    injection_scale: float = 150.0,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    print(f"GPU disponibili nel nodo: {num_gpus}")

    # Dedicate GPU 0 (or 0-1) to Actor, and GPU 1 (or 2-3) to Critic
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
    if resume_from and (Path(resume_from) / "actor").exists():
        actor_load_path = str(Path(resume_from) / "actor")
        print(f"🔄 Ripresa training Actor dal checkpoint: {actor_load_path}...")
    else:
        actor_load_path = actor_model_id
        print(f"Caricamento Actor base: {actor_load_path}...")

    actor = AutoModelForCausalLM.from_pretrained(
        actor_load_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(actor_device)
    d_model = actor.config.hidden_size

    print(f"Caricamento Critic base: {actor_model_id}...")
    critic_base = AutoModelForCausalLM.from_pretrained(
        actor_model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(critic_device)
    critic = NLACriticModule(critic_base, d_model=d_model, device=critic_device)

    if resume_from and (Path(resume_from) / "critic.pt").exists():
        critic_ckpt_path = Path(resume_from) / "critic.pt"
        print(f"🔄 Ripresa pesi Critic dal checkpoint: {critic_ckpt_path}...")
        critic.load_state_dict(torch.load(critic_ckpt_path, map_location=critic_device))

    optimizer = torch.optim.AdamW(
        list(actor.parameters()) + list(critic.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    mse_criterion = nn.MSELoss()

    prompt_prefix = "Explain the concept represented by this activation: "
    prompt_suffix = " Explanation: "

    print(f"Inizio Training Loop ({epochs} epoche, batch_size={batch_size})...")
    for epoch in range(epochs):
        actor.train()
        critic.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for orig_vectors in pbar:
            # Scale on CPU / Actor device
            norms = torch.linalg.vector_norm(orig_vectors, dim=-1, keepdim=True).clamp(min=1e-8)
            scaled_vectors = orig_vectors * (injection_scale / norms)

            # Prompts for Actor
            batch_size_cur = orig_vectors.shape[0]
            prompts = [prompt_prefix + prompt_suffix for _ in range(batch_size_cur)]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(actor_device)

            # Actor generates explanation text
            with torch.no_grad():
                gen_tokens = actor.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=True,
                    temperature=0.7,
                )

            # Critic reads explanation on critic_device
            gen_tokens_critic = gen_tokens.to(critic_device)
            attention_mask_critic = torch.ones_like(gen_tokens_critic, device=critic_device)
            reconstructed_vectors = critic(gen_tokens_critic, attention_mask_critic)  # [B, d_model] on critic_device

            # Transfer target vectors to critic_device for MSE computation
            target_vectors_critic = scaled_vectors.to(critic_device).to(reconstructed_vectors.dtype)
            loss = mse_criterion(reconstructed_vectors, target_vectors_critic)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"mse_loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / max(1, num_batches)
        print(f"Epoch {epoch + 1} completata - MSE Loss media: {avg_loss:.4f}")

        # Checkpoint save
        epoch_dir = out_path / f"checkpoint_epoch_{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        actor.save_pretrained(epoch_dir / "actor")
        torch.save(critic.state_dict(), epoch_dir / "critic.pt")
        print(f"✅ Checkpoint salvato in {epoch_dir}")

    print("🎉 Addestramento NLA completato con successo!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training NLA su 4x A100")
    parser.add_argument("--shards_dir", type=str, default="/mnt/beegfs/g.dambrosio65/nla_data/activations")
    parser.add_argument("--output_dir", type=str, default="/mnt/beegfs/g.dambrosio65/nla_checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Path a un checkpoint esistente da cui continuare (es. checkpoint_epoch_3)")
    parser.add_argument("--model", type=str, default="ibm-granite/granite-4.2-3b")
    parser.add_argument("--layer", type=str, default="layer_20")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
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
