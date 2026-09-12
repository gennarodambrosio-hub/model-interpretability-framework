#!/usr/bin/env python3
"""Stage 1 + 2 Full Hybrid Training:
1. SFT (Supervised Fine-Tuning) su frasi fluide in linguaggio naturale.
2. Critic MSE Alignment per preservare la geometria esatta dei vettori.
Total runtime designed to maximize quality over ~5-6 hours on 4x A100 GPUs.
"""

import os
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

class SFTDataset(Dataset):
    def __init__(self, sft_file: str, layer_key: str = "layer_20"):
        with safe_open(sft_file, framework="pt", device="cpu") as sf:
            self.vectors = sf.get_tensor(layer_key)
            self.target_ids = sf.get_tensor("target_input_ids")
            self.target_mask = sf.get_tensor("target_attention_mask")
        print(f"Dataset SFT caricato: {self.vectors.shape[0]} campioni.")

    def __len__(self):
        return self.vectors.shape[0]

    def __getitem__(self, idx):
        return self.vectors[idx], self.target_ids[idx], self.target_mask[idx]

class NLACriticModule(nn.Module):
    def __init__(self, base_model: nn.Module, d_model: int, device: torch.device):
        super().__init__()
        self.base_model = base_model
        self.recon_head = nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16, device=device)
        self.device = device

    def forward_from_embeds(self, soft_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        soft_embeds = soft_embeds.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.base_model(
            inputs_embeds=soft_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        return self.recon_head(last_hidden)

def train_hybrid(
    sft_file: str,
    output_dir: str,
    layer_key: str = "layer_20",
    model_id: str = "ibm-granite/granite-4.2-3b",
    epochs_sft: int = 4,
    epochs_joint: int = 4,
    batch_size: int = 8,
    lr: float = 2e-5,
    injection_scale: float = 150.0,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus >= 4:
        actor_device = torch.device("cuda:0")
        critic_device = torch.device("cuda:2")
    else:
        actor_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        critic_device = actor_device

    print(f"Actor su: {actor_device} | Critic su: {critic_device}")

    dataset = SFTDataset(sft_file, layer_key=layer_key)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Caricamento Actor...")
    actor = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(actor_device)
    d_model = actor.config.hidden_size

    print("Caricamento Critic...")
    critic_base = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(critic_device)
    critic = NLACriticModule(critic_base, d_model=d_model, device=critic_device)

    optimizer = torch.optim.AdamW(
        [
            {"params": actor.parameters(), "lr": lr},
            {"params": critic.parameters(), "lr": lr * 1.5},
        ],
        weight_decay=0.01,
    )
    ce_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    mse_criterion = nn.MSELoss()

    prompt_prefix = "Explain the concept represented by this activation: * Explanation: "
    prefix_ids = tokenizer(prompt_prefix, return_tensors="pt")["input_ids"]
    prefix_tokens = tokenizer.convert_ids_to_tokens(prefix_ids[0])
    inj_idx = 9
    for i, t in enumerate(prefix_tokens):
        if "*" in t:
            inj_idx = i
            break

    total_epochs = epochs_sft + epochs_joint
    print(f"\n🚀 Avvio Training Ibrido Totale ({total_epochs} Epoche: {epochs_sft} SFT + {epochs_joint} Joint Actor-Critic)...")

    for epoch in range(total_epochs):
        is_joint = (epoch >= epochs_sft)
        stage_name = "Joint SFT+Critic MSE" if is_joint else "Supervised SFT (Grammatica & Fluidità)"
        actor.train()
        critic.train() if is_joint else critic.eval()

        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoca {epoch + 1}/{total_epochs} [{stage_name}]")

        for orig_vectors, target_ids, target_mask in pbar:
            batch_cur = orig_vectors.shape[0]

            # 1. Scaling L2
            norms = torch.linalg.vector_norm(orig_vectors, dim=-1, keepdim=True).clamp(min=1e-8)
            scaled = orig_vectors * (injection_scale / norms)

            # 2. Prep prompt + target sequence
            b_prefix = prefix_ids.repeat(batch_cur, 1).to(actor_device)
            b_target = target_ids.to(actor_device)
            full_ids = torch.cat([b_prefix, b_target], dim=1)

            # 3. Iniezione Vettore
            full_embeds = actor.get_input_embeddings()(full_ids)
            full_embeds[:, inj_idx, :] = scaled.to(actor_device).to(full_embeds.dtype)

            # 4. Forward Actor
            out = actor(inputs_embeds=full_embeds, output_hidden_states=True)
            logits = out.logits

            # 5. Cross-Entropy Loss sui token di spiegazione
            # Shift per calcolo language modeling loss
            shift_logits = logits[:, b_prefix.shape[1] - 1 : -1, :].contiguous()
            shift_labels = b_target.contiguous()
            loss_ce = ce_criterion(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

            total_step_loss = loss_ce

            # 6. Joint Critic MSE (negli stage avanzati)
            if is_joint:
                last_hidden = out.hidden_states[-1]
                critic_embeds = last_hidden.to(critic_device).to(torch.bfloat16)
                critic_mask = torch.ones((batch_cur, critic_embeds.shape[1]), device=critic_device)
                recon = critic.forward_from_embeds(critic_embeds, critic_mask)
                loss_mse = mse_criterion(recon, scaled.to(critic_device).to(recon.dtype))
                total_step_loss = loss_ce + 0.3 * loss_mse

            optimizer.zero_grad()
            total_step_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            if is_joint:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_step_loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{total_step_loss.item():.4f}", "ce_loss": f"{loss_ce.item():.4f}"})

        avg = epoch_loss / max(1, num_batches)
        print(f"Epoca {epoch + 1} completata - Loss media: {avg:.4f}")

        # Salva checkpoint a fine epoca
        ckpt_dir = out_path / f"checkpoint_epoch_{epoch + 1}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        actor.save_pretrained(ckpt_dir / "actor")
        torch.save(critic.state_dict(), ckpt_dir / "critic.pt")
        print(f"✅ Checkpoint salvato in {ckpt_dir}")

    print("🎉 Addestramento Ibrido SFT + NLA completato con successo!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layer", type=str, default="layer_20")
    parser.add_argument("--epochs_sft", type=int, default=4)
    parser.add_argument("--epochs_joint", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train_hybrid(
        sft_file=args.sft_file,
        output_dir=args.output_dir,
        layer_key=args.layer,
        epochs_sft=args.epochs_sft,
        epochs_joint=args.epochs_joint,
        batch_size=args.batch_size,
        lr=args.lr,
    )
