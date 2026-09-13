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
    resume_actor: str = None,
    resume_critic: str = None,
    start_epoch: int = 0,
    epochs_sft: int = 4,
    epochs_joint: int = 4,
    batch_size: int = 8,
    lr: float = 2e-5,
    injection_scale: float = 150.0,
    checkpoint_interval: int = 5,
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
    actor_path = resume_actor if (resume_actor and Path(resume_actor).exists()) else model_id
    if resume_actor and Path(resume_actor).exists():
        print(f"🔄 Ripresa Actor da checkpoint pre-addestrato: {resume_actor}")
    else:
        print(f"Inizializzazione Actor da base model: {model_id}")

    actor = AutoModelForCausalLM.from_pretrained(
        actor_path,
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

    if resume_critic and Path(resume_critic).exists():
        print(f"🔄 Ripresa Critic da checkpoint: {resume_critic}")
        critic.load_state_dict(torch.load(resume_critic, map_location=critic_device))

    optimizer = torch.optim.AdamW(
        [
            {"params": actor.parameters(), "lr": lr},
            {"params": critic.parameters(), "lr": lr * 1.5},
        ],
        weight_decay=0.01,
    )
    total_epochs = epochs_sft + epochs_joint
    total_steps = total_epochs * len(dataloader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps), eta_min=1e-6)

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

    print(f"\n🚀 Avvio Training Ibrido ({total_epochs} Epoche: {epochs_sft} SFT + {epochs_joint} Joint Actor-Critic MSE)...")
    if start_epoch > 0:
        print(f"📌 Offset epoche: {start_epoch} (numerazione da {start_epoch + 1} a {start_epoch + total_epochs})")

    best_loss = float("inf")
    best_mse = float("inf")

    for local_epoch in range(total_epochs):
        curr_epoch_num = start_epoch + local_epoch + 1
        is_joint = (local_epoch >= epochs_sft)
        stage_name = "Joint SFT+Critic MSE" if is_joint else "Supervised SFT (Grammatica & Fluidità)"
        actor.train()
        critic.train() if is_joint else critic.eval()

        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_mse = 0.0
        num_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoca {curr_epoch_num}/{start_epoch + total_epochs} [{stage_name}]")

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
            shift_logits = logits[:, b_prefix.shape[1] - 1 : -1, :].contiguous()
            shift_labels = b_target.contiguous()
            loss_ce = ce_criterion(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

            total_step_loss = loss_ce
            cur_mse = 0.0

            # 6. Joint Critic MSE (negli stage avanzati)
            if is_joint:
                last_hidden = out.hidden_states[-1]
                critic_embeds = last_hidden.to(critic_device).to(torch.bfloat16)
                critic_mask = torch.ones((batch_cur, critic_embeds.shape[1]), device=critic_device)
                recon = critic.forward_from_embeds(critic_embeds, critic_mask)
                loss_mse = mse_criterion(recon, scaled.to(critic_device).to(recon.dtype))
                # Risoluzione device mismatch: sposta il loss_mse sullo stesso device di loss_ce
                total_step_loss = loss_ce + 0.3 * loss_mse.to(actor_device)
                cur_mse = loss_mse.item()

            optimizer.zero_grad()
            total_step_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            if is_joint:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += total_step_loss.item()
            epoch_ce += loss_ce.item()
            epoch_mse += cur_mse
            num_batches += 1

            cur_lr = scheduler.get_last_lr()[0]
            if is_joint:
                pbar.set_postfix({
                    "loss": f"{total_step_loss.item():.4f}",
                    "ce": f"{loss_ce.item():.4f}",
                    "mse": f"{cur_mse:.4f}",
                    "lr": f"{cur_lr:.2e}",
                })
            else:
                pbar.set_postfix({
                    "ce": f"{loss_ce.item():.4f}",
                    "lr": f"{cur_lr:.2e}",
                })

        avg_loss = epoch_loss / max(1, num_batches)
        avg_ce = epoch_ce / max(1, num_batches)
        avg_mse = epoch_mse / max(1, num_batches) if is_joint else 0.0

        mse_str = f" | MSE: {avg_mse:.4f}" if is_joint else ""
        print(f"Epoca {curr_epoch_num}/{start_epoch + total_epochs} completata - Loss totale: {avg_loss:.4f} | CE: {avg_ce:.4f}{mse_str}")

        # 1. Checkpoint Latest (sempre aggiornato ogni epoca come fail-safe)
        latest_dir = out_path / "checkpoint_latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        actor.save_pretrained(latest_dir / "actor")
        tokenizer.save_pretrained(latest_dir / "actor")
        torch.save(critic.state_dict(), latest_dir / "critic.pt")

        # 2. Checkpoint Best
        improved = False
        if is_joint and avg_mse < best_mse:
            best_mse = avg_mse
            improved = True
        elif not is_joint and avg_loss < best_loss:
            best_loss = avg_loss
            improved = True

        if improved:
            best_dir = out_path / "checkpoint_best"
            best_dir.mkdir(parents=True, exist_ok=True)
            actor.save_pretrained(best_dir / "actor")
            tokenizer.save_pretrained(best_dir / "actor")
            torch.save(critic.state_dict(), best_dir / "critic.pt")
            metric_info = f"MSE: {avg_mse:.4f}" if is_joint else f"Loss: {avg_loss:.4f}"
            print(f"🌟 Nuovo Record! Checkpoint Best salvato in {best_dir} ({metric_info})")

        # 3. Checkpoint Milestone (ogni checkpoint_interval epoche o all'ultima)
        is_milestone = (curr_epoch_num % checkpoint_interval == 0) or (local_epoch + 1 == total_epochs)
        if is_milestone:
            milestone_dir = out_path / f"checkpoint_epoch_{curr_epoch_num}"
            milestone_dir.mkdir(parents=True, exist_ok=True)
            actor.save_pretrained(milestone_dir / "actor")
            tokenizer.save_pretrained(milestone_dir / "actor")
            torch.save(critic.state_dict(), milestone_dir / "critic.pt")
            print(f"💾 Checkpoint milestone salvato in {milestone_dir}")

    print("🎉 Addestramento Ibrido SFT + NLA completato con successo!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layer", type=str, default="layer_20")
    parser.add_argument("--model_id", type=str, default="ibm-granite/granite-4.2-3b")
    parser.add_argument("--resume_actor", type=str, default=None, help="Directory del checkpoint Actor da riprendere")
    parser.add_argument("--resume_critic", type=str, default=None, help="File del checkpoint Critic (.pt) da riprendere")
    parser.add_argument("--start_epoch", type=int, default=0, help="Offset epoche per la numerazione")
    parser.add_argument("--epochs_sft", type=int, default=4, help="Epoche SFT warmup (0 per andare direttamente al Critic MSE)")
    parser.add_argument("--epochs_joint", type=int, default=4, help="Epoche Joint Actor-Critic MSE")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Intervallo salvataggio milestone (epoche)")
    args = parser.parse_args()

    train_hybrid(
        sft_file=args.sft_file,
        output_dir=args.output_dir,
        layer_key=args.layer,
        model_id=args.model_id,
        resume_actor=args.resume_actor,
        resume_critic=args.resume_critic,
        start_epoch=args.start_epoch,
        epochs_sft=args.epochs_sft,
        epochs_joint=args.epochs_joint,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_interval=args.checkpoint_interval,
    )
