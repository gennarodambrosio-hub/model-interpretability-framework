#!/usr/bin/env python3
"""Distributed Activation Extractor for IBM Granite models on 4x A100 GPUs.
Extracts residual-stream hidden states from Granite across target layers and saves them
into chunked safetensors format for NLA training.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_activations(
    model_id: str,
    output_dir: str,
    target_layers: List[int],
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    max_samples: int = 5000,
    seq_len: int = 512,
    batch_size: int = 8,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{device}] Caricamento modello: {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Locate transformer layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_modules = model.model.layers
    else:
        raise RuntimeError("Impossibile individuare i decoder layers.")

    print(f"Modello caricato: {len(layer_modules)} layer. Target layers: {target_layers}")

    # Set up hooks
    captured: Dict[int, torch.Tensor] = {}

    def make_hook(l_idx: int):
        def hook_fn(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[l_idx] = hs.detach().cpu()
        return hook_fn

    handles = []
    for l in target_layers:
        handles.append(layer_modules[l].register_forward_hook(make_hook(l)))

    # Load dataset
    print(f"Caricamento dataset: {dataset_name} ({dataset_config})...")
    ds = load_dataset(dataset_name, dataset_config, split="train")

    buffer_texts = []
    for row in ds:
        text = row.get("text", "").strip()
        if len(text) > 50:
            buffer_texts.append(text)
            if len(buffer_texts) >= max_samples:
                break

    print(f"Campioni raccolti: {len(buffer_texts)}. Inizio estrazione vettori...")

    batch_tensors: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}
    chunk_idx = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(buffer_texts), batch_size), desc="Estrazione"):
            batch = buffer_texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=seq_len,
            ).to(device)

            model(**encoded)

            # Sample random non-padding tokens from each sequence
            attention_mask = encoded["attention_mask"].cpu()
            for l in target_layers:
                acts = captured[l]  # [B, S, D]
                for b_idx in range(acts.shape[0]):
                    valid_len = int(attention_mask[b_idx].sum().item())
                    if valid_len > 1:
                        # Extract activation at last valid position and a middle position
                        v_last = acts[b_idx, valid_len - 1].clone()
                        v_mid = acts[b_idx, valid_len // 2].clone()
                        batch_tensors[l].extend([v_last, v_mid])

            # Save chunks every 2000 vectors
            if len(batch_tensors[target_layers[0]]) >= 2000:
                chunk_file = out_path / f"activations_chunk_{chunk_idx:04d}.safetensors"
                save_dict = {}
                for l in target_layers:
                    stacked = torch.stack(batch_tensors[l]).float()
                    save_dict[f"layer_{l}"] = stacked
                save_file(save_dict, chunk_file)
                print(f"Salvato chunk: {chunk_file} ({stacked.shape[0]} vettori)")
                batch_tensors = {l: [] for l in target_layers}
                chunk_idx += 1

    # Save remaining
    if len(batch_tensors[target_layers[0]]) > 0:
        chunk_file = out_path / f"activations_chunk_{chunk_idx:04d}.safetensors"
        save_dict = {}
        for l in target_layers:
            stacked = torch.stack(batch_tensors[l]).float()
            save_dict[f"layer_{l}"] = stacked
        save_file(save_dict, chunk_file)
        print(f"Salvato chunk finale: {chunk_file}")

    for h in handles:
        h.remove()
    print("Estrazione completata con successo!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estrazione attivazioni su cluster HPC")
    parser.add_argument("--model", type=str, default="ibm-granite/granite-4.2-3b")
    parser.add_argument("--output_dir", type=str, default="/home/G.DAMBROSIO65/nla_data/activations")
    parser.add_argument("--layers", type=int, nargs="+", default=[16, 20, 28])
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()

    extract_activations(
        model_id=args.model,
        output_dir=args.output_dir,
        target_layers=args.layers,
        max_samples=args.max_samples,
    )
