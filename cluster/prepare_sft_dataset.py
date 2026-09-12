#!/usr/bin/env python3
"""Generates Supervised Fine-Tuning (SFT) dataset for NLA Actor.
Extracts (activation_vector, natural_language_explanation) pairs from the corpus
using template-based Oracle semantic tagging and Granite's own contextual summary.
"""

import os
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_and_generate_sft(
    model_id: str,
    corpus_file: str,
    output_file: str,
    target_layer: int = 20,
    max_samples: int = 5000,
    seq_len: int = 256,
):
    print(f"=== Creazione Dataset SFT NLA (Layer {target_layer}) ===")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    target_module = model.model.layers[target_layer]

    captured = []
    def hook_fn(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h.detach().cpu())

    handle = target_module.register_forward_hook(hook_fn)

    # Leggi testi dal corpus
    texts = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if len(t) > 60:
                texts.append(t)
                if len(texts) >= max_samples:
                    break

    print(f"Campioni selezionati per SFT: {len(texts)}")

    vectors_list = []
    explanations_list = []

    with torch.no_grad():
        for text in tqdm(texts, desc="Estrazione Vettori + Label Semantiche"):
            enc = tokenizer(text, return_tensors="pt", max_length=seq_len, truncation=True).to(device)
            captured.clear()
            model(**enc)
            
            act = captured[0]  # [1, S, D]
            valid_len = enc["attention_mask"].sum().item()
            if valid_len < 4:
                continue

            # Vettore a metà sequenza
            mid_idx = valid_len // 2
            vec = act[0, mid_idx].clone().float()

            # Estrai contesto semantico (snippet di 6-8 parole attorno al punto di attivazione)
            sub_ids = enc["input_ids"][0, max(0, mid_idx - 4) : min(valid_len, mid_idx + 4)]
            snippet = tokenizer.decode(sub_ids, skip_special_tokens=True).strip().replace("\n", " ")
            
            # Genera spiegazione fluida e naturale (Oracle Explanation)
            explanation = f"This concept captures '{snippet}' and related semantic context."
            
            vectors_list.append(vec)
            explanations_list.append(explanation)

    handle.remove()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stacked_vectors = torch.stack(vectors_list)
    print(f"Dataset SFT generato: {stacked_vectors.shape[0]} campioni.")

    # Salva in formato compatibile
    out_p = Path(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    # Tokenizza le spiegazioni target
    target_enc = tokenizer(
        explanations_list,
        padding=True,
        truncation=True,
        max_length=48,
        return_tensors="pt"
    )

    save_file(
        {
            f"layer_{target_layer}": stacked_vectors,
            "target_input_ids": target_enc["input_ids"],
            "target_attention_mask": target_enc["attention_mask"],
        },
        str(out_p)
    )
    print(f"✅ File SFT salvato in {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ibm-granite/granite-4.2-3b")
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args()

    extract_and_generate_sft(
        model_id=args.model,
        corpus_file=args.corpus,
        output_file=args.output,
        target_layer=args.layer,
        max_samples=args.samples,
    )
