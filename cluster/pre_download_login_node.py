#!/usr/bin/env python3
"""Script da eseguire ESCLUSIVAMENTE sul LOGIN NODE (lnode02) prima di lanciare Slurm.
Scarica in anticipo modello e dataset nella cache locale ~/hf_cache, in modo che i nodi
di calcolo GPU (gnodeXX), che sono privi di connessione internet, trovino tutto offline.
"""

import os
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "ibm-granite/granite-4.2-3b")
HF_HOME = os.path.expanduser(os.getenv("HF_HOME", "~/hf_cache"))

os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")

print(f"=== [LOGIN NODE] Pre-download Modello: {MODEL_ID} in {HF_HOME} ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)
print("✅ Modello e tokenizer scaricati e memorizzati in cache locale!")

print("\n=== [LOGIN NODE] Pre-download Dataset: wikitext ===")
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
print(f"✅ Dataset scaricato ({len(ds)} righe) in cache locale!")

print("\n🎉 Tutto pronto! Ora i nodi di calcolo (gnodeXX) potranno lavorare 100% OFFLINE senza internet.")
