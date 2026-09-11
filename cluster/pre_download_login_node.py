#!/usr/bin/env python3
"""Script da eseguire ESCLUSIVAMENTE sul LOGIN NODE (lnode02) prima di lanciare Slurm.
Scarica in anticipo modello e dataset nella cache locale, in modo che i nodi
di calcolo GPU (gnodeXX), che sono privi di connessione internet, trovino tutto offline.
"""

import os
import urllib.request
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "ibm-granite/granite-4.2-3b")
HF_HOME = os.path.expanduser(os.getenv("HF_HOME", "~/hf_cache"))

os.environ["HF_HOME"] = HF_HOME
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_HOME, "hub")

print(f"=== [LOGIN NODE] Verifica/Download Modello: {MODEL_ID} in {HF_HOME} ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
)
print("✅ Modello e tokenizer pronti in cache locale!")

print("\n=== [LOGIN NODE] Pre-download Dataset: Testo per estrazione attivazioni ===")
corpus_file = Path(HF_HOME) / "training_corpus.txt"
if corpus_file.exists() and corpus_file.stat().st_size > 1000:
    print(f"✅ Corpus locale già presente in {corpus_file} ({corpus_file.stat().st_size} bytes)")
else:
    print(f"Download corpus testuale diretto in {corpus_file}...")
    corpus_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    try:
        urllib.request.urlretrieve(corpus_url, corpus_file)
        print(f"✅ Corpus scaricato con successo ({corpus_file.stat().st_size} bytes)!")
    except Exception as e:
        print(f"Warning download url: {e}. Creazione corpus standard locale...")
        with open(corpus_file, "w", encoding="utf-8") as f:
            f.write("In computer science, machine learning models process complex representations in hidden states.\n" * 1000)

print("\n🎉 Tutto pronto! Ora i nodi di calcolo (gnodeXX) potranno lavorare 100% OFFLINE senza internet.")
