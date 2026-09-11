#!/usr/bin/env bash
# Script di setup ambiente virtuale per 4x A100 sul cluster HPC UNISA (partizione gpuq)
set -euo pipefail

echo "=== [1/4] Creazione directory su HPC ==="
mkdir -p ~/hf_cache
mkdir -p ~/nla_data/activations
mkdir -p ~/nla_checkpoints
mkdir -p ~/nla_logs
mkdir -p ~/venvs

echo "=== [2/4] Creazione Virtualenv Python 3.10/3.12 con supporto CUDA 12 ==="
module purge || true
module load cuda/12.2 || module load cuda/12.1 || module load cuda/12.4 || true

python3 -m venv ~/venvs/nla-training-cu12
source ~/venvs/nla-training-cu12/bin/activate

echo "=== [3/4] Upgrade pip e installazione PyTorch CUDA ==="
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== [4/4] Installazione librerie LLM, Accelerate, FSDP, SGLang, Datasets ==="
pip install transformers>=4.48.0 accelerate>=0.34.0 safetensors datasets trl
pip install sentencepiece tiktoken pyyaml rich orjson httpx pydantic pyarrow
pip install deepspeed packaging

echo "Setup completato con successo in ~/venvs/nla-training-cu12 !"
