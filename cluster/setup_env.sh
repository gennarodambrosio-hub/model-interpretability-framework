#!/usr/bin/env bash
# Script di setup ambiente virtuale per 4x A100 sul cluster HPC UNISA (partizione gpuq)
set -euo pipefail

echo "=== [1/5] Creazione directory su HPC ==="
mkdir -p ~/tools
mkdir -p ~/venvs
mkdir -p ~/hf_cache
mkdir -p ~/nla_data/activations
mkdir -p ~/nla_checkpoints
mkdir -p ~/nla_logs

echo "=== [2/5] Creazione Virtualenv senza ensurepip (usando virtualenv) ==="
# Il Python di sistema Ubuntu su lnode02 non ha ensurepip installato.
# Usiamo virtualenv come da documentazione cluster UNISA.
if [ -d "$HOME/venvs/vllm-qwen36-cu129" ]; then
    echo "Trovato ambiente esistente vllm-qwen36-cu129, lo uso per bootstrap virtualenv..."
    source "$HOME/venvs/vllm-qwen36-cu129/bin/activate"
    python -m pip install -U virtualenv
    python -m virtualenv "$HOME/venvs/nla-training-cu12"
    deactivate
elif [ -d "$HOME/venvs/sglang-qwen36" ]; then
    echo "Trovato ambiente esistente sglang-qwen36, lo uso per bootstrap virtualenv..."
    source "$HOME/venvs/sglang-qwen36/bin/activate"
    python -m pip install -U virtualenv
    python -m virtualenv "$HOME/venvs/nla-training-cu12"
    deactivate
else
    echo "Installazione standalone di virtualenv in ~/tools/virtualenv..."
    python3 -m pip install --target "$HOME/tools/virtualenv" virtualenv
    PYTHONPATH="$HOME/tools/virtualenv" python3 -m virtualenv "$HOME/venvs/nla-training-cu12"
fi

source "$HOME/venvs/nla-training-cu12/bin/activate"

echo "=== [3/5] Installazione PyTorch con supporto CUDA 12 ==="
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== [4/5] Installazione Transformers, Accelerate, Datasets ==="
pip install "transformers>=4.48.0" "accelerate>=0.34.0" safetensors datasets trl
pip install sentencepiece tiktoken pyyaml rich orjson httpx pydantic pyarrow

echo "=== [5/5] Pre-download offline di Granite 4.2 e dataset su LOGIN NODE ==="
export HF_HOME="$HOME/hf_cache"
export HUGGINGFACE_HUB_CACHE="$HOME/hf_cache/hub"
python cluster/pre_download_login_node.py

echo "=========================================================================="
echo "🎉 Setup e pre-download completati con successo!"
echo "Ambiente pronto in: ~/venvs/nla-training-cu12"
echo "Puoi ora lanciare: sbatch cluster/run_nla_4xa100.sbatch"
echo "=========================================================================="
