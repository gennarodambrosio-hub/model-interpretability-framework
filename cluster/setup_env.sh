#!/usr/bin/env bash
# Script di setup ambiente virtuale per 4x A100 sul cluster HPC UNISA (partizione gpuq)
set -euo pipefail

echo "=== [1/5] Gestione Storage ad alte prestazioni BeeGFS (quota home) ==="
# Se esiste BeeGFS, spostiamo la cache dei modelli lì per non saturare la home (/home ha quota ridotta)
if [ -d "/mnt/beegfs/g.dambrosio65" ]; then
    echo "Rilevato storage BeeGFS in /mnt/beegfs/g.dambrosio65. Configuro cache e symlink..."
    mkdir -p /mnt/beegfs/g.dambrosio65/hf_cache/hub
    mkdir -p /mnt/beegfs/g.dambrosio65/nla_data/activations
    mkdir -p /mnt/beegfs/g.dambrosio65/nla_checkpoints

    # Se esiste già ~/hf_cache come cartella reale con file dentro, spostiamo i dati su BeeGFS
    if [ -d "$HOME/hf_cache" ] && [ ! -L "$HOME/hf_cache" ]; then
        echo "Sposto i file già scaricati da ~/hf_cache su BeeGFS per liberare spazio su /home..."
        cp -rn "$HOME/hf_cache/"* /mnt/beegfs/g.dambrosio65/hf_cache/ 2>/dev/null || true
        rm -rf "$HOME/hf_cache"
    fi
    ln -sfn /mnt/beegfs/g.dambrosio65/hf_cache "$HOME/hf_cache"
    mkdir -p "$HOME/.cache"
    ln -sfn /mnt/beegfs/g.dambrosio65/hf_cache "$HOME/.cache/huggingface"

    export HF_HOME="/mnt/beegfs/g.dambrosio65/hf_cache"
    export HUGGINGFACE_HUB_CACHE="/mnt/beegfs/g.dambrosio65/hf_cache/hub"
else
    mkdir -p "$HOME/hf_cache"
    mkdir -p "$HOME/nla_data/activations"
    mkdir -p "$HOME/nla_checkpoints"
    export HF_HOME="$HOME/hf_cache"
    export HUGGINGFACE_HUB_CACHE="$HOME/hf_cache/hub"
fi

mkdir -p "$HOME/tools"
mkdir -p "$HOME/venvs"
mkdir -p "$HOME/nla_logs"

echo "=== [2/5] Creazione Virtualenv senza ensurepip (usando virtualenv) ==="
if [ -d "$HOME/venvs/nla-training-cu12" ]; then
    echo "Ambiente nla-training-cu12 già presente, lo attivo..."
elif [ -d "$HOME/venvs/vllm-qwen36-cu129" ]; then
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
python cluster/pre_download_login_node.py

echo "=========================================================================="
echo "🎉 Setup e pre-download completati con successo!"
echo "Ambiente pronto in: ~/venvs/nla-training-cu12"
echo "Puoi ora lanciare: sbatch cluster/run_nla_4xa100.sbatch"
echo "=========================================================================="
