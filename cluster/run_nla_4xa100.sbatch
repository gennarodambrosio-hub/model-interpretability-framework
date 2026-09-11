#!/usr/bin/env bash
#SBATCH --job-name=granite-nla-4xa100
#SBATCH --partition=gpuq
#SBATCH -A did_tesi_nlp_330
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/home/G.DAMBROSIO65/nla_logs/nla-%j.out
#SBATCH --error=/home/G.DAMBROSIO65/nla_logs/nla-%j.err

set -euo pipefail

echo "================================================================="
echo "Avvio Job SLURM ID: ${SLURM_JOB_ID} su nodo: $(hostname)"
echo "Data inizio: $(date)"
echo "GPU assegnate:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "================================================================="

# Caricamento ambiente
module purge || true
module load cuda/12.2 || module load cuda/12.1 || true
source /home/G.DAMBROSIO65/venvs/nla-training-cu12/bin/activate

export HF_HOME=/home/G.DAMBROSIO65/hf_cache
export HUGGINGFACE_HUB_CACHE=/home/G.DAMBROSIO65/hf_cache/hub
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

WORKDIR="/home/G.DAMBROSIO65/model-interpretability-framework"
cd "${WORKDIR}"

echo "=== [Fase 1/2] Estrazione Attivazioni su Granite 4.2 3B ==="
python cluster/extract_activations.py \
    --model "ibm-granite/granite-4.2-3b" \
    --output_dir "/home/G.DAMBROSIO65/nla_data/activations" \
    --layers 16 20 28 \
    --max_samples 5000

echo "=== [Fase 2/2] Addestramento Natural Language Autoencoder (Layer 20) ==="
python cluster/train_nla.py \
    --shards_dir "/home/G.DAMBROSIO65/nla_data/activations" \
    --output_dir "/home/G.DAMBROSIO65/nla_checkpoints" \
    --model "ibm-granite/granite-4.2-3b" \
    --layer "layer_20" \
    --epochs 3 \
    --batch_size 8 \
    --lr 1e-5

echo "================================================================="
echo "Job NLA completato con successo!"
echo "Data fine: $(date)"
echo "================================================================="
