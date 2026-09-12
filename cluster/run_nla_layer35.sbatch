#!/usr/bin/env bash
#SBATCH --job-name=granite-nla-l35
#SBATCH --partition=gpuq
#SBATCH -A did_tesi_nlp_330
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:4
#SBATCH --time=06:45:00
#SBATCH --output=/home/G.DAMBROSIO65/nla_logs/nla-l35-%j.out
#SBATCH --error=/home/G.DAMBROSIO65/nla_logs/nla-l35-%j.err

set -euo pipefail

echo "================================================================="
echo "Avvio Job NLA Layer 35 ID: ${SLURM_JOB_ID} su $(hostname)"
echo "Data inizio: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "================================================================="

source /home/G.DAMBROSIO65/venvs/nla-training-cu12/bin/activate

export HF_HOME="/mnt/beegfs/g.dambrosio65/hf_cache"
DATA_DIR="/mnt/beegfs/g.dambrosio65/nla_data_l35"
CKPT_DIR="/mnt/beegfs/g.dambrosio65/nla_checkpoints_l35"

export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /home/G.DAMBROSIO65/model-interpretability-framework

echo "=== [1/2] Estrazione Attivazioni Layer 35 da Granite 4.2 3B ==="
python cluster/extract_activations.py \
    --model "ibm-granite/granite-4.2-3b" \
    --output_dir "${DATA_DIR}/activations" \
    --layers 35 \
    --max_samples 25000

echo "=== [2/2] Training NLA Actor-Critic Layer 35 (3 Epoche) ==="
python cluster/train_nla.py \
    --shards_dir "${DATA_DIR}/activations" \
    --output_dir "${CKPT_DIR}" \
    --model "ibm-granite/granite-4.2-3b" \
    --layer "layer_35" \
    --epochs 3 \
    --batch_size 8 \
    --lr 1e-5

echo "================================================================="
echo "Job NLA Layer 35 completato con successo!"
echo "Data fine: $(date)"
echo "================================================================="
