#!/usr/bin/env bash
#SBATCH --job-name=nla-l35-real
#SBATCH --partition=gpuq
#SBATCH -A did_tesi_nlp_330
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:4
#SBATCH --time=06:45:00
#SBATCH --output=/home/G.DAMBROSIO65/nla_logs/nla-l35-real-%j.out
#SBATCH --error=/home/G.DAMBROSIO65/nla_logs/nla-l35-real-%j.err

set -euo pipefail

echo "================================================================="
echo "Avvio Job NLA Layer 35 (Differentiable) ID: ${SLURM_JOB_ID} su $(hostname)"
echo "Data inizio: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "================================================================="

source /home/G.DAMBROSIO65/venvs/nla-training-cu12/bin/activate

export HF_HOME="/mnt/beegfs/g.dambrosio65/hf_cache"
DATA_DIR="/mnt/beegfs/g.dambrosio65/nla_data_l35/activations"
CKPT_DIR="/mnt/beegfs/g.dambrosio65/nla_checkpoints_v2_l35"

export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /home/G.DAMBROSIO65/model-interpretability-framework

echo "=== Addestramento NLA Differenziabile Actor-Critic (Layer 35) ==="
python cluster/train_nla.py \
    --shards_dir "${DATA_DIR}" \
    --output_dir "${CKPT_DIR}" \
    --model "ibm-granite/granite-4.2-3b" \
    --layer "layer_35" \
    --epochs 3 \
    --batch_size 8 \
    --lr 2e-5

echo "================================================================="
echo "Job Layer 35 completato con successo!"
echo "Data fine: $(date)"
echo "================================================================="
