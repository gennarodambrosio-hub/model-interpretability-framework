#!/usr/bin/env bash
#SBATCH --job-name=nla-critic-l20
#SBATCH --partition=gpuq
#SBATCH -A did_tesi_nlp_330
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:2
#SBATCH --time=06:45:00
#SBATCH --output=/home/G.DAMBROSIO65/nla_logs/nla-critic-l20-%j.out
#SBATCH --error=/home/G.DAMBROSIO65/nla_logs/nla-critic-l20-%j.err

set -euo pipefail

echo "================================================================="
echo "Avvio Job Fase 2: NLA Critic MSE Alignment (Layer 20 - ~7h) ID: ${SLURM_JOB_ID} su $(hostname)"
echo "Data inizio: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "================================================================="

source /home/G.DAMBROSIO65/venvs/nla-training-cu12/bin/activate

export HF_HOME="/mnt/beegfs/g.dambrosio65/hf_cache"
SFT_FILE="/mnt/beegfs/g.dambrosio65/nla_data_sft/sft_dataset_l20.safetensors"
RESUME_ACTOR="/mnt/beegfs/g.dambrosio65/nla_checkpoints_hybrid_l20/checkpoint_epoch_4/actor"
CKPT_DIR="/mnt/beegfs/g.dambrosio65/nla_checkpoints_hybrid_l20"

export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=6

cd /home/G.DAMBROSIO65/model-interpretability-framework

echo "=== Training Fase 2: Critic MSE Alignment da Checkpoint Epoch 4 (35 Epoche) ==="
python cluster/train_sft_nla.py \
    --sft_file "${SFT_FILE}" \
    --resume_actor "${RESUME_ACTOR}" \
    --output_dir "${CKPT_DIR}" \
    --layer "layer_20" \
    --start_epoch 4 \
    --epochs_sft 0 \
    --epochs_joint 35 \
    --batch_size 8 \
    --lr 2e-5 \
    --checkpoint_interval 5

echo "================================================================="
echo "Job Critic MSE Layer 20 completato con successo!"
echo "Data fine: $(date)"
echo "================================================================="
