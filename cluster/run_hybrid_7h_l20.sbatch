#!/usr/bin/env bash
#SBATCH --job-name=nla-hybrid-l20
#SBATCH --partition=gpuq
#SBATCH -A did_tesi_nlp_330
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:4
#SBATCH --time=06:45:00
#SBATCH --output=/home/G.DAMBROSIO65/nla_logs/nla-hybrid-l20-%j.out
#SBATCH --error=/home/G.DAMBROSIO65/nla_logs/nla-hybrid-l20-%j.err

set -euo pipefail

echo "================================================================="
echo "Avvio Job Full Hybrid NLA (SFT + Critic MSE - 7h) ID: ${SLURM_JOB_ID} su $(hostname)"
echo "Data inizio: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "================================================================="

source /home/G.DAMBROSIO65/venvs/nla-training-cu12/bin/activate

export HF_HOME="/mnt/beegfs/g.dambrosio65/hf_cache"
SFT_FILE="/mnt/beegfs/g.dambrosio65/nla_data_sft/sft_dataset_l20.safetensors"
CKPT_DIR="/mnt/beegfs/g.dambrosio65/nla_checkpoints_hybrid_l20"

export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /home/G.DAMBROSIO65/model-interpretability-framework

echo "=== [1/2] Generazione Dataset SFT Oracle da 5.000 testi (Layer 20) ==="
python cluster/prepare_sft_dataset.py \
    --model "ibm-granite/granite-4.2-3b" \
    --corpus "${HF_HOME}/training_corpus.txt" \
    --output "${SFT_FILE}" \
    --layer 20 \
    --samples 5000

echo "=== [2/2] Training Ibrido SFT + Critic MSE (8 Epoche Totali) ==="
python cluster/train_sft_nla.py \
    --sft_file "${SFT_FILE}" \
    --output_dir "${CKPT_DIR}" \
    --layer "layer_20" \
    --epochs_sft 4 \
    --epochs_joint 4 \
    --batch_size 8 \
    --lr 2e-5

echo "================================================================="
echo "Job Hybrid NLA Layer 20 completato con successo!"
echo "Data fine: $(date)"
echo "================================================================="
