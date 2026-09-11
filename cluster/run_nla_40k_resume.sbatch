#!/usr/bin/env bash
#SBATCH --job-name=granite-nla-40k
#SBATCH --partition=gpuq
#SBATCH -A did_tesi_nlp_330
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a100:4
#SBATCH --time=08:00:00
#SBATCH --output=/home/G.DAMBROSIO65/nla_logs/nla-40k-%j.out
#SBATCH --error=/home/G.DAMBROSIO65/nla_logs/nla-40k-%j.err

set -euo pipefail

echo "================================================================="
echo "Avvio Job Extended NLA (40.000 vettori) ID: ${SLURM_JOB_ID} su $(hostname)"
echo "Data inizio: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "================================================================="

source /home/G.DAMBROSIO65/venvs/nla-training-cu12/bin/activate

export HF_HOME="/mnt/beegfs/g.dambrosio65/hf_cache"
DATA_DIR="/mnt/beegfs/g.dambrosio65/nla_data_40k"
CKPT_DIR="/mnt/beegfs/g.dambrosio65/nla_checkpoints_40k"
PREV_CKPT="/mnt/beegfs/g.dambrosio65/nla_checkpoints/checkpoint_epoch_3"

export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd /home/G.DAMBROSIO65/model-interpretability-framework

echo "=== [1/2] Estrazione 40.000 Attivazioni da Granite 4.2 3B ==="
python cluster/extract_activations.py \
    --model "ibm-granite/granite-4.2-3b" \
    --output_dir "${DATA_DIR}/activations" \
    --layers 20 \
    --max_samples 20000

echo "=== [2/2] Training Continuativo NLA da Checkpoint Epoca 3 (40.000 vettori, 3 epoche) ==="
python cluster/train_nla.py \
    --shards_dir "${DATA_DIR}/activations" \
    --output_dir "${CKPT_DIR}" \
    --resume_from "${PREV_CKPT}" \
    --model "ibm-granite/granite-4.2-3b" \
    --layer "layer_20" \
    --epochs 3 \
    --batch_size 8 \
    --lr 5e-6

echo "================================================================="
echo "Job Extended NLA 40k completato con successo!"
echo "Data fine: $(date)"
echo "================================================================="
