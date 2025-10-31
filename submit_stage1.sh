#!/usr/bin/env bash
#
# SPIGA Stage 1: CNN Backbone Pre-training
# Distributed training with DDP and SLURM
#
# Submit with: sbatch submit_stage1.sh

#SBATCH --job-name=spiga_stage1
#SBATCH --partition=prioritized
#SBATCH --nodelist=a768-l40s-06
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=spiga_stage1.out

# ============================================================================
# Setup paths and environment
# ============================================================================

cd /home/create.aau.dk/az66ep/UMBRAL/SPIGA || exit 1

# ============================================================================
# W&B Configuration
# ============================================================================
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_PROJECT="${WANDB_PROJECT:-train_SPIGA}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-spiga_stage1_${SLURM_JOB_ID}}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# ============================================================================
# Python and Torch Configuration
# ============================================================================
export PYTHONPATH=/home/create.aau.dk/az66ep/UMBRAL/SPIGA:$PYTHONPATH
export TORCH_HOME=/home/create.aau.dk/az66ep/UMBRAL/SPIGA/.cache/torch

# Set NCCL environment variables for multi-GPU training
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_BUFFSIZE=2097152
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# Training Configuration
# ============================================================================
DATASET="wflw"
CHECKPOINT_DIR="./checkpoints"
LOG_DIR="./logs"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "SPIGA Stage 1: CNN Backbone Pre-training"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Epochs: 450"
echo "Learning Rate: 1e-3"
echo "Batch Size: 12 per GPU"
echo "GPUs: ${SLURM_NTASKS}"
echo "Master Addr: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"
echo "=========================================="

# ============================================================================
# Launch distributed training
# ============================================================================

srun --ntasks=8 --ntasks-per-node=8 bash -c " singularity exec \
    --nv \
    --env TORCH_HOME=$TORCH_HOME \
    --env PYTHONPATH=$PYTHONPATH \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_PROJECT=$WANDB_PROJECT \
    --env WANDB_RUN_NAME=$WANDB_RUN_NAME \
    --env WANDB_ENTITY=$WANDB_ENTITY \
    --env MASTER_ADDR=$MASTER_ADDR \
    --env MASTER_PORT=$MASTER_PORT \
    --env NCCL_DEBUG=$NCCL_DEBUG \
    --env NCCL_IB_DISABLE=$NCCL_IB_DISABLE \
    --env NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
    --env NCCL_P2P_LEVEL=$NCCL_P2P_LEVEL \
    --env NCCL_BUFFSIZE=$NCCL_BUFFSIZE \
    --env CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING \
    --env OMP_NUM_THREADS=$OMP_NUM_THREADS \
    --env PYTHONUNBUFFERED=$PYTHONUNBUFFERED \
    --env PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF \
    spiga_pytorch22.04.sif torchrun \
        --nnodes=1 \
        --nproc-per-node=8 \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train_stage1.py \
        --dataset "${DATASET}" \
        --epochs_stage1 450 \
        --lr_stage1 1e-3 \
        --batch_size 12 \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        --log_dir "${LOG_DIR}" \
        --save_interval 50
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Stage 1 training completed!"
    echo "Best model: ${CHECKPOINT_DIR}/checkpoint_stage1/best_model.pth"
else
    echo "ERROR: Stage 1 training failed!"
    exit 1
fi
