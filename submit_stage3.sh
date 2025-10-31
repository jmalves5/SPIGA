#!/usr/bin/env bash
#
# SPIGA Stage 3: GAT Regressor Training (Backbone Frozen)
# Distributed training with DDP and SLURM
#
# Submit with: sbatch submit_stage3.sh
# Requires: Stage 2 checkpoint (checkpoints/checkpoint_stage2/best_model.pth)

#SBATCH --job-name=spiga_stage3
#SBATCH --partition=prioritized
#SBATCH --nodelist=a768-l40s-06
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=spiga_stage3.out


# ============================================================================
# Setup paths and environment
# ============================================================================

cd /home/create.aau.dk/az66ep/UMBRAL/SPIGA || exit 1

# ============================================================================
# W&B Configuration
# ============================================================================
export WANDB_API_KEY=${WANDB_API_KEY:-}
export WANDB_PROJECT="${WANDB_PROJECT:-train_SPIGA}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-spiga_stage3_${SLURM_JOB_ID}}"
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
STAGE2_MODEL="${CHECKPOINT_DIR}/checkpoint_stage2/best_model.pth"

mkdir -p "${LOG_DIR}"

# Verify Stage 2 checkpoint exists
if [ ! -f "${STAGE2_MODEL}" ]; then
    echo "ERROR: Stage 2 checkpoint not found at ${STAGE2_MODEL}"
    echo "Please run Stage 2 training first!"
    exit 1
fi

echo "=========================================="
echo "SPIGA Stage 3: GAT Regressor Training"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Epochs: 150"
echo "Learning Rate: 1e-4"
echo "GAT Steps: 3"
echo "Batch Size: 12 per GPU"
echo "GPUs: ${SLURM_NTASKS}"
echo "Master Addr: ${MASTER_ADDR}"
echo "Master Port: ${MASTER_PORT}"
echo "Backbone: FROZEN"
echo "Pretrained: ${STAGE2_MODEL}"
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
        train_stage3.py \
        --dataset "${DATASET}" \
        --epochs_gat 150 \
        --lr_gat 1e-4 \
        --gat_steps 3 \
        --batch_size 12 \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        --log_dir "${LOG_DIR}" \
        --save_interval 50
"

echo "✓ Stage 3 completed successfully"
