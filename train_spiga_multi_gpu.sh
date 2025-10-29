#!/usr/bin/env bash
#
# SPIGA Multi-GPU Training Script
# Distributed training across 8 GPUs using SLURM and Singularity
# 
# This script uses srun to launch distributed training across multiple GPUs.
# SLURM automatically sets RANK, WORLD_SIZE, MASTER_ADDR, and MASTER_PORT.
#

#SBATCH --job-name=train_spiga_multigpu
#SBATCH --partition=prioritized
#SBATCH --nodelist=a768-l40s-05
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --ntasks=8
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --output=spiga_train.out

# ============================================================================
# Setup paths and environment
# ============================================================================

cd /home/create.aau.dk/az66ep/UMBRAL/SPIGA

# ============================================================================
# W&B Configuration
# ============================================================================
export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_PROJECT="train_SPIGA"
export WANDB_RUN_NAME="train_SPIGA_8GPU"
export WANDB_ENTITY="joaomalves"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500


# ============================================================================
# Python and Torch Configuration
# ============================================================================
export PYTHONPATH=/home/create.aau.dk/az66ep/UMBRAL/SPIGA:$PYTHONPATH
export TORCH_HOME=/home/create.aau.dk/az66ep/UMBRAL/SPIGA/.cache/torch

# Set NCCL environment variables for multi-node
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_P2P_LEVEL=NVL
export NCCL_BUFFSIZE=2097152
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
        train_distributed.py \
        --dataset wflw \
        --stage all \
        --batch_size 24 \
        --lr_stage1 1e-3 \
        --checkpoint_dir ./checkpoints \
        --log_dir ./logs \
"
