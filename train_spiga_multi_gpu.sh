#!/usr/bin/env bash
#
# SPIGA Multi-GPU Training Script
# Distributed training across 4 GPUs using SLURM and Singularity
# 
# This script uses srun to launch distributed training across multiple GPUs.
# SLURM automatically sets RANK, WORLD_SIZE, MASTER_ADDR, and MASTER_PORT.
#

#SBATCH --job-name=train_spiga_multigpu
#SBATCH --partition=prioritized
#SBATCH --nodelist=i256-a10-10
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
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
export WANDB_RUN_NAME="train_SPIGA_4GPU"
export WANDB_ENTITY="joaomalves"



export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# ============================================================================
# Python and Torch Configuration
# ============================================================================
export PYTHONPATH=/home/create.aau.dk/az66ep/UMBRAL/SPIGA:$PYTHONPATH
export TORCH_HOME=/home/create.aau.dk/az66ep/UMBRAL/SPIGA/.cache/torch
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set NCCL environment variables for multi-node
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c " singularity exec \
    --nv \
    --env TORCH_HOME=$TORCH_HOME \
    --env PYTHONPATH=$PYTHONPATH \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_PROJECT=$WANDB_PROJECT \
    --env WANDB_RUN_NAME=$WANDB_RUN_NAME \
    --env WANDB_ENTITY=$WANDB_ENTITY \
    --env MASTER_ADDR=$MASTER_ADDR \
    --env MASTER_PORT=$MASTER_PORT \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    spiga_pytorch22.04.sif torchrun \
        --node_rank=\$SLURM_PROCID \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        train_distributed.py \
        --dataset wflw \
        --stage all \
        --batch_size 12 \
        --lr_stage1 5e-4 \
        --use_amp \
        --checkpoint_dir ./checkpoints \
        --log_dir ./logs 
"
