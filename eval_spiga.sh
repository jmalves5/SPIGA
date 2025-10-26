#!/usr/bin/env bash
#SBATCH --job-name=eval_spiga
#SBATCH --partition=prioritized
#SBATCH --nodelist=a512-l4-06
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00

# Setup wandb env variables
export WANDB_API_KEY=$WANDB_API_KEY

# Point torch cache to your writable SPIGA folder
export TORCH_HOME=/home/create.aau.dk/az66ep/UMBRAL/SPIGA/.cache/torch
mkdir -p $TORCH_HOME/hub/checkpoints

# Dataset should be a list of strings
#DATASET_LIST=('wflw' '300wpublic' '300wprivate' "merlrav" "cofw68")
DATASET_LIST=('wflw' '300wpublic' '300wprivate' "cofw68")

cd /home/create.aau.dk/az66ep/UMBRAL/SPIGA

# Run results gen and evaluator for each dataset
for DATASET in "${DATASET_LIST[@]}"; do
    export WANDB_PROJECT="eval_SPIGA"
    export WANDB_RUN_NAME="eval_SPIGA_$DATASET"
    export WANDB_ENTITY="joaomalves"

    echo "Generating results for $DATASET"

    srun singularity exec \
        --nv \
        --env TORCH_HOME=/home/create.aau.dk/az66ep/UMBRAL/SPIGA/.cache/torch \
        --env PYTHONPATH=/home/create.aau.dk/az66ep/UMBRAL/SPIGA \
        --env WANDB_API_KEY=$WANDB_API_KEY \
        --env WANDB_PROJECT=$WANDB_PROJECT \
        --env WANDB_RUN_NAME=$WANDB_RUN_NAME \
        --env WANDB_ENTITY=$WANDB_ENTITY \
        --env DATASET=$DATASET \
        spiga_pytorch22.04.sif \
        python3 spiga/eval/results_gen.py ${DATASET}

    echo "Evaluating results for $DATASET"
    srun singularity exec \
        --nv \
        --env TORCH_HOME=/home/create.aau.dk/az66ep/UMBRAL/SPIGA/.cache/torch \
        --env PYTHONPATH=/home/create.aau.dk/az66ep/UMBRAL/SPIGA \
        --env WANDB_API_KEY=$WANDB_API_KEY \
        --env WANDB_PROJECT=$WANDB_PROJECT \
        --env WANDB_RUN_NAME=$WANDB_RUN_NAME \
        --env WANDB_ENTITY=$WANDB_ENTITY \
        --env DATASET=$DATASET \
        spiga_pytorch22.04.sif \
        python3 spiga/eval/benchmark/evaluator.py spiga/eval/results/results_${DATASET}_test.json --eval lnd pose -s --log_wandb
done
