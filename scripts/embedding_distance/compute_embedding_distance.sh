#!/bin/bash
#SBATCH --job-name=emb_distance
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=h100:4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

srun scripts/infrastructure/config_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

source $SLURM_TMPDIR/env/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/n/niklas/persona_playground

# -------------- Actual command --------------
srun python src/embedding_distance/compute_embedding_distance.py "$@"
