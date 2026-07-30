#!/bin/bash
#SBATCH --job-name=steering_exp
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=50G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

srun scripts/infrastructure/config_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

source $SLURM_TMPDIR/env/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# -------------- Actual command --------------
srun python -m src.vector_generation.generate_vec_logistic_regression "$@"
