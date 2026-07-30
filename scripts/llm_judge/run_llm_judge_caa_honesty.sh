#!/bin/bash
#SBATCH --job-name=steering_exp
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=70G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

srun scripts/infrastructure/config_env_vllm.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

source $SLURM_TMPDIR/env/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

export CUDA_HOME=$EBROOTCUDA
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-/scratch/n/niklas/tiktoken_encodings}"

# -------------- Actual command --------------
srun python -m src.llm_judge.runners.caa_dataset --trait honesty "$@"
