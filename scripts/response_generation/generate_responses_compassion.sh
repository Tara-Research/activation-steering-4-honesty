#!/bin/bash
# Usage: sbatch scripts/response_generation/generate_responses_compassion.sh [MODEL_ID]
#   MODEL_ID defaults to meta-llama/Llama-3.3-70B-Instruct
#SBATCH --job-name=steering_exp
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

MODEL_ID="${1:-meta-llama/Llama-3.3-70B-Instruct}"

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

srun scripts/infrastructure/config_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

source $SLURM_TMPDIR/env/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# -------------- Actual command --------------
srun python -m src.vector_generation.generate_responses --trait compassion --model-id "${MODEL_ID}" --n-responses 2
