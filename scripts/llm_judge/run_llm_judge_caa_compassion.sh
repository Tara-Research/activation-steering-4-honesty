#!/bin/bash
# Usage: sbatch scripts/llm_judge/run_llm_judge_caa_compassion.sh [MODEL_ID]
#   MODEL_ID defaults to meta-llama/Llama-3.3-70B-Instruct
#SBATCH --job-name=llm_judge_caa_compassion
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h200:8
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

MODEL_ID="${1:-meta-llama/Llama-3.3-70B-Instruct}"

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

srun scripts/infrastructure/config_env_vllm.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

source $SLURM_TMPDIR/env/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# -------------- Actual command --------------
srun python -m src.llm_judge.runners.caa_dataset --trait compassion --model-id "${MODEL_ID}"
