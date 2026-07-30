#!/bin/bash
#SBATCH --job-name=alpaca_eval_judge
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=90G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# -------------- Parameters (override via CLI args or env vars) --------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-id)          MODEL_ID="$2";          shift 2 ;;
    --trait)             TRAIT="$2";              shift 2 ;;
    --steering-mode)     STEERING_MODE="$2";      shift 2 ;;
    --exp-name)          EXP_NAME="$2";           shift 2 ;;
    --techniques)        TECHNIQUES="$2";         shift 2 ;;
    --force)             FORCE=1;                 shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.3-70B-Instruct}"
TRAIT="${TRAIT:-compassion}"
STEERING_MODE="${STEERING_MODE:-all_token_mode}"

if [[ -z "${EXP_NAME:-}" ]]; then
  echo "Error: --exp-name is required"
  exit 1
fi

if [[ -z "${TECHNIQUES:-}" ]]; then
  echo "Error: --techniques is required"
  exit 1
fi

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

srun scripts/infrastructure/config_env_vllm.sh

export HF_HOME="${HF_HOME:-/project/aip-irina/shared/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

unset HF_HUB_CACHE
unset TRANSFORMERS_CACHE
unset XDG_CACHE_HOME

source $SLURM_TMPDIR/env/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# -------------- Actual command --------------
FORCE_ARGS=()
if [[ -n "${FORCE:-}" ]]; then
  FORCE_ARGS=(--force)
fi

srun python src/benchmarking/alpaca_eval/alpaca_eval_judge.py \
  --model-id "$MODEL_ID" \
  --trait "$TRAIT" \
  --steering-mode "$STEERING_MODE" \
  --exp_name "$EXP_NAME" \
  --techniques ${TECHNIQUES} \
  "${FORCE_ARGS[@]}"
