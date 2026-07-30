#!/bin/bash
#SBATCH --job-name=mmlu_benchmark
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=20G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# -------------- Parameters (override via CLI args or env vars) --------------
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-id)       MODEL_ID="$2";       shift 2 ;;
    --trait)          TRAIT="$2";           shift 2 ;;
    --steering-type)  STEERING_TYPE="$2";   shift 2 ;;
    --honesty-category) HONESTY_CATEGORY="$2"; shift 2 ;;
    --exp_name)       EXP_NAME="$2";        shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.3-70B-Instruct}"
TRAIT="${TRAIT:-honesty}"
STEERING_TYPE="${STEERING_TYPE:-all}"
EXP_NAME="${EXP_NAME:-default}"

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

srun scripts/infrastructure/config_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

source $SLURM_TMPDIR/env/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# -------------- Actual command --------------
HONESTY_ARGS=()
if [[ -n "${HONESTY_CATEGORY:-}" ]]; then
  HONESTY_ARGS=(--honesty-category "$HONESTY_CATEGORY")
fi

srun python src/benchmarking/mmlu/lm_eval_mmlu.py \
  --model-id "$MODEL_ID" \
  --trait "$TRAIT" \
  --steering-type "$STEERING_TYPE" \
  --exp_name "$EXP_NAME" \
  "${HONESTY_ARGS[@]}"
