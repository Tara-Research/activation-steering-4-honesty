#!/bin/bash
#SBATCH --job-name=mtb_eval
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
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
# Required env vars: MODEL_ID, TRAIT, TECHNIQUES (space-separated), EXP_NAME
# Optional env vars: SAVE_EVERY (default 10)
SAVE_EVERY_ARG=""
if [[ -n "${SAVE_EVERY:-}" ]]; then
    SAVE_EVERY_ARG="--save_every ${SAVE_EVERY}"
fi

srun python -m src.benchmarking.mt_bench.evaluate \
    --model_id "${MODEL_ID}" \
    --trait "${TRAIT}" \
    --techniques ${TECHNIQUES} \
    --exp_name "${EXP_NAME}" \
    ${SAVE_EVERY_ARG}
