#!/bin/bash
#SBATCH --job-name=caa_resp_and_judge_compassion
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h200:8
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Combined CAA compassion response generation + LLM judge in one SLURM job.
# Compassion analogue of scripts/pipelines/responses_and_judge_caa_honesty.sh.
#
# Phase 1: build a non-vllm venv (config_env.sh), generate responses.
# Phase 2: rebuild the venv with vllm (config_env_vllm.sh), run the judge
#          which mutates the responses JSON in place.
#
# Both env scripts write to $SLURM_TMPDIR/env, so the second call overwrites
# the first. Same pattern as responses_and_judge_caa_honesty.sh.
#
# --model-id is forwarded to BOTH phases. The compassion entry points derive
# their input/output JSON paths from the model id, so Phase 1's output
#   data/compassion/CAA/conversation_data_compassion_generated_<model>.json
# is exactly what Phase 2 reads back in place — no path args needed.
#
# Usage:
#   sbatch scripts/pipelines/responses_and_judge_caa_compassion.sh \
#     [--model-id meta-llama/Llama-3.3-70B-Instruct] \
#     [--gen-extra <arg>]... [--judge-extra <arg>]...

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

# ── Parse named arguments ─────────────────────────────────────────────
MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"

# Optional pass-through to each phase
GEN_EXTRA=()
JUDGE_EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id)      MODEL_ID="$2";       shift 2 ;;
        --gen-extra)     GEN_EXTRA+=("$2");   shift 2 ;;
        --judge-extra)   JUDGE_EXTRA+=("$2"); shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: sbatch $0 [--model-id <id>] [--gen-extra <arg>]... [--judge-extra <arg>]..." >&2
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_ID" ]]; then
    echo "Error: --model-id must not be empty." >&2
    exit 1
fi

REPO_ROOT="$(pwd)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "======== JOB START ========"
echo "Date:        $(date)"
echo "Model:       $MODEL_ID"

# ── Build args for each phase ────────────────────────────────────────
GEN_ARGS=(
    --model-id "$MODEL_ID"
)
GEN_ARGS+=("${GEN_EXTRA[@]+"${GEN_EXTRA[@]}"}")

JUDGE_ARGS=(
    --model-id "$MODEL_ID"
)
JUDGE_ARGS+=("${JUDGE_EXTRA[@]+"${JUDGE_EXTRA[@]}"}")

# ── Phase 1: response generation (no-vllm venv) ──────────────────────
echo ""
echo "======== Phase 1: setting up non-vllm env ========"
srun --ntasks=1 --exact scripts/infrastructure/config_env.sh
source "$SLURM_TMPDIR/env/bin/activate"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"

echo ""
echo "======== Phase 1: generating responses ========"
srun python -m src.vector_generation.generate_responses --trait compassion --n-responses 2 "${GEN_ARGS[@]+"${GEN_ARGS[@]}"}"

deactivate || true

# ── Phase 2: LLM judge (vllm venv) ───────────────────────────────────
echo ""
echo "======== Phase 2: setting up vllm env ========"
srun --ntasks=1 --exact scripts/infrastructure/config_env_vllm.sh
source "$SLURM_TMPDIR/env/bin/activate"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"
export CUDA_HOME=$EBROOTCUDA
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-/scratch/n/niklas/tiktoken_encodings}"

echo ""
echo "======== Phase 2: running LLM judge ========"
srun python -m src.llm_judge.runners.caa_dataset --trait compassion "${JUDGE_ARGS[@]}"

echo ""
echo "======== JOB DONE ========"
echo "Date: $(date)"
