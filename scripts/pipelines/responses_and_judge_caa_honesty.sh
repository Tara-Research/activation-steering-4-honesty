#!/bin/bash
#SBATCH --job-name=caa_resp_and_judge
#SBATCH --time=2:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=80G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# Combined CAA honesty response generation + LLM judge in one SLURM job.
#
# Phase 1: build a non-vllm venv (config_env.sh), generate responses.
# Phase 2: rebuild the venv with vllm (config_env_vllm.sh), run the judge
#          which mutates the responses JSON in place.
#
# Both env scripts write to $SLURM_TMPDIR/env, so the second call overwrites
# the first. Same pattern as scripts/steering/run_test_steering_and_judge_honesty.sh.
#
# Usage:
#   sbatch scripts/pipelines/responses_and_judge_caa_honesty.sh \
#     --model-id google/gemma-4-31B-it \
#     --n-responses 5 \
#     --input-json data/honesty/CAA/self_report_expanded/self_report_scenarios.json \
#     --output-json data/honesty/CAA/self_report_expanded/self_report_responses_google_gemma-4-31B-it.json

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

# ── Parse named arguments ─────────────────────────────────────────────
MODEL_ID=""
N_RESPONSES=""
INPUT_JSON=""    # scenarios JSON (input to response generation)
OUTPUT_JSON=""   # responses JSON (output of gen, input to judge)
DATASET=""       # only needed if you want auto path-detection in either script

# Optional pass-through to each phase
GEN_EXTRA=()
JUDGE_EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id)      MODEL_ID="$2";       shift 2 ;;
        --n-responses)   N_RESPONSES="$2";    shift 2 ;;
        --input-json)    INPUT_JSON="$2";     shift 2 ;;
        --output-json)   OUTPUT_JSON="$2";    shift 2 ;;
        --dataset)       DATASET="$2";        shift 2 ;;
        --gen-extra)     GEN_EXTRA+=("$2");   shift 2 ;;
        --judge-extra)   JUDGE_EXTRA+=("$2"); shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: sbatch $0 --model-id <id> --input-json <scenarios.json> --output-json <responses.json> [--n-responses <N>] [--dataset <name>] [--gen-extra <arg>]... [--judge-extra <arg>]..." >&2
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_ID" || -z "$INPUT_JSON" || -z "$OUTPUT_JSON" ]]; then
    echo "Error: --model-id, --input-json, --output-json are all required." >&2
    exit 1
fi

REPO_ROOT="$(pwd)"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "======== JOB START ========"
echo "Date:        $(date)"
echo "Model:       $MODEL_ID"
echo "Scenarios:   $INPUT_JSON"
echo "Responses:   $OUTPUT_JSON"
echo "n_responses: ${N_RESPONSES:-<gen default>}"

# ── Build args for each phase ────────────────────────────────────────
GEN_ARGS=(
    --model-id "$MODEL_ID"
    --input-json "$INPUT_JSON"
    --output-json "$OUTPUT_JSON"
)
if [[ -n "$N_RESPONSES" ]]; then
    GEN_ARGS+=(--n-responses "$N_RESPONSES")
fi
if [[ -n "$DATASET" ]]; then
    GEN_ARGS+=(--dataset "$DATASET")
fi
GEN_ARGS+=("${GEN_EXTRA[@]+"${GEN_EXTRA[@]}"}")

JUDGE_ARGS=(
    --input-json "$OUTPUT_JSON"
    --model-id "$MODEL_ID"
)
if [[ -n "$DATASET" ]]; then
    JUDGE_ARGS+=(--dataset "$DATASET")
fi
JUDGE_ARGS+=("${JUDGE_EXTRA[@]+"${JUDGE_EXTRA[@]}"}")

# ── Phase 1: response generation (no-vllm venv) ──────────────────────
echo ""
echo "======== Phase 1: setting up non-vllm env ========"
srun --ntasks=1 --exact scripts/infrastructure/config_env.sh
source "$SLURM_TMPDIR/env/bin/activate"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"

echo ""
echo "======== Phase 1: generating responses ========"
srun python -m src.vector_generation.generate_responses --trait honesty "${GEN_ARGS[@]}"

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
srun python -m src.llm_judge.runners.caa_dataset --trait honesty "${JUDGE_ARGS[@]}"

echo ""
echo "======== JOB DONE ========"
echo "Date: $(date)"
