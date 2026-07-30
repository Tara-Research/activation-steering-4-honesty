#!/bin/bash
#SBATCH --job-name=steering_exp
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=80G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#
# CAA extraction + logistic-regression vector generation (both traits).
#
# Step 1: runs the unified CAA extractor, which saves the response-average
#         steering vector and per-side embeddings.
# Step 2: fits the logistic-regression vector on those embeddings.
#
# Usage:
#   sbatch scripts/vector_generation/generate_vec_and_logreg.sh \
#     --model-id meta-llama/Llama-3.3-70B-Instruct \
#     --trait honesty \
#     --dataset self_report_experimental \
#     [--input-json /path/to/responses.json]      # optional; otherwise derived
#     [--normalization-method projection]         # LR normalization (none|projection)
#     [--per-token-layer 26]                      # also save ragged per-token response
#                                                 # embeddings for this layer (PCA/histograms)
#     [--extract-extra "--no-filter-by-judge"]    # forward extra args to Step 1
#     [--logreg-extra "--C 0.01"]                 # forward extra args to Step 2

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

srun scripts/infrastructure/config_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

source $SLURM_TMPDIR/env/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# ── Parse named arguments ─────────────────────────────────────────────
MODEL_ID=""
TRAIT=""
DATASET=""
INPUT_JSON=""
NORMALIZATION_METHOD="projection"
PER_TOKEN_LAYER=""

EXTRACT_EXTRA=()
LOGREG_EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-id)                MODEL_ID="$2";              shift 2 ;;
        --trait)                   TRAIT="$2";                 shift 2 ;;
        --dataset)                 DATASET="$2";               shift 2 ;;
        --input-json)              INPUT_JSON="$2";            shift 2 ;;
        --normalization-method)    NORMALIZATION_METHOD="$2";  shift 2 ;;
        --per-token-layer)         PER_TOKEN_LAYER="$2";       shift 2 ;;
        --extract-extra)           EXTRACT_EXTRA+=("$2");      shift 2 ;;
        --logreg-extra)            LOGREG_EXTRA+=("$2");       shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: sbatch $0 --model-id <id> --trait honesty|compassion --dataset <dataset> [--input-json <path>] [--normalization-method none|projection] [--per-token-layer <N>]" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_ID" || -z "$TRAIT" || -z "$DATASET" ]]; then
    echo "Error: --model-id, --trait, --dataset are required." >&2
    exit 1
fi

# Embedding side names match the extractor's per-trait naming.
case "$TRAIT" in
    honesty)    POS_GROUP="honest";   NEG_GROUP="dishonest" ;;
    compassion) POS_GROUP="positive"; NEG_GROUP="negative" ;;
    *)
        echo "Error: --trait must be honesty or compassion (got '$TRAIT')." >&2
        exit 1 ;;
esac

# Mirror the path-derivation logic from generate_vec_caa.py / constants.py
PERSONA_VECTORS_FOLDER="${PERSONA_VECTORS_FOLDER:-/project/aip-irina/shared/steering/persona_vectors}"
MODEL_SLUG="${MODEL_ID//\//_}"
CAA_SAVE_DIR="${PERSONA_VECTORS_FOLDER}/caa/${MODEL_SLUG}/${TRAIT}/${DATASET}"

# -------------- Step 1: CAA extraction --------------
echo "=== Step 1: CAA extraction ($TRAIT / $DATASET) ==="
EXTRACT_ARGS=(
    --trait "$TRAIT"
    --dataset "$DATASET"
    --model-id "$MODEL_ID"
)
if [[ -n "$INPUT_JSON" ]]; then
    EXTRACT_ARGS+=(--input-json "$INPUT_JSON")
fi
if [[ -n "$PER_TOKEN_LAYER" ]]; then
    EXTRACT_ARGS+=(--per-token-layer "$PER_TOKEN_LAYER")
fi
EXTRACT_ARGS+=("${EXTRACT_EXTRA[@]+"${EXTRACT_EXTRA[@]}"}")

srun python -m src.vector_generation.generate_vec_caa "${EXTRACT_ARGS[@]}"

# -------------- Step 2: Logistic regression vector --------------
echo "=== Step 2: Logistic regression fit ==="
POS_FILE="${CAA_SAVE_DIR}/${TRAIT}_${DATASET}_${POS_GROUP}_embeddings_response_avg_${MODEL_SLUG}.pt"
NEG_FILE="${CAA_SAVE_DIR}/${TRAIT}_${DATASET}_${NEG_GROUP}_embeddings_response_avg_${MODEL_SLUG}.pt"

srun python -m src.vector_generation.generate_vec_logistic_regression \
    --pos-features "$POS_FILE" \
    --neg-features "$NEG_FILE" \
    --trait "$TRAIT" \
    --dataset "$DATASET" \
    --model-id "$MODEL_ID" \
    --normalization-method "$NORMALIZATION_METHOD" \
    "${LOGREG_EXTRA[@]+"${LOGREG_EXTRA[@]}"}"

echo "=== Done ==="
