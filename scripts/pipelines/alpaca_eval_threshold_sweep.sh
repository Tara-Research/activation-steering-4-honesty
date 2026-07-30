#!/bin/bash
#
# AlpacaEval threshold-alpha SWEEP launcher.
#
# Runs the full AlpacaEval pipeline (gen -> judge) once per neg-cluster
# gating-threshold alpha, so you can see how the conservative StTP/StMP gate
# (m = mu_neg + alpha*std_neg) trades benign over-steering against win-rate.
#
# Each alpha is launched by calling scripts/pipelines/alpaca_eval_pipeline.sh with
# --threshold_mode neg_mean_std --threshold_alpha <a>. The pipeline injects the
# STEERING_THRESHOLD_* env into the generation jobs (read by the steerer via
# src.steering.threshold_config) and TAGS --exp_name per alpha, so the results of
# one alpha never overwrite another's. Optionally also run a
# bias-mode reference pipeline (--include_bias) for an apples-to-apples baseline.
#
# Intended for --test mode (1 prompt/category), where re-running the full pipeline
# per alpha is cheap.
#
# Usage:
#   bash scripts/pipelines/alpaca_eval_threshold_sweep.sh \
#       --model_id meta-llama/Llama-3.3-70B-Instruct \
#       --trait honesty \
#       --honesty_category self_report_experimental \
#       --techniques "no_steering swfc sttp stmp" \
#       --steering_mode all_token_mode \
#       --exp_name v2_alpaca_honesty --alphas "-1 0 1 2 3" \
#       --include_bias --test
#
# All flags except --alphas / --include_bias are forwarded verbatim to
# alpaca_eval_pipeline.sh.
#
set -euo pipefail

# ── Parse named arguments ─────────────────────────────────────────────
MODEL_ID=""
TRAIT=""
TECHNIQUES=""
EXP_NAME=""
STEERING_MODE=""
HONESTY_CATEGORY=""
COEFFICIENTS=""
ALPHAS=""
INCLUDE_BIAS=""
TEST_FLAG=""
SKIP_GEN=""
FORCE_FLAG=""

# Forwarded args (everything the pipeline understands), rebuilt explicitly so this
# wrapper validates its own input and the runbook flag-checker sees the flags.
FWD=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_id)           MODEL_ID="$2";         FWD+=(--model_id "$2");         shift 2 ;;
        --trait)              TRAIT="$2";            FWD+=(--trait "$2");            shift 2 ;;
        --techniques)         TECHNIQUES="$2";       FWD+=(--techniques "$2");       shift 2 ;;
        --exp_name)           EXP_NAME="$2";         FWD+=(--exp_name "$2");         shift 2 ;;
        --steering_mode)      STEERING_MODE="$2";    FWD+=(--steering_mode "$2");    shift 2 ;;
        --honesty_category)   HONESTY_CATEGORY="$2"; FWD+=(--honesty_category "$2"); shift 2 ;;
        --coefficients)       COEFFICIENTS="$2";     FWD+=(--coefficients "$2");     shift 2 ;;
        --test)               TEST_FLAG="1";         FWD+=(--test);                  shift 1 ;;
        --skip_gen)           SKIP_GEN="1";          FWD+=(--skip_gen);              shift 1 ;;
        --force)              FORCE_FLAG="1";        FWD+=(--force);                 shift 1 ;;
        --alphas)             ALPHAS="$2";                                            shift 2 ;;
        --include_bias)       INCLUDE_BIAS="1";                                       shift 1 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$ALPHAS" ]]; then echo "Error: --alphas is required (e.g. --alphas \"-1 0 1 2 3\")"; exit 1; fi
if [[ -z "$EXP_NAME" ]]; then echo "Error: --exp_name is required"; exit 1; fi

PIPELINE="scripts/pipelines/alpaca_eval_pipeline.sh"

echo "=== AlpacaEval threshold-alpha sweep ==="
echo "  base exp_name = ${EXP_NAME}"
echo "  alphas        = ${ALPHAS}"
echo "  include_bias  = ${INCLUDE_BIAS:-no}"
echo ""

# ── Optional bias-mode reference pipeline ─────────────────────────────
if [[ -n "$INCLUDE_BIAS" ]]; then
    echo ">>> launching pipeline: threshold_mode=bias"
    bash "$PIPELINE" "${FWD[@]}" --threshold_mode bias
    echo ""
fi

# ── One pipeline per alpha ────────────────────────────────────────────
read -ra ALPHA_ARRAY <<< "$ALPHAS"
for ALPHA in "${ALPHA_ARRAY[@]}"; do
    echo ">>> launching pipeline: threshold_mode=neg_mean_std alpha=${ALPHA}"
    bash "$PIPELINE" "${FWD[@]}" --threshold_mode neg_mean_std --threshold_alpha "$ALPHA"
    echo ""
done

N_PIPELINES=${#ALPHA_ARRAY[@]}
[[ -n "$INCLUDE_BIAS" ]] && N_PIPELINES=$((N_PIPELINES + 1))
echo "=== sweep launched: ${N_PIPELINES} pipeline(s) ==="
echo "Monitor: squeue -u \$USER"
