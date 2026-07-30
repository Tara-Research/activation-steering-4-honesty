#!/bin/bash
# Launch sequential honesty steering configurations as parallel SLURM jobs,
# then submit LLM judge jobs that depend on the steering jobs.
#
# Pipeline: 3 dual-mode steering + 2 baseline = 5 steering jobs → 4 judge jobs (9 total)
#   swfc:        dual (resp + all_token) → judge
#   sttp:        dual (resp + all_token) → judge
#   stmp:        dual (resp + all_token) → judge
#   no_steering: aligned + malicious     → judge
#
# By default all techniques run in dual mode (both response_mode + all_token_mode).
# Use --no_dual to run a single steering mode instead, and --steering_mode to pick
# which one. Use --techniques to select a subset of techniques/baselines to run.
#
# Usage:
#   bash scripts/steering/launch_sequential_honesty_all_configs.sh
#   bash scripts/steering/launch_sequential_honesty_all_configs.sh --include_raw_output
#   bash scripts/steering/launch_sequential_honesty_all_configs.sh --model_id meta-llama/Llama-3.3-70B-Instruct
#   bash scripts/steering/launch_sequential_honesty_all_configs.sh --no_dual                              # single mode (response_mode)
#   bash scripts/steering/launch_sequential_honesty_all_configs.sh --no_dual --steering_mode all_token_mode
#   bash scripts/steering/launch_sequential_honesty_all_configs.sh --techniques swfc,stmp                 # subset of techniques
#   bash scripts/steering/launch_sequential_honesty_all_configs.sh --techniques aligned_baseline,malicious_baseline
#   bash scripts/steering/launch_sequential_honesty_all_configs.sh --test_mode  # 5 layers only, reduced SLURM times
set -euo pipefail

HONESTY_CATEGORY="self_report_experimental"
MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"
INCLUDE_RAW_OUTPUT=false
TEST_MODE=false
EXP_NAME=""
SKIP_GENERATION=false
DUAL_MODE=true
STEERING_MODE="response_mode"      # only used when DUAL_MODE=false
TECHNIQUES="swfc,sttp,stmp,aligned_baseline,malicious_baseline"

USAGE="Usage: $0 [--honesty_category self_report_experimental] [--model_id MODEL] [--include_raw_output] [--test_mode] [--exp_name NAME] [--skip_generation] [--no_dual] [--steering_mode response_mode|all_token_mode] [--techniques swfc,sttp,stmp,aligned_baseline,malicious_baseline]"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --honesty_category)
            HONESTY_CATEGORY="$2"
            shift 2
            ;;
        --model_id)
            MODEL_ID="$2"
            shift 2
            ;;
        --include_raw_output)
            INCLUDE_RAW_OUTPUT=true
            shift
            ;;
        --test_mode)
            TEST_MODE=true
            shift
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --skip_generation)
            SKIP_GENERATION=true
            shift
            ;;
        --no_dual)
            DUAL_MODE=false
            shift
            ;;
        --steering_mode)
            STEERING_MODE="$2"
            shift 2
            ;;
        --techniques)
            TECHNIQUES="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "$USAGE"
            exit 1
            ;;
    esac
done

if [[ "$DUAL_MODE" == "true" && "$STEERING_MODE" != "response_mode" ]]; then
    echo "WARNING: --steering_mode '$STEERING_MODE' ignored in dual mode (use --no_dual to run a single mode)"
fi

# -------------- Parse + validate --techniques --------------
# Steering techniques get their own steering + judge job each; baselines share a single
# judge over the 'no_steering' subdir. Lookup maps preserve the historical job names.
declare -A STEER_JOBNAME=([swfc]="seq_hon_fixed" [sttp]="seq_hon_dist" [stmp]="seq_hon_mirror")
declare -A STEER_JUDGENAME=([swfc]="judge_hon_fixed" [sttp]="judge_hon_dist" [stmp]="judge_hon_mirror")

STEER_TECHS=()
BASELINE_TECHS=()
IFS=',' read -ra _REQUESTED_TECHS <<< "$TECHNIQUES"
for tech in "${_REQUESTED_TECHS[@]}"; do
    case "$tech" in
        swfc|sttp|stmp)
            STEER_TECHS+=("$tech")
            ;;
        aligned_baseline|malicious_baseline)
            BASELINE_TECHS+=("$tech")
            ;;
        "")
            ;;  # tolerate empty tokens from trailing/double commas
        *)
            echo "ERROR: unknown technique '$tech' in --techniques"
            echo "Valid techniques: swfc, sttp, stmp, aligned_baseline, malicious_baseline"
            exit 1
            ;;
    esac
done

if [[ ${#STEER_TECHS[@]} -eq 0 && ${#BASELINE_TECHS[@]} -eq 0 ]]; then
    echo "ERROR: no valid techniques selected via --techniques '$TECHNIQUES'"
    exit 1
fi

MODEL_SAVE_NAME="${MODEL_ID//\//_}"

STEER_SCRIPT="scripts/steering/sequential_steer_over_all_layers_honesty.sh"
JUDGE_SCRIPT="scripts/llm_judge/run_llm_judge_honesty_universal.sh"

for script in "$STEER_SCRIPT" "$JUDGE_SCRIPT"; do
    if [[ ! -f "$script" ]]; then
        echo "ERROR: script not found: $script"
        exit 1
    fi
done

mkdir -p logs

RESULTS_BASE="results/logistic_regression/honesty/${MODEL_SAVE_NAME}/all_layers/${HONESTY_CATEGORY}"
ALL_JOB_IDS=()

# --- Phase 1: Submit steering jobs ---

echo "Model: ${MODEL_ID} (save name: ${MODEL_SAVE_NAME})"

# submit_steer sets _JOB_ID as a side effect (avoids subshell issues with arrays)
submit_steer() {
    local job_name="$1"
    shift
    local sbatch_extra=()
    local script_extra=()
    if [[ "$TEST_MODE" == "true" ]]; then
        sbatch_extra+=(--time=02:00:00)
        script_extra+=(--test_mode)
    fi
    if [[ -n "$EXP_NAME" ]]; then
        script_extra+=(--exp_name "$EXP_NAME")
    fi
    local output
    output=$(sbatch --job-name="$job_name" "${sbatch_extra[@]}" "$STEER_SCRIPT" "$@" --honesty_category "$HONESTY_CATEGORY" --model_id "$MODEL_ID" "${script_extra[@]}")
    _JOB_ID=$(echo "$output" | awk '{print $NF}')
    ALL_JOB_IDS+=("$_JOB_ID")
    printf "  %-28s  JobID: %s\n" "$job_name" "$_JOB_ID"
}

# Per-technique steering job IDs (assoc array keyed by technique) and the list of
# baseline steering job IDs. Initialized empty so the judge dependency code (which
# tolerates empty/colon-only deps) works under --skip_generation.
declare -A STEER_JID
for tech in "${STEER_TECHS[@]}"; do STEER_JID[$tech]=""; done
BASELINE_JIDS=()

# Build the dual / single-mode args once.
MODE_ARGS=()
if [[ "$DUAL_MODE" == "true" ]]; then
    MODE_ARGS=(--dual_mode)
    MODE_SUFFIX="dual"
else
    MODE_ARGS=(--steering_mode "$STEERING_MODE")
    MODE_SUFFIX="$STEERING_MODE"
fi

if [[ "$SKIP_GENERATION" == "true" ]]; then
    echo "=== Phase 1: SKIPPED (--skip_generation) ==="
    echo ""
else
    echo "=== Phase 1: Steering jobs (category=${HONESTY_CATEGORY}, mode=${MODE_SUFFIX}) ==="
    echo ""

    # Steering techniques (one job each, dual or single mode)
    for tech in "${STEER_TECHS[@]}"; do
        submit_steer "${STEER_JOBNAME[$tech]}_${MODE_SUFFIX}" --steering_technique "$tech" "${MODE_ARGS[@]}"
        STEER_JID[$tech]=$_JOB_ID
    done

    # Baselines (one job each, never dual-mode)
    for tech in "${BASELINE_TECHS[@]}"; do
        submit_steer "seq_hon_${tech%_baseline}" --steering_technique "$tech"
        BASELINE_JIDS+=("$_JOB_ID")
    done
fi

# --- Phase 2: Submit judge jobs (depend on steering pairs) ---

echo ""
echo "=== Phase 2: Judge jobs (depend on steering pairs) ==="
echo ""

submit_judge() {
    local job_name="$1"
    local dep_ids="$2"
    local results_subdir="$3"
    local extra_args=""
    extra_args="--honesty_category=${HONESTY_CATEGORY}"
    if [[ "$INCLUDE_RAW_OUTPUT" == "true" ]]; then
        extra_args="$extra_args --include_raw_output"
    fi
    local sbatch_extra=()
    if [[ "$TEST_MODE" == "true" ]]; then
        sbatch_extra+=(--time=01:00:00)
    fi
    # Strip leading/trailing colons (left over when some JID_* are empty under
    # --skip_generation, e.g. ":12345" or "12345:"), and treat an empty/colons-only
    # string as "no dependency".
    local clean_dep_ids="${dep_ids#:}"
    clean_dep_ids="${clean_dep_ids%:}"
    local dep_args=()
    if [[ -n "$clean_dep_ids" && "$clean_dep_ids" != ":" ]]; then
        dep_args+=(--dependency="afterok:${clean_dep_ids}")
    fi
    local output
    output=$(sbatch \
        --job-name="$job_name" \
        "${dep_args[@]}" \
        "${sbatch_extra[@]}" \
        "$JUDGE_SCRIPT" \
        --results_dir="${RESULTS_BASE}/${results_subdir}" \
        $extra_args)
    local job_id
    job_id=$(echo "$output" | awk '{print $NF}')
    _JOB_ID=$job_id
    ALL_JOB_IDS+=("$job_id")
    printf "  %-28s  JobID: %s  (after %s)\n" "$job_name" "$job_id" "${clean_dep_ids:-<none>}"
}

JUDGE_JIDS=()

# One judge per steering technique, over its own results subdir (named after the technique).
for tech in "${STEER_TECHS[@]}"; do
    submit_judge "${STEER_JUDGENAME[$tech]}" "${STEER_JID[$tech]}" "$tech"
    JUDGE_JIDS+=("$_JOB_ID")
done

# Single baseline judge over the shared 'no_steering' subdir (depends on both baselines).
if [[ ${#BASELINE_TECHS[@]} -gt 0 ]]; then
    baseline_dep=$(IFS=':'; echo "${BASELINE_JIDS[*]}")
    submit_judge "judge_hon_baseline" "$baseline_dep" "no_steering"
    JUDGE_JIDS+=("$_JOB_ID")
fi

echo ""
echo "=== Submitted ${#ALL_JOB_IDS[@]} jobs (${#STEER_TECHS[@]} steering + ${#BASELINE_TECHS[@]} baseline + ${#JUDGE_JIDS[@]} judge) ==="
echo "Monitor: squeue -u \$USER"
echo "Cancel all: scancel ${ALL_JOB_IDS[*]}"
