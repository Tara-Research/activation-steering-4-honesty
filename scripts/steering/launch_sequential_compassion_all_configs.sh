#!/bin/bash
# Launch sequential compassion steering configurations as parallel SLURM jobs,
# then submit LLM judge jobs that depend on the steering jobs.
#
# Pipeline: 3 dual-mode steering + 2 baseline = 5 steering jobs → 4 judge jobs (9 total)
#   swfc:        dual (resp + all_token) → judge
#   sttp:        dual (resp + all_token) → judge
#   stmp:        dual (resp + all_token) → judge
#   no_steering: aligned + malicious     → judge
#
# Use --techniques to select a subset of techniques/baselines to run.
#
# Usage:
#   bash scripts/steering/launch_sequential_compassion_all_configs.sh
#   bash scripts/steering/launch_sequential_compassion_all_configs.sh --model_id meta-llama/Llama-3.3-70B-Instruct
#   bash scripts/steering/launch_sequential_compassion_all_configs.sh --techniques swfc,stmp                 # subset of techniques
#   bash scripts/steering/launch_sequential_compassion_all_configs.sh --techniques aligned_baseline,malicious_baseline
#   bash scripts/steering/launch_sequential_compassion_all_configs.sh --test_mode  # 5 layers only, reduced SLURM times
set -euo pipefail

MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"
TEST_MODE=false
EXP_NAME=""
SKIP_GENERATION=false
TECHNIQUES="swfc,sttp,stmp,aligned_baseline,malicious_baseline"

USAGE="Usage: $0 [--model_id MODEL_ID] [--test_mode] [--exp_name NAME] [--skip_generation] [--techniques swfc,sttp,stmp,aligned_baseline,malicious_baseline]"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_id)
            MODEL_ID="$2"
            shift 2
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

# -------------- Parse + validate --techniques --------------
# Steering techniques get their own steering + judge job each; baselines share a single
# judge over the 'no_steering' subdir. Lookup maps preserve the historical job names.
declare -A STEER_JOBNAME=([swfc]="seq_comp_fixed" [sttp]="seq_comp_dist" [stmp]="seq_comp_mirror")
declare -A STEER_JUDGENAME=([swfc]="judge_comp_fixed" [sttp]="judge_comp_dist" [stmp]="judge_comp_mirror")

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

STEER_SCRIPT="scripts/steering/sequential_steer_over_all_layers_compassion.sh"
JUDGE_SCRIPT="scripts/llm_judge/run_llm_judge_compassion_optimized.sh"

for script in "$STEER_SCRIPT" "$JUDGE_SCRIPT"; do
    if [[ ! -f "$script" ]]; then
        echo "ERROR: script not found: $script"
        exit 1
    fi
done

mkdir -p logs

echo "Model: ${MODEL_ID} (save name: ${MODEL_SAVE_NAME})"
RESULTS_BASE="results/logistic_regression/compassion/${MODEL_SAVE_NAME}/all_layers"
ALL_JOB_IDS=()

# --- Phase 1: Submit steering jobs ---

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
    output=$(sbatch --job-name="$job_name" "${sbatch_extra[@]}" "$STEER_SCRIPT" "$@" --model_id "$MODEL_ID" "${script_extra[@]}")
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

if [[ "$SKIP_GENERATION" == "true" ]]; then
    echo "=== Phase 1: SKIPPED (--skip_generation) ==="
    echo ""
else
    echo "=== Phase 1: Steering jobs ==="
    echo ""

    # Steering techniques (one dual-mode job each)
    for tech in "${STEER_TECHS[@]}"; do
        submit_steer "${STEER_JOBNAME[$tech]}_dual" --steering_technique "$tech" --dual_mode
        STEER_JID[$tech]=$_JOB_ID
    done

    # Baselines (one job each, never dual-mode)
    for tech in "${BASELINE_TECHS[@]}"; do
        submit_steer "seq_comp_${tech%_baseline}" --steering_technique "$tech"
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
    local sbatch_extra=()
    if [[ "$TEST_MODE" == "true" ]]; then
        sbatch_extra+=(--time=01:00:00)
    fi
    # Strip leading/trailing colons (left over when some JID_* are empty under
    # --skip_generation, e.g. ":12345" or "12345:" or ":"), and treat an
    # empty/colons-only string as "no dependency".
    local clean_dep_ids="${dep_ids//:/}"
    if [[ -n "$clean_dep_ids" ]]; then
        clean_dep_ids="${dep_ids#:}"
        clean_dep_ids="${clean_dep_ids%:}"
    fi
    local dep_args=()
    if [[ -n "$clean_dep_ids" ]]; then
        dep_args+=(--dependency="afterok:${clean_dep_ids}")
    fi
    local output
    output=$(sbatch \
        --job-name="$job_name" \
        "${dep_args[@]}" \
        "${sbatch_extra[@]}" \
        "$JUDGE_SCRIPT" \
        --results_dir="${RESULTS_BASE}/${results_subdir}")
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
    submit_judge "judge_comp_baseline" "$baseline_dep" "no_steering"
    JUDGE_JIDS+=("$_JOB_ID")
fi

echo ""
echo "=== Submitted ${#ALL_JOB_IDS[@]} jobs (${#STEER_TECHS[@]} steering + ${#BASELINE_TECHS[@]} baseline + ${#JUDGE_JIDS[@]} judge) ==="
echo "Monitor: squeue -u \$USER"
echo "Cancel all: scancel ${ALL_JOB_IDS[*]}"
