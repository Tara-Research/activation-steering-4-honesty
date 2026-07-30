#!/bin/bash
#
# Multi-turn honesty pipeline launcher.
# Generates and submits 7 sequential SLURM jobs with dependencies:
#   1. Generate steered conversations    (Llama 70B, 4× H100)
#   2. Phase B:  LLM judge               (vLLM, 4× H100)
#   3. Phase B2: Consistency judge        (vLLM, 4× H100)
#   4. Phase A1: activation projections   (Llama 70B, 4× H100)
#   5. Phase A2: qualitative metrics      (BERT, 1× H100)
#   6. Fill judge scores into metrics     (CPU only)
#   7. Compute N-run metrics              (CPU only)
#
# Usage:
#   bash scripts/pipelines/multi_turn_honesty_pipeline.sh --scenario_category <cat> --layer_mode <mode> [--n_runs <N>] [--exp_name <name>] [--max_batch_size <N>]
#
# Examples:
#   bash scripts/pipelines/multi_turn_honesty_pipeline.sh --scenario_category factual_claims --layer_mode single_layer
#   bash scripts/pipelines/multi_turn_honesty_pipeline.sh --scenario_category professional_advice --n_runs 3
#   bash scripts/pipelines/multi_turn_honesty_pipeline.sh --scenario_category professional_advice --layer_mode multi_layer
#   bash scripts/pipelines/multi_turn_honesty_pipeline.sh --scenario_category factual_claims --exp_name exp_1
#
set -euo pipefail
mkdir -p logs

# ── Parse named arguments ─────────────────────────────────────────────
SCENARIO_CATEGORY="factual_claims"
LAYER_MODE="single_layer"
N_RUNS=5
EXP_NAME="15_06_mult_layer_full_run_sre"
MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"
HONESTY_CATEGORY="self_report_experimental"
SCENARIO_FILTER=""
MAX_BATCH_SIZE=""
SKIP_GENERATION="false"   # if true: skip stage 1 (gen), resume from Phase B (judge) on the
                          # existing generation already in the run dir (matched via exp_name).
TEST_MODE="false"         # if true: n_runs=1 + short (1h) GPU walltimes for a fast end-to-end
                          # smoke test. Combine with --scenario_filter <id> for a single scenario.

# Single source of truth for the technique set; the Python-list literal is built from it so the
# two never drift. Also drives --split_judge (one judge job per technique).
TECHNIQUE_LIST=(aligned_baseline malicious_baseline stmp sttp swfc)
_tq=""; for _t in "${TECHNIQUE_LIST[@]}"; do _tq="${_tq}'${_t}', "; done
TECHNIQUES="[${_tq%, }]"
MODE="all_token_mode"
MULTI_LAYER_FLAG="False"
SPLIT_JUDGE="false"       # if true: fan the judge stage into one job per technique (concurrent).

while [[ $# -gt 0 ]]; do
    case "$1" in
        --scenario_category) SCENARIO_CATEGORY="$2"; shift 2 ;;
        --layer_mode)        LAYER_MODE="$2";        shift 2 ;;
        --n_runs)            N_RUNS="$2";            shift 2 ;;
        --exp_name)          EXP_NAME="$2";          shift 2 ;;
        --model_id)          MODEL_ID="$2";          shift 2 ;;
        --honesty_category)  HONESTY_CATEGORY="$2";  shift 2 ;;
        --scenario_filter)   SCENARIO_FILTER="$2";   shift 2 ;;
        --max_batch_size)    MAX_BATCH_SIZE="$2";    shift 2 ;;
        --skip_generation)   SKIP_GENERATION="true"; shift ;;
        --test_mode)         TEST_MODE="true";       shift ;;
        --split_judge)       SPLIT_JUDGE="true";     shift ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash $0 --scenario_category <cat> --layer_mode <mode> [--n_runs <N>] [--exp_name <name>] [--model_id <id>] [--honesty_category <cat>] [--scenario_filter <id>] [--max_batch_size <N>] [--skip_generation] [--test_mode] [--split_judge]"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_ID" ]]; then
    echo "Error: --model_id is required"
    exit 1
fi
MODEL_SAVE_NAME="${MODEL_ID//\//_}"

# ── Walltimes (test mode shortens the GPU stages for a fast smoke test) ──
T_GEN="10:00:00"; T_JUDGE="12:00:00"; T_QUAL="1:30:00"
if [[ "$TEST_MODE" == "true" ]]; then
    N_RUNS=1
    T_GEN="0:30:00"; T_JUDGE="0:30:00"; T_QUAL="0:20:00"
    echo ">>> TEST_MODE=true — n_runs=1; walltimes gen=30m judge=30m qual=20m (covers model-load + 1 scenario). Add --scenario_filter <id> for a single-scenario smoke test."
fi

CONFIG_CONST="constants.CONFIG_HONESTY"

# Build Python expression for exp_name: None if empty, 'exp_1' if set
if [[ -z "$EXP_NAME" ]]; then
    EXP_NAME_PY="None"
else
    EXP_NAME_PY="'${EXP_NAME}'"
fi

if [[ -z "$MAX_BATCH_SIZE" ]]; then
    MAX_BATCH_SIZE_PY="None"
else
    MAX_BATCH_SIZE_PY="${MAX_BATCH_SIZE}"
fi

if [[ -z "$SCENARIO_FILTER" ]]; then
    SCENARIO_FILTER_PY="None"
else
    SCENARIO_FILTER_PY="'${SCENARIO_FILTER}'"
fi

SCRIPT_DIR="scripts/pipelines/generated_honesty_pipeline"
mkdir -p "$SCRIPT_DIR"

if [[ "$LAYER_MODE" == "multi_layer" ]]; then
    MULTI_LAYER_FLAG="True"
fi

# ── Helper: write a job script ─────────────────────────────────────────
write_job_script() {
    local script_path="$1"
    local job_name="$2"
    local time="$3"
    local gpus="$4"
    local cpus="$5"
    local python_code="$6"

    local gpu_line=""
    local run_prefix="srun"
    if [[ "$gpus" != "0" ]]; then
        gpu_line="#SBATCH --gpus-per-node=h100:${gpus}"
#    else
#        run_prefix=""
    fi

    cat > "$script_path" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=${time}
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err
${gpu_line}

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

${run_prefix} scripts/infrastructure/config_env_vllm.sh

export HF_HOME="\${HF_HOME:-/project/aip-irina/shared/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

unset HF_HUB_CACHE
unset TRANSFORMERS_CACHE
unset XDG_CACHE_HOME

source \$SLURM_TMPDIR/env/bin/activate
export PYTHONPATH=\${PYTHONPATH:-}:\$(pwd)

echo "=== ${job_name} ==="
echo "Job ID: \$SLURM_JOB_ID"
date

${run_prefix} python -c "${python_code}"

echo "=== ${job_name} done ==="
date
SBATCH_EOF
    chmod +x "$script_path"
}

# ── Generate job scripts ───────────────────────────────────────────────

write_job_script "$SCRIPT_DIR/1_gen.sh" "mt_hon_gen" "$T_GEN" 4 2 "
from src.multi_turn_steering.honesty.multi_turn_steering_honesty import run_steered_conversation_honesty
from src.constants import constants
run_steered_conversation_honesty(
    model_id='${MODEL_ID}',
    steering_techniques=${TECHNIQUES},
    steering_mode='${MODE}',
    multi_layer=${MULTI_LAYER_FLAG},
    scenario_category='${SCENARIO_CATEGORY}',
    n_runs=${N_RUNS},
    exp_name=${EXP_NAME_PY},
    config_file=${CONFIG_CONST},
    honesty_category='${HONESTY_CATEGORY}',
    scenario_filter=${SCENARIO_FILTER_PY},
    max_batch_size=${MAX_BATCH_SIZE_PY},
)
"

write_job_script "$SCRIPT_DIR/2_b_judge.sh" "mt_hon_b_judge" "$T_JUDGE" 4 8 "
import os
from src.llm_judge.runners.multi_turn import run_llm_judge_honesty_multi_turn
from src.utils.exp_utils import resolve_experiment_dir
from src.constants import constants
base = os.path.join('results', 'logistic_regression', 'honesty', '${MODEL_SAVE_NAME}', 'multi_turn', '${HONESTY_CATEGORY}', '${SCENARIO_CATEGORY}', '${LAYER_MODE}')
exp = resolve_experiment_dir(base, exp_name=${EXP_NAME_PY}, create=False)
results_dir = os.path.join(base, exp)
run_llm_judge_honesty_multi_turn(
    config_file=${CONFIG_CONST},
    results_dir=results_dir,
)
"

# 2b. Per-technique judge jobs (used when --split_judge): same as Phase B but each job judges
#     only one technique's files, so the N jobs run concurrently (~1/N walltime each).
if [[ "$SPLIT_JUDGE" == "true" ]]; then
    for t in "${TECHNIQUE_LIST[@]}"; do
        write_job_script "$SCRIPT_DIR/2_b_judge_${t}.sh" "mt_hon_b_judge_${t}" "$T_JUDGE" 4 8 "
import os
from src.llm_judge.runners.multi_turn import run_llm_judge_honesty_multi_turn
from src.utils.exp_utils import resolve_experiment_dir
from src.constants import constants
base = os.path.join('results', 'logistic_regression', 'honesty', '${MODEL_SAVE_NAME}', 'multi_turn', '${HONESTY_CATEGORY}', '${SCENARIO_CATEGORY}', '${LAYER_MODE}')
exp = resolve_experiment_dir(base, exp_name=${EXP_NAME_PY}, create=False)
results_dir = os.path.join(base, exp)
run_llm_judge_honesty_multi_turn(
    config_file=${CONFIG_CONST},
    results_dir=results_dir,
    technique='${t}',
)
"
    done
fi

write_job_script "$SCRIPT_DIR/3_b2_consistency.sh" "mt_hon_b2_consist" "10:00:00" 4 8 "
import os
from src.llm_judge.runners.multi_turn import run_consistency_judge_honesty_multi_turn
from src.utils.exp_utils import resolve_experiment_dir
from src.constants import constants
base = os.path.join('results', 'logistic_regression', 'honesty', '${MODEL_SAVE_NAME}', 'multi_turn', '${HONESTY_CATEGORY}', '${SCENARIO_CATEGORY}', '${LAYER_MODE}')
exp = resolve_experiment_dir(base, exp_name=${EXP_NAME_PY}, create=False)
results_dir = os.path.join(base, exp)
run_consistency_judge_honesty_multi_turn(
    config_file=${CONFIG_CONST},
    results_dir=results_dir,
)
"

write_job_script "$SCRIPT_DIR/4_a1_proj.sh" "mt_hon_a1_proj" "0:30:00" 4 2 "
from src.multi_turn_steering.honesty.multi_turn_qualitative_eval_suite import phase_a1_activation_projection_for_all
from src.constants import constants
phase_a1_activation_projection_for_all(
    techniques=${TECHNIQUES},
    config_file=${CONFIG_CONST},
    scenario_category='${SCENARIO_CATEGORY}',
    layer_mode='${LAYER_MODE}',
    exp_name=${EXP_NAME_PY},
    model_id='${MODEL_ID}',
    honesty_category='${HONESTY_CATEGORY}',
)
"

write_job_script "$SCRIPT_DIR/5_a2_qual.sh" "mt_hon_a2_qual" "$T_QUAL" 4 2 "
from src.multi_turn_steering.honesty.multi_turn_qualitative_eval_suite import phase_a2_qualitative_metrics_for_all
phase_a2_qualitative_metrics_for_all(
    techniques=${TECHNIQUES},
    scenario_category='${SCENARIO_CATEGORY}',
    layer_mode='${LAYER_MODE}',
    exp_name=${EXP_NAME_PY},
    model_id='${MODEL_ID}',
    honesty_category='${HONESTY_CATEGORY}',
)
"

write_job_script "$SCRIPT_DIR/6_fill_judge.sh" "mt_hon_fill_judge" "0:10:00" 0 8 "
from src.multi_turn_steering.honesty.multi_turn_qualitative_eval_suite import fill_judge_scores_from_results
fill_judge_scores_from_results(
    techniques=${TECHNIQUES},
    scenario_category='${SCENARIO_CATEGORY}',
    layer_mode='${LAYER_MODE}',
    exp_name=${EXP_NAME_PY},
    model_id='${MODEL_ID}',
    honesty_category='${HONESTY_CATEGORY}',
)
"

write_job_script "$SCRIPT_DIR/7_n_run_metrics.sh" "mt_hon_nrun" "0:15:00" 0 8 "
from src.multi_turn_steering.honesty.multi_turn_qualitative_eval_suite import phase_n_run_honesty_metrics
phase_n_run_honesty_metrics(
    model_id='${MODEL_ID}',
    techniques=${TECHNIQUES},
    scenario_category='${SCENARIO_CATEGORY}',
    layer_mode='${LAYER_MODE}',
    exp_name=${EXP_NAME_PY},
    honesty_category='${HONESTY_CATEGORY}',
)
"

# ── Submit chain ───────────────────────────────────────────────────────
echo "Submitting multi-turn honesty pipeline..."
echo "  scenario_category  = ${SCENARIO_CATEGORY}"
echo "  layer_mode         = ${LAYER_MODE}"
echo "  n_runs             = ${N_RUNS}"
echo "  exp_name           = ${EXP_NAME:-<auto>}"
echo "  honesty_category   = ${HONESTY_CATEGORY}"
echo "  scenario_filter    = ${SCENARIO_FILTER:-<all>}"
echo ""

# Stage 1 (gen) → defines the dependency Phase B (judge) waits on.
if [[ "$SKIP_GENERATION" == "true" ]]; then
    # Resume mode: stage 1 (gen) is skipped; Phase B (judge) runs immediately and reads the
    # existing generation from the run dir (resolved via exp_name, create=False). Fails fast
    # in the judge job if no generation exists for these (model, category, exp_name) args.
    echo ">>> SKIP_GENERATION=true — skipping stage 1; resuming from Phase B (judge)."
    JOB1="(skipped)"
    GEN_DEP=""
else
    JOB1=$(sbatch "$SCRIPT_DIR/1_gen.sh"      | awk '{print $NF}')
    GEN_DEP="--dependency=afterok:$JOB1"
fi

# Phase B (judge): single job, or one job per technique (--split_judge). Either way we end up
# with JUDGE_DEP = the afterok dependency the downstream stages must wait on.
if [[ "$SPLIT_JUDGE" == "true" ]]; then
    echo ">>> SPLIT_JUDGE=true — fanning Phase B into one job per technique (${#TECHNIQUE_LIST[@]} concurrent jobs)."
    JUDGE_IDS=()
    for t in "${TECHNIQUE_LIST[@]}"; do
        jid=$(sbatch $GEN_DEP "$SCRIPT_DIR/2_b_judge_${t}.sh" | awk '{print $NF}')
        JUDGE_IDS+=("$jid")
        echo "     judge[$t] : $jid"
    done
    JOB2="${JUDGE_IDS[*]}"                       # space-joined, for display / scancel
    JUDGE_DEP="--dependency=afterok:$(IFS=:; echo "${JUDGE_IDS[*]}")"
else
    JOB2=$(sbatch $GEN_DEP "$SCRIPT_DIR/2_b_judge.sh" | awk '{print $NF}')
    JUDGE_DEP="--dependency=afterok:$JOB2"
fi
#JOB3=$(sbatch $JUDGE_DEP                   "$SCRIPT_DIR/3_b2_consistency.sh"    | awk '{print $NF}')
JOB4=$(sbatch $JUDGE_DEP                   "$SCRIPT_DIR/4_a1_proj.sh"           | awk '{print $NF}')
JOB5=$(sbatch --dependency=afterok:$JOB4  "$SCRIPT_DIR/5_a2_qual.sh"           | awk '{print $NF}')
JOB6=$(sbatch --dependency=afterok:$JOB5  "$SCRIPT_DIR/6_fill_judge.sh"        | awk '{print $NF}')
#JOB7=$(sbatch --dependency=afterok:$JOB6  "$SCRIPT_DIR/7_n_run_metrics.sh"     | awk '{print $NF}')

echo ""
echo "Pipeline submitted:"
echo "  1. Generate conversations   : $JOB1"
echo "  2. Phase B  (LLM judge)     : $JOB2  (after $JOB1)"
#echo "  3. Phase B2 (consistency)   : $JOB3  (after judge)"
echo "  4. Phase A1 (projections)   : $JOB4  (after judge)"
echo "  5. Phase A2 (qualitative)   : $JOB5  (after $JOB4)"
echo "  6. Fill judge scores        : $JOB6  (after $JOB5)"
#echo "  7. N-run metrics            : $JOB7  (after $JOB6)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel all: scancel $JOB2 $JOB4 $JOB5 $JOB6"
