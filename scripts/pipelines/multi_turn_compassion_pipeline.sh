#!/bin/bash
#
# Multi-turn compassion pipeline launcher (N-run version).
# Generates and submits 6 sequential SLURM jobs with dependencies:
#   1. Generate steered conversations    (batched, N-run, 4× H100)
#   2. LLM Judge (compassion + coherence)(vLLM, 4× H100)
#   3. Phase A1: activation projections  (4× H100)
#   4. Phase A2: qualitative metrics     (BERT, 1× H100)
#   5. Fill judge scores → metrics       (CPU only)
#   6. N-run metrics                     (CPU only)
#
# No consistency judge for compassion (no factual scenario context).
#
# Usage:
#   bash scripts/pipelines/multi_turn_compassion_pipeline.sh [--model_id <id>] [--n_runs <N>] [--n_turns <N>] [--exp_name <name>] [--max_batch_size <N>]
#
# Examples:
#   bash scripts/pipelines/multi_turn_compassion_pipeline.sh
#   bash scripts/pipelines/multi_turn_compassion_pipeline.sh --model_id "Qwen/Qwen3-32B" --n_runs 3
#   bash scripts/pipelines/multi_turn_compassion_pipeline.sh --exp_name exp_1 --n_turns 5
#
set -euo pipefail
mkdir -p logs

# ── Parse named arguments ─────────────────────────────────────────────
# Single source of truth for the technique set; the Python-list literal below is
# built from it so the two never drift. Also drives --split_judge (one job per technique).
TECHNIQUE_LIST=(aligned_baseline malicious_baseline stmp sttp swfc)
_tq=""; for _t in "${TECHNIQUE_LIST[@]}"; do _tq="${_tq}'${_t}', "; done
TECHNIQUES="[${_tq%, }]"
MODE="all_token_mode"
MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"
FORCE_RECOMPUTE=False
N_RUNS=5
N_TURNS=10
N_QUESTIONS=""            # empty → None → use full test set (20); --test_mode defaults this to 5.
EXP_NAME=""
MAX_BATCH_SIZE=""
SKIP_GENERATION="false"   # if true: skip stage 1 (gen), resume from the judge on the existing
                          # generation already in the run dir (matched via exp_name).
TEST_MODE="false"         # if true: n_runs=1 + short (1h) GPU walltimes for a fast smoke test.
SPLIT_JUDGE="false"       # if true: fan the judge stage into one job per technique (run concurrently,
                          # each ~1/N of the work). Downstream proj waits on afterok of ALL of them.

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_id) MODEL_ID="$2"; shift 2 ;;
        --force_recompute) FORCE_RECOMPUTE=True; shift ;;
        --n_runs) N_RUNS="$2"; shift 2 ;;
        --n_turns) N_TURNS="$2"; shift 2 ;;
        --n_questions) N_QUESTIONS="$2"; shift 2 ;;
        --exp_name) EXP_NAME="$2"; shift 2 ;;
        --max_batch_size) MAX_BATCH_SIZE="$2"; shift 2 ;;
        --skip_generation) SKIP_GENERATION="true"; shift ;;
        --test_mode) TEST_MODE="true"; shift ;;
        --split_judge) SPLIT_JUDGE="true"; shift ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash $0 [--model_id <id>] [--force_recompute] [--n_runs <N>] [--n_turns <N>] [--n_questions <N>] [--exp_name <name>] [--max_batch_size <N>] [--skip_generation] [--test_mode] [--split_judge]"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_ID" ]]; then
    echo "Error: --model_id cannot be empty"
    exit 1
fi
MODEL_SAVE_NAME="${MODEL_ID//\//_}"

# ── Walltimes (test mode shortens the GPU stages for a fast smoke test) ──
T_GEN="11:50:00"; T_JUDGE="11:50:00"; T_QUAL="1:30:00"
if [[ "$TEST_MODE" == "true" ]]; then
    N_RUNS=1
    [[ -z "$N_QUESTIONS" ]] && N_QUESTIONS=5   # smoke test: small scenario subset unless overridden
    T_GEN="1:00:00"; T_JUDGE="1:00:00"; T_QUAL="1:00:00"
    echo ">>> TEST_MODE=true — n_runs=1; n_questions=${N_QUESTIONS}; gen/judge/qual walltimes capped at 1h."
fi

CONFIG_CONST="constants.CONFIG_COMPASSION"

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

if [[ -z "$N_QUESTIONS" ]]; then
    N_QUESTIONS_PY="None"
else
    N_QUESTIONS_PY="${N_QUESTIONS}"
fi

SCRIPT_DIR="scripts/pipelines/generated_compassion_pipeline"
mkdir -p "$SCRIPT_DIR"

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
    fi

    cat > "$script_path" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${cpus}
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

# 1. Generate steered conversations (batched, N-run)
write_job_script "$SCRIPT_DIR/1_gen.sh" "mt_comp_gen" "$T_GEN" 4 2 "
from src.multi_turn_steering.compassion.multi_turn_steering_compassion import run_steered_conversation_compassion
run_steered_conversation_compassion(
    steering_techniques=${TECHNIQUES},
    steering_mode='${MODE}',
    model_id='${MODEL_ID}',
    n_runs=${N_RUNS},
    n_turns=${N_TURNS},
    n_questions=${N_QUESTIONS_PY},
    exp_name=${EXP_NAME_PY},
    batch=True,
    max_batch_size=${MAX_BATCH_SIZE_PY},
)
"

# 2. LLM Judge (compassion trait + coherence)
write_job_script "$SCRIPT_DIR/2_judge.sh" "mt_comp_judge" "$T_JUDGE" 4 8 "
import os
from src.llm_judge.runners.multi_turn import run_llm_judge_compassion_multi_turn
from src.utils.exp_utils import resolve_experiment_dir
from src.constants import constants
base = os.path.join('results', 'logistic_regression', 'compassion', '${MODEL_SAVE_NAME}', 'multi_turn')
exp = resolve_experiment_dir(base, exp_name=${EXP_NAME_PY}, create=False)
results_dir = os.path.join(base, exp)
run_llm_judge_compassion_multi_turn(
    config_file=${CONFIG_CONST},
    results_dir=results_dir,
    force_recompute=${FORCE_RECOMPUTE},
)
"

# 2b. Per-technique judge jobs (used when --split_judge): same as stage 2 but each
#     job judges only one technique's files, so the N jobs run concurrently.
if [[ "$SPLIT_JUDGE" == "true" ]]; then
    for t in "${TECHNIQUE_LIST[@]}"; do
        write_job_script "$SCRIPT_DIR/2_judge_${t}.sh" "mt_comp_judge_${t}" "$T_JUDGE" 4 8 "
import os
from src.llm_judge.runners.multi_turn import run_llm_judge_compassion_multi_turn
from src.utils.exp_utils import resolve_experiment_dir
from src.constants import constants
base = os.path.join('results', 'logistic_regression', 'compassion', '${MODEL_SAVE_NAME}', 'multi_turn')
exp = resolve_experiment_dir(base, exp_name=${EXP_NAME_PY}, create=False)
results_dir = os.path.join(base, exp)
run_llm_judge_compassion_multi_turn(
    config_file=${CONFIG_CONST},
    results_dir=results_dir,
    force_recompute=${FORCE_RECOMPUTE},
    technique='${t}',
)
"
    done
fi

# 3. Phase A1: activation projections
write_job_script "$SCRIPT_DIR/3_a1_proj.sh" "mt_comp_a1_proj" "00:30:00" 4 2 "
from src.multi_turn_steering.compassion.multi_turn_qualitative_eval_suite import phase_a1_activation_projection_for_all
phase_a1_activation_projection_for_all(
    steering_techniques=${TECHNIQUES},
    steering_mode='${MODE}',
    exp_name=${EXP_NAME_PY},
    model_id='${MODEL_ID}',
)
"

# 4. Phase A2: qualitative metrics
write_job_script "$SCRIPT_DIR/4_a2_qual.sh" "mt_comp_a2_qual" "$T_QUAL" 4 4 "
from src.multi_turn_steering.compassion.multi_turn_qualitative_eval_suite import phase_a2_qualitative_metrics_for_all
phase_a2_qualitative_metrics_for_all(
    steering_techniques=${TECHNIQUES},
    steering_mode='${MODE}',
    exp_name=${EXP_NAME_PY},
    model_id='${MODEL_ID}',
)
"

# 5. Fill judge scores into metrics
write_job_script "$SCRIPT_DIR/5_fill_judge.sh" "mt_comp_fill_judge" "0:20:00" 0 8 "
from src.multi_turn_steering.compassion.multi_turn_qualitative_eval_suite import fill_judge_scores_from_results
fill_judge_scores_from_results(
    exp_name=${EXP_NAME_PY},
    model_id='${MODEL_ID}',
)
"

# 6. N-run metrics
write_job_script "$SCRIPT_DIR/6_n_run_metrics.sh" "mt_comp_nrun" "0:15:00" 0 8 "
from src.multi_turn_steering.compassion.multi_turn_qualitative_eval_suite import phase_n_run_compassion_metrics
phase_n_run_compassion_metrics(
    model_id='${MODEL_ID}',
    exp_name=${EXP_NAME_PY},
)
"

# ── Submit chain ───────────────────────────────────────────────────────
echo "Submitting multi-turn compassion pipeline..."
echo "  Model ID:    ${MODEL_ID}"
echo "  n_runs:      ${N_RUNS}"
echo "  n_turns:     ${N_TURNS}"
echo "  n_questions: ${N_QUESTIONS:-<all>}"
echo "  exp_name:    ${EXP_NAME:-<auto>}"
echo ""

# Stage 1 (gen) → defines the dependency the judge waits on.
if [[ "$SKIP_GENERATION" == "true" ]]; then
    # Resume mode: skip stage 1 (gen); the judge runs immediately and reads the existing
    # generation from the run dir (resolved via exp_name). Fails fast if no generation exists.
    echo ">>> SKIP_GENERATION=true — skipping stage 1; resuming from the judge."
    JOB1="(skipped)"
    GEN_DEP=""
else
    JOB1=$(sbatch "$SCRIPT_DIR/1_gen.sh"      | awk '{print $NF}')
    GEN_DEP="--dependency=afterok:$JOB1"
fi

# Stage 2 (judge): single job, or one job per technique (--split_judge). Either way we end
# up with JUDGE_DEP = the afterok dependency the downstream proj stage must wait on.
if [[ "$SPLIT_JUDGE" == "true" ]]; then
    echo ">>> SPLIT_JUDGE=true — fanning the judge into one job per technique (${#TECHNIQUE_LIST[@]} concurrent jobs)."
    JUDGE_IDS=()
    for t in "${TECHNIQUE_LIST[@]}"; do
        jid=$(sbatch $GEN_DEP "$SCRIPT_DIR/2_judge_${t}.sh" | awk '{print $NF}')
        JUDGE_IDS+=("$jid")
        echo "     judge[$t] : $jid"
    done
    JOB2="${JUDGE_IDS[*]}"                       # space-joined, for display / scancel
    JUDGE_DEP="--dependency=afterok:$(IFS=:; echo "${JUDGE_IDS[*]}")"
else
    JOB2=$(sbatch $GEN_DEP "$SCRIPT_DIR/2_judge.sh" | awk '{print $NF}')
    JUDGE_DEP="--dependency=afterok:$JOB2"
fi

JOB3=$(sbatch $JUDGE_DEP                   "$SCRIPT_DIR/3_a1_proj.sh"      | awk '{print $NF}')
JOB4=$(sbatch --dependency=afterok:$JOB3  "$SCRIPT_DIR/4_a2_qual.sh"      | awk '{print $NF}')
JOB5=$(sbatch --dependency=afterok:$JOB4  "$SCRIPT_DIR/5_fill_judge.sh"   | awk '{print $NF}')
JOB6=$(sbatch --dependency=afterok:$JOB5  "$SCRIPT_DIR/6_n_run_metrics.sh"| awk '{print $NF}')

echo ""
echo "Pipeline submitted:"
echo "  1. Generate conversations   : $JOB1"
echo "  2. LLM judge (compassion)   : $JOB2  (after $JOB1)"
echo "  3. Phase A1 (projections)   : $JOB3  (after judge)"
echo "  4. Phase A2 (qualitative)   : $JOB4  (after $JOB3)"
echo "  5. Fill judge scores        : $JOB5  (after $JOB4)"
echo "  6. N-run metrics            : $JOB6  (after $JOB5)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel all: scancel $JOB2 $JOB3 $JOB4 $JOB5 $JOB6"
