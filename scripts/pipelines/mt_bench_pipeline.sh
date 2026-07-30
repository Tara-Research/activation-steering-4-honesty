#!/bin/bash
#
# MT-Bench pipeline launcher with per-technique parallelization.
# Submits independent SLURM jobs per technique for generation and evaluation,
# then a single metrics job once all evaluations complete.
#
# Dependency graph (each column runs in parallel):
#   gen_no_steering ---------> eval_no_steering --------\
#   gen_swfc -> eval_swfc -+-> metrics
#   gen_distribution --------> eval_distribution --------/
#   gen_stmp -----> eval_stmp ----/
#
# Usage:
#   bash scripts/pipelines/mt_bench_pipeline.sh \
#       --model_id meta-llama/Llama-3.3-70B-Instruct \
#       --trait honesty \
#       --techniques "no_steering swfc sttp stmp" \
#       --exp_name my_experiment \
#       [--steering_mode all_token_mode] \
#       [--honesty_category diverse_with_sycophancy] \
#       [--test] \
#       [--skip_gen]   # skip generation jobs; eval starts immediately
#
set -euo pipefail
mkdir -p logs

# ── Parse named arguments ─────────────────────────────────────────────
MODEL_ID=""
TRAIT=""
TECHNIQUES=""
EXP_NAME=""
STEERING_MODE=""
HONESTY_CATEGORY=""
TEST_FLAG=""
SKIP_GEN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_id)           MODEL_ID="$2";           shift 2 ;;
        --trait)              TRAIT="$2";               shift 2 ;;
        --techniques)         TECHNIQUES="$2";         shift 2 ;;
        --exp_name)           EXP_NAME="$2";           shift 2 ;;
        --steering_mode)      STEERING_MODE="$2";      shift 2 ;;
        --honesty_category)   HONESTY_CATEGORY="$2";   shift 2 ;;
        --test)               TEST_FLAG="1";           shift 1 ;;
        --skip_gen)           SKIP_GEN="1";            shift 1 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_ID" ]]; then echo "Error: --model_id is required"; exit 1; fi
if [[ -z "$TRAIT" ]]; then echo "Error: --trait is required"; exit 1; fi
if [[ -z "$TECHNIQUES" ]]; then echo "Error: --techniques is required"; exit 1; fi
if [[ -z "$EXP_NAME" ]]; then echo "Error: --exp_name is required"; exit 1; fi

# ── Time limits ───────────────────────────────────────────────────────
if [[ -n "$TEST_FLAG" ]]; then
    TIME_GEN="0:30:00"
    TIME_EVAL="1:30:00"
    TIME_METRICS="0:10:00"
else
    TIME_GEN="2:30:00"
    TIME_EVAL="2:30:00"
    TIME_METRICS="0:15:00"
fi

# ── Build optional argument fragments ─────────────────────────────────
OPT_ARGS_GEN=""
[[ -n "$STEERING_MODE" ]]    && OPT_ARGS_GEN+=" --steering_mode ${STEERING_MODE}"
[[ -n "$HONESTY_CATEGORY" ]] && OPT_ARGS_GEN+=" --honesty_category ${HONESTY_CATEGORY}"
[[ -n "$TEST_FLAG" ]]        && OPT_ARGS_GEN+=" --test"

# ── Derived paths ─────────────────────────────────────────────────────
MODEL_SAVE_NAME=$(echo "$MODEL_ID" | tr '/' '_')
METRICS_FILE="results/mt_bench/logistic_regression/${TRAIT}/${MODEL_SAVE_NAME}/${EXP_NAME}/mt_bench_metrics.json"

SCRIPT_DIR="scripts/pipelines/generated_mt_bench_pipeline"
mkdir -p "$SCRIPT_DIR"

# ── Helper: write a self-contained SLURM job script ──────────────────
write_job_script() {
    local script_path="$1"
    local job_name="$2"
    local time="$3"
    local gpus="$4"
    local shell_command="$5"

    local gpu_line=""
    local mem="16G"
    local cpus="2"
    if [[ "$gpus" != "0" ]]; then
        gpu_line="#SBATCH --gpus-per-node=h100:${gpus}"
        mem="30G"
        cpus="2"
    fi

    cat > "$script_path" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err
${gpu_line}

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

srun scripts/infrastructure/config_env_vllm.sh

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

${shell_command}

echo "=== ${job_name} done ==="
date
SBATCH_EOF
    chmod +x "$script_path"
}

# ── Per-technique generation and evaluation jobs ──────────────────────
read -ra TECHNIQUE_ARRAY <<< "$TECHNIQUES"
ALL_EVAL_JOBS=""
ALL_JOB_IDS=""

echo "Submitting MT-Bench pipeline..."
echo "  model_id           = ${MODEL_ID}"
echo "  trait              = ${TRAIT}"
echo "  techniques         = ${TECHNIQUES}"
echo "  exp_name           = ${EXP_NAME}"
echo "  steering_mode      = ${STEERING_MODE:-<default>}"
echo "  honesty_category   = ${HONESTY_CATEGORY:-<default>}"
echo "  test               = ${TEST_FLAG:-no}"
echo "  skip_gen           = ${SKIP_GEN:-no}"
echo ""

for TECHNIQUE in "${TECHNIQUE_ARRAY[@]}"; do
    # ── Generation script ─────────────────────────────────────────────
    GEN_CMD="srun python -m src.benchmarking.mt_bench.generate_responses --model_id ${MODEL_ID} --trait ${TRAIT} --techniques ${TECHNIQUE} --exp_name ${EXP_NAME}${OPT_ARGS_GEN}"

    write_job_script \
        "$SCRIPT_DIR/1_gen_${TECHNIQUE}.sh" \
        "mtb_gen_${TECHNIQUE}" \
        "$TIME_GEN" \
        4 \
        "$GEN_CMD"

    # ── Evaluation script ─────────────────────────────────────────────
    EVAL_CMD="srun python -m src.benchmarking.mt_bench.evaluate --model_id ${MODEL_ID} --trait ${TRAIT} --techniques ${TECHNIQUE} --exp_name ${EXP_NAME}"

    write_job_script \
        "$SCRIPT_DIR/2_eval_${TECHNIQUE}.sh" \
        "mtb_eval_${TECHNIQUE}" \
        "$TIME_EVAL" \
        4 \
        "$EVAL_CMD"

    # ── Submit gen -> eval chain ──────────────────────────────────────
    if [[ -n "$SKIP_GEN" ]]; then
        EVAL_JOB=$(sbatch "$SCRIPT_DIR/2_eval_${TECHNIQUE}.sh" | awk '{print $NF}')
        echo "  ${TECHNIQUE}:"
        echo "    gen  = SKIPPED"
        echo "    eval = ${EVAL_JOB} (no dependency)"
    else
        GEN_JOB=$(sbatch "$SCRIPT_DIR/1_gen_${TECHNIQUE}.sh" | awk '{print $NF}')
        EVAL_JOB=$(sbatch --dependency=afterok:${GEN_JOB} "$SCRIPT_DIR/2_eval_${TECHNIQUE}.sh" | awk '{print $NF}')
        echo "  ${TECHNIQUE}:"
        echo "    gen  = ${GEN_JOB}"
        echo "    eval = ${EVAL_JOB} (after ${GEN_JOB})"
    fi

    # Collect eval job IDs for the metrics dependency
    if [[ -n "$ALL_EVAL_JOBS" ]]; then
        ALL_EVAL_JOBS+=":${EVAL_JOB}"
    else
        ALL_EVAL_JOBS="${EVAL_JOB}"
    fi
    if [[ -z "$SKIP_GEN" ]]; then
        ALL_JOB_IDS+=" ${GEN_JOB}"
    fi
    ALL_JOB_IDS+=" ${EVAL_JOB}"
done

# ── Metrics job (depends on all eval jobs) ────────────────────────────
METRICS_CMD="python -m src.benchmarking.mt_bench.metrics --model_id ${MODEL_ID} --trait ${TRAIT} --techniques ${TECHNIQUES} --exp_name ${EXP_NAME}"

write_job_script \
    "$SCRIPT_DIR/3_metrics.sh" \
    "mtb_metrics" \
    "$TIME_METRICS" \
    0 \
    "$METRICS_CMD"

METRICS_JOB=$(sbatch --dependency=afterok:${ALL_EVAL_JOBS} "$SCRIPT_DIR/3_metrics.sh" | awk '{print $NF}')

ALL_JOB_IDS+=" ${METRICS_JOB}"

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "  metrics = ${METRICS_JOB} (after all evals: ${ALL_EVAL_JOBS})"
echo "  results = ${METRICS_FILE}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel all: scancel${ALL_JOB_IDS}"
