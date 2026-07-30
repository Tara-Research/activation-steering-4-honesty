#!/bin/bash
#
# ELO Stage 1 pipeline: Per-method coefficient ranking.
# Submits 3 parallel SLURM jobs (one per steering method).
#
# Dependency graph:
#   judge_fixed / judge_distribution / judge_mirror  (all in parallel)
#
# Usage:
#   bash scripts/pipelines/elo_stage1_pipeline.sh \
#       --model_id meta-llama/Llama-3.3-70B-Instruct \
#       --trait compassion \
#       [--steering_mode all_token_mode] \
#       [--honesty_category self_report_experimental] \
#       [--exp_name 04_06_llama_hon] \
#       [--results_dir results/.../all_layers/self_report_experimental] \
#       [--batch_size 16] \
#       [--test]
#
set -euo pipefail
mkdir -p logs

# ── Parse named arguments ─────────────────────────────────────────────
MODEL_ID=""
TRAIT=""
STEERING_MODE="all_token_mode"
HONESTY_CATEGORY="self_report_experimental"
EXP_NAME=""
RESULTS_DIR=""
BATCH_SIZE=""
TEST_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_id)           MODEL_ID="$2";           shift 2 ;;
        --trait)              TRAIT="$2";               shift 2 ;;
        --steering_mode)      STEERING_MODE="$2";      shift 2 ;;
        --honesty_category)   HONESTY_CATEGORY="$2";   shift 2 ;;
        --exp_name)           EXP_NAME="$2";           shift 2 ;;
        --results_dir)        RESULTS_DIR="$2";        shift 2 ;;
        --batch_size)         BATCH_SIZE="$2";         shift 2 ;;
        --test)               TEST_FLAG="1";           shift 1 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_ID" ]]; then echo "Error: --model_id is required"; exit 1; fi
if [[ -z "$TRAIT" ]]; then echo "Error: --trait is required"; exit 1; fi

# ── Time limits ───────────────────────────────────────────────────────
if [[ -n "$TEST_FLAG" ]]; then
    TIME_JUDGE="0:10:00"
else
    TIME_JUDGE="04:50:00"
fi

# ── Build optional argument fragments ─────────────────────────────────
OPT_ARGS=""
OPT_ARGS+=" --steering_mode ${STEERING_MODE}"
[[ "$TRAIT" == "honesty" ]] && OPT_ARGS+=" --honesty_category ${HONESTY_CATEGORY}"
[[ -n "$EXP_NAME" ]]        && OPT_ARGS+=" --exp_name ${EXP_NAME}"
[[ -n "$RESULTS_DIR" ]]     && OPT_ARGS+=" --results_dir ${RESULTS_DIR}"
[[ -n "$BATCH_SIZE" ]]      && OPT_ARGS+=" --batch_size ${BATCH_SIZE}"
[[ -n "$TEST_FLAG" ]]       && OPT_ARGS+=" --test"

SCRIPT_DIR="scripts/pipelines/generated_elo_stage1"
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
        mem="90G"
        cpus="4"
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

# ── Print summary ────────────────────────────────────────────────────
echo "ELO Stage 1 Pipeline"
echo "  model_id        = ${MODEL_ID}"
echo "  trait            = ${TRAIT}"
echo "  steering_mode    = ${STEERING_MODE}"
echo "  honesty_category = ${HONESTY_CATEGORY}"
echo "  exp_name         = ${EXP_NAME:-none}"
echo "  results_dir      = ${RESULTS_DIR:-auto}"
echo "  test             = ${TEST_FLAG:-no}"
echo ""

# ── Submit per-method judge jobs in parallel ─────────────────────────
METHODS="swfc sttp stmp"
ALL_JUDGE_JOBS=""
ALL_JOB_IDS=""

for METHOD in $METHODS; do
    JUDGE_CMD="srun python -m src.elo_evaluation.run_elo_evaluation --stage 1 --model_id ${MODEL_ID} --trait ${TRAIT} --method ${METHOD}${OPT_ARGS}"

    write_job_script \
        "$SCRIPT_DIR/1_judge_${METHOD}.sh" \
        "elo_s1_${METHOD}" \
        "$TIME_JUDGE" \
        4 \
        "$JUDGE_CMD"

    JOB_ID=$(sbatch "$SCRIPT_DIR/1_judge_${METHOD}.sh" | awk '{print $NF}')
    echo "  ${METHOD}: judge = ${JOB_ID}"
    ALL_JOB_IDS+=" ${JOB_ID}"

    if [[ -n "$ALL_JUDGE_JOBS" ]]; then
        ALL_JUDGE_JOBS+=":${JOB_ID}"
    else
        ALL_JUDGE_JOBS="${JOB_ID}"
    fi
done

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "  judges = ${ALL_JUDGE_JOBS}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel all: scancel${ALL_JOB_IDS}"
