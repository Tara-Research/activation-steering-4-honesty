#!/bin/bash
#
# AlpacaEval pipeline launcher with per-technique parallelization.
# Submits independent SLURM jobs per technique for generation and judging.
#
# Dependency graph (each column runs in parallel):
#   gen_baseline
#   gen_swfc ──→ judge_swfc
#   gen_sttp ──→ judge_sttp
#   gen_stmp ──→ judge_stmp
#
# - Baseline gen runs in parallel with all steered gen jobs (no judge for baseline)
# - Each judge job depends on its own gen job + baseline gen job
#
# Usage:
#   bash scripts/pipelines/alpaca_eval_pipeline.sh \
#       --model_id meta-llama/Llama-3.3-70B-Instruct \
#       --trait compassion \
#       --techniques "swfc sttp stmp" \
#       --exp_name my_experiment \
#       [--steering_mode all_token_mode] \
#       [--honesty_category diverse_with_sycophancy] \
#       [--test] \
#       [--skip_gen]   # skip generation, judge starts immediately
#       [--force]      # passed to generate and judge
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
FORCE_FLAG=""
THRESHOLD_MODE=""
THRESHOLD_ALPHA=""
COEFFICIENTS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_id)           MODEL_ID="$2";           shift 2 ;;
        --trait)              TRAIT="$2";               shift 2 ;;
        --techniques)         TECHNIQUES="$2";         shift 2 ;;
        --exp_name)           EXP_NAME="$2";           shift 2 ;;
        --steering_mode)      STEERING_MODE="$2";      shift 2 ;;
        --honesty_category)   HONESTY_CATEGORY="$2";   shift 2 ;;
        --threshold_mode)     THRESHOLD_MODE="$2";     shift 2 ;;
        --threshold_alpha)    THRESHOLD_ALPHA="$2";    shift 2 ;;
        --coefficients)       COEFFICIENTS="$2";       shift 2 ;;
        --test)               TEST_FLAG="1";           shift 1 ;;
        --skip_gen)           SKIP_GEN="1";            shift 1 ;;
        --force)              FORCE_FLAG="1";          shift 1 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ── StTP/StMP gating-threshold override (optional alpha sweep support) ─────────
# When --threshold_mode / --threshold_alpha are given, inject STEERING_THRESHOLD_*
# env into the GENERATION jobs (the steerer reads them via
# src.steering.threshold_config.load_threshold_config, overriding
# configs/steering_threshold.yaml) and TAG exp_name so per-alpha results never
# overwrite each other. When omitted, behavior is unchanged (the committed YAML
# decides the threshold).
THRESHOLD_ENV=""
if [[ -n "$THRESHOLD_ALPHA" && -z "$THRESHOLD_MODE" ]]; then
    THRESHOLD_MODE="neg_mean_std"   # alpha only matters for the neg-cluster rule
fi
if [[ -n "$THRESHOLD_MODE" ]]; then
    case "$THRESHOLD_MODE" in
        bias)
            THRESHOLD_ENV="export STEERING_THRESHOLD_MODE=bias; "
            EXP_NAME="${EXP_NAME}_thr_bias"
            ;;
        neg_mean_std)
            if [[ -z "$THRESHOLD_ALPHA" ]]; then
                echo "Error: --threshold_mode neg_mean_std requires --threshold_alpha"; exit 1
            fi
            # Filesystem-safe alpha tag: 3.0 -> 3p0, -1.0 -> m1p0
            ALPHA_TAG=$(echo "$THRESHOLD_ALPHA" | sed 's/-/m/g; s/\./p/g')
            THRESHOLD_ENV="export STEERING_THRESHOLD_MODE=neg_mean_std; export STEERING_THRESHOLD_ALPHA=${THRESHOLD_ALPHA}; "
            EXP_NAME="${EXP_NAME}_thr_a${ALPHA_TAG}"
            ;;
        *)
            echo "Error: --threshold_mode must be 'bias' or 'neg_mean_std'"; exit 1
            ;;
    esac
fi

if [[ -z "$MODEL_ID" ]]; then echo "Error: --model_id is required"; exit 1; fi
if [[ -z "$TRAIT" ]]; then echo "Error: --trait is required"; exit 1; fi
if [[ -z "$TECHNIQUES" ]]; then echo "Error: --techniques is required"; exit 1; fi
if [[ -z "$EXP_NAME" ]]; then echo "Error: --exp_name is required"; exit 1; fi

# ── Time limits ───────────────────────────────────────────────────────
if [[ -n "$TEST_FLAG" ]]; then
    TIME_GEN="1:30:00"
    TIME_JUDGE="1:30:00"
else
    TIME_GEN="11:50:00"
    TIME_JUDGE="11:50:00"
fi

# ── Build optional argument fragments ─────────────────────────────────
OPT_ARGS=""
[[ -n "$STEERING_MODE" ]]    && OPT_ARGS+=" --steering-mode ${STEERING_MODE}"
[[ -n "$HONESTY_CATEGORY" ]] && OPT_ARGS+=" --honesty-category ${HONESTY_CATEGORY}"
[[ -n "$TEST_FLAG" ]]        && OPT_ARGS+=" --test"
[[ -n "$FORCE_FLAG" ]]       && OPT_ARGS+=" --force"

# Optional coefficient-subset override (steered generation only; baseline has none).
# Passed unquoted so each value becomes a separate argparse token.
COEF_ARGS=""
[[ -n "$COEFFICIENTS" ]] && COEF_ARGS=" --coefficients ${COEFFICIENTS}"

FORCE_ARGS=""
[[ -n "$FORCE_FLAG" ]]       && FORCE_ARGS+=" --force"

# ── Derived paths ─────────────────────────────────────────────────────
MODEL_SAVE_NAME=$(echo "$MODEL_ID" | tr '/' '_')
RESULTS_DIR="results/alpaca_eval/logistic_regression/${TRAIT}/${MODEL_SAVE_NAME}/${EXP_NAME}"

SCRIPT_DIR="scripts/pipelines/generated_alpaca_eval_pipeline"
mkdir -p "$SCRIPT_DIR"

# ── Helper: write a self-contained SLURM job script ──────────────────
# Args: script_path job_name time gpus shell_command [env_script] [python_module]
#   env_script    — infrastructure env-setup script (default: config_env.sh).
#                   Judge jobs use config_env_vllm.sh because the LLM judge
#                   runs on vLLM, which config_env.sh does not install.
#   python_module — module to load (default: python/3.12). Must match the
#                   python version the chosen env_script builds its venv with
#                   (config_env_vllm.sh uses python/3.11).
write_job_script() {
    local script_path="$1"
    local job_name="$2"
    local time="$3"
    local gpus="$4"
    local shell_command="$5"
    local env_script="${6:-scripts/infrastructure/config_env.sh}"
    local python_module="${7:-python/3.12}"

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

module load gcc ${python_module} arrow/24 opencv/4.13 cuda
set -euo pipefail

srun ${env_script}

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

# ── Submit baseline generation job ────────────────────────────────────
read -ra TECHNIQUE_ARRAY <<< "$TECHNIQUES"
ALL_JUDGE_JOBS=""
ALL_JOB_IDS=""

echo "Submitting AlpacaEval pipeline..."
echo "  model_id           = ${MODEL_ID}"
echo "  trait              = ${TRAIT}"
echo "  techniques         = ${TECHNIQUES}"
echo "  exp_name           = ${EXP_NAME}"
echo "  steering_mode      = ${STEERING_MODE:-<default>}"
echo "  honesty_category   = ${HONESTY_CATEGORY:-<default>}"
echo "  threshold_mode     = ${THRESHOLD_MODE:-<yaml default>}"
echo "  threshold_alpha    = ${THRESHOLD_ALPHA:-<n/a>}"
echo "  test               = ${TEST_FLAG:-no}"
echo "  skip_gen           = ${SKIP_GEN:-no}"
echo "  force              = ${FORCE_FLAG:-no}"
echo ""

BASELINE_JOB=""
if [[ -z "$SKIP_GEN" ]]; then
    BASELINE_GEN_CMD="${THRESHOLD_ENV}srun python -m src.benchmarking.alpaca_eval.alpaca_eval_generate --model-id ${MODEL_ID} --trait ${TRAIT} --exp_name ${EXP_NAME} --techniques baseline_no_steering${OPT_ARGS}"

    write_job_script \
        "$SCRIPT_DIR/1_gen_baseline.sh" \
        "ae_gen_baseline" \
        "$TIME_GEN" \
        4 \
        "$BASELINE_GEN_CMD"

    BASELINE_JOB=$(sbatch "$SCRIPT_DIR/1_gen_baseline.sh" | awk '{print $NF}')
    ALL_JOB_IDS+=" ${BASELINE_JOB}"
    echo "  baseline:"
    echo "    gen  = ${BASELINE_JOB}"
else
    echo "  baseline:"
    echo "    gen  = SKIPPED"
fi

# ── Per-technique generation and judge jobs ───────────────────────────
for TECHNIQUE in "${TECHNIQUE_ARRAY[@]}"; do
    # ── Generation script ─────────────────────────────────────────────
    GEN_CMD="${THRESHOLD_ENV}srun python -m src.benchmarking.alpaca_eval.alpaca_eval_generate --model-id ${MODEL_ID} --trait ${TRAIT} --exp_name ${EXP_NAME} --techniques ${TECHNIQUE}${OPT_ARGS}${COEF_ARGS}"

    write_job_script \
        "$SCRIPT_DIR/1_gen_${TECHNIQUE}.sh" \
        "ae_gen_${TECHNIQUE}" \
        "$TIME_GEN" \
        4 \
        "$GEN_CMD"

    # ── Judge script ──────────────────────────────────────────────────
    JUDGE_CMD="srun python -m src.benchmarking.alpaca_eval.alpaca_eval_judge --model-id ${MODEL_ID} --trait ${TRAIT} --steering-mode ${STEERING_MODE:-all_token_mode} --exp_name ${EXP_NAME} --techniques ${TECHNIQUE}${FORCE_ARGS}"

    write_job_script \
        "$SCRIPT_DIR/2_judge_${TECHNIQUE}.sh" \
        "ae_judge_${TECHNIQUE}" \
        "$TIME_JUDGE" \
        4 \
        "$JUDGE_CMD" \
        "scripts/infrastructure/config_env_vllm.sh" \
        "python/3.11"

    # ── Submit gen -> judge chain ─────────────────────────────────────
    if [[ -n "$SKIP_GEN" ]]; then
        JUDGE_JOB=$(sbatch "$SCRIPT_DIR/2_judge_${TECHNIQUE}.sh" | awk '{print $NF}')
        echo "  ${TECHNIQUE}:"
        echo "    gen   = SKIPPED"
        echo "    judge = ${JUDGE_JOB} (no dependency)"
    else
        GEN_JOB=$(sbatch "$SCRIPT_DIR/1_gen_${TECHNIQUE}.sh" | awk '{print $NF}')
        JUDGE_JOB=$(sbatch --dependency=afterok:${GEN_JOB}:${BASELINE_JOB} \
            "$SCRIPT_DIR/2_judge_${TECHNIQUE}.sh" | awk '{print $NF}')
        echo "  ${TECHNIQUE}:"
        echo "    gen   = ${GEN_JOB}"
        echo "    judge = ${JUDGE_JOB} (after ${GEN_JOB}, ${BASELINE_JOB})"
        ALL_JOB_IDS+=" ${GEN_JOB}"
    fi

    if [[ -n "$ALL_JUDGE_JOBS" ]]; then
        ALL_JUDGE_JOBS+=":${JUDGE_JOB}"
    else
        ALL_JUDGE_JOBS="${JUDGE_JOB}"
    fi
    ALL_JOB_IDS+=" ${JUDGE_JOB}"
done

# ── Summary ───────────────────────────────────────────────────────────
echo ""
echo "  judges  = ${ALL_JUDGE_JOBS}"
echo "  results = ${RESULTS_DIR}"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel all: scancel${ALL_JOB_IDS}"
