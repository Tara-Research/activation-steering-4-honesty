#!/bin/bash
#SBATCH --job-name=steering_exp
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=50G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

echo "=== Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) ==="
echo "Args: $@"
echo "Started: $(date)"

srun --ntasks=1 --exact scripts/infrastructure/config_env.sh

export HF_HOME="${HF_HOME:-$HOME/links/projects/aip-irina/shared/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HF_HUB_CACHE

source $SLURM_TMPDIR/env/bin/activate
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)

# -------------- Parse dual_mode / steering_mode --------------
DUAL_MODE=false
PYTHON_ARGS=()
CAPTURED_STEERING_MODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dual_mode)  DUAL_MODE=true; shift ;;
        --steering_mode) CAPTURED_STEERING_MODE="$2"; shift 2 ;;
        *) PYTHON_ARGS+=("$1"); shift ;;
    esac
done

# -------------- Actual command --------------
PYTHON_MODULE="src.sequential_steering.sequential_steering_all_layers_honesty"

if [[ "$DUAL_MODE" == "true" ]]; then
    if [[ -n "$CAPTURED_STEERING_MODE" ]]; then
        echo "WARNING: --steering_mode '$CAPTURED_STEERING_MODE' ignored in dual mode (both modes run automatically)"
    fi
    echo "=== Dual-mode: running response_mode + all_token_mode via srun --exclusive ==="

    srun --ntasks=1 --gres=gpu:h100:2 --exact --exclusive \
        python -m "$PYTHON_MODULE" ${PYTHON_ARGS[@]+"${PYTHON_ARGS[@]}"} --steering_mode response_mode \
        > "$SLURM_TMPDIR/response_mode.log" 2>&1 &

    srun --ntasks=1 --gres=gpu:h100:2 --exact --exclusive \
        python -m "$PYTHON_MODULE" ${PYTHON_ARGS[@]+"${PYTHON_ARGS[@]}"} --steering_mode all_token_mode \
        > "$SLURM_TMPDIR/all_token_mode.log" 2>&1 &

    set +e
    wait -n; EXIT1=$?
    wait -n; EXIT2=$?
    set -e

    echo ""
    echo "========== response_mode log =========="
    cat "$SLURM_TMPDIR/response_mode.log"
    echo ""
    echo "========== all_token_mode log =========="
    cat "$SLURM_TMPDIR/all_token_mode.log"
    echo ""

    echo "Exit codes: step1=$EXIT1, step2=$EXIT2"
    if [[ $EXIT1 -ne 0 || $EXIT2 -ne 0 ]]; then
        echo "ERROR: one or both steering modes failed"
        exit 1
    fi
else
    # Single mode (backward compatible).
    # The SBATCH header requests --ntasks=2 for dual mode's two parallel srun steps.
    # Here we run ONE process that shards the model across all 4 GPUs, so pin the step
    # to a single task — otherwise srun inherits ntasks=2 and launches two identical
    # processes that each load the full model onto the same GPUs (CUDA OOM).
    echo "=== Single-mode: running ${CAPTURED_STEERING_MODE:-default mode} on all GPUs (1 task) ==="
    SINGLE_ARGS=(${PYTHON_ARGS[@]+"${PYTHON_ARGS[@]}"})
    if [[ -n "$CAPTURED_STEERING_MODE" ]]; then
        SINGLE_ARGS+=(--steering_mode "$CAPTURED_STEERING_MODE")
    fi
    srun --ntasks=1 python -m "$PYTHON_MODULE" ${SINGLE_ARGS[@]+"${SINGLE_ARGS[@]}"}
fi

echo "Finished: $(date)"
