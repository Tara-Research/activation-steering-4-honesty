#!/bin/bash
#SBATCH --job-name=latency_bench
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Latency benchmark for steering methods on MT-Bench turn-1 prompts.
# All CLI args after the script name are forwarded directly to the Python module,
# e.g.:
#   sbatch scripts/benchmarks/latency_benchmark.sh --methods baseline
#   sbatch scripts/benchmarks/latency_benchmark.sh --methods swfc --batch_sizes 1 8
# See `python -m src.latency.benchmark_latency --help` for available flags.

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

srun scripts/infrastructure/config_env.sh

export HF_HOME="${HF_HOME:-/project/aip-irina/shared/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

unset HF_HUB_CACHE
unset TRANSFORMERS_CACHE
unset XDG_CACHE_HOME

source $SLURM_TMPDIR/env/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# -------------- Actual command --------------
srun python -m src.latency.benchmark_latency "$@"
