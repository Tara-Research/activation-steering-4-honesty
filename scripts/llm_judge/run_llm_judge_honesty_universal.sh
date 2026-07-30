#!/bin/bash
#SBATCH --job-name=judge_honesty_universal_h_200
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=110G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# =============================================================================
# Single-turn honesty judge over steering sweep results.
#
# Runs:  python -m src.llm_judge.runners.sweep --trait honesty
#
# Usage:
#   sbatch scripts/llm_judge/run_llm_judge_honesty_universal.sh [OPTIONS]
#
# Exactly one of --results_dir / --single_file is required.
#   --results_dir=PATH        Judge every *.json under this directory
#   --single_file=PATH        Judge exactly this file
#
# Options (all forwarded to the module; run with --help for the full list):
#   --honesty_category=STR    Scenario category (default: self_report_experimental)
#   --model_save_name=STR     Selects the common_steering_config_* section
#   --checkpoint_interval=N   Save every N evaluations (default: 4000, 0 = end only)
#   --force_recompute         Re-judge items that already have scores
#   --skip_trait              Skip the trait phase
#   --skip_coherence          Skip the coherence phase
#   --include_raw_output      Store the judge's raw text inside judge_score
#   --sampling_mode           Judge N extra times into suffixed keys
#   --n_samples=N             Extra judge runs under --sampling_mode (default: 5)
#   --target_layers=L1,L2     Only these layers
#   --target_coefs=C1,C2      Only these coefficients
#
# NOTE: booleans are now bare flags. `--force_recompute=True` is no longer
# valid; write `--force_recompute`. `--evaluate_coherence=False` is now
# `--skip_coherence`.
#
# Examples:
#   sbatch scripts/llm_judge/run_llm_judge_honesty_universal.sh #       --results_dir=results/logistic_regression/honesty/<model>/all_layers/swfc #       --honesty_category=self_report_experimental
#
#   sbatch scripts/llm_judge/run_llm_judge_honesty_universal.sh #       --single_file=<path>.json --force_recompute --include_raw_output
# =============================================================================

# Hardcode repo root (SLURM copies script to temp location, breaking BASH_SOURCE)
REPO_ROOT="$(pwd)"

echo "======== STARTING JOB ========"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "REPO_ROOT: $REPO_ROOT"
echo "CLI args: $@"

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

cd "$REPO_ROOT"
echo "PWD: $(pwd)"

echo "======== SETTING UP ENVIRONMENT ========"
echo "Running config_env_vllm.sh..."
srun bash "$REPO_ROOT/scripts/infrastructure/config_env_vllm.sh"
echo "config_env_vllm.sh completed"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "Activating virtualenv at: $SLURM_TMPDIR/env"
source $SLURM_TMPDIR/env/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

echo "======== RUNNING UNIVERSAL HONESTY JUDGE (MEGA-BATCHED) ========"
echo "Module: src.llm_judge.runners.sweep --trait honesty"

# Pass through any CLI arguments to the Python script
srun python -m src.llm_judge.runners.sweep --trait honesty "$@"

echo "======== JOB COMPLETED ========"
echo "Date: $(date)"
