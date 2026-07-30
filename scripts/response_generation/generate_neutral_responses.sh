#!/bin/bash
# Neutral generation on training scenarios (no steering). Saves responses to JSON.
# Produces the model's natural responses, used to characterize its projection
# distribution in the activation-projection analysis. Works for any trait via
# --trait / --data-path.
#
# Usage:
#   sbatch scripts/response_generation/generate_neutral_responses.sh \
#       --trait honesty --model-id Qwen/Qwen3.6-27B \
#       --data-path data/honesty/CAA/generalized/iterations/v4/generalized_honesty_scenarios.json \
#       --honesty-category generalized
#   sbatch scripts/response_generation/generate_neutral_responses.sh \
#       --trait compassion --model-id meta-llama/Llama-3.3-70B-Instruct \
#       --data-path data/compassion/CAA/conversation_data_compassion_generated_meta-llama_Llama-3.3-70B-Instruct.json
#
#   # Generate under each scenario's neutral system prompt (directive-stripped setup)
#   # instead of no system prompt -> neutral_answers_neutral_sysprompt.json:
#   sbatch scripts/response_generation/generate_neutral_responses.sh \
#       --trait honesty --model-id meta-llama/Llama-3.3-70B-Instruct \
#       --honesty-category self_report_experimental \
#       --data-path data/honesty/CAA/self_report_experimental/self_report_experimental_scenarios.json \
#       --neutral-system-prompt
#
# All arguments are forwarded to the Python module (see its --help). With no args
# it defaults to Llama-3.3-70B + the self_report_experimental honesty scenarios.
#SBATCH --job-name=neutral_gen
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=40GB
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

srun scripts/infrastructure/config_env.sh

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

source $SLURM_TMPDIR/env/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# -------------- Actual command --------------
srun python -m src.sequential_steering.neutral_generation "$@"
