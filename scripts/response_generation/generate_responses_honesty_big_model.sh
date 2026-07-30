#!/bin/bash
#SBATCH --job-name=response_gen_honesty_big
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=h200:8
#SBATCH --mem=30G
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

# GLM-5.1 MoE requires transformers >= 5.3 (GlmMoeDsaConfig support).
# Upgrade in-place before running; --no-index keeps us on the cluster wheelhouse.
pip install --no-index --upgrade "transformers==5.3.*"
python -c "import transformers; print(f'[env] transformers=={transformers.__version__}')"

# -------------- Actual command --------------
# Forward all CLI args to the Python script, e.g.:
#   sbatch scripts/response_generation/generate_responses_honesty.sh \
#       --model-id meta-llama/Llama-3.3-70B-Instruct \
#       --n-responses 5 --batch-size 64 \
#       --input-json /path/to/input.json --output-json /path/to/output.json
srun python -m src.vector_generation.generate_responses --trait honesty "$@"
