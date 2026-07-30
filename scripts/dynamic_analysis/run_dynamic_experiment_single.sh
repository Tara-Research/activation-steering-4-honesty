#!/bin/bash
#
# Dynamic steering analysis pipeline launcher.
# Generates and submits 2 sequential SLURM jobs with dependencies:
#   1. Run dynamic steering analysis  (4x H100)
#   2. LLM judge evaluation            (4x H100, vLLM)
#
# Usage:
#   bash scripts/dynamic_analysis/run_dynamic_experiment_single.sh --test --config configs/config_compassion.yaml
#   bash scripts/dynamic_analysis/run_dynamic_experiment_single.sh --full --config configs/config_honesty.yaml
#   bash scripts/dynamic_analysis/run_dynamic_experiment_single.sh --test --n_prompts 5 --max_tokens 300
#   bash scripts/dynamic_analysis/run_dynamic_experiment_single.sh --full --skip_judge
#
set -euo pipefail
mkdir -p logs

# ── Defaults ─────────────────────────────────────────────────────────
CONFIG="configs/config_compassion.yaml"
TRAIT="honesty"
MODEL_ID="meta-llama/Llama-3.3-70B-Instruct"
EXP_NAME="correct_same_layer_full_run"
SEED=42
PRESET="full"
N_PROMPTS=""
MAX_TOKENS=""
SKIP_JUDGE=false
HONESTY_CATEGORY=""
LOAD_FROM_RESULTS=false

# ── Parse CLI arguments ──────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2";    shift 2 ;;
        --trait)        TRAIT="$2";     shift 2 ;;
        --honesty_category) HONESTY_CATEGORY="$2";     shift 2 ;;
        --model_id)     MODEL_ID="$2";  shift 2 ;;
        --exp_name)     EXP_NAME="$2";  shift 2 ;;
        --seed)         SEED="$2";      shift 2 ;;
        --test)         PRESET="test";  shift ;;
        --full)         PRESET="full";  shift ;;
        --n_prompts)    N_PROMPTS="$2"; shift 2 ;;
        --max_tokens)   MAX_TOKENS="$2"; shift 2 ;;
        --skip_judge)   SKIP_JUDGE=true; shift ;;
        --load_from_results) LOAD_FROM_RESULTS=true; shift ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash $0 --test|--full [--config PATH] [--model_id ID] [--exp_name NAME] [--seed N] [--n_prompts N] [--max_tokens N] [--skip_judge] [--load_from_results]"
            exit 1
            ;;
    esac
done

# ── Apply presets, then allow overrides ──────────────────────────────
if [[ "$PRESET" == "test" ]]; then
    : "${N_PROMPTS:=3}"
    : "${MAX_TOKENS:=1024}"
elif [[ "$PRESET" == "full" ]]; then
    : "${N_PROMPTS:=20}"
    : "${MAX_TOKENS:=1024}"
else
    echo "Error: must specify --test or --full"
    exit 1
fi

# ── Derive paths ─────────────────────────────────────────────────────
MODEL_SAVE_NAME="${MODEL_ID//\//_}"
OUTPUT_DIR="results/${TRAIT}/${MODEL_SAVE_NAME}/dynamic_analysis/${EXP_NAME}"

SCRIPT_DIR="scripts/dynamic_analysis/generated_dynamic_pipeline"
mkdir -p "$SCRIPT_DIR"

# ── Helper: write a job script ───────────────────────────────────────
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
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=${time}
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err
${gpu_line}

module load gcc python/3.12 arrow/24 opencv/4.13 cuda
set -euo pipefail

${run_prefix} scripts/infrastructure/config_env.sh

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

# ── Generate job scripts ─────────────────────────────────────────────

write_job_script "$SCRIPT_DIR/1_experiment.sh" "dyn_experiment" "16:00:00" 4 8 "
from src.dynamic_analysis.dynamic_steering_analysis import run_dynamic_analysis
run_dynamic_analysis(
    config_path='${CONFIG}',
    model_id_or_path='${MODEL_ID}',
    honesty_category='${HONESTY_CATEGORY}',
    methods='fixed,distribution,mirror',
    max_tokens=${MAX_TOKENS},
    n_prompts=${N_PROMPTS},
    seed_base=${SEED},
    exp_name='${EXP_NAME}',
    load_from_results=${LOAD_FROM_RESULTS^}
)
"

write_job_script "$SCRIPT_DIR/2_judge.sh" "dyn_judge" "8:00:00" 4 8 "
import glob
from src.dynamic_analysis.judge_evaluation import evaluate_dynamic_results

output_dir = '${OUTPUT_DIR}'
candidates = sorted(
    [f for f in glob.glob(f'{output_dir}/dynamic_analysis_*.json')
     if '_with_judge' not in f]
)
if not candidates:
    raise FileNotFoundError(f'No dynamic_analysis JSON found in {output_dir}')
input_json = candidates[-1]
print(f'Evaluating: {input_json}')
kwargs = {}
if '${TRAIT}' == 'honesty':
    from src.llm_judge.runners.ground_truth import honesty_scenarios_path
    kwargs['scenarios_json'] = honesty_scenarios_path('${MODEL_SAVE_NAME}', '${HONESTY_CATEGORY}')
evaluate_dynamic_results(
    input_json=input_json,
    trait='${TRAIT}',
    **kwargs
)
"

# ── Submit chain ─────────────────────────────────────────────────────
echo "Submitting dynamic steering analysis pipeline..."
echo "  Config:     ${CONFIG}"
echo "  Trait:      ${TRAIT}"
echo "  Model:      ${MODEL_ID}"
echo "  Exp name:   ${EXP_NAME}"
echo "  N prompts:  ${N_PROMPTS}"
echo "  Max tokens: ${MAX_TOKENS}"
echo "  Seed:       ${SEED}"
echo "  Skip judge: ${SKIP_JUDGE}"
echo "  Load from results: ${LOAD_FROM_RESULTS}"
echo "  Output dir: ${OUTPUT_DIR}"
echo ""

JOB1=$(sbatch "$SCRIPT_DIR/1_experiment.sh" | awk '{print $NF}')

if [[ "$SKIP_JUDGE" == "false" ]]; then
   JOB2=$(sbatch --dependency=afterok:$JOB1 "$SCRIPT_DIR/2_judge.sh" | awk '{print $NF}')

    echo "Pipeline submitted:"
    echo "  1. Experiment : $JOB1"
    echo "  2. Judge      : $JOB2  (after $JOB1)"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "Cancel all: scancel $JOB1 $JOB2"
else
    echo "Pipeline submitted (judge skipped):"
    echo "  1. Experiment : $JOB1"
    echo ""
    echo "Monitor: squeue -u \$USER"
    echo "Cancel: scancel $JOB1"
fi
