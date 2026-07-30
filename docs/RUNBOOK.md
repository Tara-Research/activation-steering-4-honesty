# RUNBOOK — command index

Single source of truth for how to run things in this repo. Organized by pipeline stage; one
canonical command per entry. To re-run the **whole paper in order**, follow
[REPRODUCTION.md](REPRODUCTION.md). Notion command pages now point here.

## Conventions

- **Launch on the tamia SLURM cluster from the repo root.** Job scripts: `sbatch <script>.sh <args>`.
  Launcher scripts that fan out their own jobs: `bash <script>.sh <args>`. GPU-free steps (the
  cosine check, the LR fit) run with `python -m` on a login node.
- **Specify every CLI argument explicitly — never rely on implicit defaults.** Exception: never paste
  absolute artifact paths; they derive from `(trait, dataset, model)` via
  `src/constants/constants.py`.
- **Models in scope:** `meta-llama/Llama-3.3-70B-Instruct` (primary), `Qwen/Qwen3.6-27B`
  (replication); `google/gemma-4-31B-it` is also configured.
- **Traits / datasets:** honesty with dataset `self_report_experimental`; compassion with dataset
  `compassion`. The honesty steering **test set** is
  `data/honesty/test/generalized/v4/generalized_honesty_test_scenarios.json`.
- **Methods:** `swfc` (SwFC), `sttp` (StTP), `stmp` (StMP). Steering always uses the
  **logistic-regression** vectors; the CAA vector is extracted only for the CAA-vs-LR
  direction-equivalence figure.
- Only **response-average** activations exist; artifact filenames keep the `response_avg` infix,
  and results paths have no feature-mode segment.
- **Discovery:** `just --list` (see the [justfile](../justfile)).

### Best layers (`configs/config_{honesty,compassion}.yaml`)

| Trait | Llama-3.3-70B | Qwen3.6-27B | gemma-4-31B-it |
|---|---|---|---|
| honesty (`self_report_experimental`) | 26 | 32 | 34 |
| compassion | 29 | 32 | 34 |

## Stages

### 1. Vector generation → [vector_generation.md](vector_generation.md)
Generate training responses (judge annotation runs in the same job), then extract the CAA
response-average activations and fit the logistic-regression steering vector.
```bash
sbatch scripts/pipelines/responses_and_judge_caa_honesty.sh \
  --model-id meta-llama/Llama-3.3-70B-Instruct --n-responses 5 \
  --input-json data/honesty/CAA/self_report_experimental/self_report_experimental_scenarios.json \
  --output-json data/honesty/CAA/self_report_experimental/self_report_experimental_responses_meta-llama_Llama-3.3-70B-Instruct.json
sbatch scripts/vector_generation/generate_vec_and_logreg.sh \
  --model-id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --dataset self_report_experimental --normalization-method projection
```

### 2. Single-turn layer sweep + LLM judge → [steering.md](steering.md)
One launcher per trait runs the whole stage — 5 steering → 4 judge, `afterok`-chained,
both position modes by default. (Multi-turn self-play is in [steering.md](steering.md).)
```bash
bash scripts/steering/launch_sequential_honesty_all_configs.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --honesty_category self_report_experimental \
  --techniques swfc,sttp,stmp,aligned_baseline,malicious_baseline --exp_name v2_sweep_honesty
```
The launcher submits the judge too — there is no separate judge stage. Launch a subset with
`--techniques`, one position mode with `--no_dual --steering_mode …`, or just the judge over an
existing sweep with `--skip_generation`.

#### LLM judge entry points

Every judge in the repo is one local vLLM engine (`src/llm_judge/engine.py`) reading its model
settings from `configs/config_llm_judge.yaml` and its rubric from
`src/llm_judge/system_prompts/`. There are three judge CLIs, all under `src/llm_judge/runners/`:

```bash
# single-turn sweep: trait + coherence over steering results (what the launchers submit)
python -m src.llm_judge.runners.sweep --trait honesty \
  --results_dir results/logistic_regression/honesty/<model>/all_layers/<category>/swfc \
  --honesty_category self_report_experimental \
  --model_save_name meta-llama_Llama-3.3-70B-Instruct \
  --checkpoint_interval 4000

python -m src.llm_judge.runners.sweep --trait compassion \
  --results_dir results/logistic_regression/compassion/<model>/all_layers/swfc \
  --sys_prompt_type dismissive_system_prompt --checkpoint_interval 5000

# CAA training-response annotation (what the response pipelines submit)
python -m src.llm_judge.runners.caa_dataset --trait honesty \
  --model-id meta-llama/Llama-3.3-70B-Instruct --dataset self_report_experimental \
  --batch-size 200 --save-every-items 200 --score-threshold 65

python -m src.llm_judge.runners.caa_dataset --trait compassion \
  --model-id meta-llama/Llama-3.3-70B-Instruct --batch-size 200 --save-every-items 200

# multi-turn: trait + coherence per assistant turn, plus the N-run consistency judge
python -m src.llm_judge.runners.multi_turn --trait honesty \
  --results_dir results/logistic_regression/honesty/<model>/multi_turn/<exp_name>
```

Shared flags on `sweep`: exactly one of `--results_dir` / `--single_file`; then
`--force_recompute`, `--skip_trait`, `--skip_coherence`, `--sampling_mode --n_samples N`,
`--target_layers L1,L2`, `--target_coefs coef_1,coef_2`. `--include_raw_output` is honesty-only
and `--no_raw_score` / `--sys_prompt_type` are compassion-only; passing one to the wrong trait
raises instead of being ignored. Run any of the three with `--help` for the authoritative list.

> **Flag change:** these were `fire` CLIs and are now `argparse`. Booleans are bare flags —
> `--force_recompute=True` is invalid, write `--force_recompute`; `--evaluate_coherence=False`
> is now `--skip_coherence` and `--evaluate_trait=False` is now `--skip_trait`.

### 3. Validation metrics & geometry → [analysis.md](analysis.md)
Per-token target distance + aggregate/cross-entropy (dynamic analysis), embedding distance,
coefficient-tournament ELO, CAA-vs-LR cosine.
```bash
bash scripts/dynamic_analysis/run_dynamic_experiment_single.sh \
  --full --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --honesty_category self_report_experimental --exp_name v2_dynamic_honesty
```

### 4. Capability benchmarks → [benchmarks.md](benchmarks.md)
MMLU, MT-Bench, AlpacaEval (per trait).
```bash
sbatch scripts/benchmarks/mmlu_benchmark.sh \
  --model-id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --honesty-category self_report_experimental --steering-type all --exp_name v2_mmlu_honesty
```

### 5. Generalization → [generalization.md](generalization.md)
The out-of-distribution experiments (MASK, Among Us, AuditBench, Emergent Misalignment) live in
the **companion repository**, not here.

### 6. Exploratory (not in the paper)
Kept in the repo but **not part of paper reproduction**: the latency benchmark
(`scripts/benchmarks/latency_benchmark.sh`).

### 7. Job monitoring & ops → [monitoring.md](monitoring.md)
Standard SLURM tools (`squeue`, `sacct`) plus the per-job logs under `logs/`.

## Keeping this current

When you add or change an experiment wrapper or its CLI, update this file +
[REPRODUCTION.md](REPRODUCTION.md) (and the relevant detail doc + justfile). The runbook
docs + justfile are the command source of truth; verify flags by hand against the argparse
definitions / shell-script case statements.
