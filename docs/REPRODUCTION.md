# Paper reproduction recipe

Ordered `sbatch`/`bash` commands to regenerate every result in the paper
(*Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence*),
in dependency order. This is the doc to follow when re-running the whole paper. For a
per-stage lookup index see [RUNBOOK.md](RUNBOOK.md).

## How to run

- Everything runs on the **tamia** SLURM cluster, launched from the repo root. Job
  scripts are submitted with `sbatch <script>.sh <args>`; launcher scripts that fan out
  their own jobs are run with `bash <script>.sh <args>`. Lightweight, GPU-free steps
  (the CAA-vs-LR cosine, the LR fit) run directly with `python -m` on a login node.
- **Specify every CLI argument explicitly — do not rely on implicit defaults.** Paths are
  the exception: never paste absolute artifact paths; they are derived from
  `(trait, dataset, model)` by `src/constants/constants.py`.
- Steering methods: `swfc` (SwFC), `sttp` (StTP), `stmp` (StMP). Steering always uses the
  **logistic-regression** vectors; the CAA vector is extracted only for the P1
  direction-equivalence figure.

## New-version delta (vs. the published v1 figures)

Re-running for the new version changes two parameters everywhere; the commands below are
already written for the new version:

- Second architecture: `Qwen/Qwen3-32B` → **`Qwen/Qwen3.6-27B`**.
- Honesty dataset / category: `diverse_with_sycophancy` → **`self_report_experimental`**.

Current figure filenames embed `qwen3-32b` / `diverse_with_sycophancy`; they will change
accordingly. Run every phase once per model: `meta-llama/Llama-3.3-70B-Instruct` (primary)
and `Qwen/Qwen3.6-27B` (replication). Commands show Llama; substitute the model id for the
replication.

## Best layers (`configs/config_{honesty,compassion}.yaml`)

| Trait | Llama-3.3-70B | Qwen3.6-27B | gemma-4-31B-it |
|---|---|---|---|
| honesty (`self_report_experimental`) | 26 | 32 | 34 |
| compassion | 29 | 32 | 34 |

Layer/coefficient are resolved from the config; pass `--layer`/`--coefficient` only to override.

---

## P0 — Response generation + vector extraction (CAA + LR)

Per model × trait. First generate the training responses (the LLM-judge annotation runs in the
same job), then extract the CAA response-average activations and fit the logistic-regression
steering vector in one job.

```bash
# honesty training responses + judge annotation (5 responses per scenario side)
sbatch scripts/pipelines/responses_and_judge_caa_honesty.sh \
  --model-id meta-llama/Llama-3.3-70B-Instruct --n-responses 5 \
  --input-json data/honesty/CAA/self_report_experimental/self_report_experimental_scenarios.json \
  --output-json data/honesty/CAA/self_report_experimental/self_report_experimental_responses_meta-llama_Llama-3.3-70B-Instruct.json
# compassion training responses + judge annotation (2 responses per pair, fixed in the script)
sbatch scripts/pipelines/responses_and_judge_caa_compassion.sh \
  --model-id meta-llama/Llama-3.3-70B-Instruct

# CAA extraction + LR fit (one job per model x trait)
sbatch scripts/vector_generation/generate_vec_and_logreg.sh \
  --model-id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --dataset self_report_experimental --normalization-method projection
sbatch scripts/vector_generation/generate_vec_and_logreg.sh \
  --model-id meta-llama/Llama-3.3-70B-Instruct --trait compassion \
  --dataset compassion --normalization-method projection
```

Detail + the standalone extractor/LR wrappers: [vector_generation.md](vector_generation.md).

## P1 — CAA-vs-LR direction equivalence (right after extraction)

Consumes only the extracted vectors (no steering, no GPU; `--model` takes model *slugs*):

```bash
python -m scripts.verify_lr_vector_direction \
  --trait honesty --dataset self_report_experimental --model meta-llama_Llama-3.3-70B-Instruct
python -m scripts.verify_lr_vector_direction \
  --trait compassion --dataset compassion --model meta-llama_Llama-3.3-70B-Instruct
```

Writes the per-layer cosine table (CSV + LaTeX) alongside the summary figure. See
[analysis.md](analysis.md).

## P2 — Single-turn layer sweep + LLM judge → operating points

Config-driven sweeps over all layers (both `all_token_mode` and `response_mode`), then the
GPT-oss-120B judge scores trait + coherence. Operating points (best layer/coefficient) are
selected from the judged sweep and recorded in the configs.

```bash
# Canonical: one launcher per trait runs the whole stage (5 steering -> 4 judge,
# afterok-chained; both position modes by default).
bash scripts/steering/launch_sequential_honesty_all_configs.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --honesty_category self_report_experimental \
  --techniques swfc,sttp,stmp,aligned_baseline,malicious_baseline --exp_name v2_sweep_honesty
bash scripts/steering/launch_sequential_compassion_all_configs.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct \
  --techniques swfc,sttp,stmp,aligned_baseline,malicious_baseline --exp_name v2_sweep_compassion
```

The launcher reads layers/coefficients from the configs and submits the judge itself — there
is no separate judge step; the judge module is `src.llm_judge.runners.sweep --trait {honesty,compassion}`
(see [the judge entry points](RUNBOOK.md#llm-judge-entry-points)). See
[steering.md](steering.md) for details.

**Re-judge note:** the coherence rubric is now called the same way for both traits (rubric in the
system message, data in a user message). Honesty coherence scores produced before this change used
a different message shape, so re-run the honesty judge with `--force_recompute` before comparing
honesty and compassion coherence, or before re-deriving operating points.

## P3 — Validation at operating points

Judge-independent corroboration. Per-token target distance + aggregate/cross-entropy come from
the dynamic-analysis launcher (`src/dynamic_analysis/*`); embedding distance; and the pairwise
coefficient-tournament **ELO** (`elo_stage1_pipeline.sh` → `stage1_*_bt_normalized`; the only ELO
in the paper).

```bash
# per-token target distance + aggregate metrics + cross-entropy
bash scripts/dynamic_analysis/run_dynamic_experiment_single.sh \
  --full --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --honesty_category self_report_experimental --exp_name v2_dynamic_honesty
# embedding distance to the aligned baseline
bash scripts/embedding_distance/compute_embedding_distance.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --honesty_category self_report_experimental --techniques "swfc sttp stmp" \
  --exp_name v2_embdist_honesty
# pairwise coefficient-tournament ELO (stage 1 then stage 2)
bash scripts/pipelines/elo_stage1_pipeline.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --steering_mode all_token_mode --honesty_category self_report_experimental \
  --exp_name v2_elo_honesty
bash scripts/pipelines/elo_stage2_pipeline.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --steering_mode all_token_mode --honesty_category self_report_experimental \
  --exp_name v2_elo_honesty
```

For compassion, drop `--honesty_category` and use `--trait compassion`. See [analysis.md](analysis.md).

## P4 — Multi-turn steering

Self-play: honesty 5 turns / compassion 10 turns, N=5 runs, plus the judge, n-run, and
text-quality (sentence reuse, cross-turn 4-gram) metrics. The pipelines chain all stages.

```bash
# honesty (5 turns) — scenario_category from data/honesty/ (e.g. factual_claims)
bash scripts/pipelines/multi_turn_honesty_pipeline.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --honesty_category self_report_experimental \
  --scenario_category factual_claims --layer_mode single_layer --n_runs 5 --exp_name v2_mt_honesty
# compassion (10 turns)
bash scripts/pipelines/multi_turn_compassion_pipeline.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --n_runs 5 --exp_name v2_mt_compassion
```

## P5 — Capability benchmarks

MMLU, MT-Bench, AlpacaEval for both traits.

```bash
# MMLU
sbatch scripts/benchmarks/mmlu_benchmark.sh \
  --model-id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --honesty-category self_report_experimental --steering-type all --exp_name v2_mmlu_honesty
# MT-Bench
bash scripts/pipelines/mt_bench_pipeline.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --honesty_category self_report_experimental --techniques "no_steering swfc sttp stmp" \
  --steering_mode all_token_mode --exp_name v2_mtbench_honesty
# AlpacaEval
bash scripts/pipelines/alpaca_eval_pipeline.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --honesty_category self_report_experimental --techniques "no_steering swfc sttp stmp" \
  --steering_mode all_token_mode --exp_name v2_alpaca_honesty
```

Repeat MMLU/MT-Bench/AlpacaEval with `--trait compassion` (drop `--honesty_category`). See
[benchmarks.md](benchmarks.md).

## P6 — Generalization (honesty vector only) — companion repository

MASK, Among Us, AuditBench, and Emergent Misalignment are released separately; run them from the
companion repository. See [generalization.md](generalization.md) for the pointer.

---

## Paper result → command provenance

Keyed by paper figure/section. Run each per model.

| Paper result | Phase | Producing command |
|---|---|---|
| PCA + projection histograms of the training embeddings (Fig 1) | P0/P1 | `generate_vec_and_logreg.sh` (per-side embeddings + LR boundary) |
| CAA-vs-LR cosine per layer (App) | P1 | `python -m scripts.verify_lr_vector_direction` |
| Layer sweep: trait + coherence vs. layer, per coefficient (Fig 3 / App) | P2 | `launch_sequential_*_all_configs.sh` |
| Per-token target distance, method comparison (Fig 5) | P3 | `run_dynamic_experiment_single.sh` |
| Aggregate divergence metrics + model cross-entropy (App) | P3 | `run_dynamic_experiment_single.sh` |
| Embedding distance to the aligned baseline (App) | P3 | `compute_embedding_distance.sh` |
| Pairwise coefficient-tournament ELO / Bradley-Terry ratings (App) | P3 | `elo_stage1_pipeline.sh` → `elo_stage2_pipeline.sh` |
| Multi-turn trait/coherence per turn, N-run consistency, text-repetition metrics (Fig 6 / App) | P4 | `multi_turn_{honesty,compassion}_pipeline.sh` |
| AlpacaEval length-controlled win rates (Fig 4) | P5 | `alpaca_eval_pipeline.sh` |
| MT-Bench per-turn scores (Fig 4 / App) | P5 | `mt_bench_pipeline.sh` |
| MMLU accuracy under steering (App) | P5 | `mmlu_benchmark.sh` |
| MASK / Among Us / AuditBench / Emergent Misalignment (Fig 7 / App) | P6 | companion repository |
