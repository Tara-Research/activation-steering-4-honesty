# Validation metrics & geometry

Judge-independent corroboration of the steering results, plus the representational-geometry
analyses. Run at the operating points selected in [steering.md](steering.md).

## Per-token target distance + aggregate metrics / cross-entropy

The dynamic-analysis launcher (`src/dynamic_analysis/*`) does paired generation with activation
recording and computes per-token target distance, divergence, and model cross-entropy, then
judges. It generates and chains two jobs (experiment -> judge).

The judge stage runs `python -m src.dynamic_analysis.judge_evaluation --input_json <path>
--trait {compassion,honesty}`. `--trait` is required and selects the rubric: compassion is graded
0-100 into `compassion_score`, honesty is judged against each scenario's gold truth into
`honesty_score` and additionally needs `--scenarios_json <path>` (the wrapper derives it from the
trait config). Both also write `coherence_score`. Judge model settings come from
`configs/config_llm_judge.yaml` — there are no per-call model/prompt-path overrides.

```bash
bash scripts/dynamic_analysis/run_dynamic_experiment_single.sh \
  --full --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --honesty_category self_report_experimental --exp_name v2_dynamic_honesty
```

For compassion, use `--trait compassion` and drop `--honesty_category`. Results (per-token target
distance and the aggregate metrics behind Fig 5 and the appendix) are written to
`results/logistic_regression/{trait}/{model}/dynamic_analysis/{exp_name}/`.

## Embedding distance

Cosine similarity of steered responses' sentence embeddings to the aligned-baseline responses,
across layers. The wrapper forwards `"$@"` to `src/embedding_distance/`.

```bash
bash scripts/embedding_distance/compute_embedding_distance.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --honesty_category self_report_experimental --techniques "swfc sttp stmp" \
  --exp_name v2_embdist_honesty
```

## Pairwise coefficient-tournament ELO

The judge compares response pairs across 5 coefficient variants + 2 baselines and ranks them
(Bradley-Terry). **This is the only ELO in the paper.**

> The ELO judge previously hardcoded `Qwen/Qwen3-32B` at `temperature=0.0`, bypassing
> `configs/config_llm_judge.yaml`. It now uses the shared judge like every other stage
> (`openai/gpt-oss-120b`, `temperature=1.0`), so **ratings produced before that change are not
> comparable with new ones** and the tournament must be re-run. `--recompute_bt` still works on
> saved match files without re-judging.

```bash
bash scripts/pipelines/elo_stage1_pipeline.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --steering_mode all_token_mode --honesty_category self_report_experimental \
  --exp_name v2_elo_honesty
bash scripts/pipelines/elo_stage2_pipeline.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --steering_mode all_token_mode --honesty_category self_report_experimental \
  --exp_name v2_elo_honesty
```

Stage 1 writes `stage1_{method}_{steering_mode}_ratings.json` per method under
`results/elo_evaluation/{trait}/{model}/`; stage 2 reads those to rank the best config per method.
`src.elo_evaluation.recompute_bt_from_matches` recomputes ratings from stored match records
without re-judging.

## Representational geometry (Fig 1 + appendix)

The CAA-vs-LR direction-equivalence cosine consumes only the extracted vectors -- run it right
after [vector generation](vector_generation.md), no steering or GPU required. `--model` takes
one or more model *slugs*; pass two or more to also produce the combined multi-model summary.

```bash
python -m scripts.verify_lr_vector_direction \
  --trait honesty --dataset self_report_experimental --model meta-llama_Llama-3.3-70B-Instruct
python -m scripts.verify_lr_vector_direction \
  --trait compassion --dataset compassion --model meta-llama_Llama-3.3-70B-Instruct
```

It writes the per-layer cosine table (CSV + LaTeX) and a summary figure under
`plots/vector_comparison/`; `--no-plot` / `--no-latex` skip either output.

The PCA and projection-histogram analysis behind Fig 1 consumes the per-side embeddings and the
`--per-token-layer` outputs from [vector generation](vector_generation.md).
