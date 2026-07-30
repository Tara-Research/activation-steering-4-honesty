# Steering experiments

Three methods (`src/steering/`): **SwFC** (`swfc`, adds `coef·vector`), **StTP** (`sttp`, steers
tokens below the decision boundary to a target projection), **StMP** (`stmp`, mirrors them across
the boundary). Position mode: `--steering_mode all_token_mode` (steer prompt + response) or
`response_mode` (response only). Steering always uses the logistic-regression vectors; layer and
coefficient are resolved from `configs/config_{honesty,compassion}.yaml` per the best-layer table
in [RUNBOOK.md](RUNBOOK.md).

## Single-turn layer sweep

Config-driven: the model, dataset, methods, coefficients, and layers are read from the configs
(edit `configs/config_{honesty,compassion}.yaml` to change model/dataset). Runs every layer for
both position modes on the honesty test set
(`data/honesty/test/generalized/v4/generalized_honesty_test_scenarios.json`).

**Canonical (one launcher, all configs):** `launch_sequential_{trait}_all_configs.sh` fans out the
whole stage — 5 steering → 4 judge, `afterok`-chained, both position modes by default.
Launch a subset with `--techniques`, one mode with `--no_dual --steering_mode …` (honesty
launcher only), or just the judge with `--skip_generation`:

```bash
bash scripts/steering/launch_sequential_honesty_all_configs.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --honesty_category self_report_experimental \
  --techniques swfc,sttp,stmp,aligned_baseline,malicious_baseline --exp_name v2_sweep_honesty
bash scripts/steering/launch_sequential_compassion_all_configs.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct \
  --techniques swfc,sttp,stmp,aligned_baseline,malicious_baseline --exp_name v2_sweep_compassion
```

The launcher submits the judge itself (no separate judge step); the honesty judge is always
`scripts/llm_judge/run_llm_judge_honesty_universal.sh`, the compassion judge
`run_llm_judge_compassion_optimized.sh`. Both wrappers run the same module,
`src.llm_judge.runners.sweep`, pinning `--trait honesty` / `--trait compassion` respectively —
see [the judge entry points](RUNBOOK.md#llm-judge-entry-points). Operating points (best
layer/coefficient at ≥90% of aligned coherence) are selected from the judged sweep and recorded
back into the configs.

Results are written under
`results/logistic_regression/{trait}/{model}/all_layers/{honesty_category}/{technique}/`
(honesty; compassion has no category segment). Results paths are flat — there is no
`features_<mode>/` subdirectory and no feature-mode suffix on `exp_name`.

## Multi-turn (self-play)

Honesty over 5 turns, compassion over 10, N=5 independent runs per scenario. The pipelines chain
generation → judge → consistency → activation projection → qualitative metrics → fill-judge →
n-run metrics, with SLURM `--dependency`. Text-quality metrics (sentence reuse, within-turn
and cross-turn 4-gram repetition) and the mean projection are computed alongside trait + coherence.

```bash
# honesty (5 turns); scenario_category is a folder under data/honesty/ (e.g. factual_claims)
bash scripts/pipelines/multi_turn_honesty_pipeline.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --honesty_category self_report_experimental \
  --scenario_category factual_claims --layer_mode single_layer --n_runs 5 --exp_name v2_mt_honesty
# compassion (10 turns)
bash scripts/pipelines/multi_turn_compassion_pipeline.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --n_runs 5 --exp_name v2_mt_compassion
```

Results: `results/logistic_regression/{trait}/{model}/multi_turn/...`.
