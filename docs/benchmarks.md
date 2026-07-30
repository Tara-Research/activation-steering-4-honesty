# Capability benchmarks

Capability/quality retention under steering: **MMLU**, **MT-Bench**, **AlpacaEval**, run per trait.
(Latency is **not** in the paper — see the Exploratory section of [RUNBOOK.md](RUNBOOK.md).)

## MMLU

```bash
sbatch scripts/benchmarks/mmlu_benchmark.sh \
  --model-id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --honesty-category self_report_experimental --steering-type all --exp_name v2_mmlu_honesty
```

## MT-Bench

Pipeline: generate → evaluate → metrics, one job set per technique.

```bash
bash scripts/pipelines/mt_bench_pipeline.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --honesty_category self_report_experimental --techniques "no_steering swfc sttp stmp" \
  --steering_mode all_token_mode --exp_name v2_mtbench_honesty
```

## AlpacaEval

Length-controlled win rate vs. the unsteered model. Pipeline: generate → judge.

```bash
bash scripts/pipelines/alpaca_eval_pipeline.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --honesty_category self_report_experimental --techniques "no_steering swfc sttp stmp" \
  --steering_mode all_token_mode --exp_name v2_alpaca_honesty
```

> The AlpacaEval judge now runs through the shared engine (`src/llm_judge/engine.py`), which
> applies `reasoning_effort: high` via `llm.chat(chat_template_kwargs=…)`. The previous code
> passed it through `apply_chat_template` in a way the harmony template did not pick up, so the
> judge very likely ran at default effort. **Win rates from before this change are not comparable
> with new ones.** The commented-out logprob scoring path was removed with the migration; scoring
> is binary `assistantfinalA`/`assistantfinalB` as before.

Repeat MMLU / MT-Bench / AlpacaEval with `--trait compassion` (drop `--honesty_category` /
`--honesty-category`). Results paths are flat — no `features_<mode>/` subdirectory.

### Threshold-alpha sweep (over-steering)

To measure how the conservative StTP/StMP gating threshold `m = mu_neg + alpha*std_neg`
trades benign over-steering against win rate, sweep alpha. `alpaca_eval_threshold_sweep.sh`
runs the full pipeline once per alpha, tagging `--exp_name` per run so the results of one
alpha never overwrite another's (`--include_bias` adds a bias-mode reference run).
Forwarded flags are passed to `alpaca_eval_pipeline.sh`; intended for `--test` mode.

```bash
bash scripts/pipelines/alpaca_eval_threshold_sweep.sh \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --honesty_category self_report_experimental --techniques "sttp" \
  --steering_mode all_token_mode --exp_name v2_alpaca_honesty \
  --alphas "0 0.5 1.0 1.5" --include_bias --coefficients "24 30 36" --test
```

`--coefficients` (optional) overrides `all_layers.<technique>.coefficients` for the steered
arm — use it to subset the target-coefficient list (e.g. the last three sttp coefficients)
and keep the sweep cheap; omit it to run the full per-technique list.

A single pipeline run can also target one threshold directly via
`alpaca_eval_pipeline.sh --threshold_mode neg_mean_std --threshold_alpha <a>` (or
`--threshold_mode bias`); both override `configs/steering_threshold.yaml` for that run
via the `STEERING_THRESHOLD_MODE` / `STEERING_THRESHOLD_ALPHA` env vars and tag `--exp_name`.

The internal per-stage scripts (`mt_bench_generate.sh`, `mt_bench_evaluate.sh`,
`alpaca_eval_generate.sh`, `alpaca_eval_judge.sh`) are driven by their pipeline and are not called
directly.
