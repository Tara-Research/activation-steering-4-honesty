# Vector generation

Extract the steering directions. Two methods share one pipeline: **CAA** (mean difference of
contrastive response-average activations) and **logistic regression** (classifier weight vector +
a calibrated decision boundary used by StTP/StMP). The paper uses logistic regression for
steering; the CAA vector is extracted too for the CAA-vs-LR direction-equivalence figure
(`python -m scripts.verify_lr_vector_direction`).

Three entry points in `src/vector_generation/`, run in order:

1. **`generate_responses`** — generate the pos/neg training responses per trait.
2. **`generate_vec_caa`** — extract the response-average CAA steering vector + per-side embeddings.
3. **`generate_vec_logistic_regression`** — fit the LR vector on those embeddings (no GPU).

Only **response-average** activations exist (per-sample mean over all response tokens); artifact
filenames keep the `response_avg` infix.

## Step 0 — Training responses (+ judge annotation)

`python -m src.vector_generation.generate_responses --trait {honesty,compassion} --model-id <id>
--n-responses <N> [--max-tokens 1024] [--temperature 0.6] [--batch-size 50] [--input-json PATH]
[--output-json PATH]`. Paper runs: honesty 5 responses per scenario side, compassion 2 per
(question, system prompt). The combined pipeline scripts generate and then run the LLM-judge
annotation pass in one job (mutates the responses JSON in place: `judge_caa_*` fields for
honesty, `judge_score` for compassion):

```bash
# honesty: generation + judge annotation in one job
sbatch scripts/pipelines/responses_and_judge_caa_honesty.sh \
  --model-id meta-llama/Llama-3.3-70B-Instruct --n-responses 5 \
  --input-json data/honesty/CAA/self_report_experimental/self_report_experimental_scenarios.json \
  --output-json data/honesty/CAA/self_report_experimental/self_report_experimental_responses_meta-llama_Llama-3.3-70B-Instruct.json
# compassion: paths are derived from --model-id; n-responses is fixed to 2 in the script
sbatch scripts/pipelines/responses_and_judge_caa_compassion.sh \
  --model-id meta-llama/Llama-3.3-70B-Instruct
```

The annotation pass on its own is
`python -m src.llm_judge.runners.caa_dataset --trait {honesty,compassion} --model-id <id>`, with
`--dataset <d>` and `--score-threshold 65` (honesty only), `--save-raw-eval` (compassion only),
and `--input-json PATH --batch-size 200 --save-every-items 200 --force` for both. It is
idempotent: generations that already carry judge fields are skipped unless `--force` is given.
See [RUNBOOK](RUNBOOK.md#llm-judge-entry-points) for the other two judge CLIs.

Generation-only wrappers: `scripts/response_generation/generate_responses_honesty.sh` and
`scripts/response_generation/generate_responses_honesty_big_model.sh` (both forward `"$@"` to
the module with `--trait honesty`; the latter targets big-model nodes) and
`scripts/response_generation/generate_responses_compassion.sh` (positional `MODEL_ID`).
`scripts/response_generation/generate_neutral_responses.sh` generates the model's natural
(no-persona) responses via `src.sequential_steering.neutral_generation` — used to
characterize the projection distribution, not for vector extraction.

## Steps 1+2 — CAA extraction + LR fit

The combined `generate_vec_and_logreg.sh` job does both in one job: it extracts the CAA vector
and per-side embeddings, then fits the LR vector on them (embedding paths are derived, not
pasted):

```bash
# honesty
sbatch scripts/vector_generation/generate_vec_and_logreg.sh \
  --model-id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --dataset self_report_experimental --normalization-method projection
# compassion
sbatch scripts/vector_generation/generate_vec_and_logreg.sh \
  --model-id meta-llama/Llama-3.3-70B-Instruct --trait compassion \
  --dataset compassion --normalization-method projection
```

Optional flags: `--input-json <responses.json>` (otherwise derived), `--per-token-layer <L>`
(additionally saves ragged per-token response embeddings for that layer, used by the
PCA/histogram analyses), `--extract-extra "<args>"` (forwarded to the extractor, e.g.
`--no-filter-by-judge`), `--logreg-extra "<args>"` (forwarded to the LR fit, e.g. `--C 0.01`).

## Standalone steps

`scripts/vector_generation/generate_vec_caa.sh` and
`scripts/vector_generation/generate_vec_logistic_regression.sh` are thin sbatch wrappers that
forward `"$@"` to the modules. Prefer the combined script so the embedding paths are derived.

**CAA extraction** (`python -m src.vector_generation.generate_vec_caa`): required
`--trait {honesty,compassion} --dataset <d> --model-id <id>`; optional `--input-json PATH`,
`--save-dir PATH`, `--no-filter-by-judge` (disables judge-score filtering / best-sample
selection), `--per-token-layer <L>`. With judge annotation present, filtering selects one pair
per scenario: best honest (highest score) vs worst dishonest (lowest score); compassion pools
across the prompt ids and keeps generations with `judge_score >= 75`.

**LR fit** (`python -m src.vector_generation.generate_vec_logistic_regression`; no GPU):
required `--pos-features <pt> --neg-features <pt> --trait <t> --dataset <d> --model-id <id>`;
optional `--save-dir PATH`, `--C 0.001`, `--max-iter 1000`, `--penalty l2`, `--solver lbfgs`,
`--class-weight None`, `--random-state 42`, `--normalization-method {none,projection}`.

## Normalization

The LR direction is normalized to the pos/neg projection gap (`--normalization-method
projection`, the default) so a coefficient of 1 shifts activations by one unit of class
separation; the bias is rescaled by the same factor so the decision boundary is unchanged.
`--normalization-method none` keeps the raw LR weights.

## Outputs

Artifact paths derive from `(trait, dataset, model)`. Relative to `PERSONA_VECTORS_FOLDER`
(side = `honest`/`dishonest` for honesty, `positive`/`negative` for compassion):

- CAA vector `[L, D]`:
  `caa/{model_slug}/{trait}/{dataset}/{trait}_{dataset}_steering_vector_response_avg_{model_slug}.pt`
- Per-side embeddings `[L, N, D]`:
  `caa/{model_slug}/{trait}/{dataset}/{trait}_{dataset}_{side}_embeddings_response_avg_{model_slug}.pt`
- Ragged per-token embeddings (only with `--per-token-layer`):
  `caa/{model_slug}/{trait}/{dataset}/per_token/{trait}_{dataset}_{side}_per_token_embeddings_{model_slug}.pt`
- LR vector / bias / metadata:
  `logistic_regression/{model_slug}/{trait}/{dataset}/{trait}_{dataset}_logreg_{steering_vector,bias}_response_avg_{model_slug}.pt`
  + `..._logreg_metadata_response_avg_{model_slug}.json`

They feed [steering.md](steering.md), the geometry analyses ([analysis.md](analysis.md)), and
every downstream stage.
