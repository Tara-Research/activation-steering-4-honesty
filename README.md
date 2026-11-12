# Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence

Niklas Herbster, Martin Zborowski, Alberto Tosato, Gauthier Gidel, Tommaso Tosato

[arXiv:2604.08169](https://arxiv.org/abs/2604.08169)

Alignment in LLMs is more brittle than commonly assumed, and recent evidence suggests some
misalignment behaviors are encoded as linear structure in activation space — which makes them
tractable via activation steering as a lightweight runtime defense. This repository implements
three steering methods and the evaluation protocol used to test whether they restore alignment
**without** degrading text coherence or general capability, in open-ended single-turn and
multi-turn generation. Malicious system prompts serve as a controlled proxy for misalignment,
across two threat models: **dishonesty** and **dismissiveness**.

## Methods

Three steering methods (`src/steering/`), applied via PyTorch forward hooks that modify hidden
states at a target layer ℓ during generation:

| Paper name | Code id | Mechanism |
|---|---|---|
| Steer-with-Fixed-Coeff (**SwFC**) | `swfc` | Uniform additive steering: `h' = h + α·v` on every token |
| Steer-to-Target-Projection (**StTP**) | `sttp` | Only tokens whose projection falls below the decision boundary are projected to a target value; already-aligned tokens are left unchanged |
| Steer-to-Mirror-Projection (**StMP**) | `stmp` | Tokens below the boundary are reflected across it |

StTP and StMP are the projection-aware methods introduced in the paper. Both gate on a
**logistic-regression decision boundary**: a binary classifier is fit per layer on
response-averaged activations from contrastive on-policy response pairs, its weight vector
(normalized to the pos/neg projection gap) is the steering direction, and its rescaled bias is the
threshold. All steering uses the logistic-regression direction; the CAA mean-difference vector is
extracted as well, but only for the direction-equivalence check.

Every method supports two position modes via `--steering_mode`: `all_token_mode` (steer prompt +
response) and `response_mode` (generated tokens only).

## Scope

| | Honesty | Compassion |
|---|---|---|
| Threat model | dishonesty | dismissiveness |
| Dataset name | `self_report_experimental` | `compassion` |
| Training | 50 scenarios × contrastive system prompts | 50 prompts × 5 system-prompt variants |
| Test | 112 scenarios, 8 categories | 40 held-out prompts |
| Multi-turn | 5 turns, 20 scenarios | 10 turns, 40 conversations |

Models: `meta-llama/Llama-3.3-70B-Instruct` (primary), `Qwen/Qwen3.6-27B` (cross-architecture
replication); `google/gemma-4-31B-it` is also configured. Judge: GPT-oss-120B via vLLM, scoring
trait expression and coherence 0–100 independently.

Best layer per (trait, model), as recorded in `configs/config_{honesty,compassion}.yaml`:

| Trait | Llama-3.3-70B | Qwen3.6-27B | gemma-4-31B-it |
|---|---|---|---|
| honesty | 26 | 32 | 34 |
| compassion | 29 | 32 | 34 |

The out-of-distribution generalization experiments (MASK, Among Us, AuditBench, Emergent
Misalignment) are released in a companion repository — see [docs/generalization.md](docs/generalization.md).

## Setup

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements_dynamic.txt
```

`requirements_dynamic.txt` uses flexible version ranges and excludes vLLM. `requirements.txt`
records the exact environment used for the paper runs (Python 3.12, torch 2.6,
transformers 4.52) — it is a provenance record rather than an installable file, since a few pins
are Compute Canada wheelhouse builds (`+computecanada`) that resolve only on that cluster.

- **vLLM** (Linux + CUDA, Python 3.11) is required by the LLM-judge, ELO, MT-Bench-judge and
  AlpacaEval-judge stages; install it separately. Judge model and decoding settings live in
  `configs/config_llm_judge.yaml`.
- Create a `.env` at the repo root with `OPENAI_API_KEY` (AlpacaEval judge); `ANTHROPIC_API_KEY`
  is optional.
- Models are loaded with `local_files_only=True` — download the weights into your HuggingFace
  cache first. Device and dtype are auto-detected (CUDA bfloat16 / MPS float16 / CPU float32).
- Steering runs distribute a 70B model across 4 GPUs via `torch.multiprocessing`.
- Run everything **from the repository root** as a module (`python -m src....`).

The datasets ship with the repository under `data/`:

```
data/honesty/CAA/self_report_experimental/   # 50 contrastive training scenarios
data/honesty/test/generalized/v4/            # 112 honesty test scenarios across 8 categories
data/compassion/CAA/                         # 50 training questions + 5 contrastive system-prompt variants
data/compassion/test/                        # 40 held-out prompts + the test system-prompt pair
data/mt_bench/, data/alpaca_eval/            # benchmark question sets + judge prompts
data/trait_descriptions.yaml                 # trait definitions used in prompt construction
```

These are the inputs, not the outputs: the on-policy model responses that vector extraction
consumes are **not** included, since they are model-specific and large. Generate them for your
model with stage 1 below — that is what stage 1 is for.

## Pipeline

Commands below are the direct entry points, one per stage. Each stage links to its detail doc.
Flags are spelled out explicitly rather than relying on defaults; artifact paths are derived from
`(trait, dataset, model)` by `src/constants/constants.py` and never passed as absolute paths.
Substitute `--trait compassion` (dropping `--honesty_category`) for the compassion arm, and the
replication model id for the second architecture.

`scripts/` holds SLURM wrappers around these same commands for cluster runs; see
[docs/RUNBOOK.md](docs/RUNBOOK.md). They target a Slurm cluster with H100 nodes and read four
environment variables, so no allocation details are hardcoded:

| Variable | Purpose | Default |
|---|---|---|
| `SBATCH_ACCOUNT` | Slurm accounting group (native Slurm variable; the scripts declare no `--account`) | none — set it before submitting |
| `HF_HOME` | HuggingFace cache holding the model weights | a shared project path |
| `PERSONA_VECTORS_FOLDER` | Where steering vectors are written and read | `<repo>/persona_vectors` off-cluster |
| `TIKTOKEN_ENCODINGS_BASE` | Offline tiktoken encodings for the vLLM judge | a shared scratch path |

The job scripts also `module load` a specific toolchain and build their venv via
`scripts/infrastructure/config_env*.sh`, which assume a Compute Canada wheelhouse — adapt those
two things for your site. The `python -m` commands above have no such assumptions.

### 1. Training responses + judge annotation

Generate the contrastive on-policy response pairs, then annotate them with the judge (the
annotation filters which pairs feed vector extraction).

```bash
python -m src.vector_generation.generate_responses \
  --trait honesty --model-id meta-llama/Llama-3.3-70B-Instruct --n-responses 5 \
  --max-tokens 1024 --temperature 0.6 --batch-size 50

python -m src.llm_judge.runners.caa_dataset \
  --trait honesty --model-id meta-llama/Llama-3.3-70B-Instruct \
  --dataset self_report_experimental \
  --batch-size 200 --save-every-items 200 --score-threshold 65
```

Compassion uses `--n-responses 2` and takes no `--dataset` / `--score-threshold`.
→ [docs/vector_generation.md](docs/vector_generation.md)

### 2. Steering-vector extraction

Extract the response-average CAA vector plus the per-side embeddings, then fit the
logistic-regression direction on those embeddings (CPU only).

```bash
python -m src.vector_generation.generate_vec_caa \
  --trait honesty --dataset self_report_experimental \
  --model-id meta-llama/Llama-3.3-70B-Instruct --per-token-layer 26

python -m src.vector_generation.generate_vec_logistic_regression \
  --pos-features <persona_vectors>/caa/<model_slug>/honesty/self_report_experimental/honesty_self_report_experimental_honest_embeddings_response_avg_<model_slug>.pt \
  --neg-features <persona_vectors>/caa/<model_slug>/honesty/self_report_experimental/honesty_self_report_experimental_dishonest_embeddings_response_avg_<model_slug>.pt \
  --trait honesty --dataset self_report_experimental \
  --model-id meta-llama/Llama-3.3-70B-Instruct \
  --normalization-method projection --C 1.0 --max-iter 1000
```

`<model_slug>` is the model id with `/` replaced by `_`. Compassion sides are named
`positive` / `negative`. `--per-token-layer` additionally saves ragged per-token embeddings, used
by the representational-geometry analysis. → [docs/vector_generation.md](docs/vector_generation.md)

### 3. CAA-vs-LR direction equivalence (CPU only)

Confirms the logistic-regression direction aligns with the CAA mean-difference direction across
layers. Consumes only the extracted vectors; `--model` takes model slugs.

```bash
python -m scripts.verify_lr_vector_direction \
  --trait honesty --dataset self_report_experimental \
  --model meta-llama_Llama-3.3-70B-Instruct Qwen_Qwen3.6-27B
```

### 4. Single-turn layer sweep + LLM judge

Sweeps every layer at every configured coefficient, then scores trait expression and coherence.
Layers and coefficients come from the trait config; the module loops over layers internally
(`--test_mode` subsamples 5 evenly-spaced layers). Run once per technique and position mode.

```bash
python -m src.sequential_steering.sequential_steering_all_layers_honesty \
  --steering_technique swfc --steering_mode all_token_mode \
  --honesty_category self_report_experimental \
  --model_id meta-llama/Llama-3.3-70B-Instruct --exp_name sweep_honesty

python -m src.llm_judge.runners.sweep --trait honesty \
  --results_dir results/logistic_regression/honesty/<model_slug>/all_layers/self_report_experimental/swfc \
  --honesty_category self_report_experimental \
  --model_save_name meta-llama_Llama-3.3-70B-Instruct --checkpoint_interval 4000
```

Techniques: `swfc`, `sttp`, `stmp`, `aligned_baseline`, `malicious_baseline` (the two baselines
take no `--steering_mode`). The compassion module is
`sequential_steering_all_layers_compassion` and takes no `--honesty_category`. Operating points —
the layer/coefficient maximizing trait expression at ≥90% of aligned-baseline coherence — are
selected from the judged sweep and written back into the configs.
→ [docs/steering.md](docs/steering.md)

### 5. Validation at the operating points

Three judge-independent corroborations of the sweep result.

```bash
# per-token target distance, divergence, model cross-entropy (paired generation)
python -m src.dynamic_analysis.dynamic_steering_analysis \
  --config_path configs/config_honesty.yaml \
  --model_id_or_path meta-llama/Llama-3.3-70B-Instruct \
  --honesty_category self_report_experimental \
  --methods fixed,distribution,mirror \
  --n_prompts 50 --max_tokens 2048 --seed_base 42 \
  --temperature 0.6 --top_p 0.9 --load_from_results True --exp_name dynamic_honesty

python -m src.dynamic_analysis.judge_evaluation \
  --input_json <dynamic_analysis_*.json> --trait honesty \
  --scenarios_json <honesty_test_scenarios.json>

# sentence-embedding distance of steered responses to the aligned baseline
python -m src.embedding_distance.compute_embedding_distance \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --honesty_category self_report_experimental \
  --techniques swfc sttp stmp --modes all_token_mode response_mode \
  --method whole --exp_name embdist_honesty

# pairwise coefficient-tournament ELO (Bradley-Terry): stage 1 per method, then stage 2 across methods
python -m src.elo_evaluation.run_elo_evaluation --stage 1 \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty --method swfc \
  --steering_mode all_token_mode --honesty_category self_report_experimental

python -m src.elo_evaluation.run_elo_evaluation --stage 2 \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --steering_mode all_token_mode --honesty_category self_report_experimental
```

The dynamic analysis takes its trait from the config file, not a `--trait` flag, and reuses the
sweep generations (`--load_from_results True`). Its judge stage requires `--scenarios_json` for
honesty and rejects it for compassion. → [docs/analysis.md](docs/analysis.md)

### 6. Multi-turn self-play

Two copies of the model converse: one acts as the steered assistant under the malicious system
prompt, the other as an unsteered user simulator. Honesty runs 5 turns, compassion 10, with N=5
independent runs.

```bash
python -m src.multi_turn_steering.honesty.multi_turn_steering_honesty \
  --model_id meta-llama/Llama-3.3-70B-Instruct --steering_mode all_token_mode \
  --techniques aligned_baseline malicious_baseline swfc sttp stmp \
  --scenario_category factual_claims --honesty_category self_report_experimental \
  --n_runs 5 --exp_name mt_honesty

python -m src.llm_judge.runners.multi_turn --trait honesty \
  --results_dir results/logistic_regression/honesty/<model_slug>/multi_turn/<...>/<exp_name>
```

Compassion: `src.multi_turn_steering.compassion.multi_turn_steering_compassion` with `--n_runs 5
--n_turns 10 --n_questions 20`. Text-repetition metrics (sentence reuse rate, cross-turn 4-gram
repetition) and mean projection come from the qualitative-evaluation suites
(`multi_turn_qualitative_eval_suite`), which the pipeline in
[docs/steering.md](docs/steering.md) chains after the judge.

### 7. Capability benchmarks

Capability retention under steering, at the operating points.

```bash
# MMLU (via lm-eval-harness)
python -m src.benchmarking.mmlu.lm_eval_mmlu \
  --model-id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --honesty-category self_report_experimental --steering-type all --exp_name mmlu_honesty

# MT-Bench: generate + judge per technique, then aggregate
python -m src.benchmarking.mt_bench.generate_responses \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --techniques swfc --steering_mode all_token_mode \
  --honesty_category self_report_experimental --batch_size 100 --exp_name mtbench_honesty
python -m src.benchmarking.mt_bench.evaluate \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --techniques swfc --exp_name mtbench_honesty
python -m src.benchmarking.mt_bench.metrics \
  --model_id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --techniques no_steering swfc sttp stmp --exp_name mtbench_honesty

# AlpacaEval length-controlled win rate (note: hyphenated flags here)
python -m src.benchmarking.alpaca_eval.alpaca_eval_generate \
  --model-id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --honesty-category self_report_experimental --steering-mode all_token_mode \
  --techniques swfc --exp_name alpaca_honesty
python -m src.benchmarking.alpaca_eval.alpaca_eval_judge \
  --model-id meta-llama/Llama-3.3-70B-Instruct --trait honesty \
  --steering-mode all_token_mode --techniques swfc --exp_name alpaca_honesty
```

The StTP/StMP gating threshold can be overridden per run via the `STEERING_THRESHOLD_MODE` /
`STEERING_THRESHOLD_ALPHA` environment variables (default in `configs/steering_threshold.yaml`),
which is how the over-steering alpha sweep is run. → [docs/benchmarks.md](docs/benchmarks.md)

## Repository layout

| Path | Contents |
|---|---|
| `src/steering/` | The three steerers (`swfc.py`, `sttp.py`, `stmp.py`), the factory, and batched steered generation |
| `src/vector_generation/` | Training-response generation, CAA extraction, logistic-regression fit |
| `src/sequential_steering/` | Single-turn layer sweep, per-method sweep drivers, neutral-baseline generation |
| `src/embedding_distance/` | Embedding distance of steered responses to the aligned baseline |
| `src/multi_turn_steering/` | Multi-turn self-play + qualitative evaluation, per trait |
| `src/dynamic_analysis/` | Paired generation, divergence metrics, cross-entropy, judge pass |
| `src/llm_judge/` | Shared vLLM judge engine, rubrics (`system_prompts/`), three runner CLIs |
| `src/elo_evaluation/` | Pairwise tournament judging and Bradley-Terry ratings |
| `src/benchmarking/` | `mmlu/`, `mt_bench/`, `alpaca_eval/` |
| `src/activation_projection/` | Activation recording and projection analysis |
| `src/latency/` | Throughput / memory cost per steering method (not reported in the paper) |
| `src/constants/` | All derived path constants and baseline scores |
| `src/utils/` | Model loading, path and experiment helpers |
| `configs/` | Per-trait steering configs, judge config, threshold config |
| `scripts/` | SLURM wrappers and chained pipelines for the commands above |
| `docs/` | Per-stage detail docs and the full reproduction recipe |

Outputs: steering vectors to `persona_vectors/`, generations and judge scores to `results/`.

## Documentation

| Doc | Contents |
|---|---|
| [docs/REPRODUCTION.md](docs/REPRODUCTION.md) | The whole paper in dependency order, phase by phase |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Per-stage command index, including all three judge CLIs |
| [docs/vector_generation.md](docs/vector_generation.md) | Extraction, normalization, artifact path templates |
| [docs/steering.md](docs/steering.md) | Single-turn sweep and multi-turn self-play |
| [docs/analysis.md](docs/analysis.md) | Validation metrics and representational geometry |
| [docs/benchmarks.md](docs/benchmarks.md) | MMLU, MT-Bench, AlpacaEval, threshold sweep |
| [docs/generalization.md](docs/generalization.md) | The out-of-distribution settings (companion repository) |

Those docs give the `sbatch`/`bash` cluster forms of the commands above; `just --list` enumerates
them via the [justfile](justfile).

## Citation

```bibtex
@misc{herbster2026activationsteeringalignedopenended,
      title={Activation Steering for Aligned Open-ended Generation without Sacrificing Coherence},
      author={Niklas Herbster and Martin Zborowski and Alberto Tosato and Gauthier Gidel and Tommaso Tosato},
      year={2026},
      eprint={2604.08169},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.08169},
}
```

## License

MIT — see [LICENSE](LICENSE).
