# persona_playground command runner.
#
# `just` is a command runner: named shortcuts for the long sbatch/bash commands, so you
# don't have to memorize flags. NEW TO just? Three things to know:
#   1. install once on the cluster (no sudo): download the prebuilt `just` binary into
#      ~/bin and add it to PATH, or `cargo install just` / `conda install -c conda-forge just`.
#   2. `just --list`            -> see every recipe with its description.
#      `just <recipe>`          -> run it (submits the sbatch/bash command shown in the body).
#      `just var=val <recipe>`  -> override a default (e.g. `just model=Qwen/Qwen3.6-27B mmlu-honesty`).
#   3. recipe bodies below ARE the canonical commands (same as docs/REPRODUCTION.md); `{{var}}`
#      substitutes a variable. Per the explicit-args rule, every flag is written out.
#
# Full ordered paper reproduction: docs/REPRODUCTION.md. Per-stage index: docs/RUNBOOK.md.

# ---- overridable defaults (override on the CLI: `just var=val <recipe>`) ----
model       := "meta-llama/Llama-3.3-70B-Instruct"   # second model: Qwen/Qwen3.6-27B
honesty_cat := "self_report_experimental"
mode        := "all_token_mode"
techniques  := "no_steering swfc sttp stmp"

# Default: list recipes.
default:
    @just --list

# ============================ P0 — vector generation ============================
# Generate training responses (judge annotation runs in the same job).
gen-honesty-responses:
    sbatch scripts/pipelines/responses_and_judge_caa_honesty.sh --model-id {{model}} --n-responses 5 --input-json data/honesty/CAA/self_report_experimental/self_report_experimental_scenarios.json --output-json data/honesty/CAA/self_report_experimental/self_report_experimental_responses_{{replace(model, "/", "_")}}.json

gen-compassion-responses:
    sbatch scripts/pipelines/responses_and_judge_caa_compassion.sh --model-id {{model}}

# Extract CAA response-average activations + fit the logistic-regression vector.
gen-honesty-vec:
    sbatch scripts/vector_generation/generate_vec_and_logreg.sh --model-id {{model}} --trait honesty --dataset {{honesty_cat}} --normalization-method projection

gen-compassion-vec:
    sbatch scripts/vector_generation/generate_vec_and_logreg.sh --model-id {{model}} --trait compassion --dataset compassion --normalization-method projection

# ============================ P1 — geometry figures =============================
# CAA-vs-LR direction-equivalence cosine (login node, no GPU).
caa-vs-logreg:
    python -m scripts.verify_lr_vector_direction --trait honesty --dataset {{honesty_cat}} --model {{replace(model, "/", "_")}}

# ============================ P2 — sweep + judge ================================
# One launcher per trait runs the whole stage (5 steering -> 4 judge, afterok-chained).
# The launcher submits the judge itself; there is no separate judge recipe.
sweep-honesty:
    bash scripts/steering/launch_sequential_honesty_all_configs.sh --model_id {{model}} --honesty_category {{honesty_cat}} --techniques swfc,sttp,stmp,aligned_baseline,malicious_baseline --exp_name v2_sweep_honesty

sweep-compassion:
    bash scripts/steering/launch_sequential_compassion_all_configs.sh --model_id {{model}} --techniques swfc,sttp,stmp,aligned_baseline,malicious_baseline --exp_name v2_sweep_compassion

# ============================ P3 — validation ==================================
# Per-token target distance + aggregate metrics + cross-entropy (src/dynamic_analysis).
dynamic-honesty:
    bash scripts/dynamic_analysis/run_dynamic_experiment_single.sh --model_id {{model}} --trait honesty --honesty_category {{honesty_cat}} --exp_name v2_dynamic_honesty

embedding-distance-honesty:
    bash scripts/embedding_distance/compute_embedding_distance.sh --model_id {{model}} --trait honesty --honesty_category {{honesty_cat}} --techniques "swfc sttp stmp" --exp_name v2_embdist_honesty

# Coefficient-tournament ELO (the only ELO in the paper); run stage 1 then stage 2.
coef-elo-honesty:
    bash scripts/pipelines/elo_stage1_pipeline.sh --model_id {{model}} --trait honesty --steering_mode {{mode}} --honesty_category {{honesty_cat}} --exp_name v2_elo_honesty

# ============================ P4 — multi-turn ==================================

# Multi-turn self-play: honesty 5 turns + judge + n-run + text-quality metrics.
multiturn-honesty:
    bash scripts/pipelines/multi_turn_honesty_pipeline.sh --model_id {{model}} --honesty_category {{honesty_cat}} --scenario_category factual_claims --layer_mode single_layer --n_runs 5 --exp_name v2_mt_honesty

multiturn-compassion:
    bash scripts/pipelines/multi_turn_compassion_pipeline.sh --model_id {{model}} --n_runs 5 --exp_name v2_mt_compassion

# ============================ P5 — capability benchmarks =======================

# MMLU capability retention under steering.
mmlu-honesty:
    sbatch scripts/benchmarks/mmlu_benchmark.sh --model-id {{model}} --trait honesty --honesty-category {{honesty_cat}} --steering-type all --exp_name v2_mmlu_honesty

mt-bench-honesty:
    bash scripts/pipelines/mt_bench_pipeline.sh --model_id {{model}} --trait honesty --honesty_category {{honesty_cat}} --techniques "{{techniques}}" --steering_mode {{mode}} --exp_name v2_mtbench_honesty

alpaca-honesty:
    bash scripts/pipelines/alpaca_eval_pipeline.sh --model_id {{model}} --trait honesty --honesty_category {{honesty_cat}} --techniques "{{techniques}}" --steering_mode {{mode}} --exp_name v2_alpaca_honesty

# ============================ phase umbrellas ==================================
# Submit a whole phase (honesty path; see docs/REPRODUCTION.md for compassion + full ordering).
repro-p0: gen-honesty-vec gen-compassion-vec
repro-p1: caa-vs-logreg
repro-p2: sweep-honesty sweep-compassion
repro-p3: dynamic-honesty embedding-distance-honesty coef-elo-honesty
repro-p4: multiturn-honesty multiturn-compassion
repro-p5: mmlu-honesty mt-bench-honesty alpaca-honesty
