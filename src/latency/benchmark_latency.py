"""Latency benchmark for steering methods on MT-Bench turn-1 prompts.

Measures tokens/sec, ms/token, and peak GPU memory for the unsteered baseline and
each steering method (SwFC, StTP, StMP), at one or more batch sizes. Reuses the
existing wrappers in src.steering.steering_sampling so the measured code path is
identical to the experiment path.

Example:
    python -m src.latency.benchmark_latency \\
        --model meta-llama/Llama-3.3-70B-Instruct \\
        --vector_dataset self_report_experimental \\
        --positions all \\
        --batch_sizes 1 8 \\
        --methods baseline swfc sttp stmp
"""

import argparse
import gc
import json
import math
import os
import statistics
from datetime import datetime
from typing import Optional

import torch
import yaml

from src.constants.constants import (
    CONFIG_HONESTY,
    PERSONA_VECTORS_FOLDER,
    RESULTS_FOLDER,
)
from src.latency.mt_bench_prompts import load_mt_bench_turn1
from src.latency.timing import (
    CudaTimer,
    peak_memory_gb_all_devices,
    reset_peak_memory_all_devices,
)
from src.steering.steering_sampling import (
    sample,
    sample_stmp,
    sample_sttp,
    sample_swfc,
)
from src.utils.model_utils import load_model


METHOD_CHOICES = ["baseline", "swfc", "sttp", "stmp"]
POSITION_TO_MODE_KEY = {"all": "all_token_mode", "response": "response_mode"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    p.add_argument("--vector_dataset", default="self_report_experimental",
                   help="Dataset key used in configs/config_honesty.yaml "
                        "(e.g. self_report_experimental, diverse_with_sycophancy, generalized).")
    p.add_argument("--positions", default="all", choices=["all", "response"])
    p.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 8])
    p.add_argument("--methods", nargs="+", default=METHOD_CHOICES, choices=METHOD_CHOICES)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--n_prompts", type=int, default=80, help="Number of MT-Bench turn-1 prompts.")
    p.add_argument("--warmup_batches", type=int, default=2)
    p.add_argument("--output_dir", default=None,
                   help="Defaults to {RESULTS_FOLDER}/latency/{model_slug}/{timestamp}/")
    return p.parse_args()


def resolve_paths_and_layers(model: str, vector_dataset: str, positions: str) -> dict:
    """Read vector paths + per-method layer/coefficient from the honesty config.

    Raises KeyError if any expected key is missing (no silent fallbacks).
    """
    with open(CONFIG_HONESTY, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_slug = model.replace("/", "_")
    paths_key = f"common_steering_config_{model_slug}_{vector_dataset}"
    layers_key = f"best_layer_{model_slug}_{vector_dataset}"
    mode_key = POSITION_TO_MODE_KEY[positions]

    paths_block = config[paths_key]
    layers_block = config[layers_key][mode_key]

    return {
        "vector_path": os.path.join(PERSONA_VECTORS_FOLDER, paths_block["vector_path"]),
        "bias_path": os.path.join(PERSONA_VECTORS_FOLDER, paths_block["bias_path"]),
        "neg_embeddings_path": os.path.join(PERSONA_VECTORS_FOLDER, paths_block["neg_embeddings_path"]),
        "pos_embeddings_path": os.path.join(PERSONA_VECTORS_FOLDER, paths_block["pos_embeddings_path"]),
        "layer_per_method": {m: layers_block[m]["layer"] for m in ("swfc", "sttp", "stmp")},
        "coef_per_method": {m: layers_block[m]["coefficient"] for m in ("swfc", "sttp", "stmp")},
    }


def chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def count_output_tokens(tokenizer, outputs: list[str]) -> int:
    return sum(len(tokenizer(o, add_special_tokens=False).input_ids) for o in outputs)


def run_one_batch(
    method: str,
    chunk_prompts: list[str],
    *,
    model,
    tokenizer,
    steering_vectors: torch.Tensor,
    neg_embeddings: Optional[torch.Tensor],
    pos_embeddings: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    layer: int,
    coef: float,
    positions: str,
    batch_size: int,
    max_new_tokens: int,
) -> tuple[list[str], float]:
    """Run one batch through the chosen method, return (outputs, elapsed_ms)."""
    with CudaTimer() as timer:
        if method == "baseline":
            outputs = sample(
                model, tokenizer, conversations=None, prompts=chunk_prompts,
                bs=batch_size, max_tokens=max_new_tokens, temperature=0, min_tokens=1,
            )
        elif method == "swfc":
            outputs = sample_swfc(
                model, tokenizer, conversations=None, prompts=chunk_prompts,
                vector=steering_vectors[layer], layer=layer, coef=coef,
                bs=batch_size, max_tokens=max_new_tokens, temperature=0, min_tokens=1,
                steering_type=positions,
            )
        elif method == "sttp":
            outputs = sample_sttp(
                model, tokenizer, conversations=None, prompts=chunk_prompts,
                steering_vectors=steering_vectors,
                negative_embeddings=neg_embeddings,
                positive_embeddings=pos_embeddings,
                layer=layer, coefficient=coef,
                bs=batch_size, max_tokens=max_new_tokens, temperature=0, min_tokens=1,
                steering_type=positions, bias=bias,
            )
        elif method == "stmp":
            outputs = sample_stmp(
                model, tokenizer, conversations=None, prompts=chunk_prompts,
                steering_vectors=steering_vectors,
                negative_embeddings=neg_embeddings,
                positive_embeddings=pos_embeddings,
                layer=layer, coefficient=coef,
                bs=batch_size, max_tokens=max_new_tokens, temperature=0, min_tokens=1,
                steering_type=positions, bias=bias,
            )
        else:
            raise ValueError(f"Unknown method {method!r}")
    return outputs, timer.elapsed_ms


def main() -> None:
    args = parse_args()

    resolved = resolve_paths_and_layers(args.model, args.vector_dataset, args.positions)
    layer_per_method = resolved["layer_per_method"]
    coef_per_method = resolved["coef_per_method"]

    output_dir = args.output_dir or os.path.join(
        RESULTS_FOLDER, "latency",
        args.model.replace("/", "_"),
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"[latency] loading model {args.model}")
    model, tokenizer = load_model(args.model)

    print(f"[latency] loading steering vectors from {resolved['vector_path']}")
    steering_vectors = torch.load(resolved["vector_path"], weights_only=False)

    needs_distributions = any(m in args.methods for m in ("sttp", "stmp"))
    neg_embeddings = pos_embeddings = bias = None
    if needs_distributions:
        print(f"[latency] loading neg/pos embeddings + bias for StTP/StMP")
        neg_embeddings = torch.load(resolved["neg_embeddings_path"], weights_only=False)
        pos_embeddings = torch.load(resolved["pos_embeddings_path"], weights_only=False)
        bias = torch.load(resolved["bias_path"], weights_only=False)

    print(f"[latency] loading MT-Bench turn-1 prompts (n={args.n_prompts})")
    mt_bench = load_mt_bench_turn1(n=args.n_prompts)
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": q["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in mt_bench
    ]

    results: list[dict] = []
    for batch_size in args.batch_sizes:
        chunks = list(chunked(formatted_prompts, batch_size))
        n_warmup = min(args.warmup_batches, len(chunks))

        for method in args.methods:
            layer = layer_per_method.get(method, -1) if method != "baseline" else -1
            coef = coef_per_method.get(method, 0.0) if method != "baseline" else 0.0
            print(f"[latency] === method={method} batch_size={batch_size} layer={layer} coef={coef} ===")

            print(f"[latency]   warmup ({n_warmup} batches)")
            for warmup_chunk in chunks[:n_warmup]:
                _ = run_one_batch(
                    method, warmup_chunk,
                    model=model, tokenizer=tokenizer,
                    steering_vectors=steering_vectors,
                    neg_embeddings=neg_embeddings, pos_embeddings=pos_embeddings, bias=bias,
                    layer=layer, coef=coef, positions=args.positions,
                    batch_size=batch_size, max_new_tokens=args.max_new_tokens,
                )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            reset_peak_memory_all_devices()

            per_batch_ms: list[float] = []
            per_batch_output_tokens: list[int] = []
            total_output_tokens = 0
            sample_outputs: list[str] = []
            for i, chunk_prompts in enumerate(chunks):
                outputs, ms = run_one_batch(
                    method, chunk_prompts,
                    model=model, tokenizer=tokenizer,
                    steering_vectors=steering_vectors,
                    neg_embeddings=neg_embeddings, pos_embeddings=pos_embeddings, bias=bias,
                    layer=layer, coef=coef, positions=args.positions,
                    batch_size=batch_size, max_new_tokens=args.max_new_tokens,
                )
                per_batch_ms.append(ms)
                n_toks = count_output_tokens(tokenizer, outputs)
                per_batch_output_tokens.append(n_toks)
                total_output_tokens += n_toks
                if i == 0:
                    sample_outputs = outputs[: min(5, len(outputs))]

            peak_mem_gb = peak_memory_gb_all_devices()
            wall_time_s = sum(per_batch_ms) / 1000.0

            # Per-batch ms/token, then mean + standard error across batches.
            per_batch_ms_per_token = [
                ms / n for ms, n in zip(per_batch_ms, per_batch_output_tokens) if n > 0
            ]
            n_ok = len(per_batch_ms_per_token)
            if n_ok > 1:
                mean_mpt = statistics.mean(per_batch_ms_per_token)
                sd_mpt = statistics.stdev(per_batch_ms_per_token)
                sem_mpt = sd_mpt / math.sqrt(n_ok)
            elif n_ok == 1:
                mean_mpt = per_batch_ms_per_token[0]
                sd_mpt = 0.0
                sem_mpt = 0.0
            else:
                mean_mpt = sd_mpt = sem_mpt = 0.0

            cell = {
                "method": method,
                "batch_size": batch_size,
                "layer": layer,
                "coefficient": coef,
                "wall_time_s": wall_time_s,
                "total_output_tokens": total_output_tokens,
                "tokens_per_sec": (total_output_tokens / wall_time_s) if wall_time_s > 0 else 0.0,
                "ms_per_token": (1000.0 * wall_time_s / total_output_tokens) if total_output_tokens > 0 else 0.0,
                "ms_per_token_mean_across_batches": mean_mpt,
                "ms_per_token_stdev_across_batches": sd_mpt,
                "ms_per_token_sem_across_batches": sem_mpt,
                "n_batches": n_ok,
                "peak_gpu_memory_gb": peak_mem_gb,
                "per_batch_ms": per_batch_ms,
                "per_batch_output_tokens": per_batch_output_tokens,
                "sample_outputs": sample_outputs,
            }
            results.append(cell)
            print(f"[latency]   wall={wall_time_s:.2f}s tokens={total_output_tokens} "
                  f"tok/s={cell['tokens_per_sec']:.2f} "
                  f"ms/tok={mean_mpt:.3f}±{sem_mpt:.3f}(SEM,n={n_ok}) "
                  f"peak_mem={peak_mem_gb:.2f}GB")

    report = {
        "config": {
            "model": args.model,
            "vector_dataset": args.vector_dataset,
            "positions": args.positions,
            "max_new_tokens": args.max_new_tokens,
            "n_prompts": args.n_prompts,
            "warmup_batches": args.warmup_batches,
            "layer_per_method": layer_per_method,
            "coefficient_per_method": coef_per_method,
        },
        "results": results,
    }

    out_path = os.path.join(output_dir, "benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[latency] wrote {out_path}")


if __name__ == "__main__":
    main()
