"""
Compute per-answer cosine similarity between steered responses and
aligned-baseline responses using sentence-transformer embeddings.

Inserts ``embedding_similarity`` fields into each answer dict of the
steered result JSONs (same pattern as the LLM judge inserting scores).

Usage:
    python src/embedding_distance/compute_embedding_distance.py \
        --model_id meta-llama/Llama-3.3-70B-Instruct \
        --method both
"""

import json
import os
from typing import List

import numpy as np
import torch
from sentence_transformers import util

from src.constants.constants import RESULTS_FOLDER, VECTOR_TYPE, VECTOR_TYPE_LABEL
from src.multi_turn_steering.qualitative_evaluation.sentence_template_reuse import (
    _split_sentences,
    SentenceEmbedder,
)


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def _cosine_similarities(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity (embeddings are already L2-normalised)."""
    return np.sum(emb_a * emb_b, axis=1)


def _sentence_nn_similarity(
    baseline_text: str,
    steered_text: str,
    embedder: SentenceEmbedder,
) -> float:
    """Symmetric nearest-neighbour cosine similarity (BERTScore-style F1).

    Computes precision (steered→baseline NN) and recall (baseline→steered NN),
    returns their harmonic mean (F1).
    """
    baseline_sents = _split_sentences(baseline_text)
    steered_sents = _split_sentences(steered_text)

    if not steered_sents or not baseline_sents:
        return 0.0

    baseline_emb = embedder.encode(baseline_sents)
    steered_emb = embedder.encode(steered_sents)

    # [num_steered, num_baseline]
    sims = util.cos_sim(torch.from_numpy(steered_emb), torch.from_numpy(baseline_emb))

    # Precision: for each steered sentence, best match in baseline
    precision = float(torch.max(sims, dim=1).values.mean().item())

    # Recall: for each baseline sentence, best match in steered
    recall = float(torch.max(sims, dim=0).values.mean().item())

    # F1: harmonic mean
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return f1


# ---------------------------------------------------------------------------
# Baseline loading
# ---------------------------------------------------------------------------

def _baseline_candidates(trait: str, model_save_name: str, role: str) -> list:
    """Return candidate baseline filenames for the given trait and role ('aligned' or 'malicious').

    Baselines:  {trait}_{role}_baseline_{model}[_{label}].json
    """
    return [
        f"{trait}_{role}_baseline_{model_save_name}_{VECTOR_TYPE_LABEL}.json",
        f"{trait}_{role}_baseline_{model_save_name}.json",
    ]


def _load_malicious_baseline(results_base: str, model_save_name: str, trait: str,
                             subdir: str = "") -> tuple:
    """Load malicious baseline answers. Returns (path, answers) or ("", [])."""
    candidates = _baseline_candidates(trait, model_save_name, "malicious")

    for subfolder in ("no_steering", "malicious_baseline"):
        for filename in candidates:
            p = os.path.join(results_base, subfolder, subdir, filename)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return p, [item["steered_answer"] for item in data]
    return "", []


def _load_aligned_baseline(results_base: str, model_save_name: str, trait: str,
                           subdir: str = "") -> tuple:
    """Load aligned baseline answers. Returns (path, answers) or ("", [])."""
    candidates = _baseline_candidates(trait, model_save_name, "aligned")

    for filename in candidates:
        p = os.path.join(results_base, "no_steering", subdir, filename)
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return p, [item["steered_answer"] for item in data]

    return "", []


# ---------------------------------------------------------------------------
# Core: compute embedding similarity scores and write to JSON
# ---------------------------------------------------------------------------

_FIELD_WHOLE = "embedding_similarity"
_FIELD_SENT_NN = "embedding_similarity_sentence_nn"


def similarity_score_steered_results(
    results_path: str,
    baseline_answers: List[str],
    baseline_emb: np.ndarray,
    embedder: SentenceEmbedder,
    method: str = "both",
):
    """Add embedding similarity fields to every answer in a steered results JSON."""
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    do_whole = method in ("whole", "both")
    do_sent_nn = method in ("sentence_nn", "both")

    layers_dict = results.get("layers", {})

    for layer_key, per_layer in layers_dict.items():
        if not isinstance(per_layer, dict):
            continue

        for coef_key, answers in per_layer.items():
            if not isinstance(answers, list) or not answers:
                continue

            need_whole = do_whole
            need_sent_nn = do_sent_nn

            steered_texts = [item["steered_answer"] for item in answers]
            if len(steered_texts) != len(baseline_answers):
                raise ValueError(
                    f"Mismatch: {len(steered_texts)} steered vs {len(baseline_answers)} baseline answers"
                )

            if need_whole:
                steered_emb = embedder.encode(steered_texts)
                sims = _cosine_similarities(baseline_emb, steered_emb)
                for i, entry in enumerate(answers):
                    entry[_FIELD_WHOLE] = round(float(sims[i]), 6)

            if need_sent_nn:
                for i, entry in enumerate(answers):
                    entry[_FIELD_SENT_NN] = round(
                        _sentence_nn_similarity(baseline_answers[i], steered_texts[i], embedder),
                        6,
                    )

        print(f"  layer {layer_key} done")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[save] {results_path}")


def similarity_score_baseline(
    baseline_path: str,
    reference_answers: List[str],
    reference_emb: np.ndarray,
    embedder: SentenceEmbedder,
    method: str = "both",
):
    """Add embedding similarity fields to a baseline JSON (flat list format)."""
    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[skip] Baseline is not a flat list: {baseline_path}")
        return

    do_whole = method in ("whole", "both")
    do_sent_nn = method in ("sentence_nn", "both")

    need_whole = do_whole
    need_sent_nn = do_sent_nn

    target_texts = [item["steered_answer"] for item in data]
    if len(target_texts) != len(reference_answers):
        raise ValueError(
            f"Mismatch: {len(target_texts)} target vs {len(reference_answers)} reference answers"
        )

    if need_whole:
        target_emb = embedder.encode(target_texts)
        sims = _cosine_similarities(reference_emb, target_emb)
        for i, entry in enumerate(data):
            entry[_FIELD_WHOLE] = round(float(sims[i]), 6)

    if need_sent_nn:
        for i, entry in enumerate(data):
            entry[_FIELD_SENT_NN] = round(
                _sentence_nn_similarity(reference_answers[i], target_texts[i], embedder),
                6,
            )

    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[save] {baseline_path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def pipeline(model_id: str, trait: str, steering_configs: list, method: str = "both",
             honesty_category: str = None,
             results_dir: str = None, exp_name: str = None):
    model_save_name = model_id.replace("/", "_")

    # Base dir holding the technique subdirs (swfc/sttp/stmp/no_steering).
    if results_dir is not None:
        results_base = results_dir
    elif trait == "honesty":
        # For honesty the path includes a category subdirectory (e.g. self_report_experimental)
        if honesty_category is None:
            raise ValueError("honesty_category is required when trait is 'honesty'")
        results_base = os.path.join(
            RESULTS_FOLDER, VECTOR_TYPE, trait, model_save_name, "all_layers", honesty_category,
        )
    else:
        results_base = os.path.join(
            RESULTS_FOLDER, VECTOR_TYPE, trait, model_save_name, "all_layers",
        )

    # Run-specific nesting below each technique / no_steering dir.
    subdir = exp_name or ""

    print(f"[path] results_base: {results_base}")
    print(f"[path] run subdir: {subdir or '(flat)'}")

    # Load aligned baseline
    aligned_path, aligned_answers = _load_aligned_baseline(results_base, model_save_name, trait, subdir)
    if not aligned_path:
        print(f"[error] Aligned baseline not found for {model_save_name}")
        return

    # Load sentence-transformer
    print("[model] Loading sentence-transformer (all-mpnet-base-v2)...")
    embedder = SentenceEmbedder()

    # Embed aligned baseline once (used for whole-response similarity)
    print("[embed] Embedding aligned baseline answers...")
    aligned_emb = embedder.encode(aligned_answers)
    print(f"[embed] {len(aligned_answers)} answers embedded")

    # Score aligned baseline (similarity to itself = 1.0, but store for consistency)
    print("[score] Aligned baseline...")
    similarity_score_baseline(aligned_path, aligned_answers, aligned_emb, embedder, method)

    # Score malicious baseline (similarity to aligned baseline)
    malicious_path, _ = _load_malicious_baseline(results_base, model_save_name, trait, subdir)
    if malicious_path:
        print("[score] Malicious baseline...")
        similarity_score_baseline(malicious_path, aligned_answers, aligned_emb, embedder, method)
    else:
        print(f"[warn] Malicious baseline not found for {model_save_name}")

    # Score steered results
    for technique, mode in steering_configs:
        technique_dir = os.path.join(results_base, technique, subdir)
        if not os.path.isdir(technique_dir):
            print(f"[skip] {technique_dir}")
            continue

        json_files = [
            f for f in os.listdir(technique_dir)
            if f.endswith(".json") and mode in f and model_save_name in f
        ]
        for json_file in json_files:
            results_path = os.path.join(technique_dir, json_file)
            print(f"[score] {technique}/{json_file}")
            similarity_score_steered_results(results_path, aligned_answers, aligned_emb, embedder, method)

    print("[done] Embedding distance computation complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--trait", default="compassion")
    parser.add_argument("--techniques", nargs="+",
                        default=["swfc", "sttp", "stmp"])
    parser.add_argument("--modes", nargs="+",
                        default=["response_mode", "all_token_mode"])
    parser.add_argument("--method", choices=["whole", "sentence_nn", "both"], default="whole")
    parser.add_argument("--honesty_category", default="self_report_experimental",
                        help="Honesty scenario category (e.g. self_report_experimental). Required when trait is honesty.")
    parser.add_argument("--results_dir", default=None,
                        help="Directory holding the technique subdirs (swfc/sttp/stmp/no_steering). "
                             "Overrides the path auto-derived from trait/model/honesty_category.")
    parser.add_argument("--exp_name", default=None,
                        help="Experiment-name subdir of the steering results (e.g. 04_06_llama_hon).")
    args = parser.parse_args()

    steering_configs = [(t, m) for t in args.techniques for m in args.modes]
    pipeline(
        model_id=args.model_id,
        trait=args.trait,
        steering_configs=steering_configs,
        method=args.method,
        honesty_category=args.honesty_category,
        results_dir=args.results_dir,
        exp_name=args.exp_name,
    )
