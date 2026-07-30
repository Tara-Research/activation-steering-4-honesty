"""LLM-as-judge scoring for dynamic steering analysis results.

Scores every generated response on trait expression and coherence, writes the
scores back into the dynamic-analysis JSON, and adds per-condition means with
95% confidence intervals.

Two traits are supported and they are scored differently, because they measure
different things:

* ``compassion`` -- graded 0-100 against the compassion rubric, written to
  ``compassion_score``.
* ``honesty`` -- judged against the scenario's gold truth, written to
  ``honesty_score``. The dynamic results only carry the user prompt, so the gold
  truth is looked up from the scenario file the prompts came from (the same
  lookup the single-turn sweep uses).

    python -m src.dynamic_analysis.judge_evaluation --input_json=PATH --trait=compassion
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from src.llm_judge.config import JudgeConfig
from src.llm_judge.engine import VLLMJudge
from src.llm_judge.parsing import parse_int_score, parse_judge_json
from src.llm_judge.prompts import (
    build_honesty_messages,
    build_qa_messages,
    load_prompt,
    load_trait_prompt,
)
from src.llm_judge.runners.ground_truth import load_ground_truth_data, normalize_message

TRAIT_SCORE_KEY = {"compassion": "compassion_score", "honesty": "honesty_score"}


def load_dynamic_results(json_path: str) -> dict:
    """Load dynamic analysis results from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_ci_95(values: List[float]) -> Tuple[float, float]:
    """95% confidence interval of the mean, via the t-distribution."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    return tuple(stats.t.interval(0.95, len(values) - 1,
                                  loc=np.mean(values), scale=stats.sem(values)))


def _iter_responses(results: dict):
    """Yield ``(container, user_prompt, response_text)`` for every generation.

    Steered generations live directly on the prompt dict; baselines live in a
    nested ``baseline`` / ``malicious_baseline`` block.
    """
    for result_data in results.get("results", {}).values():
        for prompt_data in result_data.get("prompts", []):
            user_prompt = prompt_data.get("user_prompt", "")

            if "baseline" in prompt_data:
                yield prompt_data["baseline"], user_prompt, prompt_data["baseline"].get("text", "")
            elif "experimental_text" in prompt_data:
                yield prompt_data, user_prompt, prompt_data.get("experimental_text", "")
            elif "malicious_baseline" in prompt_data:
                block = prompt_data["malicious_baseline"]
                yield block, user_prompt, block.get("text", "")


def _trait_conversations(trait: str, pairs, trait_prompt, ground_truth):
    """Build the trait-phase conversation for each (question, answer) pair."""
    if trait == "compassion":
        return [build_qa_messages(trait_prompt, q, a) for q, a in pairs], list(range(len(pairs)))

    conversations, kept = [], []
    for idx, (question, answer) in enumerate(pairs):
        truth = ground_truth.get(normalize_message(question))
        if truth is None:
            continue
        conversations.append(build_honesty_messages(
            trait_prompt,
            gold_truth=truth["gold_truth"],
            user_message=truth["user_message"],
            model_response=answer,
            category=truth["category"],
        ))
        kept.append(idx)
    return conversations, kept


def evaluate_dynamic_results(
    input_json: str,
    trait: str,
    scenarios_json: Optional[str] = None,
    output_json: Optional[str] = None,
) -> dict:
    """Score every response in a dynamic-analysis JSON and save the augmented file.

    Args:
        input_json: Path to a ``dynamic_analysis_*.json``.
        trait: ``compassion`` or ``honesty``.
        scenarios_json: Scenario file holding the gold truths. Required for
            ``--trait honesty``; rejected for compassion.
        output_json: Defaults to ``<input>_with_judge.json``.
    """
    if trait not in TRAIT_SCORE_KEY:
        raise ValueError(f"Unsupported trait {trait!r}. Expected one of {sorted(TRAIT_SCORE_KEY)}.")
    if trait == "honesty" and not scenarios_json:
        raise ValueError(
            "--scenarios_json is required for --trait honesty: honesty is judged "
            "against each scenario's gold truth, which dynamic results do not carry."
        )
    if trait == "compassion" and scenarios_json:
        raise ValueError("--scenarios_json is only used with --trait honesty.")

    print(f"[JudgeEvaluation] Loading results from: {input_json}")
    results = load_dynamic_results(input_json)
    output_json = output_json or f"{input_json.rsplit('.', 1)[0]}_with_judge.json"

    containers, pairs = [], []
    for container, question, answer in _iter_responses(results):
        containers.append(container)
        pairs.append((question, answer))
    print(f"[JudgeEvaluation] Found {len(pairs)} responses to evaluate")

    if trait == "honesty":
        trait_prompt = load_prompt("honesty_universal")
        ground_truth = load_ground_truth_data(scenarios_json)
    else:
        trait_prompt = load_trait_prompt("compassion", "compassion")
        ground_truth = None

    judge = VLLMJudge(JudgeConfig.load())
    score_key = TRAIT_SCORE_KEY[trait]

    print(f"[JudgeEvaluation] Scoring {trait}...")
    conversations, kept = _trait_conversations(trait, pairs, trait_prompt, ground_truth)
    if len(kept) < len(pairs):
        print(f"[JudgeEvaluation] {len(pairs) - len(kept)} responses had no matching "
              f"ground-truth scenario and were skipped")
    for idx, raw in zip(kept, judge.generate(conversations)):
        parsed = parse_judge_json(raw) if trait == "honesty" else None
        containers[idx][score_key] = (
            parsed["honesty_score"] if parsed else
            (None if trait == "honesty" else parse_int_score(raw))
        )
        containers[idx][f"{score_key}_raw"] = raw

    print("[JudgeEvaluation] Scoring coherence...")
    coherence_prompt = load_prompt("coherence")
    coherence_conversations = [build_qa_messages(coherence_prompt, q, a) for q, a in pairs]
    for container, raw in zip(containers, judge.generate(coherence_conversations)):
        container["coherence_score"] = parse_int_score(raw)
        container["coherence_raw"] = raw

    _add_aggregates(results, score_key)

    results["judge_config"] = {
        "backend": "vllm",
        "trait": trait,
        **vars(JudgeConfig.load()),
    }

    print(f"[JudgeEvaluation] Saving results to: {output_json}")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _print_summary(results, score_key, trait)
    return results


def _add_aggregates(results: dict, score_key: str) -> None:
    """Per-condition mean + 95% CI for the trait and coherence scores."""
    for result_data in results.get("results", {}).values():
        collected: Dict[str, List[float]] = {score_key: [], "coherence_score": []}

        for prompt_data in result_data.get("prompts", []):
            for response_type in ("baseline", "malicious_baseline"):
                if response_type in prompt_data:
                    source = prompt_data[response_type]
                    break
            else:
                source = prompt_data
            for key in collected:
                value = source.get(key)
                if value is not None:
                    collected[key].append(value)

        aggregate = result_data.setdefault("aggregate", {})
        for key, values in collected.items():
            if not values:
                continue
            label = "coherence" if key == "coherence_score" else key.replace("_score", "")
            aggregate[f"mean_{key}"] = float(np.mean(values))
            low, high = compute_ci_95(values)
            aggregate[f"ci_95_{label}"] = [float(low), float(high)]
            aggregate[f"n_{label}_valid"] = len(values)


def _print_summary(results: dict, score_key: str, trait: str) -> None:
    print("\n" + "=" * 60)
    print(f"JUDGE EVALUATION SUMMARY ({trait})")
    print("=" * 60)
    for result_key, result_data in results.get("results", {}).items():
        aggregate = result_data.get("aggregate", {})

        def fmt(key):
            value = aggregate.get(key, "N/A")
            return f"{value:.1f}" if isinstance(value, float) else value

        print(f"{result_key:45} | {trait[:10]:>10}: {fmt(f'mean_{score_key}'):>6} "
              f"| Coherence: {fmt('mean_coherence_score'):>6}")
    print("=" * 60)


def main():
    """CLI entry point using Fire."""
    import fire
    fire.Fire(evaluate_dynamic_results)


if __name__ == "__main__":
    main()
