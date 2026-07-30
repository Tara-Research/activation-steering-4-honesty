"""AlpacaEval 2.0 — Binary Pairwise Judge.

Compares steered outputs against the unsteered baseline using
``openai/gpt-oss-120b`` via vLLM.

Scoring (binary, default):
    1. Let the model generate its judgement (``max_tokens=4096``).
    2. Parse the ``assistantfinalA`` / ``assistantfinalB`` tag from the output.
    3. ``win_prob = 1.0`` if steered wins, ``0.0`` otherwise.
    4. ``win_rate = mean(win_probs)``  (over all prompts)

Position-bias control: for each prompt the assignment of steered/baseline
to model identifiers "A"/"B" is randomised (seeded per technique+coefficient
for deterministic resume).

Input:  per-technique ``generations_{steering_type}_{technique}.json`` files.
Output: per-technique ``alpaca_eval_results_{steering_type}_{technique}.json``.
"""

import argparse
import hashlib
import json
import os
import random
import re

import yaml

from src.benchmarking.alpaca_eval.alpaca_eval_generate import MODE_TO_TYPE, STEERING_TECHNIQUES
from src.constants import constants
from src.llm_judge.config import JudgeConfig
from src.llm_judge.engine import VLLMJudge


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_PATH = os.path.join(constants.DATA_FOLDER, "alpaca_eval", "judge_prompt.yaml")


def _load_judge_prompt():
    """Load system prompt and user template from YAML."""
    with open(_JUDGE_PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_cfg = yaml.safe_load(f)
    return prompt_cfg["system_prompt"], prompt_cfg["user_template"]


_SYSTEM_PROMPT, _USER_TEMPLATE = _load_judge_prompt()


def _build_messages(instruction: str, output_1: str, output_2: str):
    user_text = _USER_TEMPLATE.format(
        instruction=instruction, output_1=output_1, output_2=output_2,
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_from_output(text: str):
    """Extract the binary win-score from one judge output.

    The rubric asks the judge to reason and then emit ``assistantfinalA`` or
    ``assistantfinalB``. Single-caller contract, so it is parsed here rather
    than in src/llm_judge/parsing.py.

    Returns 1.0 if the judge chose B, 0.0 if it chose A, None if unjudgeable.
    """
    match = re.search(r"assistantfinal\s*([AB])", text, flags=re.IGNORECASE)
    if match:
        return 1.0 if match.group(1).upper() == "B" else 0.0

    print(
        f"WARNING: No 'assistantfinalA/B' found in output. "
        f"Generated text (last 200 chars): ...{text[-200:]!r}. "
        f"Returning None (unjudgeable)."
    )
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _stable_seed(seed: int, technique: str, coef_str: str) -> int:
    """Deterministic 32-bit seed, stable across processes/runs."""
    raw = f"{seed}|{technique}|{coef_str}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], byteorder="big")


def evaluate_technique(
    judge, baseline_items, steered_items_by_coef, out_path, result_data, technique
):
    """Run pairwise judging for one steering technique across all coefficients.

    RNG is seeded per (technique, coefficient) for deterministic position
    assignments that are stable across resume.
    """
    technique_results = result_data.get(technique, {})

    for coef_str, steered_items in steered_items_by_coef.items():
        assert len(baseline_items) == len(steered_items), (
            f"Mismatch: {len(baseline_items)} baseline vs {len(steered_items)} steered"
        )

        # Deterministic per-coefficient RNG for position-bias assignment
        coef_rng = random.Random(
            _stable_seed(constants.SEED, technique, coef_str)
        )

        # Build prompts with random position assignment
        all_messages = []
        steered_is_B_flags = []
        for base, steered in zip(baseline_items, steered_items):
            instruction = base["instruction"]
            base_output = base["output"]
            steered_output = steered["output"]

            # steered_is_B: if True, steered output is identifier "B" (output_2)
            steered_is_B = coef_rng.random() < 0.5
            steered_is_B_flags.append(steered_is_B)

            if steered_is_B:
                msgs = _build_messages(instruction, base_output, steered_output)
            else:
                msgs = _build_messages(instruction, steered_output, base_output)
            all_messages.append(msgs)

        raw_texts = judge.generate(all_messages)

        # Score
        per_prompt_results = []
        for idx, (text, steered_is_B, base, steered) in enumerate(zip(
            raw_texts, steered_is_B_flags, baseline_items, steered_items
        )):
            raw_score = _score_from_output(text)
            # raw_score = P(B wins). If steered is B, that's the steered win prob.
            # If steered is A, we need 1 - raw_score.
            if raw_score is None:
                win_prob = None
                print(f"  WARNING: prompt {idx} unjudgeable (coef={coef_str})")
            else:
                win_prob = raw_score if steered_is_B else (1.0 - raw_score)
            per_prompt_results.append({
                "dataset": base.get("dataset"),
                "instruction": base["instruction"],
                "steered_output": steered["output"],
                "baseline_output": base["output"],
                "win_prob": win_prob,
                "steered_is": "B" if steered_is_B else "A",
                "raw_judge_output": text,
            })

        valid_win_probs = [r["win_prob"] for r in per_prompt_results if r["win_prob"] is not None]
        n_unjudgeable = len(per_prompt_results) - len(valid_win_probs)
        if n_unjudgeable > 0:
            print(f"  WARNING: {n_unjudgeable}/{len(per_prompt_results)} prompts unjudgeable "
                  f"(excluded from win_rate)")
        steered_lengths = [s["output_tokens"] for s in steered_items]
        baseline_lengths = [b["output_tokens"] for b in baseline_items]

        technique_results[coef_str] = {
            "win_rate": round(sum(valid_win_probs) / len(valid_win_probs), 4) if valid_win_probs else None,
            "avg_length_steered": round(sum(steered_lengths) / len(steered_lengths), 1),
            "avg_length_baseline": round(sum(baseline_lengths) / len(baseline_lengths), 1),
            "num_prompts": len(per_prompt_results),
            "num_judged": len(valid_win_probs),
            "per_prompt_results": per_prompt_results,
        }
        wr = technique_results[coef_str]['win_rate']
        print(f"  coef={coef_str}: win_rate={wr:.4f}" if wr is not None else f"  coef={coef_str}: win_rate=N/A (no judgeable prompts)")

        # Incremental save after each coefficient
        result_data[technique] = technique_results
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

    return technique_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AlpacaEval 2.0 — logprobs pairwise judge"
    )
    parser.add_argument("--model-id", type=str,
                        default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--trait", type=str, default="compassion")
    parser.add_argument("--steering-mode", type=str, default="all_token_mode",
                        choices=list(MODE_TO_TYPE),
                        help="Steering mode: 'all_token_mode' or 'response_mode'")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name (used in result directory)")
    parser.add_argument("--techniques", nargs="+", required=True,
                        choices=STEERING_TECHNIQUES,
                        help="Steering techniques to evaluate")
    parser.add_argument("--force", action="store_true",
                        help="Force re-evaluation even if results exist")
    args = parser.parse_args()

    # ---- Derive paths (same logic as alpaca_eval_generate.py) ----
    model_save_name = args.model_id.replace("/", "_")
    steering_type = MODE_TO_TYPE[args.steering_mode]
    output_dir = os.path.join(
        constants.ALPACA_EVAL_RESULTS_DIR, constants.VECTOR_TYPE, args.trait,
        model_save_name, args.exp_name,
    )

    # ---- Load baseline ----
    baseline_path = os.path.join(
        output_dir, f"generations_{steering_type}_baseline_no_steering.json"
    )
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    baseline_items = baseline_data["items"]
    baseline_metadata = baseline_data["metadata"]
    print(f"Loaded {len(baseline_items)} baseline items from {baseline_path}")

    # ---- Load judge ----
    judge = VLLMJudge(JudgeConfig.load())

    # ---- Evaluate each technique ----
    for technique in args.techniques:
        gen_path = os.path.join(
            output_dir, f"generations_{steering_type}_{technique}.json"
        )
        out_path = os.path.join(
            output_dir, f"alpaca_eval_results_{steering_type}_{technique}.json"
        )

        with open(gen_path, "r", encoding="utf-8") as f:
            tech_data = json.load(f)

        steered_by_coef = tech_data["items"]

        # Resume: load existing results, skip completed coefficients
        done_coefs = set()
        if os.path.exists(out_path) and not args.force:
            with open(out_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
            done_coefs = set(result_data.get(technique, {}).keys())
            if done_coefs:
                print(f"Resuming {technique}: {len(done_coefs)} coefficients already evaluated")
        else:
            result_data = {
                "metadata": {**baseline_metadata, "exp_name": args.exp_name, "technique": technique},
            }

        todo_coefs = {k: v for k, v in steered_by_coef.items() if k not in done_coefs}
        if not todo_coefs:
            print(f"\n=== {technique}: all coefficients already evaluated, skipping ===")
            continue

        print(f"\n=== Judging: {technique} ({len(todo_coefs)} coefficients remaining) ===")

        evaluate_technique(
            judge, baseline_items, todo_coefs, out_path, result_data, technique
        )
