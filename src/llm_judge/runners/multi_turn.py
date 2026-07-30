"""LLM judge over multi-turn self-play results.

Two stages, both writing back into the result JSON in place:

* **trait + coherence** -- every assistant turn is scored twice, once against the
  trait rubric and once against the coherence rubric.
* **consistency** (honesty only) -- for each turn, the N independent runs are
  shown to the judge together and scored for factual agreement.

The multi-turn pipelines import these functions directly, so their names and
signatures are part of the interface.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from src.llm_judge.config import JudgeConfig
from src.llm_judge.engine import VLLMJudge
from src.llm_judge.parsing import parse_int_score
from src.llm_judge.prompts import build_qa_messages, load_prompt, load_trait_prompt, partial_format

_TRAIT_RUBRIC = {"honesty": "honesty_multi_turn", "compassion": "compassion_multi_turn"}


def _build_scenario_context_str(judge_context: dict) -> str:
    """Render a scenario's judge_context into the {SCENARIO_CONTEXT} block."""
    lines = []
    knows = judge_context.get("what_the_judge_knows", {})

    if "latent_truth" in knows:
        lines.append(f"**Latent Truth:** {knows['latent_truth']}")
    if "user_goal" in knows:
        lines.append(f"**User Goal:** {knows['user_goal']}")
    if "honest_behavior_definition" in knows:
        lines.append("**Honest Behavior:**")
        lines.extend(f"  - {item}" for item in knows["honest_behavior_definition"])
    if "dishonest_behavior_definition" in knows:
        lines.append("**Dishonest Behavior:**")
        lines.extend(f"  - {item}" for item in knows["dishonest_behavior_definition"])

    return "\n".join(lines)


def multi_turn_qa_pairs(conversation: List[dict], trait: str) -> List[tuple]:
    """One (question, answer) pair per assistant turn in a conversation.

    For compassion the question is the opening user message plus the most recent
    one, so the judge sees what the conversation is about; honesty uses only the
    most recent user message. That asymmetry is longstanding behaviour and is
    preserved deliberately.
    """
    initial_user = ""
    for message in conversation:
        if message.get("role") == "user" and message.get("content"):
            initial_user = message["content"]
            break

    pairs = []
    last_user = ""
    for message in conversation:
        role, content = message["role"], message["content"]
        if role == "user" and content:
            last_user = content
        elif role == "assistant" and content:
            if initial_user and last_user and trait != "honesty":
                question = f"{initial_user}\n\n{last_user}"
            else:
                question = last_user
            pairs.append((question, content))
    return pairs


def evaluate_multi_turn_file(
    input_json: str,
    *,
    trait: str,
    trait_prompt_template: str,
    coherence_prompt: str,
    judge: VLLMJudge,
    batch_size: int = 20,
    force_recompute: bool = False,
) -> None:
    """Score every assistant turn in one multi-turn result file."""
    with open(input_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    is_single = isinstance(results, dict)
    conversations = [results] if is_single else results

    for conversation in conversations:
        if "conversation_histories" in conversation:
            histories = conversation["conversation_histories"]
        else:
            histories = [conversation.get("history") or conversation.get("conversation_history", [])]

        # Honesty rubrics need the scenario's ground truth to judge disclosure.
        if trait == "honesty":
            judge_context = conversation.get("scenario", {}).get("judge_context", {})
            trait_prompt = partial_format(
                trait_prompt_template,
                SCENARIO_CONTEXT=(
                    _build_scenario_context_str(judge_context)
                    if judge_context else "No scenario context available."
                ),
            )
        else:
            trait_prompt = trait_prompt_template

        for history in histories:
            assistant_entries = [m for m in history if m.get("role") == "assistant"]
            if not assistant_entries:
                continue
            if not force_recompute and "trait_expression_score" in assistant_entries[0]:
                continue

            pairs = multi_turn_qa_pairs(history, trait)

            for system_prompt, score_key in (
                (trait_prompt, "trait_expression_score"),
                (coherence_prompt, "coherence_score"),
            ):
                raw_outputs = judge.generate(
                    [build_qa_messages(system_prompt, q, a) for q, a in pairs],
                    batch_size=batch_size,
                )
                if len(raw_outputs) != len(assistant_entries):
                    raise RuntimeError(
                        f"Judge returned {len(raw_outputs)} scores for "
                        f"{len(assistant_entries)} assistant messages in {input_json}."
                    )
                for entry, raw in zip(assistant_entries, raw_outputs):
                    entry[f"{score_key}_raw"] = raw
                    entry[score_key] = parse_int_score(raw)

    with open(input_json, "w", encoding="utf-8") as f:
        json.dump(conversations[0] if is_single else conversations, f,
                  ensure_ascii=False, indent=2)
    print(f"[save] updated entries with judge scores -> {input_json}")


def _iter_result_files(results_dir: str, technique: Optional[str], require_multi_turn_dir: bool):
    """Yield judgeable result files, optionally restricted to one technique.

    Restricting by technique is how the pipelines fan this stage out into one
    SLURM job per technique.
    """
    for root, _, files in os.walk(results_dir):
        if require_multi_turn_dir and "multi_turn" not in root:
            continue
        for fname in sorted(files):
            if not fname.endswith(".json"):
                continue
            if fname.endswith("_metrics.json") or fname.endswith("_n_run_metrics.json"):
                continue
            if technique is not None and not fname.startswith(technique + "_"):
                continue
            yield os.path.join(root, fname)


def _run_trait_judge(
    trait: str,
    results_dir: Optional[str],
    force_recompute: bool,
    technique: Optional[str],
    require_multi_turn_dir: bool,
) -> None:
    if results_dir is None:
        raise ValueError("results_dir must be specified.")

    trait_prompt_template = load_trait_prompt(_TRAIT_RUBRIC[trait], trait)
    coherence_prompt = load_prompt("coherence")
    judge = VLLMJudge(JudgeConfig.load())

    for fpath in _iter_result_files(results_dir, technique, require_multi_turn_dir):
        print(f"[judge] evaluating: {fpath}")
        evaluate_multi_turn_file(
            fpath,
            trait=trait,
            trait_prompt_template=trait_prompt_template,
            coherence_prompt=coherence_prompt,
            judge=judge,
            batch_size=20,
            force_recompute=force_recompute,
        )


# ---------------------------------------------------------------------------
# Entry points imported by scripts/pipelines/multi_turn_*_pipeline.sh
# ---------------------------------------------------------------------------

def run_llm_judge_honesty_multi_turn(
    config_file: str = None,
    results_dir: str = None,
    force_recompute: bool = False,
    technique: str = None,
) -> None:
    """Score trait + coherence over multi-turn honesty results.

    ``config_file`` is accepted for call-site compatibility and is unused: judge
    settings come from configs/config_llm_judge.yaml.
    """
    _run_trait_judge("honesty", results_dir, force_recompute, technique,
                     require_multi_turn_dir=True)


def run_llm_judge_compassion_multi_turn(
    config_file: str = None,
    results_dir: str = None,
    force_recompute: bool = False,
    technique: str = None,
) -> None:
    """Score trait + coherence over multi-turn compassion results.

    ``config_file`` is accepted for call-site compatibility and is unused.
    """
    _run_trait_judge("compassion", results_dir, force_recompute, technique,
                     require_multi_turn_dir=False)


def evaluate_consistency_across_runs(
    all_histories: List[List[dict]],
    judge: VLLMJudge,
    scenario: dict,
    consistency_prompt_template: str,
) -> Dict[str, Optional[int]]:
    """Score factual agreement across the N runs, per turn.

    All N responses for a turn go into one prompt as ``[Response i]`` blocks, so
    the judge compares them directly.
    """
    judge_context = scenario.get("judge_context", {})
    consistency_prompt = partial_format(
        consistency_prompt_template,
        SCENARIO_CONTEXT=(
            _build_scenario_context_str(judge_context)
            if judge_context else "No scenario context available."
        ),
    )

    def assistant_turns(history):
        return [m for m in history if m.get("role") == "assistant"]

    user_turns = [m for m in all_histories[0] if m.get("role") == "user"]
    n_turns = len(assistant_turns(all_histories[0]))

    scores: Dict[str, Optional[int]] = {}
    for turn_idx in range(n_turns):
        responses = [
            assistant_turns(h)[turn_idx]["content"]
            for h in all_histories
            if turn_idx < len(assistant_turns(h))
        ]
        question = user_turns[turn_idx]["content"] if turn_idx < len(user_turns) else ""
        formatted = "\n\n".join(f"[Response {i + 1}]\n{r}" for i, r in enumerate(responses))

        messages = build_qa_messages(consistency_prompt, question, formatted)
        raw_outputs = judge.generate([messages], batch_size=1)
        scores[f"turn_{turn_idx}"] = parse_int_score(raw_outputs[0]) if raw_outputs else None

    return scores


def run_consistency_judge_honesty_multi_turn(
    config_file: str = None,
    results_dir: str = None,
    force_recompute: bool = False,
    technique: str = None,
) -> None:
    """Write per-turn ``consistency_scores`` into every N-run honesty result file.

    ``config_file`` is accepted for call-site compatibility and is unused.
    """
    if results_dir is None:
        raise ValueError("results_dir must be specified.")

    consistency_prompt_template = load_prompt("consistency")
    judge = VLLMJudge(JudgeConfig.load())

    for fpath in _iter_result_files(results_dir, technique, require_multi_turn_dir=True):
        with open(fpath, "r", encoding="utf-8") as f:
            result = json.load(f)

        if not isinstance(result, dict) or "conversation_histories" not in result:
            continue
        if not force_recompute and "consistency_scores" in result:
            print(f"[consistency] already scored, skipping: {fpath}")
            continue

        print(f"[consistency] evaluating: {fpath}")
        result["consistency_scores"] = evaluate_consistency_across_runs(
            all_histories=result["conversation_histories"],
            judge=judge,
            scenario=result.get("scenario", {}),
            consistency_prompt_template=consistency_prompt_template,
        )

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[save] consistency scores -> {fpath}")


def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the LLM judge over multi-turn self-play results.")
    parser.add_argument("--trait", default="compassion", choices=["compassion", "honesty"])
    parser.add_argument("--results_dir", required=True,
                        help="Judge every multi-turn result file under this directory")
    parser.add_argument("--force_recompute", action="store_true")
    parser.add_argument("--technique", default=None,
                        help="Only judge files named <technique>_*.json")
    args = parser.parse_args(argv)

    runner = (run_llm_judge_honesty_multi_turn if args.trait == "honesty"
              else run_llm_judge_compassion_multi_turn)
    runner(results_dir=args.results_dir, force_recompute=args.force_recompute,
           technique=args.technique)


if __name__ == "__main__":
    main()
