"""Stage 2: Judge MT-Bench answers with vLLM using FastChat judge prompts.

Usage:
    python -m src.mt_bench.evaluate \
        --model_id meta-llama/Llama-3.3-70B-Instruct \
        --trait honesty \
        --techniques no_steering swfc
"""

import argparse
import json
import os
import re

import yaml

from src.constants.constants import (
    MT_BENCH_DATA_DIR,
    MT_BENCH_RESULTS_DIR,
    CONFIG_MT_BENCH,
    VECTOR_TYPE,
)
from src.llm_judge.config import JudgeConfig
from src.llm_judge.engine import VLLMJudge
from src.benchmarking.mt_bench.data_utils import (
    load_judge_prompts,
    load_reference_answers,
    select_judge_prompt_name,
    REFERENCE_ANSWER_CATEGORIES,
)


def parse_mt_bench_score(judge_output: str) -> float:
    """Extract a 1-10 score from judge output using [[score]] format.

    Returns:
        Score as float, or -1.0 if parsing fails.
    """
    match = re.search(r"\[\[(\d+\.?\d*)\]\]", judge_output)
    if match:
        score = float(match.group(1))
        if 1.0 <= score <= 10.0:
            return score
    return -1.0


def _build_judge_message(
    prompt_template_data: dict,
    question: dict,
    model_answer: dict,
    turn: int,
) -> list:
    """Build judge messages for a single (question, turn) pair.

    Args:
        prompt_template_data: Judge prompt dict with 'system_prompt' and 'prompt_template'.
        question: Question dict with 'turns' and 'question_id'.
        model_answer: Answer dict with 'turns'.
        reference_answers: Dict mapping question_id -> list of reference turn answers.
        turn: 1 or 2.

    Returns:
        List of message dicts for the judge.
    """
    system_prompt = prompt_template_data["system_prompt"]
    template = prompt_template_data["prompt_template"]

    qid = question["question_id"]
    q_turns = question["turns"]
    a_turns = model_answer["turns"]

    ref_turns = question["reference"] if "reference" in question else []

    if turn == 1:
        content = template.format(
            question=q_turns[0],
            answer=a_turns[0],
            ref_answer_1=ref_turns[0] if ref_turns else "",
        )
    else:
        content = template.format(
            question_1=q_turns[0],
            answer_1=a_turns[0],
            question_2=q_turns[1],
            answer_2=a_turns[1],
            ref_answer_1=ref_turns[0] if ref_turns else "",
            ref_answer_2=ref_turns[1] if len(ref_turns) > 1 else "",
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _load_model_answers(answer_file: str) -> dict:
    """Load model answers from JSONL, keyed by (question_id, coefficient)."""
    answers = {}
    with open(answer_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                coef = data.get("coefficient")
                answers[(data["question_id"], coef)] = data
    return answers


def _load_existing_judgments(judgment_file: str) -> set:
    """Load (question_id, turn, coefficient) triples already judged for resume."""
    judged = set()
    if os.path.exists(judgment_file):
        with open(judgment_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    judged.add((data["question_id"], data["turn"], data.get("coefficient")))
    return judged


def evaluate_technique(
    judge, judge_prompts, questions_by_id,
    answer_file, judgment_file,
):
    """Judge all answers for a single technique, writing results to judgment JSONL.

    Answers are grouped by coefficient and each coefficient's judgments are
    flushed to disk before proceeding to the next, so progress survives
    mid-run crashes.
    """
    model_answers = _load_model_answers(answer_file)
    judged = _load_existing_judgments(judgment_file)
    if judged:
        print(f"  Resuming: {len(judged)} judgments already completed")

    os.makedirs(os.path.dirname(judgment_file), exist_ok=True)

    # Group answers by coefficient
    coef_groups = {}
    for (qid, coef), answer in sorted(model_answers.items()):
        coef_groups.setdefault(coef, []).append((qid, coef, answer))

    total_scored = 0
    total_failed = 0

    for coef, entries in sorted(coef_groups.items(), key=lambda x: (x[0] is not None, x[0] or 0)):
        # Collect work items for this coefficient
        work = []
        for qid, coef_val, answer in entries:
            question = questions_by_id[qid]
            category = question["category"]

            for turn in [1, 2]:
                if (qid, turn, coef_val) in judged:
                    continue
                prompt_name = select_judge_prompt_name(category, turn)
                prompt_data = judge_prompts[prompt_name]
                messages = _build_judge_message(
                    prompt_data, question, answer, turn,
                )
                work.append((qid, turn, category, coef_val, messages))

        if not work:
            continue

        coef_label = f"coefficient={coef}" if coef is not None else "baseline"
        print(f"  Evaluating {len(work)} judgments for {coef_label}...")

        batch_messages = [item[4] for item in work]
        raw_responses = judge.generate(batch_messages)

        with open(judgment_file, "a", encoding="utf-8") as f:
            for (qid, turn, category, coef_val, _), raw in zip(work, raw_responses):
                score = parse_mt_bench_score(raw)
                record = {
                    "question_id": qid,
                    "turn": turn,
                    "category": category,
                    "coefficient": coef_val,
                    "score": score,
                    "judge_output": raw,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        batch_failed = sum(1 for raw in raw_responses if parse_mt_bench_score(raw) < 0)
        total_scored += len(work)
        total_failed += batch_failed
        print(f"  {coef_label}: {len(work)} scored ({batch_failed} failures)")

    if total_scored == 0:
        print("  All judgments already completed")
    else:
        print(f"  Done: {total_scored} scored, {total_failed} parse failures")
        print(f"  Saved to {judgment_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MT-Bench responses with LLM judge"
    )
    parser.add_argument("--model_id", required=True, help="HuggingFace model ID")
    parser.add_argument("--trait", required=True, help="Trait name")
    parser.add_argument(
        "--techniques", nargs="+", required=True,
        help="Techniques to evaluate",
    )
    parser.add_argument("--exp_name", required=True, help="Experiment name (used in result directory)")

    args = parser.parse_args()

    # Load MT-Bench config
    with open(CONFIG_MT_BENCH, "r", encoding="utf-8") as f:
        mt_bench_cfg = yaml.safe_load(f)

    mt_cfg = mt_bench_cfg["mt_bench"]

    judge = VLLMJudge(JudgeConfig.load())

    # Load data
    judge_prompts = load_judge_prompts(
        os.path.join(MT_BENCH_DATA_DIR, mt_cfg["judge_prompts_file"])
    )
    # reference_answers = load_reference_answers(
    #     os.path.join(MT_BENCH_DATA_DIR, mt_cfg["reference_answer_dir"])
    # )

    from src.benchmarking.mt_bench.data_utils import load_questions
    questions = load_questions(
        os.path.join(MT_BENCH_DATA_DIR, mt_cfg["question_file"])
    )
    questions_by_id = {q["question_id"]: q for q in questions}

    model_save_name = args.model_id.replace("/", "_")
    results_dir = os.path.join(
        MT_BENCH_RESULTS_DIR, VECTOR_TYPE, args.trait, model_save_name,
        args.exp_name,
    )

    for technique in args.techniques:
        print(f"\n=== Judging technique: {technique} ===")
        answer_file = os.path.join(results_dir, f"model_answer_{technique}.jsonl")
        judgment_file = os.path.join(results_dir, f"model_judgment_{technique}.jsonl")

        if not os.path.exists(answer_file):
            print(f"  Skipping: answer file not found: {answer_file}")
            continue

        evaluate_technique(
            judge, judge_prompts, questions_by_id,
            answer_file, judgment_file,
        )


if __name__ == "__main__":
    main()
