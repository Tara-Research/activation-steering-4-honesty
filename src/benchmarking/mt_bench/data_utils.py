"""Utilities for loading MT-Bench data files (questions, references, judge prompts)."""

import json
import os
from typing import Dict, List

from src.constants.constants import MT_BENCH_DATA_DIR

MT_BENCH_CATEGORIES = [
    "writing",
    "roleplay",
    "reasoning",
    "math",
    "coding",
    "extraction",
    "stem",
    "humanities",
]

BASELINE_TECHNIQUES = {"no_steering", "aligned_baseline", "malicious_baseline"}

REFERENCE_ANSWER_CATEGORIES = {"math", "coding", "reasoning"}


def load_questions(question_file: str = None) -> List[Dict]:
    """Load MT-Bench questions from JSONL.

    Each line: {"question_id": int, "category": str, "turns": [str, str]}

    Returns:
        List of question dicts, one per line.
    """
    if question_file is None:
        question_file = os.path.join(MT_BENCH_DATA_DIR, "question.jsonl")
    questions = []
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def load_reference_answers(
    reference_dir: str = None, source: str = "gpt-4"
) -> Dict[int, List[str]]:
    """Load reference answers keyed by question_id.

    Args:
        reference_dir: Directory containing reference answer JSONL files.
        source: Reference model name (default "gpt-4").

    Returns:
        Dict mapping question_id -> list of turn answers.
    """
    if reference_dir is None:
        reference_dir = os.path.join(MT_BENCH_DATA_DIR, "reference_answer")
    ref_file = os.path.join(reference_dir, f"{source}.jsonl")
    references = {}
    with open(ref_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            qid = data["question_id"]
            turns = data["choices"][0]["turns"]
            references[qid] = turns
    return references


def load_judge_prompts(judge_prompts_file: str = None) -> Dict[str, Dict]:
    """Load judge prompt templates keyed by prompt name.

    Each line: {"name": str, "type": str, "system_prompt": str, "prompt_template": str, ...}

    Returns:
        Dict mapping prompt name -> full prompt dict.
    """
    if judge_prompts_file is None:
        judge_prompts_file = os.path.join(MT_BENCH_DATA_DIR, "judge_prompts.jsonl")
    prompts = {}
    with open(judge_prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prompts[data["name"]] = data
    return prompts


def filter_one_per_category(questions: list) -> list:
    """Return the first question from each category (for test mode).

    Args:
        questions: Full list of MT-Bench question dicts.

    Returns:
        List with one question per category (8 total).
    """
    seen = set()
    filtered = []
    for q in questions:
        cat = q["category"]
        if cat not in seen:
            seen.add(cat)
            filtered.append(q)
    return filtered


def select_judge_prompt_name(category: str, turn: int) -> str:
    """Select the correct judge prompt template for a given category and turn.

    Args:
        category: MT-Bench category (e.g. "math", "writing").
        turn: 1 or 2.

    Returns:
        Judge prompt name string.
    """
    if category in REFERENCE_ANSWER_CATEGORIES:
        if turn == 1:
            return "single-math-v1"
        return "single-math-v1-multi-turn"
    if turn == 1:
        return "single-v1"
    return "single-v1-multi-turn"
