import json
import os
from typing import Optional

from src.constants.constants import MT_BENCH_DATA_DIR


def load_mt_bench_turn1(n: Optional[int] = None) -> list[dict]:
    """Load MT-Bench turn-1 prompts.

    Returns a list of {"question_id": int, "category": str, "prompt": str} dicts.
    If `n` is given, truncates to the first n entries (order matches the JSONL).
    """
    path = os.path.join(MT_BENCH_DATA_DIR, "question.jsonl")
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            items.append({
                "question_id": row["question_id"],
                "category": row["category"],
                "prompt": row["turns"][0],
            })
    if n is not None:
        items = items[:n]
    return items
