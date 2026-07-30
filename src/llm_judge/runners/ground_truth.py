"""Honesty scenario ground truth, keyed by normalized user message.

Steering results record the user message but not the gold truth, so the honesty
judge has to look it up in the scenario file the responses were generated from.

No vLLM and no torch, so score aggregation and backfill tools can import this on
a login node.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

import yaml

from src.constants.constants import DATA_FOLDER, get_config_for_trait
from src.utils.path_utils import merge_generation_defaults


def normalize_message(text: str) -> str:
    """Normalize user message text for matching."""
    text = " ".join(text.split())
    return text.strip().lower()


def honesty_scenarios_path(model_save_name: str, honesty_category: str) -> str:
    """Scenario file the honesty responses were generated from.

    Derived from the trait config rather than passed around as a literal path,
    so callers never hardcode artifact locations.
    """
    with open(get_config_for_trait("honesty"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    section = f"common_steering_config_{model_save_name}_{honesty_category}"
    return os.path.join(DATA_FOLDER, merge_generation_defaults(cfg, section)["data_path"])


def load_ground_truth_data(data_path: str) -> Dict[str, Dict[str, Any]]:
    """Index a scenario file by normalized ``user_message``.

    Returns ``{normalized_user_message: {gold_truth, user_message, category}}``.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    message_to_data = {}
    for scenario in data.get("scenarios", []):
        user_message = scenario.get("user_message", "").strip()
        message_to_data[normalize_message(user_message)] = {
            "gold_truth": scenario.get("gold_truth", ""),
            "user_message": user_message,
            "category": scenario.get("category", ""),
        }

    return message_to_data
