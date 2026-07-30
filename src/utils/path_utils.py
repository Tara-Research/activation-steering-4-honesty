"""Helpers for building multi-turn result / metrics file paths."""

import os

import yaml

from src.constants import constants
from src.constants.constants import (
    RESULTS_FOLDER,
    PLOTS_FOLDER,
    VECTOR_TYPE,
    VECTOR_TYPE_LABEL,
)


def build_multi_turn_results_base(trait: str, model_save_name: str) -> str:
    """Base directory for multi-turn results: results/logistic_regression/{trait}/{model}/multi_turn"""
    return os.path.join(RESULTS_FOLDER, VECTOR_TYPE, trait, model_save_name, "multi_turn")


def build_multi_turn_plots_base(trait: str, model_save_name: str) -> str:
    """Base directory for multi-turn plots: plots/logistic_regression/{trait}/{model}/multi_turn"""
    return os.path.join(PLOTS_FOLDER, VECTOR_TYPE, trait, model_save_name, "multi_turn")


def load_llm_judge_config(config_path: str = None) -> tuple:
    """Load unified LLM judge config. Returns (defaults_dict, prompt_paths_dict)."""
    if config_path is None:
        config_path = str(constants.CONFIG_LLM_JUDGE)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["llm_judge_defaults"], cfg["system_prompt_paths"]


def merge_generation_defaults(cfg: dict, section_key: str) -> dict:
    """Return cfg[section_key] with generation_defaults merged underneath."""
    gen_defaults = cfg.get("generation_defaults", {})
    merged = {**gen_defaults, **cfg[section_key]}
    required_keys = ["batch_size", "top_p", "max_tokens", "temperature", "min_tokens"]
    missing = [k for k in required_keys if k not in merged]
    if missing:
        raise KeyError(
            f"Missing required generation keys {missing} after merging "
            f"generation_defaults into '{section_key}'. "
            f"Add them to generation_defaults or the section itself."
        )
    return merged

_SHORT_TECHNIQUE_NAME = {
    "sttp": "sttp",
    "swfc": "swfc",
    "stmp": "stmp",
}

_SHORT_MODE_NAME = {
    "all_token_mode": "all",
    "response_mode": "response",
}

_BASELINES = {"no_steering", "aligned_baseline", "malicious_baseline"}


_TECHNIQUE_DIR = {
    "aligned_baseline": "no_steering",
    "malicious_baseline": "no_steering",
}


def build_single_turn_output_path(
    trait: str,
    technique: str,
    mode: str = None,
) -> str:
    """Build the relative output_json path for a single-turn all-layers experiment.

    Returns paths like::

        all_layers/no_steering/{trait}_{baseline}_logreg.json
        all_layers/{technique}/{trait}_{technique}_{mode}_logreg.json
    """
    vt = VECTOR_TYPE_LABEL

    technique_dir = _TECHNIQUE_DIR.get(
        technique,
        technique.replace("_mult_layers", "").replace("_multi_layer", ""),
    )

    if technique in _BASELINES:
        return f"all_layers/{technique_dir}/{trait}_{technique}_{vt}.json"

    if mode is None:
        raise ValueError(f"mode required for non-baseline technique '{technique}'")

    is_multi = "mult_layers" in technique or "multi_layer" in technique
    suffix = "_mult_layer" if is_multi else ""

    return f"all_layers/{technique_dir}/{trait}_{technique_dir}_{mode}{suffix}_{vt}.json"


def build_multi_turn_output_path(
    trait: str, technique: str, mode: str = None,
) -> str:
    """Build the ``output_json`` portion of a multi-turn result path.

    For baselines::

        {technique}/{trait}_multi_turn_{technique}_logreg.json

    For steering methods::

        {technique}/{trait}_multi_turn_{short_name}_{mode_short}_logreg.json

    Parameters
    ----------
    trait : str
        ``"compassion"`` or ``"honesty"``.
    technique : str
        Steering technique name (e.g. ``"sttp"``).
    mode : str or None
        ``"all_token_mode"`` or ``"response_mode"``.  Required for non-baseline
        techniques; ignored for baselines.
    """
    vt = VECTOR_TYPE_LABEL

    if technique in _BASELINES:
        return f"{technique}/{trait}_multi_turn_{technique}_{vt}.json"

    short = _SHORT_TECHNIQUE_NAME.get(technique, technique)
    mode_short = _SHORT_MODE_NAME.get(mode)
    if mode_short is None:
        raise ValueError(
            f"mode must be one of {list(_SHORT_MODE_NAME)} for technique "
            f"'{technique}', got {mode!r}"
        )
    return f"{technique}/{trait}_multi_turn_{short}_{mode_short}_{vt}.json"
