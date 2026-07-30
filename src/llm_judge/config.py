"""Judge model configuration.

Single source of truth: ``configs/config_llm_judge.yaml`` -> ``llm_judge_defaults``.
Imports no vLLM, so this module is safe to import without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.utils.path_utils import load_llm_judge_config


@dataclass(frozen=True)
class JudgeConfig:
    """vLLM engine + sampling parameters for the judge.

    Every field is required and comes from ``llm_judge_defaults``. There are no
    defaults on purpose: a missing key raises a TypeError naming it, and an
    unknown key raises rather than being silently ignored.
    """

    model: str
    temperature: float
    top_p: float
    max_tokens: int
    min_tokens: int
    reasoning_effort: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    max_num_seqs: int

    @classmethod
    def load(cls) -> "JudgeConfig":
        defaults, _ = load_llm_judge_config()
        return cls(**defaults)
