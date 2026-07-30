"""N-run metrics for MASK-inspired multi-turn honesty evaluation.

Re-exports from the shared trait-generic module for backward compatibility.
The honesty-specific wrapper passes trait_name="honesty" by default.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.multi_turn_steering.n_run_metrics import compute_n_run_metrics as _compute_n_run_metrics

Message = Dict[str, str]


def compute_n_run_metrics(
    technique_histories: List[List[Message]],
    aligned_histories: List[List[Message]],
    consistency_scores: Dict[str, Optional[float]],
) -> Dict[str, Any]:
    """Per-turn metrics: consistency, raw honesty, knowledge-normalized honesty.

    Thin wrapper around the shared ``compute_n_run_metrics`` with
    ``trait_name="honesty"`` for backward compatibility.
    """
    return _compute_n_run_metrics(
        technique_histories=technique_histories,
        aligned_histories=aligned_histories,
        consistency_scores=consistency_scores,
        trait_name="honesty",
    )
