"""Shared factory that maps a steering technique name to a steerer instance.

Used by multi-turn qualitative eval suites to replicate the same steering
conditions during activation recording that were active during generation.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


_BASELINES = frozenset({"aligned_baseline", "malicious_baseline", "no_steering"})


def create_steerer_for_eval(
    technique: str,
    model: torch.nn.Module,
    steering_vectors: torch.Tensor,
    specific_config: dict,
    negative_embeddings: Optional[torch.Tensor] = None,
    positive_embeddings: Optional[torch.Tensor] = None,
    layer: int = None,
    coefficient: float = 1.0,
    bias: Optional[torch.Tensor] = None,
) -> Optional[Any]:
    """Return a steerer context-manager matching *technique*, or ``None`` for baselines.

    Parameters
    ----------
    technique:
        One of ``aligned_baseline``, ``malicious_baseline``, ``no_steering``,
        ``swfc`` (Steering with Fixed Coefficient),
        ``sttp`` (Steering to Target Projection),
        ``stmp`` (Steering to Mirror Projection).
    model:
        The loaded HuggingFace model (already on the correct device).
    steering_vectors:
        Tensor of shape ``[L, H]`` — one direction per layer.
    specific_config:
        Per-technique config dict (from the result JSON's
        ``specific_steering_config`` or from the YAML).  Expected keys for
        steering techniques: ``layer``, ``coefficient``, ``steering_type``,
        ``steering_direction``.
    negative_embeddings:
        Tensor ``[L, N, H]`` — needed for StTP / StMP.
    positive_embeddings:
        Tensor ``[L, N, H]`` — needed for StTP / StMP.
    """
    if technique in _BASELINES:
        return None

    if layer is None:
        raise ValueError("layer must be specified for steering techniques")

    positions = specific_config["steering_type"]
    steering_direction = specific_config["steering_direction"]

    if technique == "swfc":
        from src.steering.swfc import SteeringWithFixedCoef

        return SteeringWithFixedCoef(
            model,
            steering_vector=steering_vectors[layer],
            coeff=coefficient,
            layer_idx=layer,
            positions=positions,
        )

    if technique == "sttp":
        from src.steering.sttp import SteeringToTargetProjection

        if negative_embeddings is None or positive_embeddings is None:
            raise ValueError(
                "sttp requires negative_embeddings and positive_embeddings"
            )
        return SteeringToTargetProjection(
            model,
            steering_vectors=steering_vectors,
            negative_embeddings=negative_embeddings,
            positive_embeddings=positive_embeddings,
            layer_idx=layer,
            positions=positions,
            steering_direction=steering_direction,
            coefficient=coefficient,
            bias=bias,
        )

    if technique == "stmp":
        from src.steering.stmp import SteeringToMirrorProjection

        if negative_embeddings is None or positive_embeddings is None:
            raise ValueError(
                "stmp requires negative_embeddings and positive_embeddings"
            )
        return SteeringToMirrorProjection(
            model,
            steering_vectors=steering_vectors,
            negative_embeddings=negative_embeddings,
            positive_embeddings=positive_embeddings,
            layer_idx=layer,
            positions=positions,
            steering_direction=steering_direction,
            coefficient=coefficient,
            bias=bias,
        )

    raise ValueError(f"Unknown steering technique: {technique!r}")
