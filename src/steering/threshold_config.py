"""Load + validate the global gating-threshold config for StTP / StMP.

The steerer classes (`SteeringToTargetProjection`, `SteeringToMirrorProjection`)
read this to decide how their per-layer gating threshold is computed:

    mode == "bias"          -> threshold = -bias/||v||      (class midpoint; original behavior)
    mode == "neg_mean_std"  -> threshold = mu_neg + alpha*std_neg  (conservative, neg-cluster anchored)

This only controls *which tokens get steered* (the gate); it does not affect the
StTP target s = mu_pos + coefficient*std_pos. `alpha` is kept distinct from the
steerers' `coefficient`.

The config is intentionally a single global file (not keyed by model/trait):
`configs/steering_threshold.yaml`.

Per-run / per-job override (used by the AlpacaEval threshold-alpha sweep): the
environment variables ``STEERING_THRESHOLD_MODE`` and ``STEERING_THRESHOLD_ALPHA``
take precedence over the YAML when set. Each sweep job is its own process, so this
lets parallel jobs use different alphas without touching the shared YAML file.
"""
from __future__ import annotations

import functools
import numbers
import os
from typing import Any, Dict

import yaml

from src.constants.constants import ROOT_FOLDER

THRESHOLD_CONFIG_PATH = ROOT_FOLDER / "configs" / "steering_threshold.yaml"

VALID_MODES = {"bias", "neg_mean_std"}

MODE_ENV_VAR = "STEERING_THRESHOLD_MODE"
ALPHA_ENV_VAR = "STEERING_THRESHOLD_ALPHA"


@functools.lru_cache(maxsize=1)
def load_threshold_config() -> Dict[str, Any]:
    """Read and validate the gating-threshold config (env vars override the YAML).

    Returns a dict with keys "mode" (str) and "alpha" (float, only meaningful
    for mode == "neg_mean_std").

    ``STEERING_THRESHOLD_MODE`` / ``STEERING_THRESHOLD_ALPHA``, when set, override
    the YAML values. The result is cached for the process (env is read once on
    first call); each SLURM job is a fresh process, so a sweep gets the right alpha.

    Raises (no silent fallbacks, per project convention):
      FileNotFoundError  if the config file is missing,
      KeyError           if "mode" is absent from the YAML,
      ValueError         if the resolved "mode" is not in {"bias", "neg_mean_std"},
                         STEERING_THRESHOLD_ALPHA is non-numeric, or the resolved
                         "mode" == "neg_mean_std" and "alpha" is missing/non-numeric.
    """
    if not THRESHOLD_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Gating-threshold config not found at {THRESHOLD_CONFIG_PATH}"
        )

    with open(THRESHOLD_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or "mode" not in cfg:
        raise KeyError(
            f"'mode' missing from gating-threshold config {THRESHOLD_CONFIG_PATH}"
        )

    # Env vars override the YAML (used by the per-alpha sweep).
    mode = os.environ.get(MODE_ENV_VAR, cfg["mode"])
    if mode not in VALID_MODES:
        raise ValueError(
            f"gating-threshold 'mode' must be one of {sorted(VALID_MODES)}, got {mode!r}"
        )

    alpha_env = os.environ.get(ALPHA_ENV_VAR)
    if alpha_env is not None:
        try:
            alpha = float(alpha_env)
        except ValueError:
            raise ValueError(
                f"{ALPHA_ENV_VAR} must be numeric, got {alpha_env!r}"
            )
    else:
        alpha = cfg.get("alpha")

    if mode == "neg_mean_std" and not isinstance(alpha, numbers.Real):
        raise ValueError(
            "gating-threshold mode 'neg_mean_std' requires a numeric 'alpha' "
            f"(set 'alpha' in {THRESHOLD_CONFIG_PATH} or the {ALPHA_ENV_VAR} env var)"
        )

    return {"mode": mode, "alpha": float(alpha) if alpha is not None else 0.0}
