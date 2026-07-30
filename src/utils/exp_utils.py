"""Experiment directory versioning: exp_1, exp_2, ..."""

from __future__ import annotations

import os
import re


def resolve_experiment_dir(
    base_dir: str,
    exp_name: str | None = None,
    create: bool = False,
) -> str:
    """Resolve or create a versioned ``exp_N`` subdirectory under *base_dir*.

    Parameters
    ----------
    base_dir : str
        Parent directory that contains (or will contain) ``exp_N`` folders.
    exp_name : str | None
        Explicit experiment name (e.g. ``"exp_3"``).
        * If given with ``create=True``: creates that directory and returns the name.
        * If given with ``create=False``: returns the name if it exists, else ValueError.
        * If ``None`` with ``create=True``: auto-creates ``exp_{max+1}``.
        * If ``None`` with ``create=False``: returns the latest ``exp_{max}``, else ValueError.
    create : bool
        Whether we are in "generation" mode (may create dirs) vs. "read" mode.

    Returns
    -------
    str
        The experiment directory **name** (e.g. ``"exp_2"``), *not* a full path.
    """
    _EXP_RE = re.compile(r"^exp_(\d+)$")

    def _scan_existing() -> list[int]:
        if not os.path.isdir(base_dir):
            return []
        return sorted(
            int(m.group(1))
            for d in os.listdir(base_dir)
            if (m := _EXP_RE.match(d)) and os.path.isdir(os.path.join(base_dir, d))
        )

    if create:
        if exp_name is not None:
            os.makedirs(os.path.join(base_dir, exp_name), exist_ok=True)
            return exp_name
        existing = _scan_existing()
        next_n = (max(existing) + 1) if existing else 1
        name = f"exp_{next_n}"
        os.makedirs(os.path.join(base_dir, name), exist_ok=True)
        return name

    # read mode
    if exp_name is not None:
        path = os.path.join(base_dir, exp_name)
        if not os.path.isdir(path):
            raise ValueError(
                f"Experiment directory does not exist: {path}"
            )
        return exp_name

    existing = _scan_existing()
    if not existing:
        raise ValueError(
            f"No exp_N directories found under {base_dir}"
        )
    return f"exp_{max(existing)}"
