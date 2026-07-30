"""Shared resolver for steering vector paths and best layer/coefficient.

Reads from the central honesty config (configs/config_honesty.yaml)
to map (model_id, vector_type, technique, steering_mode) onto the
artifact paths and the (layer, coefficient) tuned for that combination.

Used by both MASK (src/mask/generate_responses.py) and AmongUs
(src/AmongUs/among-agents/amongagents/agent/steered_agent.py) so that the
honesty config remains the single source of truth.

Vector type semantics:
  - "global"    : use the top-level common_steering_config_<model>_<category>
                  paths verbatim, and look up best_layer_<model>_<category>
                  for layer/coefficient.
  - direct      : if a top-level common_steering_config_<model>_<vector_type>
                  exists, use its paths verbatim. Layer/coefficient are looked
                  up in this priority order:
                    1. best_layer_<model>_<vector_type>           (per-category
                       vectors that have their own dataset, e.g. self_report,
                       self_report_expanded, self_report_experimental)
                    2. best_layer_per_category_<model>_<vector_type> (flat
                       per_category key, e.g. self_report_inst_disc)
                    3. best_layer_per_category_<model>_<category>.<vector_type>
                       (legacy nested form)
  - per-category: legacy fallback. Paths are constructed under the
                  per_category/<vector_type>/ subdir next to the global
                  vector. Layer/coefficient come from
                  best_layer_per_category_<model>_<category>.<vector_type>.
  - combined    : a vector_type string with sorted category names joined by
                  '_' (e.g. "self_report_strategic_deception"). Resolved via
                  the "direct" path above when a dedicated common_steering_config
                  block exists.
"""
from __future__ import annotations

import posixpath
from typing import Dict, Optional

import yaml

from src.constants.constants import get_config_for_trait


def load_honesty_config():
    """Load the central honesty config YAML.

    Returns: (cfg_dict, config_path_str)
    """
    config_path = get_config_for_trait("honesty")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg, config_path


def resolve_artifact_paths(
    model_id: str,
    vector_type: str = "global",
    honesty_category: str = "generalized",
    cfg: Optional[Dict] = None,
) -> Dict[str, Optional[str]]:
    """Return the relative artifact paths (under PERSONA_VECTORS_FOLDER) for
    the (model, vector_type, honesty_category) combination.

    Returns a dict with keys: vector_path, bias_path, pos_embeddings_path,
    neg_embeddings_path. Values may be None for techniques that don't need them.
    """
    if cfg is None:
        cfg, _ = load_honesty_config()

    model_save_name = model_id.replace("/", "_")
    common_key = f"common_steering_config_{model_save_name}_{honesty_category}"
    if common_key not in cfg:
        raise KeyError(
            f"Missing config section '{common_key}' for model={model_id}, "
            f"category={honesty_category}"
        )
    common = cfg[common_key]

    if vector_type == "global":
        return {
            "vector_path": common["vector_path"],
            "bias_path": common.get("bias_path"),
            "pos_embeddings_path": common.get("pos_embeddings_path"),
            "neg_embeddings_path": common.get("neg_embeddings_path"),
        }

    # "Direct" mode: a dedicated common_steering_config_<model>_<vector_type>
    # block has the full paths spelled out (used for combined vectors like
    # "self_report_inst_disc" and for Qwen3.6-27B per-category vectors).
    direct_key = f"common_steering_config_{model_save_name}_{vector_type}"
    if direct_key in cfg:
        direct = cfg[direct_key]
        return {
            "vector_path": direct["vector_path"],
            "bias_path": direct.get("bias_path"),
            "pos_embeddings_path": direct.get("pos_embeddings_path"),
            "neg_embeddings_path": direct.get("neg_embeddings_path"),
        }

    # Legacy per-category mode: paths are constructed under
    # per_category/<vector_type>/ relative to the global paths. Paths are kept
    # as forward-slash (POSIX) strings since they are joined with
    # PERSONA_VECTORS_FOLDER (a Linux path) when loaded on the cluster.
    global_vec = common["vector_path"]
    vec_dir = posixpath.dirname(global_vec)
    cat_vec_dir = posixpath.join(vec_dir, "per_category", vector_type)

    pos_emb = common.get("pos_embeddings_path")
    neg_emb = common.get("neg_embeddings_path")
    cat_pos_emb = None
    cat_neg_emb = None
    if pos_emb is not None and neg_emb is not None:
        emb_dir = posixpath.dirname(pos_emb)
        cat_emb_dir = posixpath.join(emb_dir, "per_category", vector_type)
        cat_pos_emb = posixpath.join(
            cat_emb_dir,
            f"honesty_{honesty_category}_honest_all_embeddings_{model_save_name}.pt",
        )
        cat_neg_emb = posixpath.join(
            cat_emb_dir,
            f"honesty_{honesty_category}_dishonest_all_embeddings_{model_save_name}.pt",
        )

    return {
        "vector_path": posixpath.join(
            cat_vec_dir,
            f"honesty_{honesty_category}_{vector_type}_logreg_steering_vector_{model_save_name}.pt",
        ),
        "bias_path": posixpath.join(
            cat_vec_dir,
            f"honesty_{honesty_category}_{vector_type}_logreg_bias_{model_save_name}.pt",
        ),
        "pos_embeddings_path": cat_pos_emb,
        "neg_embeddings_path": cat_neg_emb,
    }


def resolve_layer_and_coef(
    model_id: str,
    technique: str,
    steering_mode: str,
    vector_type: str = "global",
    honesty_category: str = "generalized",
    cfg: Optional[Dict] = None,
) -> Dict:
    """Return {'layer': int, 'coefficient': float} for the (model, vector_type,
    technique, steering_mode) combination.

    Raises KeyError with a precise message if any required section/sub-key
    is missing in the YAML. No silent fallbacks.
    """
    if cfg is None:
        cfg, config_path = load_honesty_config()
    else:
        config_path = "<provided cfg>"

    model_save_name = model_id.replace("/", "_")

    if vector_type == "global":
        best_layer_key = f"best_layer_{model_save_name}_{honesty_category}"
        section_path = best_layer_key
        if best_layer_key not in cfg:
            raise KeyError(
                f"Missing required config section '{best_layer_key}' in {config_path}"
            )
        section = cfg[best_layer_key]
    else:
        # Three accepted naming patterns for per-category / combined vectors:
        #   1. best_layer_<model>_<vector_type>            — per-category
        #      vectors with their own dataset (self_report, self_report_expanded,
        #      self_report_experimental). Same shape as the global
        #      best_layer_<model>_<honesty_category> key, just keyed by the
        #      vector name instead.
        #   2. best_layer_per_category_<model>_<vector_type> — flat form used
        #      for combined vectors (e.g. "self_report_inst_disc").
        #   3. best_layer_per_category_<model>_<category>.<vector_type> —
        #      legacy nested form.
        direct_key = f"best_layer_{model_save_name}_{vector_type}"
        flat_key = f"best_layer_per_category_{model_save_name}_{vector_type}"
        nested_key = f"best_layer_per_category_{model_save_name}_{honesty_category}"
        if direct_key in cfg:
            best_layer_key = direct_key
            section_path = direct_key
            section = cfg[direct_key]
        elif flat_key in cfg:
            best_layer_key = flat_key
            section_path = flat_key
            section = cfg[flat_key]
        elif nested_key in cfg and vector_type in cfg[nested_key]:
            best_layer_key = nested_key
            section_path = f"{nested_key}.{vector_type}"
            section = cfg[nested_key][vector_type]
        else:
            raise KeyError(
                f"Missing layer/coef config for vector_type='{vector_type}': "
                f"none of '{direct_key}', '{flat_key}', or "
                f"'{nested_key}.{vector_type}' present in {config_path}"
            )

    if steering_mode not in section:
        raise KeyError(
            f"Missing steering mode '{steering_mode}' in config section "
            f"'{section_path}' in {config_path}"
        )
    mode_layers = section[steering_mode]
    if technique not in mode_layers:
        raise KeyError(
            f"Missing technique '{technique}' in config section "
            f"'{section_path}.{steering_mode}' in {config_path}"
        )
    entry = mode_layers[technique]
    if "layer" not in entry:
        raise KeyError(
            f"Entry '{section_path}.{steering_mode}.{technique}' missing 'layer' "
            f"in {config_path}"
        )
    return {
        "layer": entry["layer"],
        "coefficient": entry.get("coefficient"),
    }


def resolve_steering_config(
    model_id: str,
    technique: str,
    steering_mode: str,
    vector_type: str = "global",
    honesty_category: str = "generalized",
    cfg: Optional[Dict] = None,
) -> Dict:
    """Convenience: artifact paths + layer + coefficient in one dict."""
    if cfg is None:
        cfg, _ = load_honesty_config()
    paths = resolve_artifact_paths(
        model_id=model_id,
        vector_type=vector_type,
        honesty_category=honesty_category,
        cfg=cfg,
    )
    layer_coef = resolve_layer_and_coef(
        model_id=model_id,
        technique=technique,
        steering_mode=steering_mode,
        vector_type=vector_type,
        honesty_category=honesty_category,
        cfg=cfg,
    )
    return {**paths, **layer_coef}


def normalize_vector_type(per_category_vector: Optional[str], categories: Optional[str]) -> str:
    """Return the canonical vector_type string from MASK-style CLI flags.

    - per_category_vector="self_report"        -> "self_report"
    - categories="self_report,strategic_deception" -> "self_report_strategic_deception"
                                                       (sorted, joined by '_')
    - both None -> "global"

    Mirrors the joining the MASK code already uses for combined-categories vector paths.
    """
    if per_category_vector and categories:
        raise ValueError(
            "per_category_vector and categories are mutually exclusive; "
            "pass only one"
        )
    if per_category_vector:
        return per_category_vector
    if categories:
        selected = sorted(c.strip() for c in categories.split(","))
        return "_".join(selected)
    return "global"
