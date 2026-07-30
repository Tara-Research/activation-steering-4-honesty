#!/usr/bin/env python3
"""
Extract paper values from experimental results.

This script extracts baseline values and best operating points from the JSON result files
and saves them in structured formats (JSON, CSV) for use in the paper.

Usage:
    python -m src.utils.extract_paper_values
    python -m src.utils.extract_paper_values --output_dir results/paper_values
    python -m src.utils.extract_paper_values --coherence_threshold_pct 90
"""

import json
import csv
import math
import re
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import argparse
from datetime import datetime

from src.constants import constants

# Map display method names (used in tables/reports) to config YAML keys
DISPLAY_TO_CONFIG_METHOD = {
    'SwFC': 'swfc',
    'StTP': 'sttp',
    'StMP': 'stmp',
}

# Map internal mode names to YAML config keys
DISPLAY_TO_CONFIG_MODE = {
    'all': 'all_token_mode',
    'resp': 'response_mode',
}


def _resolve_result_file(base_dir: Path, filename: str) -> Optional[Path]:
    """Resolve a result file under base_dir, tolerating extra nesting levels
    (e.g. <exp_name>/, or the legacy features_response_avg/<exp_name>/ layout
    of old results).

    Returns the direct path if it exists, otherwise the most recently modified
    recursive match, or None if nothing is found.
    """
    direct = base_dir / filename
    if direct.exists():
        return direct
    if not base_dir.is_dir():
        return None
    matches = sorted(base_dir.rglob(filename), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  NOTE: {len(matches)} files named {filename} under {base_dir}; using most recent: {matches[-1]}")
    return matches[-1]


def _standard_error(scores: list[float]) -> float:
    """Compute standard error of the mean (sample std / sqrt(n))."""
    n = len(scores)
    if n < 2:
        return 0.0
    mean = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / (n - 1)
    return math.sqrt(variance / n)


@dataclass
class OperatingPoint:
    """Represents a single operating point (layer, coefficient, scores)."""
    trait: str  # "compassion" or "honesty"
    method: str  # "SwFC", "StTP", "StMP", "aligned", "malicious"
    mode: str  # "all", "resp", or "--" for baselines
    layer: Optional[int]
    coef: Optional[float]
    trait_score: float
    coherence_score: float
    n_samples: int
    trait_score_se: float = 0.0
    coherence_score_se: float = 0.0


@dataclass
class UniversalLayerResult:
    """Result of the universal layer selection algorithm."""
    universal_layer: Optional[int]
    universal_accepted: bool
    tolerance_used: float
    universal_points: dict  # method -> OperatingPoint at universal layer
    independent_points: dict  # method -> OperatingPoint (independent best)
    universal_min_score: float
    per_method_drops: dict  # method -> drop from independent best


def extract_compassion_baseline(filepath: str, baseline_type: str) -> Optional[OperatingPoint]:
    """Extract compassion baseline values from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    trait_scores = [r.get('trait_expression_score') for r in data if r.get('trait_expression_score') is not None]
    coherence_scores = [r.get('coherence_score') for r in data if r.get('coherence_score') is not None]

    if not trait_scores or not coherence_scores:
        return None

    return OperatingPoint(
        trait="compassion",
        method=baseline_type,
        mode="--",
        layer=None,
        coef=None,
        trait_score=round(sum(trait_scores) / len(trait_scores), 1),
        coherence_score=round(sum(coherence_scores) / len(coherence_scores), 1),
        n_samples=len(trait_scores),
        trait_score_se=round(_standard_error(trait_scores), 2),
        coherence_score_se=round(_standard_error(coherence_scores), 2),
    )


def extract_honesty_baseline(filepath: str, baseline_type: str) -> Optional[OperatingPoint]:
    """Extract honesty baseline values from JSON file (list format)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return None

    judge_scores = []
    coherence_scores = []

    for item in data:
        js = item.get('judge_score')
        if isinstance(js, dict):
            score = js.get('score')
        else:
            score = js
        if score is not None:
            judge_scores.append(score)

        cs = item.get('coherence_score')
        if cs is not None:
            coherence_scores.append(cs)

    if not judge_scores or not coherence_scores:
        return None

    return OperatingPoint(
        trait="honesty",
        method=baseline_type,
        mode="--",
        layer=None,
        coef=None,
        trait_score=round(sum(judge_scores) / len(judge_scores), 1),
        coherence_score=round(sum(coherence_scores) / len(coherence_scores), 1),
        n_samples=len(judge_scores),
        trait_score_se=round(_standard_error(judge_scores), 2),
        coherence_score_se=round(_standard_error(coherence_scores), 2),
    )


def extract_baseline_coherence(filepath: str) -> Optional[float]:
    """Extract mean coherence score from a baseline JSON file.

    Handles both formats:
      - Compassion: {"layers": {"0": [{...}, ...]}} (dict with layers key)
      - Honesty: [{...}, ...] (flat list)

    Each item must have a 'coherence_score' field.
    """
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)

    # Flat list format (honesty baselines in no_steering/)
    if isinstance(data, list):
        items = data
    # Dict with layers format (compassion baselines in no_steering/)
    elif isinstance(data, dict) and 'layers' in data:
        items = data['layers'].get('0', [])
    else:
        return None

    scores = [item['coherence_score'] for item in items if 'coherence_score' in item]
    if not scores:
        return None
    return sum(scores) / len(scores)


def compute_coherence_threshold(aligned_coherence: float, threshold_pct: float) -> float:
    """Compute absolute coherence threshold as a percentage of aligned baseline."""
    return aligned_coherence * threshold_pct / 100.0


def extract_compassion_steering(filepath: str, method_name: str, mode: str) -> list[dict]:
    """Extract all compassion steering results from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers = data.get('layers', {})
    all_results = []

    for layer_str, layer_data in layers.items():
        layer = int(layer_str)

        if not isinstance(layer_data, dict):
            continue

        for coef_key, results in layer_data.items():
            if not coef_key.startswith('coef_'):
                continue
            coef_str = coef_key.replace('coef_', '')
            coef = float(coef_str)
            if coef == int(coef):
                coef = int(coef)

            trait_scores = [r.get('trait_expression_score') for r in results if r.get('trait_expression_score') is not None]
            coherence_scores = [r.get('coherence_score') for r in results if r.get('coherence_score') is not None]

            if trait_scores and coherence_scores:
                all_results.append({
                    'trait': 'compassion',
                    'method': method_name,
                    'mode': mode,
                    'layer': layer,
                    'coef': coef,
                    'trait_score': sum(trait_scores) / len(trait_scores),
                    'coherence_score': sum(coherence_scores) / len(coherence_scores),
                    'n_samples': len(trait_scores),
                    'trait_score_se': round(_standard_error(trait_scores), 2),
                    'coherence_score_se': round(_standard_error(coherence_scores), 2),
                })

    return all_results


def extract_honesty_steering(filepath: str, method_name: str, mode: str) -> list[dict]:
    """Extract all honesty steering results from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers = data.get('layers', {})
    all_results = []

    for layer_str, layer_data in layers.items():
        layer = int(layer_str)

        if not isinstance(layer_data, dict):
            continue

        for coef_key, results in layer_data.items():
            if not coef_key.startswith('coef_'):
                continue
            coef_str = coef_key.replace('coef_', '')
            coef = float(coef_str)
            if coef == int(coef):
                coef = int(coef)

            honesty_scores = []
            coherence_scores = []

            for r in results:
                js = r.get('judge_score')
                if isinstance(js, dict):
                    score = js.get('score')
                else:
                    score = js
                if score is not None:
                    honesty_scores.append(score)

                cs = r.get('coherence_score')
                if cs is not None:
                    coherence_scores.append(cs)

            if honesty_scores and coherence_scores:
                all_results.append({
                    'trait': 'honesty',
                    'method': method_name,
                    'mode': mode,
                    'layer': layer,
                    'coef': coef,
                    'trait_score': sum(honesty_scores) / len(honesty_scores),
                    'coherence_score': sum(coherence_scores) / len(coherence_scores),
                    'n_samples': len(honesty_scores),
                    'trait_score_se': round(_standard_error(honesty_scores), 2),
                    'coherence_score_se': round(_standard_error(coherence_scores), 2),
                })

    return all_results


def find_best_operating_point(results: list[dict], coherence_threshold: float = 80.0) -> Optional[OperatingPoint]:
    """Find best trait score where coherence >= threshold."""
    filtered = [r for r in results if r['coherence_score'] >= coherence_threshold]
    if not filtered:
        return None

    best = max(filtered, key=lambda x: x['trait_score'])
    return OperatingPoint(
        trait=best['trait'],
        method=best['method'],
        mode=best['mode'],
        layer=best['layer'],
        coef=best['coef'],
        trait_score=round(best['trait_score'], 1),
        coherence_score=round(best['coherence_score'], 1),
        n_samples=best['n_samples'],
        trait_score_se=best.get('trait_score_se', 0.0),
        coherence_score_se=best.get('coherence_score_se', 0.0),
    )


def find_universal_layer(
    results_by_method: dict[str, list[dict]],
    coherence_threshold: float = 80.0,
    tolerance: float = 5.0,
) -> UniversalLayerResult:
    """
    Find a single shared layer that works well for all steering methods.

    Uses a maximin approach: pick the layer that maximizes the minimum trait
    score across all methods (each at its best coefficient for that layer).

    Args:
        results_by_method: {method_name: [raw result dicts]} for a single mode
        coherence_threshold: Minimum coherence score to consider a point valid
        tolerance: Maximum allowed drop from each method's independent best

    Returns:
        UniversalLayerResult with the chosen layer and comparison data
    """
    methods = list(results_by_method.keys())

    # 1. Find each method's independent best
    independent_points = {}
    for method, results in results_by_method.items():
        best = find_best_operating_point(results, coherence_threshold)
        if best is not None:
            independent_points[method] = best

    # If fewer than 2 methods have valid points, universal selection is meaningless
    if len(independent_points) < 2:
        return UniversalLayerResult(
            universal_layer=None,
            universal_accepted=False,
            tolerance_used=tolerance,
            universal_points={},
            independent_points=independent_points,
            universal_min_score=0.0,
            per_method_drops={},
        )

    # 2. Collect candidate layers: all layers appearing in any method's results
    all_layers = set()
    for results in results_by_method.values():
        for r in results:
            if r['coherence_score'] >= coherence_threshold:
                all_layers.add(r['layer'])

    # 3. For each candidate layer, find best coef per method and record min trait score
    best_layer = None
    best_min_score = -1.0
    best_layer_points = {}

    for layer in sorted(all_layers):
        layer_points = {}
        valid = True

        for method, results in results_by_method.items():
            if method not in independent_points:
                continue
            # Filter to this layer + coherence threshold
            at_layer = [
                r for r in results
                if r['layer'] == layer and r['coherence_score'] >= coherence_threshold
            ]
            if not at_layer:
                valid = False
                break

            best_at_layer = max(at_layer, key=lambda x: x['trait_score'])
            layer_points[method] = OperatingPoint(
                trait=best_at_layer['trait'],
                method=best_at_layer['method'],
                mode=best_at_layer['mode'],
                layer=best_at_layer['layer'],
                coef=best_at_layer['coef'],
                trait_score=round(best_at_layer['trait_score'], 1),
                coherence_score=round(best_at_layer['coherence_score'], 1),
                n_samples=best_at_layer['n_samples'],
                trait_score_se=best_at_layer.get('trait_score_se', 0.0),
                coherence_score_se=best_at_layer.get('coherence_score_se', 0.0),
            )

        if not valid or len(layer_points) != len(independent_points):
            continue

        min_score = min(p.trait_score for p in layer_points.values())
        if min_score > best_min_score:
            best_min_score = min_score
            best_layer = layer
            best_layer_points = layer_points

    # 4. Check tolerance: no method should drop more than tolerance from its independent best
    per_method_drops = {}
    universal_accepted = best_layer is not None
    if best_layer is not None:
        for method in independent_points:
            indep_score = independent_points[method].trait_score
            univ_score = best_layer_points[method].trait_score
            drop = indep_score - univ_score
            per_method_drops[method] = round(drop, 1)
            if drop > tolerance:
                universal_accepted = False

    return UniversalLayerResult(
        universal_layer=best_layer,
        universal_accepted=universal_accepted,
        tolerance_used=tolerance,
        universal_points=best_layer_points,
        independent_points=independent_points,
        universal_min_score=round(best_min_score, 1),
        per_method_drops=per_method_drops,
    )


def select_best_points(
    results_by_method: dict[str, list[dict]],
    coherence_threshold: float = 80.0,
    tolerance: float = 5.0,
    prefer_universal: bool = True,
) -> tuple[dict[str, OperatingPoint], UniversalLayerResult]:
    """
    Select the best operating points, preferring a universal shared layer.

    Args:
        results_by_method: {method_name: [raw result dicts]} for a single mode
        coherence_threshold: Minimum coherence for valid points
        tolerance: Max drop from independent best before rejecting universal layer
        prefer_universal: If True (default), use universal layer when accepted

    Returns:
        (chosen_points, universal_result) where chosen_points is
        {method -> OperatingPoint} using either universal or independent selection
    """
    result = find_universal_layer(results_by_method, coherence_threshold, tolerance)

    if prefer_universal and result.universal_accepted:
        return result.universal_points, result
    else:
        return result.independent_points, result


def select_baseline_layer_per_mode(
    points: dict[str, OperatingPoint],
) -> Optional[int]:
    """
    Select the baseline activation-projection layer for one steering mode.

    Args:
        points: {method_name: OperatingPoint} – the chosen operating points
            for a single mode (all-token or response).

    Returns:
        The most frequent layer among the points, with ties broken by
        lowest layer number for determinism.  None if *points* is empty.
    """
    if not points:
        return None

    layers = [op.layer for op in points.values() if op.layer is not None]
    if not layers:
        return None

    # Count frequencies
    freq: dict[int, int] = {}
    for layer in layers:
        freq[layer] = freq.get(layer, 0) + 1

    max_count = max(freq.values())
    # Among layers with the highest frequency, pick the lowest for determinism
    return min(layer for layer, count in freq.items() if count == max_count)


def extract_all_values(
    results_dir: str = constants.RESULTS_FOLDER,
    model_name: str = "meta-llama_Llama-3.3-70B-Instruct",
    coherence_threshold_pct: float = 90.0,
    honesty_category: str = "self_report_experimental",
) -> tuple[list[OperatingPoint], list[dict]]:
    """
    Extract all baseline and best operating point values.

    Returns:
        tuple: (list of OperatingPoint for paper table, list of all raw results)
    """
    operating_points = []
    all_raw_results = []

    # === COMPASSION ===
    compassion_base = Path(results_dir) / constants.VECTOR_TYPE / "compassion" / model_name / "all_layers"

    # Compassion baselines (now in no_steering/)
    aligned_path = compassion_base / "no_steering" / f"compassion_aligned_baseline_{model_name}_logreg.json"
    if aligned_path.exists():
        op = extract_compassion_baseline(str(aligned_path), "aligned")
        if op:
            operating_points.append(op)

    malicious_path = compassion_base / "no_steering" / f"compassion_malicious_baseline_{model_name}_logreg.json"
    if malicious_path.exists():
        op = extract_compassion_baseline(str(malicious_path), "malicious")
        if op:
            operating_points.append(op)

    # Compute effective coherence threshold from aligned baseline
    if aligned_path.exists():
        aligned_coh = extract_baseline_coherence(str(aligned_path))
        if aligned_coh is not None:
            compassion_threshold = compute_coherence_threshold(aligned_coh, coherence_threshold_pct)
        else:
            compassion_threshold = 80.0
            print(f"  WARNING: Could not extract coherence from {aligned_path}, using absolute fallback {compassion_threshold}")
    else:
        compassion_threshold = 80.0
        print(f"  WARNING: Aligned baseline not found at {aligned_path}, using absolute fallback {compassion_threshold}")

    # Compassion steering methods
    compassion_methods = {
        'SwFC': ('swfc', 'compassion_swfc'),
        'StTP': ('sttp', 'compassion_sttp'),
        'StMP': ('stmp', 'compassion_stmp'),
    }

    for method_name, (method_dir, filename_prefix) in compassion_methods.items():
        for mode_suffix, mode_name in [('all_token_mode', 'all'), ('response_mode', 'resp')]:
            filename = f"{filename_prefix}_{mode_suffix}_{model_name}_logreg.json"

            filepath = compassion_base / method_dir / filename

            if filepath.exists():
                results = extract_compassion_steering(str(filepath), method_name, mode_name)
                all_raw_results.extend(results)

                best = find_best_operating_point(results, compassion_threshold)
                if best:
                    operating_points.append(best)

    # === HONESTY ===
    honesty_base = Path(results_dir) / constants.VECTOR_TYPE / "honesty" / model_name / "all_layers" / honesty_category

    file_suffix = f"_{model_name}_logreg.json"

    # Honesty baselines
    aligned_path = honesty_base / "no_steering" / f"honesty_aligned_baseline{file_suffix}"
    if aligned_path.exists():
        op = extract_honesty_baseline(str(aligned_path), "aligned")
        if op:
            operating_points.append(op)

    malicious_path = honesty_base / "no_steering" / f"honesty_malicious_baseline{file_suffix}"
    if malicious_path.exists():
        op = extract_honesty_baseline(str(malicious_path), "malicious")
        if op:
            operating_points.append(op)

    # Compute effective coherence threshold from aligned baseline
    if aligned_path.exists():
        aligned_coh = extract_baseline_coherence(str(aligned_path))
        if aligned_coh is not None:
            honesty_threshold = compute_coherence_threshold(aligned_coh, coherence_threshold_pct)
        else:
            honesty_threshold = 80.0
            print(f"  WARNING: Could not extract coherence from {aligned_path}, using absolute fallback {honesty_threshold}")
    else:
        honesty_threshold = 80.0
        print(f"  WARNING: Aligned baseline not found at {aligned_path}, using absolute fallback {honesty_threshold}")

    # Honesty steering methods
    honesty_methods = {
        'SwFC': ('swfc', 'honesty_swfc'),
        'StTP': ('sttp', 'honesty_sttp'),
        'StMP': ('stmp', 'honesty_stmp'),
    }

    for method_name, (method_dir, filename_prefix) in honesty_methods.items():
        for mode_suffix, mode_name in [('all_token_mode', 'all'), ('response_mode', 'resp')]:
            filename = f"{filename_prefix}_{mode_suffix}{file_suffix}"
            filepath = honesty_base / method_dir / filename

            if filepath.exists():
                results = extract_honesty_steering(str(filepath), method_name, mode_name)
                all_raw_results.extend(results)

                best = find_best_operating_point(results, honesty_threshold)
                if best:
                    operating_points.append(best)

    return operating_points, all_raw_results


def save_to_json(operating_points: list[OperatingPoint], output_path: str):
    """Save operating points to JSON file."""
    data = {
        "extracted_at": datetime.now().isoformat(),
        "description": "Best operating points extracted from experimental results",
        "operating_points": [asdict(op) for op in operating_points]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Saved JSON to {output_path}")


def save_to_csv(operating_points: list[OperatingPoint], output_path: str):
    """Save operating points to CSV file."""
    fieldnames = ['trait', 'method', 'mode', 'layer', 'coef', 'trait_score', 'trait_score_se', 'coherence_score', 'coherence_score_se', 'n_samples']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for op in operating_points:
            row = asdict(op)
            # Convert None to empty string for CSV
            row = {k: ('' if v is None else v) for k, v in row.items()}
            writer.writerow(row)

    print(f"Saved CSV to {output_path}")


def generate_latex_table(operating_points: list[OperatingPoint]) -> str:
    """Generate LaTeX table content from operating points."""
    lines = []

    # Group by trait
    compassion_ops = [op for op in operating_points if op.trait == "compassion"]
    honesty_ops = [op for op in operating_points if op.trait == "honesty"]

    # Sort within each group: baselines first, then by method and mode
    method_order = {'aligned': 0, 'malicious': 1, 'SwFC': 2, 'StTP': 3, 'StMP': 4}
    mode_order = {'--': 0, 'all': 1, 'resp': 2}

    def sort_key(op):
        return (method_order.get(op.method, 99), mode_order.get(op.mode, 99))

    compassion_ops.sort(key=sort_key)
    honesty_ops.sort(key=sort_key)

    lines.append("% Auto-generated LaTeX table content")
    lines.append(f"% Generated at: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("\\multirow{8}{*}{Comp.}")

    for i, op in enumerate(compassion_ops):
        layer_str = "--" if op.layer is None else str(op.layer)
        trait_str = f"${int(round(op.trait_score))} \\pm {op.trait_score_se:.1f}$"
        coh_str = f"${int(round(op.coherence_score))} \\pm {op.coherence_score_se:.1f}$"
        coef_str = "--" if op.coef is None else (str(int(op.coef)) if op.coef == int(op.coef) else str(op.coef))
        prefix = " &" if i > 0 else ""
        lines.append(f" & {op.method:<9} & {op.mode:<4} & {layer_str:>3} & {trait_str} & {coh_str} & {coef_str:>5} \\\\")

    lines.append("\\midrule")
    lines.append("\\multirow{8}{*}{Hon.}")

    for i, op in enumerate(honesty_ops):
        layer_str = "--" if op.layer is None else str(op.layer)
        trait_str = f"${int(round(op.trait_score))} \\pm {op.trait_score_se:.1f}$"
        coh_str = f"${int(round(op.coherence_score))} \\pm {op.coherence_score_se:.1f}$"
        coef_str = "--" if op.coef is None else (str(int(op.coef)) if op.coef == int(op.coef) else str(op.coef))
        lines.append(f" & {op.method:<9} & {op.mode:<4} & {layer_str:>3} & {trait_str} & {coh_str} & {coef_str:>5} \\\\")

    return "\n".join(lines)


def generate_latex_best_ops_table(
    model_points: dict[str, dict[str, dict[str, OperatingPoint]]],
    model_baselines: dict[str, list[OperatingPoint]],
    vector_type: str,
) -> str:
    """
    Generate a full LaTeX table (Style B — both modes) for best operating points.

    Args:
        model_points: {trait: {mode: {method: OperatingPoint}}}
            e.g. {"compassion": {"all": {"SwFC": op, ...}, "resp": {...}}, "honesty": {...}}
        model_baselines: {trait: [aligned_op, malicious_op]}
        vector_type: e.g. "logistic_regression"

    Returns:
        Complete LaTeX table string
    """
    TRAIT_TO_THREAT = {
        "compassion": "Dismissive",
        "honesty": "Dishonest",
    }
    METHOD_ORDER = ["SwFC", "StTP", "StMP"]
    MODE_ORDER = ["all", "resp"]

    lines = []
    lines.append(f"% Auto-generated from extract_paper_values.py ({vector_type})")
    lines.append(f"% Generated at: {datetime.now().isoformat()}")
    lines.append(r"\begin{tabular}{@{}llccrrr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Threat & Method & Mode & Layer & Trait & Coh. & $\alpha$ \\")
    lines.append(r"\midrule")

    traits = [t for t in ["honesty", "compassion"] if t in model_points or t in model_baselines]

    for trait_idx, trait in enumerate(traits):
        threat = TRAIT_TO_THREAT.get(trait, trait.capitalize())
        baselines = model_baselines.get(trait, [])
        points_by_mode = model_points.get(trait, {})

        # Count rows: 2 baselines + (methods × modes)
        n_method_rows = sum(
            1 for mode in MODE_ORDER for method in METHOD_ORDER
            if mode in points_by_mode and method in points_by_mode[mode]
        )
        n_rows = len(baselines) + n_method_rows

        lines.append(f"\\multirow{{{n_rows}}}{{*}}{{{threat}}}")

        # Baselines
        for bl in baselines:
            trait_str = f"${int(round(bl.trait_score))} \\pm {bl.trait_score_se:.1f}$"
            coh_str = f"${int(round(bl.coherence_score))} \\pm {bl.coherence_score_se:.1f}$"
            cap_method = bl.method.capitalize()
            lines.append(f" & {cap_method:<9} & {'--':<4} & {'--':>3} & {trait_str} & {coh_str} & {'--':>5} \\\\")

        # Steering methods by mode
        for method in METHOD_ORDER:
            for mode in MODE_ORDER:
                if mode not in points_by_mode or method not in points_by_mode[mode]:
                    continue
                op = points_by_mode[mode][method]
                layer_str = "--" if op.layer is None else str(op.layer)
                trait_str = f"${int(round(op.trait_score))} \\pm {op.trait_score_se:.1f}$"
                coh_str = f"${int(round(op.coherence_score))} \\pm {op.coherence_score_se:.1f}$"
                coef_str = "--" if op.coef is None else (str(int(op.coef)) if op.coef == int(op.coef) else str(op.coef))
                lines.append(f" & {method:<9} & {mode:<4} & {layer_str:>3} & {trait_str} & {coh_str} & {coef_str:>5} \\\\")

        # Add midrule between traits, bottomrule after last
        if trait_idx < len(traits) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return "\n".join(lines)


def save_latex_best_ops_table(
    latex_content: str,
    model_name: str,
    vector_type: str,
    output_dir: str = None,
) -> str:
    """
    Write a LaTeX table string to a .tex file.

    Args:
        latex_content: The LaTeX table string
        model_name: e.g. "meta-llama_Llama-3.3-70B-Instruct"
        vector_type: e.g. "logistic_regression"
        output_dir: Directory for .tex files (default: results/paper_values/)

    Returns:
        Path to the written file
    """
    if output_dir is None:
        output_dir = os.path.join(constants.RESULTS_FOLDER, "paper_values")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{vector_type}_best_ops_{model_name}.tex"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    print(f"\nSaved LaTeX table to {filepath}")
    return filepath


# ---------------------------------------------------------------------------
# Config YAML writing (targeted text replacement, preserves surrounding YAML)
# ---------------------------------------------------------------------------

def _build_best_layer_yaml_block(
    key_name: str,
    points_by_mode: dict[str, dict[str, OperatingPoint]],
    baseline_layers_by_mode: Optional[dict[str, int]] = None,
) -> str:
    """
    Build a YAML text block for a best_layer entry.

    Args:
        key_name: Top-level YAML key, e.g. "best_layer_meta-llama_Llama-3.3-70B-Instruct_fever"
        points_by_mode: {config_mode_key: {config_method_key: OperatingPoint}}
            e.g. {"all_token_mode": {"swfc": OperatingPoint(...)}}
        baseline_layers_by_mode: {config_mode_key: layer_int} – baseline layer
            for activation projection analysis per mode.  If None, no baseline
            entry is written.

    Returns:
        YAML text block (without trailing newline)
    """
    if baseline_layers_by_mode is None:
        baseline_layers_by_mode = {}

    lines = [f"{key_name}:"]
    for mode_key in ("all_token_mode", "response_mode"):
        if mode_key not in points_by_mode:
            continue
        lines.append(f"  {mode_key}:")
        methods = points_by_mode[mode_key]
        for method_key in ("swfc", "sttp", "stmp"):
            if method_key not in methods:
                continue
            op = methods[method_key]
            # Format coefficient: use int if whole number, else float
            coef_val = op.coef
            if coef_val is not None and coef_val == int(coef_val):
                coef_str = f"{int(coef_val)}.0"
            else:
                coef_str = str(coef_val)
            lines.append(f"    {method_key}:")
            lines.append(f"      layer: {op.layer}")
            lines.append(f"      coefficient: {coef_str}")
        # Append baseline_layer_projection_activation for this mode
        if mode_key in baseline_layers_by_mode:
            lines.append(f"    baseline_layer_projection_activation:")
            lines.append(f"      layer: {baseline_layers_by_mode[mode_key]}")
    return "\n".join(lines)


def _replace_yaml_block(file_text: str, key_name: str, new_block: str) -> str:
    """
    Replace or append a top-level YAML block identified by key_name.

    Finds the block starting with `^key_name:` and ending at the next
    top-level key (line starting with a non-space, non-comment char) or EOF.
    """
    # Pattern: key_name: at start of line, then everything until next top-level key or EOF
    pattern = re.compile(
        r'^' + re.escape(key_name) + r':.*?(?=\n[^\s#]|\Z)',
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(file_text)
    if match:
        return file_text[:match.start()] + new_block + file_text[match.end():]
    else:
        # Append at end (ensure preceding newline)
        separator = "\n" if file_text.endswith("\n") else "\n\n"
        return file_text + separator + new_block + "\n"


def save_best_layers_to_config(
    points_by_mode: dict[str, dict[str, OperatingPoint]],
    trait: str,
    model_name: str,
    config_path: str,
    honesty_category: Optional[str] = None,
    baseline_layers_by_mode: Optional[dict[str, int]] = None,
    dry_run: bool = False,
) -> str:
    """
    Write (or preview) best-layer operating points into a config YAML file.

    Args:
        points_by_mode: {display_mode: {display_method: OperatingPoint}}
            e.g. {"all": {"SwFC": op, "StTP": op, ...}, "resp": {...}}
        trait: "honesty" or "compassion"
        model_name: e.g. "meta-llama_Llama-3.3-70B-Instruct"
        config_path: Path to the YAML config file
        honesty_category: For honesty, the category suffix (e.g. "fever")
        baseline_layers_by_mode: {display_mode: layer_int} – baseline layer
            for activation projection analysis per mode (auto-derived from
            operating points).  Keys use display mode names ("all", "resp").
        dry_run: If True, print what would be written without modifying the file

    Returns:
        The YAML block that was (or would be) written
    """
    # Build YAML key
    if trait == "honesty" and honesty_category:
        key_name = f"best_layer_{model_name}_{honesty_category}"
    else:
        key_name = f"best_layer_{model_name}"

    # Convert display names to config names
    config_points = {}
    for display_mode, method_points in points_by_mode.items():
        config_mode = DISPLAY_TO_CONFIG_MODE.get(display_mode, display_mode)
        config_points[config_mode] = {}
        for display_method, op in method_points.items():
            config_method = DISPLAY_TO_CONFIG_METHOD.get(display_method, display_method)
            config_points[config_mode][config_method] = op

    # Convert baseline layer display mode names to config mode names
    config_baseline_layers = None
    if baseline_layers_by_mode:
        config_baseline_layers = {}
        for display_mode, layer in baseline_layers_by_mode.items():
            config_mode = DISPLAY_TO_CONFIG_MODE.get(display_mode, display_mode)
            config_baseline_layers[config_mode] = layer

    block = _build_best_layer_yaml_block(key_name, config_points, config_baseline_layers)

    if dry_run:
        print(f"\n[DRY RUN] Would write to {config_path}:")
        print(block)
        return block

    with open(config_path, 'r', encoding='utf-8') as f:
        file_text = f.read()

    updated = _replace_yaml_block(file_text, key_name, block)

    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(updated)

    print(f"Updated {key_name} in {config_path}")
    return block


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_universal_layer_report(trait: str, mode: str, result: UniversalLayerResult):
    """Print a comparison table for universal vs independent layer selection."""
    status = "ACCEPTED" if result.universal_accepted else "REJECTED"
    max_drop = max(result.per_method_drops.values()) if result.per_method_drops else 0.0

    print(f"\n=== UNIVERSAL LAYER: {trait} / {mode} ===")
    if result.universal_layer is not None:
        print(f"Universal layer: {result.universal_layer} | Status: {status} "
              f"(max drop: {max_drop:.1f}, tolerance: {result.tolerance_used:.1f})")
    else:
        print(f"No universal layer found (not all methods have valid points)")

    if not result.independent_points:
        print("  No valid operating points found.")
        return

    # Table header
    print(f"\n{'Method':<6} | {'Indep Layer':>11} {'Coef':>5} {'Trait':>6} {'SE':>5} {'Coh':>5} {'SE':>5} "
          f"| {'Univ Coef':>9} {'Trait':>6} {'SE':>5} {'Coh':>5} {'SE':>5} | {'Drop':>5}")
    print("-" * 100)

    for method in sorted(result.independent_points.keys()):
        indep = result.independent_points[method]
        il = str(indep.layer) if indep.layer is not None else "--"
        ic = str(indep.coef) if indep.coef is not None else "--"

        if method in result.universal_points:
            univ = result.universal_points[method]
            uc = str(univ.coef) if univ.coef is not None else "--"
            drop = result.per_method_drops.get(method, 0.0)
            print(f"{method:<6} | {il:>11} {ic:>5} {indep.trait_score:>6.1f} {indep.trait_score_se:>5.2f} {indep.coherence_score:>5.1f} {indep.coherence_score_se:>5.2f} "
                  f"| {uc:>9} {univ.trait_score:>6.1f} {univ.trait_score_se:>5.2f} {univ.coherence_score:>5.1f} {univ.coherence_score_se:>5.2f} | {drop:>5.1f}")
        else:
            print(f"{method:<6} | {il:>11} {ic:>5} {indep.trait_score:>6.1f} {indep.trait_score_se:>5.2f} {indep.coherence_score:>5.1f} {indep.coherence_score_se:>5.2f} "
                  f"| {'--':>9} {'--':>6} {'--':>5} {'--':>5} {'--':>5} | {'--':>5}")


# ---------------------------------------------------------------------------
# Orchestrator: extract + universal select + save to config
# ---------------------------------------------------------------------------

def extract_and_save_all(
    results_dir: str = constants.RESULTS_FOLDER,
    coherence_threshold_pct: float = 90.0,
    tolerance: float = 5.0,
    models: list[str] = None,
    honesty_categories: list[str] = None,
    prefer_universal: bool = True,
    dry_run: bool = False,
):
    """
    Extract best operating points for all traits/models/categories, apply
    universal layer selection, and optionally save to config YAML files.

    Args:
        results_dir: Base results directory
        coherence_threshold_pct: Coherence threshold as percentage of aligned baseline (default 90%)
        tolerance: Max drop from independent best for universal layer acceptance
        models: List of model save names to process
        honesty_categories: List of honesty categories to process
        prefer_universal: Whether to prefer universal layer when accepted
        dry_run: If True, print changes without writing
    """
    if models is None:
        models = ["meta-llama_Llama-3.3-70B-Instruct", "Qwen_Qwen3-32B"]
    if honesty_categories is None:
        honesty_categories = ["self_report_experimental"]

    modes = [('all_token_mode', 'all'), ('response_mode', 'resp')]

    for model_name in models:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")

        # Accumulators for LaTeX table generation (per model)
        latex_model_points = {}   # {trait: {mode: {method: OperatingPoint}}}
        latex_model_baselines = {}  # {trait: [aligned_op, malicious_op]}

        # === COMPASSION ===
        print(f"\n--- Compassion ---")
        compassion_base = Path(results_dir) / constants.VECTOR_TYPE / "compassion" / model_name / "all_layers"

        # Compute effective coherence threshold from aligned baseline
        aligned_filename = f"compassion_aligned_baseline_{model_name}_logreg.json"
        aligned_path = _resolve_result_file(compassion_base / "no_steering", aligned_filename)
        if aligned_path is not None:
            aligned_coh = extract_baseline_coherence(str(aligned_path))
            if aligned_coh is not None:
                effective_threshold = compute_coherence_threshold(aligned_coh, coherence_threshold_pct)
                print(f"  Aligned baseline coherence: {aligned_coh:.1f} -> effective threshold ({coherence_threshold_pct}%): {effective_threshold:.1f}")
            else:
                effective_threshold = 80.0
                print(f"  WARNING: Could not extract coherence from {aligned_path}, using absolute fallback {effective_threshold}")
        else:
            effective_threshold = 80.0
            print(f"  WARNING: Aligned baseline not found under {compassion_base / 'no_steering'} ({aligned_filename}), using absolute fallback {effective_threshold}")

        # Extract compassion baselines for LaTeX table
        compassion_baselines = []
        if aligned_path is not None:
            op = extract_compassion_baseline(str(aligned_path), "aligned")
            if op:
                compassion_baselines.append(op)
        malicious_filename = f"compassion_malicious_baseline_{model_name}_logreg.json"
        malicious_path = _resolve_result_file(compassion_base / "no_steering", malicious_filename)
        if malicious_path is not None:
            op = extract_compassion_baseline(str(malicious_path), "malicious")
            if op:
                compassion_baselines.append(op)
        if compassion_baselines:
            latex_model_baselines["compassion"] = compassion_baselines

        compassion_methods_cfg = {
            'SwFC': ('swfc', 'compassion_swfc'),
            'StTP': ('sttp', 'compassion_sttp'),
            'StMP': ('stmp', 'compassion_stmp'),
        }

        compassion_points_by_mode = {}
        for mode_suffix, mode_name in modes:
            results_by_method = {}
            for method_name, (method_dir, filename_prefix) in compassion_methods_cfg.items():
                filename = f"{filename_prefix}_{mode_suffix}_{model_name}_logreg.json"

                filepath = _resolve_result_file(compassion_base / method_dir, filename)
                if filepath is not None:
                    results_by_method[method_name] = extract_compassion_steering(
                        str(filepath), method_name, mode_name
                    )

            if results_by_method:
                chosen, universal_result = select_best_points(
                    results_by_method, effective_threshold, tolerance, prefer_universal
                )
                print_universal_layer_report("compassion", mode_name, universal_result)
                compassion_points_by_mode[mode_name] = chosen

        if compassion_points_by_mode:
            # Derive baseline layer per mode from chosen operating points
            compassion_baseline_layers = {}
            for mode_name, chosen in compassion_points_by_mode.items():
                bl = select_baseline_layer_per_mode(chosen)
                if bl is not None:
                    compassion_baseline_layers[mode_name] = bl

            config_path = constants.get_config_for_trait("compassion")
            save_best_layers_to_config(
                compassion_points_by_mode, "compassion", model_name,
                config_path, baseline_layers_by_mode=compassion_baseline_layers,
                dry_run=dry_run,
            )
            latex_model_points["compassion"] = compassion_points_by_mode

        # === HONESTY (per category) ===
        for category in honesty_categories:
            print(f"\n--- Honesty / {category} ---")
            honesty_base = Path(results_dir) / constants.VECTOR_TYPE / "honesty" / model_name / "all_layers" / category
            file_suffix = f"_{model_name}_logreg.json"

            # Compute effective coherence threshold from aligned baseline
            aligned_path = _resolve_result_file(honesty_base / "no_steering", f"honesty_aligned_baseline{file_suffix}")
            if aligned_path is not None:
                aligned_coh = extract_baseline_coherence(str(aligned_path))
                if aligned_coh is not None:
                    effective_threshold = compute_coherence_threshold(aligned_coh, coherence_threshold_pct)
                    print(f"  Aligned baseline coherence: {aligned_coh:.1f} -> effective threshold ({coherence_threshold_pct}%): {effective_threshold:.1f}")
                else:
                    effective_threshold = 80.0
                    print(f"  WARNING: Could not extract coherence from {aligned_path}, using absolute fallback {effective_threshold}")
            else:
                effective_threshold = 80.0
                print(f"  WARNING: Aligned baseline not found under {honesty_base / 'no_steering'} (honesty_aligned_baseline{file_suffix}), using absolute fallback {effective_threshold}")

            # Extract honesty baselines for LaTeX table (use first category with data)
            if "honesty" not in latex_model_baselines:
                honesty_baselines = []
                if aligned_path is not None:
                    op = extract_honesty_baseline(str(aligned_path), "aligned")
                    if op:
                        honesty_baselines.append(op)
                malicious_path = _resolve_result_file(honesty_base / "no_steering", f"honesty_malicious_baseline{file_suffix}")
                if malicious_path is not None:
                    op = extract_honesty_baseline(str(malicious_path), "malicious")
                    if op:
                        honesty_baselines.append(op)
                if honesty_baselines:
                    latex_model_baselines["honesty"] = honesty_baselines

            honesty_methods_cfg = {
                'SwFC': ('swfc', 'honesty_swfc'),
                'StTP': ('sttp', 'honesty_sttp'),
                'StMP': ('stmp', 'honesty_stmp'),
            }

            honesty_points_by_mode = {}
            for mode_suffix, mode_name in modes:
                results_by_method = {}
                for method_name, (method_dir, filename_prefix) in honesty_methods_cfg.items():
                    filename = f"{filename_prefix}_{mode_suffix}{file_suffix}"
                    filepath = _resolve_result_file(honesty_base / method_dir, filename)
                    if filepath is not None:
                        results_by_method[method_name] = extract_honesty_steering(
                            str(filepath), method_name, mode_name
                        )

                if results_by_method:
                    chosen, universal_result = select_best_points(
                        results_by_method, effective_threshold, tolerance, prefer_universal
                    )
                    print_universal_layer_report(f"honesty/{category}", mode_name, universal_result)
                    honesty_points_by_mode[mode_name] = chosen

            if honesty_points_by_mode:
                # Derive baseline layer per mode from chosen operating points
                honesty_baseline_layers = {}
                for mode_name, chosen in honesty_points_by_mode.items():
                    bl = select_baseline_layer_per_mode(chosen)
                    if bl is not None:
                        honesty_baseline_layers[mode_name] = bl

                config_path = constants.get_config_for_trait("honesty")
                save_best_layers_to_config(
                    honesty_points_by_mode, "honesty", model_name,
                    config_path, honesty_category=category,
                    baseline_layers_by_mode=honesty_baseline_layers,
                    dry_run=dry_run,
                )

                # Accumulate honesty points for LaTeX (use first category with data)
                if "honesty" not in latex_model_points:
                    latex_model_points["honesty"] = honesty_points_by_mode

        # === GENERATE LATEX TABLE ===
        if latex_model_points:
            latex_content = generate_latex_best_ops_table(
                latex_model_points, latex_model_baselines, constants.VECTOR_TYPE
            )
            save_latex_best_ops_table(latex_content, model_name, constants.VECTOR_TYPE)
            print(f"\n{'='*80}")
            print(f"LaTeX TABLE ({model_name}):")
            print(f"{'='*80}")
            print(latex_content)


def print_summary(operating_points: list[OperatingPoint]):
    """Print a formatted summary of the extracted values."""
    print("\n" + "=" * 80)
    print("EXTRACTED PAPER VALUES")
    print("=" * 80)

    # Group by trait
    for trait in ["compassion", "honesty"]:
        trait_ops = [op for op in operating_points if op.trait == trait]
        print(f"\n{trait.upper()}:")
        print("-" * 80)
        print(f"{'Method':<10} {'Mode':<5} {'Layer':<6} {'Trait':<6} {'SE':>5} {'Coh':<6} {'SE':>5} {'N':<4}")
        print("-" * 80)

        for op in trait_ops:
            layer_str = "--" if op.layer is None else str(op.layer)
            print(f"{op.method:<10} {op.mode:<5} {layer_str:<6} {op.trait_score:<6.1f} {op.trait_score_se:>5.2f} {op.coherence_score:<6.1f} {op.coherence_score_se:>5.2f} {op.n_samples:<4}")


def main():
    parser = argparse.ArgumentParser(description="Extract paper values from experimental results")
    parser.add_argument("--results_dir", type=str, default=constants.RESULTS_FOLDER, help="Base results directory")
    parser.add_argument("--model_name", type=str, default="meta-llama_Llama-3.3-70B-Instruct",#"Qwen/Qwen3-32B",
                        help="Model name (for legacy single-extraction mode)")
    parser.add_argument("--output_dir", type=str, default="results/paper_values", help="Output directory")
    parser.add_argument("--coherence_threshold_pct", type=float, default=90.0,
                        help="Coherence threshold as %% of aligned baseline coherence (default 90%%)")
    parser.add_argument("--honesty_category", type=str, default="self_report_experimental",
                        choices=["self_report_experimental"],
                        help="Honesty dataset category (for legacy single-extraction mode)")
    # New CLI args for universal layer + config saving
    parser.add_argument("--tolerance", type=float, default=5.0,
                        help="Max trait-score drop from independent best for universal layer acceptance")
    parser.add_argument("--no_universal", action="store_true",
                        help="Disable universal layer selection (use independent per-method best)")
    parser.add_argument("--save_to_config", action="store_true",
                        help="Write best layers to config YAML files")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print config changes without writing (implies --save_to_config)")
    parser.add_argument("--models", nargs="+",
                        default=["meta-llama_Llama-3.3-70B-Instruct", "Qwen_Qwen3.6-27B"],
                        help="Model save names to process (for --save_to_config mode)")
    parser.add_argument("--honesty_categories", nargs="+",
                        default=["self_report_experimental"],
                        help="Honesty categories to process (for --save_to_config mode)")
    args = parser.parse_args()

    if args.save_to_config or args.dry_run:
        # Universal extraction + config writing mode
        extract_and_save_all(
            results_dir=args.results_dir,
            coherence_threshold_pct=args.coherence_threshold_pct,
            tolerance=args.tolerance,
            models=args.models,
            honesty_categories=args.honesty_categories,
            prefer_universal=not args.no_universal,
            dry_run=args.dry_run,
        )
        return

    raise NotImplementedError("Legacy single-extraction mode is no longer supported. Please use the new universal extraction and config saving mode with --save_to_config.")

    # Legacy mode: single model/category extraction with paper outputs
    #output_dir = Path(args.output_dir)
    output_dir = Path(constants.RESULTS_FOLDER, "paper_values")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting values from {args.results_dir}...")
    print(f"Model: {args.model_name}")
    print(f"Honesty category: {args.honesty_category}")
    print(f"Coherence threshold: {args.coherence_threshold_pct}% of aligned baseline")

    operating_points, all_raw_results = extract_all_values(
        results_dir=args.results_dir,
        model_name=args.model_name,
        coherence_threshold_pct=args.coherence_threshold_pct,
        honesty_category=args.honesty_category
    )

    print_summary(operating_points)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"paper_values_{timestamp}.json"
    save_to_json(operating_points, str(json_path))

    csv_path = output_dir / f"paper_values_{timestamp}.csv"
    save_to_csv(operating_points, str(csv_path))

    save_to_json(operating_points, str(output_dir / "paper_values_latest.json"))
    save_to_csv(operating_points, str(output_dir / "paper_values_latest.csv"))

    raw_json_path = output_dir / f"all_steering_results_{timestamp}.json"
    with open(raw_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "extracted_at": datetime.now().isoformat(),
            "coherence_threshold_pct": args.coherence_threshold_pct,
            "results": all_raw_results
        }, f, indent=2)
    print(f"Saved raw results to {raw_json_path}")

    latex_content = generate_latex_table(operating_points)
    latex_path = output_dir / "table_content.tex"
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    print(f"Saved LaTeX table content to {latex_path}")

    print("\n" + "=" * 80)
    print("LaTeX TABLE CONTENT:")
    print("=" * 80)
    print(latex_content)


if __name__ == "__main__":
    main()
