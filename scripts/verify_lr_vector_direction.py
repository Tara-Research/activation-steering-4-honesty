"""
Compare logistic regression and CAA steering vectors: cosine similarity,
norms, and normalization metadata.

Produces:
  - Console report (per-layer cosine similarity, norms, metadata)
  - Per-config cosine-similarity line plot (saved as PDF + PNG)
  - Combined multi-config summary plot (saved as PDF + PNG)
  - LaTeX table snippet (saved as .tex)
  - CSV with per-layer values

Usage (run from repo root):
    python -m scripts.verify_lr_vector_direction
    python -m scripts.verify_lr_vector_direction --trait honesty --dataset self_report_experimental
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.constants.constants import (
    PERSONA_VECTORS_FOLDER,
    PLOTS_FOLDER,
    get_config_for_trait,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ComparisonConfig:
    """One model--trait combination to compare."""
    model_save_name: str
    trait: str
    dataset: str  # e.g. "diverse_with_sycophancy" or "compassion"

    @property
    def short_label(self) -> str:
        model_short = self.model_save_name.split("_", 1)[-1]
        return f"{model_short} / {self.trait}"

    @property
    def _config_key_suffix(self) -> str:
        """Config section key suffix after 'common_steering_config_'."""
        if self.trait == "compassion":
            return self.model_save_name
        return f"{self.model_save_name}_{self.dataset}"


DEFAULT_CONFIGS = [
    ComparisonConfig("meta-llama_Llama-3.3-70B-Instruct", "honesty", "self_report_experimental"),
    ComparisonConfig("meta-llama_Llama-3.3-70B-Instruct", "compassion", "compassion"),
    ComparisonConfig("Qwen_Qwen3.6-27B", "honesty", "self_report_experimental"),
    ComparisonConfig("Qwen_Qwen3.6-27B", "compassion", "compassion"),
]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _resolve_paths(cfg: ComparisonConfig) -> Tuple[str, str]:
    """Resolve absolute CAA and LR vector paths.

    The CAA vector follows the naming template emitted by
    src.vector_generation.generate_vec_caa; the LR vector comes from the
    trait config's ``vector_path``. Returns (caa_path, lr_path).
    """
    caa_path = os.path.join(
        PERSONA_VECTORS_FOLDER, "caa", cfg.model_save_name, cfg.trait, cfg.dataset,
        f"{cfg.trait}_{cfg.dataset}_steering_vector_response_avg_{cfg.model_save_name}.pt",
    )

    with open(get_config_for_trait(cfg.trait)) as f:
        lr_cfg = yaml.safe_load(f)

    section_key = f"common_steering_config_{cfg._config_key_suffix}"
    lr_section = lr_cfg.get(section_key)
    if lr_section is None:
        raise KeyError(f"Config missing section '{section_key}'")

    lr_path = os.path.join(PERSONA_VECTORS_FOLDER, lr_section["vector_path"])
    return caa_path, lr_path


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compute_cosine_similarities(
    caa_vec: torch.Tensor,
    lr_vec: torch.Tensor,
) -> Tuple[List[float], float]:
    """Return (per_layer_cosines, mean_cosine)."""
    n_layers = min(caa_vec.shape[0], lr_vec.shape[0])
    cosines = []
    for layer in range(n_layers):
        cos = F.cosine_similarity(
            caa_vec[layer].unsqueeze(0),
            lr_vec[layer].unsqueeze(0),
        ).item()
        cosines.append(cos)
    return cosines, float(np.mean(cosines))


def run_comparison(cfg: ComparisonConfig) -> dict:
    """Run full comparison for one config. Returns results dict."""
    caa_path, lr_path = _resolve_paths(cfg)

    print(f"\n{'=' * 70}")
    print(f"Model:   {cfg.model_save_name}")
    print(f"Trait:   {cfg.trait}")
    print(f"Dataset: {cfg.dataset}")
    print(f"CAA vec: {caa_path}")
    print(f"LR  vec: {lr_path}")
    print(f"{'=' * 70}")

    caa_vec = torch.load(caa_path, weights_only=False, map_location="cpu").float()
    lr_vec = torch.load(lr_path, weights_only=False, map_location="cpu").float()

    n_layers = min(caa_vec.shape[0], lr_vec.shape[0])
    print(f"CAA shape: {caa_vec.shape}  |  LR shape: {lr_vec.shape}")

    # --- Cosine similarity ---
    cosines, mean_cos = compute_cosine_similarities(caa_vec, lr_vec)
    n_flipped = sum(1 for c in cosines if c < 0)

    print(f"\nCosine similarity (CAA vs LR):")
    for layer in range(n_layers):
        if layer % 10 == 0 or layer == n_layers - 1:
            tag = "OK" if cosines[layer] > 0 else "FLIPPED!"
            print(f"  Layer {layer:3d}: {cosines[layer]:+.4f}  [{tag}]")
    print(f"  Mean: {mean_cos:+.4f}  |  Flipped: {n_flipped}/{n_layers}")

    # --- Norms ---
    lr_norms = [lr_vec[l].norm().item() for l in range(n_layers)]
    caa_norms = [caa_vec[l].norm().item() for l in range(n_layers)]

    print(f"\nVector norms (every 10th layer):")
    for layer in range(n_layers):
        if layer % 10 == 0 or layer == n_layers - 1:
            ratio = lr_norms[layer] / (caa_norms[layer] + 1e-12)
            print(
                f"  Layer {layer:3d}: LR={lr_norms[layer]:.2f}  "
                f"CAA={caa_norms[layer]:.2f}  ratio={ratio:.3f}"
            )

    return {
        "config": cfg,
        "cosines": cosines,
        "mean_cosine": mean_cos,
        "n_flipped": n_flipped,
        "n_layers": n_layers,
        "lr_norms": lr_norms,
        "caa_norms": caa_norms,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_single_config(result: dict, save_dir: str) -> str:
    """Plot cosine similarity over layers for one config. Returns save path."""
    cfg = result["config"]
    cosines = result["cosines"]
    n_layers = result["n_layers"]
    layers = list(range(n_layers))

    LINE_COLOR = "#c0392b"
    MEAN_COLOR = "#922b21"

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(layers, cosines, linewidth=1.5, color=LINE_COLOR)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(
        y=result["mean_cosine"], color=MEAN_COLOR, linestyle=":",
        linewidth=1.0, label=f"Mean = {result['mean_cosine']:.3f}",
    )
    ax.set_xlabel("Layer", fontsize=18.7, fontweight="bold")
    ax.set_ylabel("Cosine Similarity", fontsize=18.7, fontweight="bold")
    ax.set_title(
        f"CAA vs Logistic Regression Vector — {cfg.short_label}\n"
        f"(mean cos = {result['mean_cosine']:.3f}, "
        f"flipped = {result['n_flipped']}/{n_layers})",
        fontsize=18.7, fontweight="bold",
    )
    ax.set_xlim(0, n_layers - 1)
    ax.set_ylim(-1.05, 1.05)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    model_short = cfg.model_save_name.replace("/", "_")
    stem = f"caa_vs_logreg_cosine_{model_short}_{cfg.trait}_{cfg.dataset}"
    for ext in ("pdf", "png"):
        path = os.path.join(save_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return os.path.join(save_dir, f"{stem}.pdf")


def plot_combined(results: List[dict], save_dir: str) -> str:
    """2x2 (or Nx1) combined plot of all configs. Returns save path."""
    n = len(results)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.2 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    LINE_COLOR = "#c0392b"
    MEAN_COLOR = "#922b21"

    for idx, result in enumerate(results):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        cfg = result["config"]
        cosines = result["cosines"]
        n_layers = result["n_layers"]
        layers = list(range(n_layers))

        ax.plot(layers, cosines, linewidth=1.5, color=LINE_COLOR)
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(
            y=result["mean_cosine"], color=MEAN_COLOR,
            linestyle=":", linewidth=1.0, alpha=0.7,
            label=f"Mean = {result['mean_cosine']:.3f}",
        )
        ax.set_xlabel("Layer", fontsize=18.7, fontweight="bold")
        ax.set_ylabel("Cosine Similarity", fontsize=18.7, fontweight="bold")
        ax.set_title(cfg.short_label, fontsize=18.7, fontweight="bold")
        ax.set_xlim(0, n_layers - 1)
        ax.set_ylim(-1.05, 1.05)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(loc="lower right", fontsize=12)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        "Cosine Similarity: CAA vs Logistic Regression Steering Vectors",
        fontsize=18.7, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    stem = "caa_vs_logreg_cosine_combined"
    for ext in ("pdf", "png"):
        path = os.path.join(save_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return os.path.join(save_dir, f"{stem}.pdf")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def generate_latex_table(results: List[dict], save_path: str) -> str:
    """Generate a LaTeX table summarizing cosine similarity across configs."""
    lines = [
        r"\begin{table}[ht]",
        r"\caption{\textbf{Cosine similarity between CAA and logistic regression "
        r"steering vectors.} Per-layer cosine similarity between the CAA "
        r"mean-difference vector and the logistic regression weight vector. "
        r"High similarity confirms that both methods recover the same direction; "
        r"logistic regression additionally provides a decision boundary for "
        r"selective steering (StTP, StMP).}",
        r"\label{tab:caa_vs_logreg}",
        r"\begin{center}",
        r"\begin{small}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Model & Trait & Layers & Mean Cos & Min Cos & Flipped \\",
        r"\midrule",
    ]

    for result in results:
        cfg = result["config"]
        model_short = cfg.model_save_name.split("_", 1)[-1]
        cosines = result["cosines"]
        lines.append(
            f"  {model_short} & {cfg.trait.capitalize()} & "
            f"{result['n_layers']} & "
            f"{result['mean_cosine']:.3f} & "
            f"{min(cosines):.3f} & "
            f"{result['n_flipped']}/{result['n_layers']} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{small}",
        r"\end{center}",
        r"\end{table}",
    ])

    tex = "\n".join(lines)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"\nLaTeX table saved → {save_path}")
    return tex


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(results: List[dict], save_dir: str):
    """Save per-layer cosine similarities to CSV files."""
    for result in results:
        cfg = result["config"]
        model_short = cfg.model_save_name.replace("/", "_")
        stem = f"caa_vs_logreg_cosine_{model_short}_{cfg.trait}_{cfg.dataset}.csv"
        path = os.path.join(save_dir, stem)
        with open(path, "w", encoding="utf-8") as f:
            f.write("layer,cosine_similarity,lr_norm,caa_norm\n")
            for layer in range(result["n_layers"]):
                f.write(
                    f"{layer},{result['cosines'][layer]:.8f},"
                    f"{result['lr_norms'][layer]:.6f},"
                    f"{result['caa_norms'][layer]:.6f}\n"
                )
            f.write(f"mean,{result['mean_cosine']:.8f},,\n")
        print(f"CSV saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare CAA and logistic regression steering vectors"
    )
    parser.add_argument(
        "--trait", default=None,
        help="Trait to check (e.g. compassion, honesty). "
             "If omitted, runs all default configs.",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Dataset / honesty category (e.g. diverse_with_sycophancy). "
             "Defaults to trait name for compassion.",
    )
    parser.add_argument(
        "--model", default=None, nargs="+",
        help="One or more model slugs (e.g. meta-llama_Llama-3.3-70B-Instruct "
             "Qwen_Qwen3.6-27B). Pass two or more to also produce the combined "
             "multi-model summary plot.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for plots, CSV, and LaTeX output. "
             "Defaults to plots/vector_comparison/.",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--no-latex", action="store_true",
        help="Skip LaTeX table generation.",
    )
    args = parser.parse_args()

    # --- Resolve output directory ---
    save_dir = args.output_dir or os.path.join(PLOTS_FOLDER, "vector_comparison")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # --- Build config list ---
    if args.trait is not None:
        dataset = args.dataset or args.trait
        models = (
            args.model if args.model
            else ["meta-llama_Llama-3.3-70B-Instruct", "Qwen_Qwen3.6-27B"]
        )
        configs = [ComparisonConfig(m, args.trait, dataset) for m in models]
    else:
        configs = DEFAULT_CONFIGS

    # --- Run comparisons ---
    all_results = []
    for cfg in configs:
        try:
            result = run_comparison(cfg)
            all_results.append(result)
        except (FileNotFoundError, KeyError) as e:
            print(f"\n  SKIPPED ({cfg.short_label}): {e}")

    if not all_results:
        print("\nNo comparisons completed — check that vector files exist.")
        return

    # --- CSV ---
    save_csv(all_results, save_dir)

    # --- Plots ---
    if not args.no_plot:
        for result in all_results:
            path = plot_single_config(result, save_dir)
            print(f"Plot saved → {path}")
        if len(all_results) > 1:
            path = plot_combined(all_results, save_dir)
            print(f"Combined plot saved → {path}")

    # --- LaTeX ---
    if not args.no_latex:
        tex_path = os.path.join(save_dir, "caa_vs_logreg_table.tex")
        generate_latex_table(all_results, tex_path)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for result in all_results:
        cfg = result["config"]
        print(
            f"  {cfg.short_label:45s}  "
            f"mean_cos={result['mean_cosine']:+.3f}  "
            f"min_cos={min(result['cosines']):+.3f}  "
            f"flipped={result['n_flipped']}/{result['n_layers']}"
        )
    print(f"\nAll outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
