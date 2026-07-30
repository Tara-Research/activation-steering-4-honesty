"""
Steering-vector extraction via logistic regression.

Loads the per-side response-average embeddings ([L, N, D] tensors written by
generate_vec_caa.py), fits an sklearn LogisticRegression per layer, and saves
the weight matrix [L, D], biases [L], and a metadata JSON.

No model / GPU required.

Usage:
    python -m src.vector_generation.generate_vec_logistic_regression \
        --pos-features <pos_embeddings.pt> --neg-features <neg_embeddings.pt> \
        --trait honesty --dataset self_report_experimental \
        --model-id meta-llama/Llama-3.3-70B-Instruct
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.constants import constants
from src.vector_generation.extraction_utils import MODE


def compute_logistic_regression_vectors(
    pos_embeddings: torch.Tensor,
    neg_embeddings: torch.Tensor,
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    penalty: str = "l2",
    random_state: int = 42,
    class_weight: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, float]]:
    """Fit logistic regression per layer and return weight vectors.

    Parameters
    ----------
    pos_embeddings : Tensor [L, N_pos, D]
        Positive-class (e.g. honest) embeddings across all layers.
    neg_embeddings : Tensor [L, N_neg, D]
        Negative-class (e.g. dishonest) embeddings across all layers.
    C : float
        Inverse regularisation strength.
    solver : str
        Solver for LogisticRegression.
    max_iter : int
        Maximum iterations for the solver.
    penalty : str
        Regularisation type ('l2', 'l1', etc.).
    random_state : int
        Random seed for reproducibility.
    class_weight : str | None
        'balanced' to auto-adjust for class imbalance, or None.

    Returns
    -------
    weights : Tensor [L, D]
        Weight vectors (steering directions) per layer.
    biases : Tensor [L]
        Intercept (decision boundary offset) per layer.
    accuracies : dict[int, float]
        Training accuracy per layer index.
    """
    n_layers = pos_embeddings.shape[0]
    D = pos_embeddings.shape[2]

    weights = torch.zeros(n_layers, D, dtype=pos_embeddings.dtype)
    biases = torch.zeros(n_layers, dtype=pos_embeddings.dtype)
    accuracies: dict[int, float] = {}

    for layer in range(n_layers):
        pos = pos_embeddings[layer].float().numpy()  # [N_pos, D]
        neg = neg_embeddings[layer].float().numpy()  # [N_neg, D]

        X = np.concatenate([pos, neg], axis=0)
        y = np.concatenate([
            np.ones(pos.shape[0], dtype=np.int32),
            np.zeros(neg.shape[0], dtype=np.int32),
        ])

        clf = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            penalty=penalty,
            random_state=random_state,
            class_weight=class_weight,
        )
        clf.fit(X, y)

        weights[layer] = torch.from_numpy(clf.coef_[0]).to(pos_embeddings.dtype)
        biases[layer] = float(clf.intercept_[0])
        accuracies[layer] = float(clf.score(X, y))

    return weights, biases, accuracies


def normalize_by_projection(
    weights: torch.Tensor,
    biases: torch.Tensor,
    pos_embeddings: torch.Tensor,
    neg_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, float]]:
    """Normalize LR vectors to the mean projection gap of pos/neg embeddings.

    For each layer, computes the unit LR direction, then measures how far
    apart the positive and negative class centroids are along that direction.
    The returned vector has magnitude equal to that gap (delta_mu), so
    ``coef=1`` moves activations by one "natural unit" of class separation.

    The bias is rescaled by the same factor (delta_mu / w_norm) so that the
    decision boundary ``-b / ‖w‖`` remains consistent after normalization:
    ``-b_rescaled / ‖w_normed‖ = -b_original / ‖w_original‖``.

    Math: ``lr_normalized = lr_hat * delta_mu``
          ``b_normalized  = b_original * (delta_mu / w_norm)``

    Parameters
    ----------
    weights : Tensor [L, D]
        Logistic regression weight vectors.
    biases : Tensor [L]
        Logistic regression intercepts (will be rescaled in-place).
    pos_embeddings : Tensor [L, N_pos, D]
        Positive-class embeddings.
    neg_embeddings : Tensor [L, N_neg, D]
        Negative-class embeddings.

    Returns
    -------
    normed : Tensor [L, D]
        Projection-normalized steering vectors.
    biases : Tensor [L]
        Rescaled biases consistent with the normalized weights.
    delta_mu_per_layer : dict[int, float]
        Per-layer projection gap (signed).
    """
    normed = weights.clone()
    biases = biases.clone()
    delta_mu_per_layer: dict[int, float] = {}

    for layer in range(weights.shape[0]):
        w = weights[layer].float()
        w_norm = w.norm()
        if w_norm == 0:
            delta_mu_per_layer[layer] = 0.0
            continue

        lr_hat = w / w_norm  # [D]

        pos_proj = (pos_embeddings[layer].float() @ lr_hat).mean().item()
        neg_proj = (neg_embeddings[layer].float() @ lr_hat).mean().item()
        delta_mu = pos_proj - neg_proj

        normed[layer] = (lr_hat * delta_mu).to(weights.dtype)
        biases[layer] *= delta_mu / w_norm.item()
        delta_mu_per_layer[layer] = delta_mu

    return normed, biases, delta_mu_per_layer


def save_logistic_regression_vectors(
    weights: torch.Tensor,
    biases: torch.Tensor,
    accuracies: dict[int, float],
    save_dir: str | Path,
    trait: str,
    dataset: str,
    model_save_name: str,
    hyperparams: dict,
    pos_embeddings_path: str,
    neg_embeddings_path: str,
    normalization_method: str = "projection",
    pos_embeddings: Optional[torch.Tensor] = None,
    neg_embeddings: Optional[torch.Tensor] = None,
) -> dict[str, Path]:
    """Save LR steering vectors, biases, and metadata.

    Parameters
    ----------
    weights : Tensor [L, D]
    biases : Tensor [L]
    accuracies : dict[int, float]
    save_dir : path
    trait, dataset, model_save_name : str
        Used to construct filenames.
    hyperparams : dict
        LR hyperparameters for metadata.
    pos_embeddings_path, neg_embeddings_path : str
        Source paths for provenance.
    normalization_method : str
        One of ``"none"``, ``"projection"``.
    pos_embeddings : Tensor | None
        Required when ``normalization_method="projection"``.
    neg_embeddings : Tensor | None
        Required when ``normalization_method="projection"``.

    Returns
    -------
    dict mapping file type to saved path.
    """
    if normalization_method not in ("none", "projection"):
        raise ValueError(
            f"Unknown normalization_method '{normalization_method}'. "
            f"Must be one of: none, projection"
        )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{trait}_{dataset}_logreg"
    vec_name = f"{stem}_steering_vector_{MODE}_{model_save_name}.pt"
    bias_name = f"{stem}_bias_{MODE}_{model_save_name}.pt"
    meta_name = f"{stem}_metadata_{MODE}_{model_save_name}.json"

    delta_mu_per_layer: Optional[dict[int, float]] = None

    if normalization_method == "projection":
        if pos_embeddings is None or neg_embeddings is None:
            raise ValueError(
                "normalization_method='projection' requires pos_embeddings "
                "and neg_embeddings"
            )
        weights, biases, delta_mu_per_layer = normalize_by_projection(
            weights, biases, pos_embeddings, neg_embeddings
        )

    vec_path = save_dir / vec_name
    bias_path = save_dir / bias_name
    meta_path = save_dir / meta_name

    torch.save(weights, vec_path)
    torch.save(biases, bias_path)

    metadata = {
        "method": "logistic_regression",
        "mode": MODE,
        "trait": trait,
        "dataset": dataset,
        "model": model_save_name,
        "hyperparams": hyperparams,
        "per_layer_train_accuracy": {str(k): v for k, v in accuracies.items()},
        "pos_embeddings_source": str(pos_embeddings_path),
        "neg_embeddings_source": str(neg_embeddings_path),
        "weight_shape": list(weights.shape),
        "bias_shape": list(biases.shape),
        "normalization_method": normalization_method,
    }
    if delta_mu_per_layer is not None:
        metadata["delta_mu_per_layer"] = {
            str(k): v for k, v in delta_mu_per_layer.items()
        }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved steering vector: {vec_path}")
    print(f"Saved bias:            {bias_path}")
    print(f"Saved metadata:        {meta_path}")

    return {"vector": vec_path, "bias": bias_path, "metadata": meta_path}


def main():
    parser = argparse.ArgumentParser(
        description="Generate steering vectors via logistic regression on response-average embeddings."
    )
    parser.add_argument(
        "--pos-features",
        required=True,
        help="Path to positive-class response-average embeddings ([L, N_pos, D] tensor "
             "written by generate_vec_caa.py).",
    )
    parser.add_argument(
        "--neg-features",
        required=True,
        help="Path to negative-class response-average embeddings ([L, N_neg, D] tensor).",
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="Output directory (default: PERSONA_VECTORS_FOLDER/logistic_regression/{model}/{trait}/{dataset})",
    )
    parser.add_argument("--trait", required=True, help="e.g. honesty, compassion")
    parser.add_argument("--dataset", required=True, help="e.g. self_report_experimental, compassion")
    parser.add_argument("--model-id", required=True, help="e.g. meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--C", type=float, default=0.001, help="Inverse regularisation strength")
    parser.add_argument("--max-iter", type=int, default=1000, help="Max solver iterations")
    parser.add_argument("--penalty", default="l2", help="Regularisation type")
    parser.add_argument("--solver", default="lbfgs", help="Solver algorithm")
    parser.add_argument("--class-weight", default=None, help="'balanced' or None")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--normalization-method",
        default="projection",
        choices=["none", "projection"],
        help=(
            "Normalization strategy: 'none' (raw LR weights), "
            "'projection' (scale to pos/neg projection gap)"
        ),
    )
    args = parser.parse_args()

    model_save_name = args.model_id.replace("/", "_")

    save_dir = args.save_dir or os.path.join(
        constants.PERSONA_VECTORS_FOLDER,
        "logistic_regression",
        model_save_name,
        args.trait,
        args.dataset,
    )

    hyperparams = {
        "C": args.C,
        "solver": args.solver,
        "max_iter": args.max_iter,
        "penalty": args.penalty,
        "random_state": args.random_state,
        "class_weight": args.class_weight,
    }

    print(f"Loading positive features: {args.pos_features}")
    pos_embeddings = torch.load(args.pos_features, weights_only=False)
    print(f"Loading negative features: {args.neg_features}")
    neg_embeddings = torch.load(args.neg_features, weights_only=False)

    print(f"Positive shape: {pos_embeddings.shape}, Negative shape: {neg_embeddings.shape}")
    assert pos_embeddings.dim() == 3, f"Expected 3D tensor [L,N,D], got {pos_embeddings.dim()}D"
    assert neg_embeddings.dim() == 3, f"Expected 3D tensor [L,N,D], got {neg_embeddings.dim()}D"
    assert pos_embeddings.shape[0] == neg_embeddings.shape[0], "Layer count mismatch"
    assert pos_embeddings.shape[2] == neg_embeddings.shape[2], "Hidden dim mismatch"

    print(f"Fitting logistic regression on {pos_embeddings.shape[0]} layers...")
    weights, biases, accuracies = compute_logistic_regression_vectors(
        pos_embeddings=pos_embeddings,
        neg_embeddings=neg_embeddings,
        C=args.C,
        solver=args.solver,
        max_iter=args.max_iter,
        penalty=args.penalty,
        random_state=args.random_state,
        class_weight=args.class_weight,
    )

    mean_acc = np.mean(list(accuracies.values()))
    print(f"Mean training accuracy: {mean_acc:.4f}")
    print(f"Weight tensor shape: {weights.shape}")

    save_logistic_regression_vectors(
        weights=weights,
        biases=biases,
        accuracies=accuracies,
        save_dir=save_dir,
        trait=args.trait,
        dataset=args.dataset,
        model_save_name=model_save_name,
        hyperparams=hyperparams,
        pos_embeddings_path=args.pos_features,
        neg_embeddings_path=args.neg_features,
        normalization_method=args.normalization_method,
        pos_embeddings=pos_embeddings,
        neg_embeddings=neg_embeddings,
    )


if __name__ == "__main__":
    main()
