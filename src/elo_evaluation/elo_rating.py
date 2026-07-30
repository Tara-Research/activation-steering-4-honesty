"""
Rating computation from pairwise match results.

Provides:
- Iterative ELO updates (online, order-dependent)
- Bradley-Terry MLE via MM algorithm (batch, order-independent, principled)
- Bootstrap confidence intervals for both methods
- Affine normalization to anchor baselines
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.constants.constants import SEED


INITIAL_RATING = 1500.0
K_FACTOR = 32.0


@dataclass
class EloResult:
    """Final ELO ratings with metadata."""
    ratings: Dict[str, float]  # player -> rating
    win_matrix: Dict[str, Dict[str, int]]  # player_a -> player_b -> wins for a
    tie_matrix: Dict[str, Dict[str, int]]  # player_a -> player_b -> ties
    n_matches: int
    bootstrap_ci: Optional[Dict[str, Tuple[float, float]]] = None  # player -> (lo, hi) 95% CI
    bootstrap_ci_normalized: Optional[Dict[str, Tuple[float, float]]] = None  # normalized (lo, hi)


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A against player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_elo_ratings(
    matches: List[dict],
    k_factor: float = K_FACTOR,
    initial_rating: float = INITIAL_RATING,
    n_rounds: int = 10,
) -> EloResult:
    """
    Compute ELO ratings from a list of match results.

    Each match dict must contain:
        - player_a: str
        - player_b: str
        - winner: str (player name or "tie")

    Processes all matches n_rounds times for convergence.

    Returns EloResult with final ratings, win/tie matrices, and match count.
    """
    # Collect all players
    players = set()
    for m in matches:
        players.add(m["player_a"])
        players.add(m["player_b"])

    # Build win/tie matrices
    win_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tie_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for m in matches:
        pa, pb, w = m["player_a"], m["player_b"], m["winner"]
        if w == "tie":
            tie_matrix[pa][pb] += 1
            tie_matrix[pb][pa] += 1
        elif w == pa:
            win_matrix[pa][pb] += 1
        elif w == pb:
            win_matrix[pb][pa] += 1

    # Iterative ELO
    rng = random.Random(SEED)
    ratings = {p: initial_rating for p in players}

    for _ in range(n_rounds):
        shuffled = list(matches)
        rng.shuffle(shuffled)
        for m in shuffled:
            pa, pb, w = m["player_a"], m["player_b"], m["winner"]
            ea = _expected_score(ratings[pa], ratings[pb])
            eb = 1.0 - ea

            if w == pa:
                sa, sb = 1.0, 0.0
            elif w == pb:
                sa, sb = 0.0, 1.0
            else:  # tie
                sa, sb = 0.5, 0.5

            ratings[pa] += k_factor * (sa - ea)
            ratings[pb] += k_factor * (sb - eb)

    # Normalize: convert defaultdicts to regular dicts
    win_dict = {p: dict(win_matrix[p]) for p in players}
    tie_dict = {p: dict(tie_matrix[p]) for p in players}

    return EloResult(
        ratings=dict(sorted(ratings.items(), key=lambda x: x[1], reverse=True)),
        win_matrix=win_dict,
        tie_matrix=tie_dict,
        n_matches=len(matches),
    )


def _build_bt_matrices(
    matches: List[dict],
) -> Tuple[List[str], Dict[str, int], np.ndarray, np.ndarray]:
    """
    Build the win and match-count matrices for Bradley-Terry.

    Returns (players_list, player_to_idx, W, N) where:
      W[i,j] = effective wins of i over j (wins + 0.5 * ties)
      N[i,j] = total matches between i and j
    """
    players = set()
    for m in matches:
        players.add(m["player_a"])
        players.add(m["player_b"])
    players_list = sorted(players)
    idx = {p: i for i, p in enumerate(players_list)}
    k = len(players_list)

    W = np.zeros((k, k))  # effective wins
    N = np.zeros((k, k))  # total matches

    for m in matches:
        i, j = idx[m["player_a"]], idx[m["player_b"]]
        winner = m["winner"]
        N[i, j] += 1
        N[j, i] += 1
        if winner == m["player_a"]:
            W[i, j] += 1.0
        elif winner == m["player_b"]:
            W[j, i] += 1.0
        else:  # tie -> 0.5 each
            W[i, j] += 0.5
            W[j, i] += 0.5

    return players_list, idx, W, N


def compute_bt_ratings(
    matches: List[dict],
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> EloResult:
    """
    Compute ratings via Bradley-Terry maximum likelihood estimation.

    Uses the MM (minorization-maximization) algorithm of Hunter (2004).
    This is the standard method used by Chatbot Arena / LMSYS.

    The BT model assumes:
        P(i beats j) = gamma_i / (gamma_i + gamma_j)

    The MM update for each player i is:
        gamma_i <- W_i / sum_j [ N_ij / (gamma_i + gamma_j) ]

    where W_i = total effective wins (ties count as 0.5).

    Final gamma values are converted to the ELO scale:
        R_i = 400 * log10(gamma_i) + 1500

    Unlike iterative ELO, BT-MLE:
    - Produces a unique solution (up to a scale factor)
    - Is independent of match ordering
    - Maximizes the likelihood of the observed outcomes
    """
    players_list, idx, W, N = _build_bt_matrices(matches)
    k = len(players_list)

    # Effective wins per player
    w_total = W.sum(axis=1)  # shape (k,)

    # Initialize strengths uniformly
    gamma = np.ones(k)

    for iteration in range(max_iter):
        gamma_old = gamma.copy()

        # NOTE: This is a Gauss-Seidel (in-place) update — when computing
        # gamma[i], players j < i already hold their (t+1) values. Hunter
        # (2004) proves both Gauss-Seidel and Jacobi variants converge to
        # the same MLE fixed point; Gauss-Seidel typically converges faster.
        for i in range(k):
            if w_total[i] == 0:
                # Player never won -> assign minimal strength
                gamma[i] = 1e-6
                continue
            denom = 0.0
            for j in range(k):
                if i == j or N[i, j] == 0:
                    continue
                denom += N[i, j] / (gamma[i] + gamma[j])
            if denom > 0:
                gamma[i] = w_total[i] / denom

        # Normalize so geometric mean = 1 (fixes the free scale parameter)
        gamma /= np.exp(np.mean(np.log(np.maximum(gamma, 1e-30))))

        # Check convergence
        change = np.max(np.abs(gamma - gamma_old) / np.maximum(gamma_old, 1e-30))
        if change < tol:
            break

    # Convert to ELO scale: R = 400 * log10(gamma) + 1500
    ratings = {}
    for i, p in enumerate(players_list):
        ratings[p] = 400.0 * math.log10(max(gamma[i], 1e-30)) + INITIAL_RATING

    # Build win/tie matrices in dict format (reuse from matches)
    win_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tie_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for m in matches:
        pa, pb, w = m["player_a"], m["player_b"], m["winner"]
        if w == "tie":
            tie_matrix[pa][pb] += 1
            tie_matrix[pb][pa] += 1
        elif w == pa:
            win_matrix[pa][pb] += 1
        elif w == pb:
            win_matrix[pb][pa] += 1

    return EloResult(
        ratings=dict(sorted(ratings.items(), key=lambda x: x[1], reverse=True)),
        win_matrix={p: dict(win_matrix[p]) for p in players_list},
        tie_matrix={p: dict(tie_matrix[p]) for p in players_list},
        n_matches=len(matches),
    )


def bootstrap_bt_ci(
    matches: List[dict],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap confidence intervals for Bradley-Terry MLE ratings."""
    rng = np.random.RandomState(SEED)
    n = len(matches)

    all_ratings: Dict[str, List[float]] = defaultdict(list)

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        sample = [matches[i] for i in indices]
        result = compute_bt_ratings(sample)
        for player, rating in result.ratings.items():
            all_ratings[player].append(rating)

    alpha_val = (1.0 - confidence) / 2.0
    ci: Dict[str, Tuple[float, float]] = {}
    for player, ratings_list in all_ratings.items():
        arr = np.array(ratings_list)
        lo = float(np.percentile(arr, 100 * alpha_val))
        hi = float(np.percentile(arr, 100 * (1.0 - alpha_val)))
        ci[player] = (lo, hi)

    return ci


def bootstrap_bt_ci_normalized(
    matches: List[dict],
    anchor_low_player: str = "malicious_baseline",
    anchor_high_player: str = "aligned_baseline",
    anchor_low_value: float = 1000.0,
    anchor_high_value: float = 2000.0,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap confidence intervals for Bradley-Terry ratings with
    per-sample normalization.

    Each bootstrap resample is independently normalized so that the two
    anchor players map to fixed values. This avoids the artifact of
    applying a single affine transform (from the point estimate) to
    all CI bounds. By construction, anchor players receive zero-width CIs.
    """
    rng = np.random.RandomState(SEED)
    n = len(matches)

    all_ratings: Dict[str, List[float]] = defaultdict(list)
    n_skipped = 0

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        sample = [matches[i] for i in indices]
        result = compute_bt_ratings(sample)

        # Skip degenerate samples where anchors are missing, identical,
        # or hit the gamma floor (zero wins → gamma=1e-6 → rating=-900).
        # Floor-hit samples distort the normalization scale because the
        # "true" BT strength is -inf, and the arbitrary floor value
        # creates an artificially large anchor spread.
        _FLOOR_RATING = 400.0 * math.log10(1e-6) + INITIAL_RATING  # -900
        r_low = result.ratings.get(anchor_low_player)
        r_high = result.ratings.get(anchor_high_player)
        if (
            r_low is None
            or r_high is None
            or r_low == r_high
            or r_low <= _FLOOR_RATING + 1.0
            or r_high <= _FLOOR_RATING + 1.0
        ):
            n_skipped += 1
            continue

        norm_ratings, _ = normalize_elo_ratings(
            result.ratings,
            anchor_low_player=anchor_low_player,
            anchor_high_player=anchor_high_player,
            anchor_low_value=anchor_low_value,
            anchor_high_value=anchor_high_value,
        )
        for player, rating in norm_ratings.items():
            all_ratings[player].append(rating)

    alpha_val = (1.0 - confidence) / 2.0
    ci: Dict[str, Tuple[float, float]] = {}
    for player, ratings_list in all_ratings.items():
        arr = np.array(ratings_list)
        lo = float(np.percentile(arr, 100 * alpha_val))
        hi = float(np.percentile(arr, 100 * (1.0 - alpha_val)))
        ci[player] = (lo, hi)

    return ci


def bootstrap_elo_ci(
    matches: List[dict],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    k_factor: float = K_FACTOR,
    initial_rating: float = INITIAL_RATING,
    n_rounds: int = 10,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute bootstrap confidence intervals for ELO ratings.

    Resamples matches with replacement and computes ELO ratings for each
    bootstrap sample. Returns per-player (lower, upper) bounds.
    """
    rng = np.random.RandomState(SEED)
    n = len(matches)

    all_ratings: Dict[str, List[float]] = defaultdict(list)

    for _ in range(n_bootstrap):
        indices = rng.randint(0, n, size=n)
        sample = [matches[i] for i in indices]
        result = compute_elo_ratings(
            sample,
            k_factor=k_factor,
            initial_rating=initial_rating,
            n_rounds=n_rounds,
        )
        for player, rating in result.ratings.items():
            all_ratings[player].append(rating)

    alpha = (1.0 - confidence) / 2.0
    ci: Dict[str, Tuple[float, float]] = {}
    for player, ratings_list in all_ratings.items():
        arr = np.array(ratings_list)
        lo = float(np.percentile(arr, 100 * alpha))
        hi = float(np.percentile(arr, 100 * (1.0 - alpha)))
        ci[player] = (lo, hi)

    return ci


def normalize_elo_ratings(
    ratings: Dict[str, float],
    anchor_low_player: str = "malicious_baseline",
    anchor_high_player: str = "aligned_baseline",
    anchor_low_value: float = 1000.0,
    anchor_high_value: float = 2000.0,
    bootstrap_ci: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[Dict[str, float], Optional[Dict[str, Tuple[float, float]]]]:
    """
    Apply an affine transformation R' = a*R + b to ELO ratings so that
    two anchor players map to fixed values.

    ELO ratings have two free parameters (translation and scale) because
    only differences matter in the expected-score formula. This function
    fixes both by mapping anchor_low_player -> anchor_low_value and
    anchor_high_player -> anchor_high_value.

    The same transformation is applied to bootstrap CIs if provided.

    Returns (normalized_ratings, normalized_bootstrap_ci).
    """
    r_low = ratings[anchor_low_player]
    r_high = ratings[anchor_high_player]

    if r_high == r_low:
        raise ValueError(
            f"Anchor players have identical ratings ({r_high}), "
            f"cannot compute normalization."
        )

    # R' = a * R + b, with a * r_low + b = anchor_low_value
    #                   and a * r_high + b = anchor_high_value
    a = (anchor_high_value - anchor_low_value) / (r_high - r_low)
    b = anchor_low_value - a * r_low

    norm_ratings = {p: a * r + b for p, r in ratings.items()}

    norm_ci = None
    if bootstrap_ci is not None:
        norm_ci = {
            p: (a * lo + b, a * hi + b)
            for p, (lo, hi) in bootstrap_ci.items()
        }

    return norm_ratings, norm_ci


def compute_win_rates(
    win_matrix: Dict[str, Dict[str, int]],
    tie_matrix: Dict[str, Dict[str, int]],
    players: List[str],
) -> np.ndarray:
    """
    Compute pairwise win-rate matrix.

    Returns NxN numpy array where entry [i, j] is win rate of player i
    against player j (wins + 0.5*ties) / total_matches.
    """
    n = len(players)
    player_idx = {p: i for i, p in enumerate(players)}
    rates = np.full((n, n), np.nan)

    for i, pi in enumerate(players):
        for j, pj in enumerate(players):
            if i == j:
                rates[i, j] = np.nan
                continue
            wins_i = win_matrix.get(pi, {}).get(pj, 0)
            wins_j = win_matrix.get(pj, {}).get(pi, 0)
            ties = tie_matrix.get(pi, {}).get(pj, 0)
            total = wins_i + wins_j + ties
            if total == 0:
                rates[i, j] = np.nan
            else:
                rates[i, j] = (wins_i + 0.5 * ties) / total

    return rates
