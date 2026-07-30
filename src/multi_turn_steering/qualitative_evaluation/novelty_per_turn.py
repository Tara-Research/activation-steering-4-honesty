from __future__ import annotations

from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer, util


# ----------------------------
# Turn embedding (SBERT-only)
# ----------------------------

class TurnEmbedder:
    def __init__(self):
        self.sbert_model_name: str = "all-mpnet-base-v2"
        self._sbert = SentenceTransformer(self.sbert_model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 1), dtype=np.float32)

        emb = self._sbert.encode(texts, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)


# ----------------------------
# Semantic novelty per turn
# ----------------------------

def semantic_novelty_per_turn(
        assistant_text: str,
        prev_assistant_texts: List[str],
        *,
        embedder: Optional[TurnEmbedder] = None,
        compute_adjacent: bool = True,
) -> Dict[str, Any]:
    """
    Semantic novelty for the current assistant turn.

    Definitions:
      - history_novelty = 1 - max_{i < t} cos(e_t, e_i)
      - adjacent_novelty = 1 - cos(e_t, e_{t-1})   (optional)

    Returns:
      - history_novelty
      - max_sim_to_history
      - adjacent_novelty (or None)
    """
    if embedder is None:
        embedder = TurnEmbedder()

    # No history => maximally novel by convention
    if len(prev_assistant_texts) == 0:
        return {
            "history_novelty": 1.0,
            "max_sim_to_history": 0.0,
            "adjacent_novelty": None,
        }

    prev_emb = embedder.encode(prev_assistant_texts)      # (T_prev, D)
    cur_emb = embedder.encode([assistant_text])           # (1, D)

    # Similarity of current turn to all previous turns: (1, T_prev)
    sims = util.cos_sim(cur_emb, prev_emb)

    max_sim = float(sims.max().item())
    history_novelty = float(max(0.0, 1.0 - max_sim))

    adjacent_novelty = None
    if compute_adjacent:
        sim_adj = float(sims[0, -1].item())
        adjacent_novelty = float(max(0.0, 1.0 - sim_adj))

    return {
        "history_novelty": history_novelty,
        "max_sim_to_history": max_sim,
        "adjacent_novelty": adjacent_novelty,
    }
