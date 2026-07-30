from __future__ import annotations

import re
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


# ----------------------------
# Sentence splitting (deterministic baseline)
# ----------------------------

def _split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter.
    - Splits on [.?!] followed by whitespace.
    - Keeps sentences as-is (trimmed).
    """
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ----------------------------
# Embedding backends
# ----------------------------

class SentenceEmbedder:
    """
    Embeds text snippets into vectors.

    mode:
      - "sbert": uses sentence-transformers (best)
      - "tfidf": uses TF-IDF vectors (no extra deps beyond sklearn)
    """

    def __init__(self):
        self.sbert_model_name: str = "all-mpnet-base-v2"
        self._sbert = SentenceTransformer(self.sbert_model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, 1), dtype=np.float32)

        emb = self._sbert.encode(texts, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)


# ----------------------------
# Sentence template reuse metric
# ----------------------------

def sentence_template_reuse(
        assistant_text: str,
        prev_assistant_texts: List[str],
        *,
        embedder: Optional[SentenceEmbedder] = None,
        threshold: float = 0.90,
        return_matches: bool = False,
) -> Dict[str, Any]:
    """
    Measures sentence template reuse in `assistant_text` relative to previous assistant turns.

    For each sentence in the current assistant response:
      - find the most similar sentence among all previous assistant sentences
      - count it as "reused" if similarity >= threshold

    Returns:
      - sentence_count
      - reuse_rate
      - nn_sim_mean / nn_sim_max
      - (optional) sentence-level nearest neighbor matches
    """
    if embedder is None:
        embedder = SentenceEmbedder()

    cur_sents = _split_sentences(assistant_text)
    prev_sents: List[str] = []
    for pt in prev_assistant_texts:
        prev_sents.extend(_split_sentences(pt))

    # Edge cases
    if len(cur_sents) == 0:
        return {
            "sentence_count": 0,
            "reuse_rate": 0.0,
            "sentence_nn_sim_mean": 0.0,
            "sentence_nn_sim_max": 0.0,
            "threshold": threshold,
            "matches": [] if return_matches else None,
        }

    if len(prev_sents) == 0:
        # Nothing to compare against: by definition no reuse
        return {
            "sentence_count": len(cur_sents),
            "reuse_rate": 0.0,
            "sentence_nn_sim_mean": 0.0,
            "sentence_nn_sim_max": 0.0,
            "threshold": threshold,
            "matches": [] if return_matches else None,
        }

    prev_emb = embedder.encode(prev_sents)
    cur_emb = embedder.encode(cur_sents)

    sims = util.cos_sim(cur_emb, prev_emb)

    nn_sim, nn_idx = torch.max(sims, dim=1)   # both are torch tensors

    reused = nn_sim >= threshold
    reuse_rate = float(reused.float().mean().item()) if reused.numel() else 0.0

    nn_sim_mean = float(nn_sim.mean().item()) if nn_sim.numel() else 0.0
    nn_sim_max  = float(nn_sim.max().item()) if nn_sim.numel() else 0.0

    out: Dict[str, Any] = {
        "sentence_count": len(cur_sents),
        "reuse_rate": reuse_rate,
        "sentence_nn_sim_mean": nn_sim_mean,
        "sentence_nn_sim_max": nn_sim_max,
        "threshold": threshold,
    }

    if return_matches:
        matches = []
        for i, (s, j, simv, flag) in enumerate(zip(cur_sents, nn_idx.tolist(), nn_sim.tolist(), reused.tolist())):
            matches.append(
                {
                    "cur_sentence_idx": i,
                    "cur_sentence": s,
                    "nn_prev_sentence_idx": j,
                    "nn_prev_sentence": prev_sents[j],
                    "nn_sim": float(simv),
                    "is_reuse": bool(flag),
                }
            )
        out["matches"] = matches

    return out
