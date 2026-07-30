
#  -------- N-gram repetition ----------------------------------

import re
from typing import List, Tuple, Dict

_WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)

def _tokenize_words(text: str) -> List[str]:
    """Lowercased word tokenization. Keeps apostrophes within words."""
    return _WORD_RE.findall(text.lower())

def _make_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def _ngram_repeat_within(text: str, n: int) -> float:
    """
    Fraction of n-gram *occurrences* that are repetitions within the same text.
    0.0 means all n-grams are unique (no repetition).
    """
    toks = _tokenize_words(text)
    ng = _make_ngrams(toks, n)
    if not ng:
        return 0.0
    total = len(ng)
    unique = len(set(ng))
    return (total - unique) / total

def _ngram_repeat_across(text: str, prev_texts: List[str], n: int) -> float:
    """
    Fraction of unique n-grams in `text` that also appeared in any previous text.
    0.0 means no overlap with history.
    """
    toks = _tokenize_words(text)
    ng_set = set(_make_ngrams(toks, n))
    if not ng_set:
        return 0.0

    prev_set = set()
    for pt in prev_texts:
        ptoks = _tokenize_words(pt)
        prev_set.update(_make_ngrams(ptoks, n))

    return len(ng_set.intersection(prev_set)) / len(ng_set)

def compute_ngram_repetition(
    assistant_text: str,
    prev_assistant_texts: List[str],
    ns: List[int] = [2, 3, 4],
) -> Dict[str, Dict[int, float]]:
    """
    Convenience wrapper returning both within-turn and across-turn repetition
    for multiple n values.
    """
    within = {n: _ngram_repeat_within(assistant_text, n) for n in ns}
    across = {n: _ngram_repeat_across(assistant_text, prev_assistant_texts, n) for n in ns}
    return {"within": within, "across_prev": across}

#  -------------------------------------------------------------------
