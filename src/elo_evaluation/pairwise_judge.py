"""
Pairwise judge for ELO evaluation.

Formats two responses as anonymized A/B comparisons, evaluates both
position orderings to eliminate position bias, and aggregates verdicts
(A wins, B wins, or TIE).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from src.llm_judge.config import JudgeConfig
from src.llm_judge.engine import VLLMJudge
from src.llm_judge.prompts import load_prompt

# The pairwise rubric asks for a single verdict line. This contract belongs to
# these two prompt files only, so it is parsed here rather than in llm_judge.parsing.
VERDICT_RE = re.compile(r"VERDICT:\s*(A|B|TIE)", re.IGNORECASE)

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of a single pairwise comparison."""
    question_idx: int
    player_a: str
    player_b: str
    winner: str  # player name or "tie"
    raw_verdict: str  # "A", "B", or "TIE" (aggregated from both positions)
    position_a_was: str  # which player was shown as A (forward direction)
    position_b_was: str  # which player was shown as B (forward direction)
    judge_raw_output: str
    parse_failed: bool = False
    raw_verdict_forward: str = ""
    raw_verdict_reverse: str = ""
    judge_raw_output_forward: str = ""
    judge_raw_output_reverse: str = ""


def _format_compassion_pairwise(
    system_prompt_template: str,
    question: str,
    response_a: str,
    response_b: str,
    trait: str,
    trait_description: str,
) -> List[dict]:
    """Format a compassion pairwise comparison as chat messages."""
    system_text = system_prompt_template.format(
        TRAIT=trait,
        TRAIT_DESCRIPTION=trait_description,
        question=question,
        response_a=response_a,
        response_b=response_b,
    )
    return [{"role": "system", "content": system_text}]


def _format_honesty_pairwise(
    system_prompt_template: str,
    question: str,
    gold_truth: str,
    response_a: str,
    response_b: str,
) -> List[dict]:
    """Format an honesty pairwise comparison as chat messages."""
    system_text = system_prompt_template.format(
        question=question,
        gold_truth=gold_truth,
        response_a=response_a,
        response_b=response_b,
    )
    return [{"role": "system", "content": system_text}]


def parse_verdict(raw_output: str) -> Tuple[str, bool]:
    """
    Parse verdict from judge output.

    Returns (verdict, parse_failed) where verdict is "A", "B", or "TIE".
    Falls back to "TIE" with parse_failed=True if regex fails to match.
    """
    match = VERDICT_RE.search(raw_output)
    if not match:
        logger.warning(
            "Failed to parse verdict from judge output: %.200s", raw_output
        )
        return ("TIE", True)
    return (match.group(1).upper(), False)


class PairwiseJudge:
    """
    Pairwise judge that compares two responses using an LLM.

    Evaluates each pair in BOTH positions (A/B and B/A) to eliminate
    position bias. Verdicts are aggregated: if both positions agree on
    the same winner, that player wins; otherwise the result is a TIE.
    """

    def __init__(self, config: JudgeConfig, trait: str, trait_description: str = ""):
        self.trait = trait
        self.trait_description = trait_description

        if trait == "compassion":
            self.system_prompt_template = load_prompt("pairwise_elo_compassion")
        elif trait == "honesty":
            self.system_prompt_template = load_prompt("pairwise_elo_honesty")
        else:
            raise ValueError(f"Unsupported trait: {trait!r}. Expected 'compassion' or 'honesty'.")

        self.judge = VLLMJudge(config)

    def _format_pair(
        self, pair: dict, response_a: str, response_b: str,
    ) -> List[dict]:
        """Format a single pairwise comparison as chat messages."""
        if self.trait == "compassion":
            return _format_compassion_pairwise(
                self.system_prompt_template,
                question=pair["question"],
                response_a=response_a,
                response_b=response_b,
                trait=self.trait,
                trait_description=self.trait_description,
            )
        else:  # honesty
            return _format_honesty_pairwise(
                self.system_prompt_template,
                question=pair["question"],
                gold_truth=pair["gold_truth"],
                response_a=response_a,
                response_b=response_b,
            )

    def evaluate_pairs_batch(
        self,
        pairs: Sequence[dict],
        batch_size: Optional[int] = None,
    ) -> List[MatchResult]:
        """
        Evaluate a batch of pairwise comparisons using dual-position judging.

        Each pair is evaluated in BOTH positions to eliminate position bias:
          - Forward:  player_1 as A, player_2 as B
          - Reverse:  player_2 as A, player_1 as B

        Aggregation rule:
          - If both positions agree on the same player winning → that player wins
          - All other cases (disagree, any TIE) → TIE

        Each pair dict must contain:
            - question_idx: int
            - question: str
            - player_1: str (player name)
            - player_2: str (player name)
            - response_1: str
            - response_2: str
            - gold_truth: str (required for honesty, ignored for compassion)

        Returns list of MatchResult in the same order as input pairs.
        """
        if not pairs:
            return []

        # Build 2N prompts: forward and reverse for each pair
        all_messages: List[List[dict]] = []
        for pair in pairs:
            # Forward: player_1 as A, player_2 as B
            all_messages.append(
                self._format_pair(pair, pair["response_1"], pair["response_2"])
            )
            # Reverse: player_2 as A, player_1 as B
            all_messages.append(
                self._format_pair(pair, pair["response_2"], pair["response_1"])
            )

        # Run batched inference (2N prompts)
        raw_outputs = self.judge.generate(all_messages, batch_size)

        # Parse verdicts, aggregate, and build results
        results: List[MatchResult] = []
        n_parse_failures = 0

        for i, pair in enumerate(pairs):
            fwd_output = raw_outputs[2 * i]
            rev_output = raw_outputs[2 * i + 1]

            fwd_verdict, fwd_failed = parse_verdict(fwd_output)
            rev_verdict, rev_failed = parse_verdict(rev_output)
            failed = fwd_failed or rev_failed

            if failed:
                n_parse_failures += 1

            # Map positional verdicts to player-level winners
            # Forward: A → player_1, B → player_2
            if fwd_verdict == "A":
                fwd_winner = pair["player_1"]
            elif fwd_verdict == "B":
                fwd_winner = pair["player_2"]
            else:
                fwd_winner = "tie"

            # Reverse: A → player_2, B → player_1
            if rev_verdict == "A":
                rev_winner = pair["player_2"]
            elif rev_verdict == "B":
                rev_winner = pair["player_1"]
            else:
                rev_winner = "tie"

            # Aggregate: both must agree on the same player to declare a winner
            if (
                fwd_winner == rev_winner
                and fwd_winner != "tie"
            ):
                winner = fwd_winner
                agg_verdict = "A" if winner == pair["player_1"] else "B"
            else:
                winner = "tie"
                agg_verdict = "TIE"

            results.append(MatchResult(
                question_idx=pair["question_idx"],
                player_a=pair["player_1"],
                player_b=pair["player_2"],
                winner=winner,
                raw_verdict=agg_verdict,
                position_a_was=pair["player_1"],
                position_b_was=pair["player_2"],
                judge_raw_output=fwd_output,
                parse_failed=failed,
                raw_verdict_forward=fwd_verdict,
                raw_verdict_reverse=rev_verdict,
                judge_raw_output_forward=fwd_output,
                judge_raw_output_reverse=rev_output,
            ))

        if n_parse_failures:
            logger.warning(
                "Parse failures: %d / %d pairs", n_parse_failures, len(pairs)
            )

        return results
