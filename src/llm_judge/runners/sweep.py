"""LLM judge over single-turn steering sweep results.

Two phases per file, each submitted to vLLM as one mega-batch: the trait phase
scores how strongly the response expresses the target trait, the coherence phase
scores whether it is still well-formed. Scores are written back into the results
JSON in place, and the file is re-saved at every checkpoint so a requeued cluster
job resumes instead of restarting.

Honesty and compassion differ only in how a trait prompt is built and how its
result is stored -- that is the entire content of the two scorer classes below.
Everything else (traversal, batching, resume, checkpointing) is shared.

    python -m src.llm_judge.runners.sweep --trait honesty --results_dir DIR
    python -m src.llm_judge.runners.sweep --trait compassion --results_dir DIR
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from src.llm_judge.config import JudgeConfig
from src.llm_judge.engine import VLLMJudge
from src.llm_judge.parsing import parse_int_score, parse_judge_json
from src.llm_judge.prompts import (
    build_honesty_messages,
    build_qa_messages,
    load_prompt,
    load_trait_prompt,
)
from src.llm_judge.runners.ground_truth import (
    honesty_scenarios_path,
    load_ground_truth_data,
    normalize_message,
)

Conversation = List[Dict[str, str]]


# ---------------------------------------------------------------------------
# Trait scorers -- the only trait-specific code in this module
# ---------------------------------------------------------------------------

class HonestyScorer:
    """Scores disclosure against a gold truth; stores a JSON verdict dict."""

    trait_key = "judge_score"
    default_checkpoint_interval = 4000
    # Honesty always keeps coherence_score_raw; only compassion can opt out.
    omit_raw = False

    def __init__(self, ground_truth: Dict[str, Dict[str, Any]], include_raw_output: bool):
        self.system_prompt = load_prompt("honesty_universal")
        self.ground_truth = ground_truth
        self.include_raw_output = include_raw_output
        self.unmatched = 0

    def build_messages(self, question: str, answer: str) -> Optional[Conversation]:
        truth = self.ground_truth.get(normalize_message(question))
        if truth is None:
            self.unmatched += 1
            return None
        return build_honesty_messages(
            self.system_prompt,
            gold_truth=truth["gold_truth"],
            user_message=truth["user_message"],
            model_response=answer,
            category=truth["category"],
        )

    def store(self, item: Dict[str, Any], keys: Dict[str, str], raw: str) -> bool:
        parsed = parse_judge_json(raw)
        if parsed is None:
            preview = raw[:500] + "..." if len(raw) > 500 else raw
            item[keys["trait"]] = {"parse_error": True, "raw_preview": preview}
            return False

        entry = {
            "score": parsed["honesty_score"],
            "honesty_score": parsed["honesty_score"],
            "verdict": parsed["verdict"],
            "reason": parsed["reason"],
        }
        if self.include_raw_output:
            entry["raw_output"] = raw
        item[keys["trait"]] = entry
        return True


class CompassionScorer:
    """Scores trait expression on a 0-100 rubric; stores an int plus its raw text."""

    trait_key = "trait_expression_score"
    default_checkpoint_interval = 5000

    def __init__(self, rubric: str, trait: str, no_raw_score: bool):
        self.system_prompt = load_trait_prompt(rubric, trait)
        self.omit_raw = no_raw_score

    def build_messages(self, question: str, answer: str) -> Optional[Conversation]:
        return build_qa_messages(self.system_prompt, question, answer)

    def store(self, item: Dict[str, Any], keys: Dict[str, str], raw: str) -> bool:
        score = parse_int_score(raw)
        if not self.omit_raw:
            item[keys["trait_raw"]] = raw
        item[keys["trait"]] = score
        return score is not None


def build_scorer(args: argparse.Namespace):
    if args.trait == "honesty":
        scenarios = honesty_scenarios_path(args.model_save_name, args.honesty_category)
        return HonestyScorer(
            ground_truth=load_ground_truth_data(scenarios),
            include_raw_output=args.include_raw_output,
        )

    # "dismissive_system_prompt" means the model was steered away from
    # dismissiveness, so what we measure is compassion -- and vice versa.
    if args.sys_prompt_type == "dismissive_system_prompt":
        return CompassionScorer("compassion", "compassion", args.no_raw_score)
    return CompassionScorer("dismissiveness", "dismissiveness", args.no_raw_score)


# ---------------------------------------------------------------------------
# Results-file traversal
# ---------------------------------------------------------------------------

def iter_answer_lists(
    results: Any,
    target_layers: Optional[List[str]],
    target_coefs: Optional[List[str]],
) -> Iterator[List[Dict[str, Any]]]:
    """Yield every list of answer items in a results file, in file order.

    Three shapes occur: a bare list (baseline files), ``layers -> [answers]``,
    and ``layers -> coef -> [answers]``.
    """
    if isinstance(results, list):
        yield results
        return

    for layer_name, layer_data in results.get("layers", {}).items():
        if target_layers is not None and layer_name not in target_layers:
            continue

        if isinstance(layer_data, list):
            yield layer_data
        elif isinstance(layer_data, dict):
            for coef_name, answers in layer_data.items():
                if target_coefs is not None and coef_name not in target_coefs:
                    continue
                if isinstance(answers, list):
                    yield answers


def score_keys(scorer, sample_run: Optional[int]) -> Dict[str, str]:
    """Result keys for this scorer, suffixed when sampling."""
    suffix = "" if sample_run is None else f"_{sample_run}"
    return {
        "trait": scorer.trait_key + suffix,
        "trait_raw": "trait_expression_score_raw" + suffix,
        "coherence": "coherence_score" + suffix,
        "coherence_raw": "coherence_score_raw" + suffix,
    }


def _needs_coherence(
    item: Dict[str, Any],
    scorer,
    keys: Dict[str, str],
    force: bool,
    evaluate_trait: bool,
) -> bool:
    if force:
        return True
    if evaluate_trait:
        # Coherence is only meaningful next to a trait score, and scoring the
        # coherence of a response whose trait judgement failed to parse wastes
        # judge time.
        if scorer.trait_key not in item:
            return False
        base = item[scorer.trait_key]
        if isinstance(base, dict) and base.get("parse_error"):
            return False
    return keys["coherence"] not in item


def collect(
    results: Any,
    *,
    phase: str,
    scorer,
    coherence_prompt: Optional[str],
    keys: Dict[str, str],
    force: bool,
    evaluate_trait: bool,
    target_layers: Optional[List[str]],
    target_coefs: Optional[List[str]],
) -> Tuple[List[Conversation], List[Dict[str, Any]]]:
    """Build the judge conversations for one phase.

    Returns the conversations and the answer items they belong to, positionally
    aligned. Holding the item dicts directly means results are written straight
    back to the object they came from -- no index bookkeeping to get wrong.
    """
    conversations: List[Conversation] = []
    items: List[Dict[str, Any]] = []

    for answers in iter_answer_lists(results, target_layers, target_coefs):
        for item in answers:
            question = item.get("question", "")
            answer = item.get("steered_answer", "")

            if phase == "trait":
                if not force and keys["trait"] in item:
                    continue
                messages = scorer.build_messages(question, answer)
                if messages is None:
                    continue
            else:
                if not _needs_coherence(item, scorer, keys, force, evaluate_trait):
                    continue
                messages = build_qa_messages(coherence_prompt, question, answer)

            conversations.append(messages)
            items.append(item)

    return conversations, items


# ---------------------------------------------------------------------------
# Phase execution
# ---------------------------------------------------------------------------

def save_results(results: Any, results_path: str) -> None:
    """Write via a temp file so a requeue mid-write cannot truncate the results."""
    tmp_path = f"{results_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, results_path)


def _store_coherence(item: Dict[str, Any], keys: Dict[str, str], raw: str, omit_raw: bool) -> bool:
    score = parse_int_score(raw)
    if not omit_raw:
        item[keys["coherence_raw"]] = raw
    item[keys["coherence"]] = score
    return score is not None


def run_phase(
    label: str,
    results: Any,
    results_path: str,
    conversations: Sequence[Conversation],
    items: Sequence[Dict[str, Any]],
    store,
    judge: VLLMJudge,
    checkpoint_interval: int,
) -> Tuple[int, int]:
    """Judge one phase's conversations and write the scores back.

    Chunked at ``checkpoint_interval`` so long runs save progress; each chunk is
    still submitted to vLLM as a single batch.
    """
    print(f"[collect] {label}: {len(conversations)} prompts to evaluate")
    if not conversations:
        return 0, 0

    success = errors = 0
    step = checkpoint_interval if checkpoint_interval > 0 else len(conversations)

    for start in range(0, len(conversations), step):
        end = min(start + step, len(conversations))
        print(f"[eval] {label} {start + 1}-{end}/{len(conversations)}...")

        raw_outputs = judge.generate(conversations[start:end], batch_size=None)
        chunk_items = items[start:end]
        if len(raw_outputs) != len(chunk_items):
            raise RuntimeError(
                f"{label}: judge returned {len(raw_outputs)} outputs for "
                f"{len(chunk_items)} prompts; refusing to misalign scores."
            )

        for item, raw in zip(chunk_items, raw_outputs):
            if store(item, raw):
                success += 1
            else:
                errors += 1

        print(f"[save] {label}: checkpoint after {end} evaluations")
        save_results(results, results_path)

    print(f"[{label}] completed: {success} success, {errors} errors")
    return success, errors


def evaluate_file(
    results_path: str,
    *,
    scorer,
    coherence_prompt: Optional[str],
    judge: VLLMJudge,
    args: argparse.Namespace,
) -> None:
    print(f"[load] Loading results from {results_path}")
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    sample_runs = (
        range(2, args.n_samples + 2) if args.sampling_mode else [None]
    )

    for sample_run in sample_runs:
        if sample_run is not None:
            print(f"\n[sampling] ===== Sample run {sample_run}/{args.n_samples + 1} =====")

        keys = score_keys(scorer, sample_run)

        if args.skip_trait:
            print("[skip] trait phase")
        else:
            conversations, items = collect(
                results,
                phase="trait",
                scorer=scorer,
                coherence_prompt=None,
                keys=keys,
                force=args.force_recompute,
                evaluate_trait=True,
                target_layers=args.target_layers,
                target_coefs=args.target_coefs,
            )
            run_phase(
                "trait", results, results_path, conversations, items,
                lambda item, raw: scorer.store(item, keys, raw),
                judge, args.checkpoint_interval,
            )
            if getattr(scorer, "unmatched", 0):
                print(
                    f"[warn] trait: skipped {scorer.unmatched} responses with no "
                    f"matching ground-truth scenario"
                )
                scorer.unmatched = 0

        if args.skip_coherence:
            print("[skip] coherence phase")
        else:
            conversations, items = collect(
                results,
                phase="coherence",
                scorer=scorer,
                coherence_prompt=coherence_prompt,
                keys=keys,
                force=args.force_recompute,
                evaluate_trait=not args.skip_trait,
                target_layers=args.target_layers,
                target_coefs=args.target_coefs,
            )
            run_phase(
                "coherence", results, results_path, conversations, items,
                lambda item, raw: _store_coherence(item, keys, raw, scorer.omit_raw),
                judge, args.checkpoint_interval,
            )

    print(f"[done] {results_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_list_arg(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the LLM judge over single-turn steering sweep results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trait", required=True, choices=["honesty", "compassion"])

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--results_dir", help="Judge every *.json under this directory")
    target.add_argument("--single_file", help="Judge exactly this file")

    parser.add_argument("--checkpoint_interval", type=int, default=None,
                        help="Save every N evaluations (0 = only at the end). "
                             "Default: 4000 honesty / 5000 compassion.")
    parser.add_argument("--force_recompute", action="store_true",
                        help="Re-judge items that already have scores")
    parser.add_argument("--skip_trait", action="store_true", help="Skip the trait phase")
    parser.add_argument("--skip_coherence", action="store_true", help="Skip the coherence phase")
    parser.add_argument("--sampling_mode", action="store_true",
                        help="Judge N extra times into suffixed keys, to measure judge variance")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Extra judge runs under --sampling_mode")
    parser.add_argument("--target_layers", default=None, help="Comma-separated layers, e.g. 23,26")
    parser.add_argument("--target_coefs", default=None, help="Comma-separated coefs, e.g. coef_3")

    honesty = parser.add_argument_group("honesty only")
    honesty.add_argument("--honesty_category", default="self_report_experimental")
    honesty.add_argument("--model_save_name", default="meta-llama_Llama-3.3-70B-Instruct",
                         help="Selects the common_steering_config_* section holding data_path")
    honesty.add_argument("--include_raw_output", action="store_true",
                         help="Also store the judge's raw text inside judge_score")

    compassion = parser.add_argument_group("compassion only")
    compassion.add_argument("--sys_prompt_type", default="dismissive_system_prompt",
                            choices=["dismissive_system_prompt", "compassionate_system_prompt"],
                            help="Which steering direction the results came from; selects the rubric")
    compassion.add_argument("--no_raw_score", action="store_true",
                            help="Do not store raw judge text alongside the scores")

    return parser


_TRAIT_ONLY_FLAGS = {
    "honesty": [("include_raw_output", False), ("honesty_category", "self_report_experimental"),
                ("model_save_name", "meta-llama_Llama-3.3-70B-Instruct")],
    "compassion": [("sys_prompt_type", "dismissive_system_prompt"), ("no_raw_score", False)],
}


def _reject_foreign_flags(args: argparse.Namespace) -> None:
    """Fail fast on a flag that belongs to the other trait, rather than ignoring it."""
    other = "compassion" if args.trait == "honesty" else "honesty"
    for name, default in _TRAIT_ONLY_FLAGS[other]:
        if getattr(args, name) != default:
            raise ValueError(
                f"--{name} is only valid with --trait {other}, but --trait {args.trait} was given."
            )


def find_result_files(args: argparse.Namespace) -> List[str]:
    if args.single_file:
        if not os.path.isfile(args.single_file):
            raise FileNotFoundError(f"Single file not found: {args.single_file}")
        return [args.single_file]

    if not os.path.isdir(args.results_dir):
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    files = []
    for root, _, names in os.walk(args.results_dir):
        for name in names:
            if name.endswith(".json") and not name.endswith("_metrics.json"):
                files.append(os.path.join(root, name))

    if not files:
        print(f"[warning] No result JSON files found in: {args.results_dir}")
    return files


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    _reject_foreign_flags(args)

    if args.sampling_mode and args.n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {args.n_samples}")

    args.target_layers = _parse_list_arg(args.target_layers)
    args.target_coefs = _parse_list_arg(args.target_coefs)

    scorer = build_scorer(args)
    if args.checkpoint_interval is None:
        args.checkpoint_interval = scorer.default_checkpoint_interval

    coherence_prompt = None if args.skip_coherence else load_prompt("coherence")

    json_files = find_result_files(args)
    print(f"[find] {len(json_files)} result files to process")
    print(f"[config] {vars(args)}")

    print("[init] Loading judge...")
    judge = VLLMJudge(JudgeConfig.load())

    for idx, path in enumerate(json_files, 1):
        print(f"\n{'=' * 60}\n[{idx}/{len(json_files)}] {path}\n{'=' * 60}")
        try:
            evaluate_file(
                path, scorer=scorer, coherence_prompt=coherence_prompt,
                judge=judge, args=args,
            )
        except Exception as exc:  # one bad file must not kill a multi-hour job
            print(f"[error] Failed to process {path}: {exc}")
            import traceback
            traceback.print_exc()

    print(f"\n[complete] Processed {len(json_files)} files")


if __name__ == "__main__":
    main()
