"""LLM judge annotation of CAA training-response datasets.

Steering vectors are fit on contrastive pairs of generated responses. This runner
scores each generation against the role it was generated for and writes the
verdict back into the responses JSON in place, so vector extraction can keep only
the generations that actually expressed their assigned role
(see ``src/vector_generation/generate_vec_caa.py``).

Annotation is idempotent and saves every N items, so a requeued job resumes.

    python -m src.llm_judge.runners.caa_dataset --trait honesty --model-id MODEL
    python -m src.llm_judge.runners.caa_dataset --trait compassion --model-id MODEL
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Tuple

from src.constants.constants import ROOT_FOLDER
from src.llm_judge.config import JudgeConfig
from src.llm_judge.engine import VLLMJudge
from src.llm_judge.parsing import parse_int_score, parse_judge_json
from src.llm_judge.prompts import build_caa_messages, build_qa_messages, load_trait_prompt, load_prompt


def iter_role_generations(
    item: Dict[str, Any],
    roles: Tuple[str, ...],
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield ``(role, generation)`` for one dataset item.

    ``responses[role]`` is a block dict holding ``generations`` (honesty) or a
    list of such blocks (compassion). Normalizing the dict case to a one-element
    list here is what keeps the rest of the module free of shape branches.
    """
    responses = item.get("responses") or {}
    for role in roles:
        blocks = responses.get(role) or []
        if isinstance(blocks, dict):
            blocks = [blocks]
        for block in blocks:
            for generation in block.get("generations") or []:
                yield role, generation


# ---------------------------------------------------------------------------
# Honesty
# ---------------------------------------------------------------------------

_EXPECTED_VERDICT = {"honest": "DISCLOSE", "dishonest": "SUPPRESS"}


class HonestyCAA:
    """DISCLOSE/SUPPRESS verdict + 0-100 score + role-match keep filter."""

    roles = ("honest", "dishonest")
    input_template = "data/honesty/CAA/{dataset}/honesty_{dataset}_responses_{model}.json"

    def __init__(self, score_threshold: int):
        self.score_threshold = score_threshold
        # Both roles share one rubric; the role is carried in its CONDITION field.
        self._prompt = load_prompt("caa_diverse_honesty")

    def system_prompt(self, role: str) -> str:
        return self._prompt

    @staticmethod
    def already_judged(generation: Dict[str, Any]) -> bool:
        return all(
            key in generation
            for key in ("judge_caa", "judge_caa_keep", "judge_caa_role_ok")
        )

    def build_messages(self, item, generation, role):
        return build_caa_messages(
            self.system_prompt(role),
            gold_truth=item["gold_truth"],
            user_message=item["user_message"],
            condition=role.upper(),
            model_response=generation.get("response", ""),
            category=item.get("category", ""),
        )

    def attach(self, generation: Dict[str, Any], raw: str, role: str) -> None:
        parsed = parse_judge_json(raw, require_role_match=True)
        expected = _EXPECTED_VERDICT[role]

        generation["judge_caa_raw"] = raw
        generation["judge_caa"] = parsed
        generation["judge_caa_honesty_score"] = parsed.get("honesty_score") if parsed else None
        generation["judge_caa_role_ok"] = bool(parsed) and parsed.get("verdict") == expected
        generation["judge_caa_keep"] = self._keep(parsed, expected)

    def _keep(self, parsed, expected: str) -> bool:
        """Keep a generation for vector training only if it played its role well."""
        if not parsed or parsed.get("verdict") != expected:
            return False
        score = parsed.get("honesty_score")
        if score is None:
            return False
        if expected == "DISCLOSE":
            return score >= self.score_threshold
        return score <= (100 - self.score_threshold)


# ---------------------------------------------------------------------------
# Compassion
# ---------------------------------------------------------------------------

class CompassionCAA:
    """0-100 trait expression, scored with the rubric matching each role."""

    roles = ("compassion", "dismissive")
    input_template = "data/compassion/CAA/conversation_data_compassion_generated_{model}.json"

    def __init__(self, save_raw_eval: bool):
        self.save_raw_eval = save_raw_eval
        # NOTE: the compassion role is deliberately scored with the *multi-turn*
        # rubric. That predates this refactor and is baked into every existing
        # responses file (and therefore into the current steering vectors), so it
        # is preserved verbatim rather than "fixed" to the single-turn rubric.
        self._prompts = {
            "compassion": load_trait_prompt("compassion_multi_turn", "compassion"),
            "dismissive": load_trait_prompt("dismissiveness", "dismissiveness"),
        }

    def system_prompt(self, role: str) -> str:
        return self._prompts[role]

    @staticmethod
    def already_judged(generation: Dict[str, Any]) -> bool:
        return generation.get("judge_score") is not None

    def build_messages(self, item, generation, role):
        return build_qa_messages(
            self.system_prompt(role),
            str(item.get("user_message", "")).strip(),
            str(generation.get("response", "")).strip(),
        )

    def attach(self, generation: Dict[str, Any], raw: str, role: str) -> None:
        if self.save_raw_eval:
            generation["judge_raw"] = raw
        generation["judge_score"] = parse_int_score(raw)


# ---------------------------------------------------------------------------
# Shared driver
# ---------------------------------------------------------------------------

def save_json(path: str, obj: Any) -> None:
    """Write via a temp file so a requeue mid-write cannot truncate the dataset."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def judge_dataset(
    dataset: List[Dict[str, Any]],
    input_json: str,
    spec,
    judge: VLLMJudge,
    *,
    batch_size: int,
    save_every_items: int,
    force: bool,
) -> None:
    """Annotate every unjudged generation, one dataset item at a time."""
    for item_idx, item in enumerate(dataset):
        pending = [
            (role, generation)
            for role, generation in iter_role_generations(item, spec.roles)
            if force or not spec.already_judged(generation)
        ]

        # Group by rubric so each distinct system prompt is one batched call.
        by_prompt = defaultdict(list)
        for role, generation in pending:
            by_prompt[spec.system_prompt(role)].append((role, generation))

        for refs in by_prompt.values():
            conversations = [
                spec.build_messages(item, generation, role) for role, generation in refs
            ]
            raw_outputs = judge.generate(conversations, batch_size=batch_size)
            if len(raw_outputs) != len(refs):
                raise RuntimeError(
                    f"Judge returned {len(raw_outputs)} outputs for {len(refs)} "
                    f"generations; refusing to misalign scores."
                )
            for (role, generation), raw in zip(refs, raw_outputs):
                spec.attach(generation, raw, role)

        if save_every_items > 0 and (item_idx + 1) % save_every_items == 0:
            save_json(input_json, dataset)
            print(f"[save] progress: judged up to item {item_idx + 1} -> {input_json}")

    save_json(input_json, dataset)
    print(f"[save] annotated dataset -> {input_json}")


def print_summary(dataset: List[Dict[str, Any]], spec) -> None:
    """Per-role counts and mean score. Informational only -- nothing reads this."""
    print("\n=== judge summary ===")
    for role in spec.roles:
        scores, n, kept = [], 0, 0
        for item in dataset:
            for gen_role, generation in iter_role_generations(item, (role,)):
                n += 1
                if isinstance(spec, HonestyCAA):
                    parsed = generation.get("judge_caa")
                    if parsed and parsed.get("honesty_score") is not None:
                        scores.append(parsed["honesty_score"])
                    kept += bool(generation.get("judge_caa_keep"))
                else:
                    score = generation.get("judge_score")
                    if isinstance(score, int):
                        scores.append(score)
                        kept += 1
        mean = round(sum(scores) / len(scores), 2) if scores else 0.0
        print(f"  {role:11s} n={n:5d}  scored={len(scores):5d}  mean={mean:6.2f}  keep={kept}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate a CAA training-response dataset with LLM judge scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trait", required=True, choices=["honesty", "compassion"])
    parser.add_argument("--model-id", "--model_id", dest="model_id",
                        default="meta-llama/Llama-3.3-70B-Instruct",
                        help="Model whose responses are being judged; selects the input file")
    parser.add_argument("--input-json", dest="input_json", default=None,
                        help="Responses JSON to annotate; derived from --trait/--model-id if omitted")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=200)
    parser.add_argument("--save-every-items", dest="save_every_items", type=int, default=200)
    parser.add_argument("--force", action="store_true",
                        help="Re-judge generations that already have scores")

    honesty = parser.add_argument_group("honesty only")
    honesty.add_argument("--dataset", default="self_report_experimental",
                         help="Subdirectory under data/honesty/CAA/")
    honesty.add_argument("--score-threshold", dest="score_threshold", type=int, default=65,
                         help="Minimum score for a generation to be kept for vector training")

    compassion = parser.add_argument_group("compassion only")
    compassion.add_argument("--save-raw-eval", dest="save_raw_eval", action="store_true",
                            help="Also store the judge's raw text as judge_raw")

    return parser


_TRAIT_ONLY_FLAGS = {
    "honesty": [("dataset", "self_report_experimental"), ("score_threshold", 65)],
    "compassion": [("save_raw_eval", False)],
}


def _reject_foreign_flags(args: argparse.Namespace) -> None:
    other = "compassion" if args.trait == "honesty" else "honesty"
    for name, default in _TRAIT_ONLY_FLAGS[other]:
        if getattr(args, name) != default:
            raise ValueError(
                f"--{name.replace('_', '-')} is only valid with --trait {other}, "
                f"but --trait {args.trait} was given."
            )


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    _reject_foreign_flags(args)

    if args.trait == "honesty":
        spec = HonestyCAA(score_threshold=args.score_threshold)
    else:
        spec = CompassionCAA(save_raw_eval=args.save_raw_eval)

    model_save_name = args.model_id.replace("/", "_")
    input_json = args.input_json or spec.input_template.format(
        dataset=getattr(args, "dataset", None), model=model_save_name
    )
    input_json = os.path.join(ROOT_FOLDER, input_json)

    with open(input_json, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError(f"CAA dataset must be a list of items: {input_json}")

    print(f"[load] {len(dataset)} items from {input_json}")
    print("[init] Loading judge...")
    judge = VLLMJudge(JudgeConfig.load())

    judge_dataset(
        dataset, input_json, spec, judge,
        batch_size=args.batch_size,
        save_every_items=args.save_every_items,
        force=args.force,
    )
    print_summary(dataset, spec)


if __name__ == "__main__":
    main()
