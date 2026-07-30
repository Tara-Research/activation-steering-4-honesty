"""Neutral generation on training scenarios (no steering). Saves responses to JSON.

Generates the model's response to each training scenario and writes them to a JSON
(no plotting / no activation recording — that is a separate downstream step). Two modes:

- default: user prompt ONLY — the scenario's ``system_prompt_honest`` /
  ``system_prompt_dishonest`` are dropped, giving the model's fully unconditioned
  ("natural") response → ``neutral_answers.json``.
- ``--neutral-system-prompt``: prepend each scenario's ``system_prompt_neutral`` (the
  directive-stripped setup) as the system turn, giving the response under the neutral
  scenario context → ``neutral_answers_neutral_sysprompt.json``.

Either output can later be fed to the activation-recording / projection-distribution
analysis to characterize where the responses sit relative to the steering threshold.

Run from the repo root, e.g.::

    python -m src.sequential_steering.neutral_generation \
        --model-id meta-llama/Llama-3.3-70B-Instruct \
        --trait honesty --honesty-category self_report_experimental \
        --data-path data/honesty/CAA/self_report_experimental/self_report_experimental_scenarios.json \
        --neutral-system-prompt
"""

import argparse
import json
import os
from typing import List, Optional, Tuple

import yaml

from src.constants import constants
from src.constants.constants import (
    RESULTS_FOLDER,
    ROOT_FOLDER,
    VECTOR_TYPE,
    get_config_for_trait,
)
from src.steering.steering_sampling import sample
from src.utils.model_utils import load_model, set_seed

# Honesty self-report training scenarios — the dataset used to fit the steering
# vectors. Each scenario carries a `user_message`; system prompts are ignored here.
DEFAULT_DATA_PATH = "data/honesty/CAA/self_report_experimental/self_report_experimental_scenarios.json"


def load_neutral_conversations(
    data_path: str,
    use_neutral_system_prompt: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """Build conversations + parallel scenario metadata from a scenarios file.

    Handles both the ``{"scenarios": [...]}`` wrapper and a bare list. Each scenario
    must carry a ``user_message`` (no silent fallback).

    By default each conversation is user-only (no system prompt) — the model's fully
    unconditioned response. With ``use_neutral_system_prompt`` each scenario's
    ``system_prompt_neutral`` (the directive-stripped setup) is prepended as a system
    turn, so the response is generated under the neutral scenario context; a scenario
    missing a non-empty ``system_prompt_neutral`` raises (no silent fallback).

    Returns ``(conversations, scenario_meta)`` as parallel lists; each ``scenario_meta``
    entry carries ``scenario_id``, ``category``, ``user_message``, ``gold_truth``, and
    the ``system_prompt`` used (``None`` when no system prompt).
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scenarios = data["scenarios"] if isinstance(data, dict) and "scenarios" in data else data

    conversations: List[dict] = []
    scenario_meta: List[dict] = []
    for s in scenarios:
        if "user_message" not in s:
            raise KeyError(f"Scenario missing required 'user_message' field: keys={sorted(s)}")
        user_message = s["user_message"]

        system_prompt: Optional[str] = None
        if use_neutral_system_prompt:
            system_prompt = s.get("system_prompt_neutral")
            if not system_prompt or not system_prompt.strip():
                raise ValueError(
                    f"Scenario {s.get('scenario_id') or s.get('id')!r} has no non-empty "
                    f"'system_prompt_neutral'; required with --neutral-system-prompt."
                )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        else:
            messages = [{"role": "user", "content": user_message}]
        conversations.append(messages)

        scenario_id = s.get("scenario_id")
        if scenario_id is None and "id" in s:
            scenario_id = str(s["id"])
        scenario_meta.append({
            "scenario_id": scenario_id,
            "category": s.get("category"),
            "user_message": user_message,
            "gold_truth": s.get("gold_truth"),
            "system_prompt": system_prompt,
        })
    return conversations, scenario_meta


def main():
    parser = argparse.ArgumentParser(
        description="Generate neutral (no system prompt, no steering) responses on training scenarios."
    )
    parser.add_argument("--model-id", default="meta-llama/Llama-3.3-70B-Instruct",
                        help="HuggingFace model ID.")
    parser.add_argument("--trait", default="honesty",
                        help="Trait whose config supplies generation defaults (honesty/compassion).")
    parser.add_argument("--honesty-category", default="self_report_experimental",
                        help="Honesty category; used only for the output subdirectory.")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                        help="Scenarios JSON (absolute, or relative to the repo root).")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override max generated tokens (default: config generation_defaults).")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Cap scenarios processed (quick smoke test).")
    parser.add_argument("--exp-name", default=None,
                        help="Optional subdirectory to keep runs separate.")
    parser.add_argument("--neutral-system-prompt", action="store_true", default=False,
                        help="Prepend each scenario's 'system_prompt_neutral' (the directive-stripped "
                             "setup) as the system turn, generating responses under the neutral scenario "
                             "context. Default: no system prompt (the model's fully unconditioned response). "
                             "Writes a separate 'neutral_answers_neutral_sysprompt.json' so it coexists with "
                             "the no-system-prompt output.")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if the output file already exists.")
    args = parser.parse_args()

    data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(ROOT_FOLDER, args.data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Scenarios file not found: {data_path}")

    config_file = get_config_for_trait(args.trait)
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gen = cfg["generation_defaults"]
    max_tokens = args.max_tokens if args.max_tokens is not None else gen["max_tokens"]
    enable_thinking = gen["enable_thinking"]

    conversations, scenario_meta = load_neutral_conversations(
        data_path, use_neutral_system_prompt=args.neutral_system_prompt
    )
    if args.max_prompts is not None:
        conversations = conversations[:args.max_prompts]
        scenario_meta = scenario_meta[:args.max_prompts]
    print(f"Loaded {len(conversations)} scenarios from {data_path}")

    model_save_name = args.model_id.replace("/", "_")
    out_dir = os.path.join(RESULTS_FOLDER, VECTOR_TYPE, args.trait, model_save_name, "neutral_generation")
    if args.trait == "honesty":
        out_dir = os.path.join(out_dir, args.honesty_category)
    if args.exp_name:
        out_dir = os.path.join(out_dir, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    out_name = "neutral_answers_neutral_sysprompt.json" if args.neutral_system_prompt else "neutral_answers.json"
    out_path = os.path.join(out_dir, out_name)

    if os.path.exists(out_path) and not args.force:
        print(f"Output already exists (use --force to regenerate): {out_path}")
        return

    model, tokenizer = load_model(args.model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    set_seed(constants.SEED)

    prompts = [
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
        )
        for messages in conversations
    ]

    answers = sample(
        model=model,
        tokenizer=tokenizer,
        conversations=conversations,
        prompts=prompts,
        bs=gen["batch_size"],
        top_p=gen["top_p"],
        max_tokens=max_tokens,
        temperature=gen["temperature"],
        min_tokens=gen["min_tokens"],
        enable_thinking=enable_thinking,
    )

    items = []
    for m, ans in zip(scenario_meta, answers):
        # 'instruction' holds the user message — the field name the projection/activation
        # recorder (record_scores) reads, kept for downstream compatibility.
        item = {
            "scenario_id": m["scenario_id"],
            "category": m["category"],
            "instruction": m["user_message"],
            "output": ans,
        }
        if m["system_prompt"] is not None:
            item["system_prompt"] = m["system_prompt"]
        if m["gold_truth"] is not None:
            item["gold_truth"] = m["gold_truth"]
        items.append(item)
    result = {
        "metadata": {
            "model_id": args.model_id,
            "trait": args.trait,
            "honesty_category": args.honesty_category if args.trait == "honesty" else None,
            "data_path": os.path.relpath(data_path, ROOT_FOLDER),
            "mode": "neutral_system_prompt" if args.neutral_system_prompt else "neutral_no_system_prompt",
            "vector_type": VECTOR_TYPE,
            "n_scenarios": len(items),
            "generation_defaults": {
                "temperature": gen["temperature"],
                "top_p": gen["top_p"],
                "min_tokens": gen["min_tokens"],
                "max_tokens": max_tokens,
                "batch_size": gen["batch_size"],
                "enable_thinking": enable_thinking,
            },
        },
        "items": items,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(items)} neutral answers to {out_path}")


if __name__ == "__main__":
    main()
