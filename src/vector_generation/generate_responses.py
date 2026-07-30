"""
Generate model responses for CAA vector-extraction training data.

Supports both traits via per-trait adapters:
- honesty: scenario dataset under data/honesty/CAA/{dataset}/ — each scenario
  carries its own system prompt pair (system_prompt_honest / system_prompt_dishonest).
- compassion: a flat user-question list combined with a pool of 5 compassionate
  + 5 dismissive system prompts from YAML.

The output JSON schemas are unchanged from the original per-trait scripts, so
the downstream LLM-judge annotation pass (judge_caa_* for honesty, judge_score
for compassion) and the vector extractor keep working untouched.

Usage:
    python -m src.vector_generation.generate_responses --trait honesty \
        --model-id meta-llama/Llama-3.3-70B-Instruct --n-responses 5
    python -m src.vector_generation.generate_responses --trait compassion \
        --model-id meta-llama/Llama-3.3-70B-Instruct --n-responses 2
"""

import argparse
import json
import os
from typing import Any, Dict, List

from src.constants import constants
from src.constants.constants import DATA_FOLDER
from src.steering.steering_sampling import sample
from src.utils.model_utils import load_model, set_seed
from src.vector_generation.extraction_utils import (
    ensure_padding,
    load_compassion_system_prompts,
    model_slug,
)

HONESTY_DATASET = "self_report_experimental"


def generate_n_responses(
    model, tokenizer, conversations, n_responses: int,
    max_tokens: int, temperature: float, batch_size: int,
) -> List[List[str]]:
    """
    Generate n_responses responses for each conversation.

    Returns:
        per_convo[i] = [resp1, resp2, ...]  (length n_responses)
    """
    all_runs = []
    for i in range(n_responses):
        print(f"  Run {i + 1}/{n_responses}")
        out = sample(
            model,
            tokenizer,
            conversations,
            max_tokens=max_tokens,
            temperature=temperature,
            bs=batch_size,
            enable_thinking=False,
            prompts=None,
        )
        all_runs.append(out)

    per_convo: List[List[str]] = []
    for convo_idx in range(len(conversations)):
        per_convo.append([all_runs[r][convo_idx] for r in range(n_responses)])
    return per_convo


# =====================================================================
# HONESTY (scenario dataset, per-scenario system prompt pair)
# =====================================================================

def run_honesty(args, model, tokenizer) -> None:
    slug = model_slug(args.model_id)
    input_json = args.input_json or os.path.join(
        DATA_FOLDER, "honesty", "CAA", HONESTY_DATASET,
        f"{HONESTY_DATASET}_scenarios.json",
    )
    output_json = args.output_json or os.path.join(
        DATA_FOLDER, "honesty", "CAA", HONESTY_DATASET,
        f"{HONESTY_DATASET}_responses_{slug}.json",
    )

    print(f"Loading scenarios from: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        scenarios = json.load(f)["scenarios"]
    print(f"Loaded {len(scenarios)} scenarios")

    # One conversation per (scenario, condition); each scenario has its own
    # unique system prompt pair, used as-is.
    honest_convos: List[List[Dict[str, str]]] = []
    dishonest_convos: List[List[Dict[str, str]]] = []
    for scenario in scenarios:
        user_message = scenario["user_message"]
        honest_convos.append([
            {"role": "system", "content": scenario["system_prompt_honest"]},
            {"role": "user", "content": user_message},
        ])
        dishonest_convos.append([
            {"role": "system", "content": scenario["system_prompt_dishonest"]},
            {"role": "user", "content": user_message},
        ])
    print(f"Built {len(honest_convos)} honest + {len(dishonest_convos)} dishonest conversations")

    print(f"\nGenerating {args.n_responses} honest responses per scenario...")
    honest_per_convo = generate_n_responses(
        model, tokenizer, honest_convos, args.n_responses,
        args.max_tokens, args.temperature, args.batch_size,
    )
    print(f"\nGenerating {args.n_responses} dishonest responses per scenario...")
    dishonest_per_convo = generate_n_responses(
        model, tokenizer, dishonest_convos, args.n_responses,
        args.max_tokens, args.temperature, args.batch_size,
    )

    # Each scenario gets a single honest and single dishonest entry. Copy all
    # scenario fields except the system prompts (which go into responses).
    _skip_keys = {"system_prompt_honest", "system_prompt_dishonest"}
    output_data: List[Dict[str, Any]] = []
    for sc_idx, scenario in enumerate(scenarios):
        item = {k: v for k, v in scenario.items() if k not in _skip_keys}
        item["responses"] = {
            "honest": {
                "system_prompt": scenario["system_prompt_honest"],
                "generations": [{"response": r} for r in honest_per_convo[sc_idx]],
            },
            "dishonest": {
                "system_prompt": scenario["system_prompt_dishonest"],
                "generations": [{"response": r} for r in dishonest_per_convo[sc_idx]],
            },
        }
        output_data.append(item)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(output_data)} scenarios with generated responses to: {output_json}")


# =====================================================================
# COMPASSION (question list x 5+5 system-prompt pool)
# =====================================================================

def run_compassion(args, model, tokenizer) -> None:
    slug = model_slug(args.model_id)
    input_json = args.input_json or os.path.join(
        DATA_FOLDER, "compassion", "CAA", "compassion_train_user_questions.json",
    )
    output_json = args.output_json or os.path.join(
        DATA_FOLDER, "compassion", "CAA",
        f"conversation_data_compassion_generated_{slug}.json",
    )
    prompts_yaml = os.path.join(
        DATA_FOLDER, "compassion", "CAA", "compassion_system_prompts_train.yaml",
    )

    print(f"Loading user questions from: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        user_questions = json.load(f)["user_questions_train"]
    print(f"Loaded {len(user_questions)} questions")

    prompts = load_compassion_system_prompts(prompts_yaml)
    pos_prompts = list(prompts["pos"].items())  # [(id, prompt)] in YAML order
    neg_prompts = list(prompts["neg"].items())

    # One conversation per (question, system prompt) on each side.
    pos_convos: List[List[Dict[str, str]]] = []
    neg_convos: List[List[Dict[str, str]]] = []
    for user_question in user_questions:
        for _pid, prompt in pos_prompts:
            pos_convos.append([
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_question},
            ])
        for _pid, prompt in neg_prompts:
            neg_convos.append([
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_question},
            ])

    print(f"\nGenerating {args.n_responses} compassionate responses per (question, system prompt)...")
    pos_per_convo = generate_n_responses(
        model, tokenizer, pos_convos, args.n_responses,
        args.max_tokens, args.temperature, args.batch_size,
    )
    print(f"\nGenerating {args.n_responses} dismissive responses per (question, system prompt)...")
    neg_per_convo = generate_n_responses(
        model, tokenizer, neg_convos, args.n_responses,
        args.max_tokens, args.temperature, args.batch_size,
    )

    # Bucketed output schema: one bucket per system prompt id, in YAML order.
    output_data: List[Dict[str, Any]] = []
    for idx, user_question in enumerate(user_questions):
        output_data.append({
            "id": idx + 1,
            "user_message": user_question,
            "responses": {
                "compassion": [
                    {"system_prompt_id": pid, "generations": []} for pid, _ in pos_prompts
                ],
                "dismissive": [
                    {"system_prompt_id": pid, "generations": []} for pid, _ in neg_prompts
                ],
            },
        })

    n_pos = len(pos_prompts)
    n_neg = len(neg_prompts)
    for convo_idx, responses in enumerate(pos_per_convo):
        ex_idx, bucket_idx = divmod(convo_idx, n_pos)
        output_data[ex_idx]["responses"]["compassion"][bucket_idx]["generations"] = [
            {"response": r} for r in responses
        ]
    for convo_idx, responses in enumerate(neg_per_convo):
        ex_idx, bucket_idx = divmod(convo_idx, n_neg)
        output_data[ex_idx]["responses"]["dismissive"][bucket_idx]["generations"] = [
            {"response": r} for r in responses
        ]

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved generated responses to: {output_json}")
    print(f"System prompts loaded from: {prompts_yaml}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CAA training responses (pos/neg) for a trait."
    )
    parser.add_argument(
        "--trait", required=True, choices=["honesty", "compassion"],
        help="Which trait to generate responses for",
    )
    parser.add_argument(
        "--model-id", "--model_id", dest="model_id", required=True,
        help="HuggingFace model ID (e.g. meta-llama/Llama-3.3-70B-Instruct)",
    )
    parser.add_argument(
        "--n-responses", type=int, required=True,
        help="Number of responses per conversation (paper runs: honesty 5, compassion 2)",
    )
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for generation")
    parser.add_argument(
        "--input-json", default=None,
        help="Path to input JSON (default: derived from --trait)",
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Path for output JSON with generated responses (default: derived from --trait and --model-id)",
    )
    args = parser.parse_args()

    set_seed(constants.SEED)

    print(f"\nLoading model: {args.model_id}")
    model, tokenizer = load_model(model_id_or_path=args.model_id)
    ensure_padding(tokenizer)
    print("Model loaded\n")

    if args.trait == "honesty":
        run_honesty(args, model, tokenizer)
    else:
        run_compassion(args, model, tokenizer)

    print(f"  Model: {args.model_id}")
    print(f"  Responses per conversation: {args.n_responses}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")


if __name__ == "__main__":
    main()
