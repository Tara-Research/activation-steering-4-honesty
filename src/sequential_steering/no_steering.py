import json
from typing import Any, Dict, List, Set

from src.steering.steering_sampling import sample


def _rows_from_answers(conversations, answers, has_sys_prompt: bool, scenario_meta=None):
    rows = []
    meta = scenario_meta or [{}] * len(conversations)
    for conv, ans, m in zip(conversations, answers, meta):
        q = conv[1]["content"] if has_sys_prompt else conv[0]["content"]
        rows.append({
            "scenario_id": m.get("scenario_id"),
            "category": m.get("category"),
            "question": q,
            "steered_answer": ans,
        })
    return rows


def run_all_layers_no_steering_sequential(
    *,
    model,
    tokenizer,
    conversations,
    steering_vectors,
    neg_embeddings,
    pos_embeddings,
    common_steering_config: Dict[str, Any],
    specific_steering_config: Dict[str, Any],
    all_layers: List[int],
    done_keys: Set[str],
    experiment_results: Dict[str, Any],
    output_json: str,
    bias=None,        # unused; kept for uniform interface with wrapper
    scenario_meta=None,
) -> None:
    has_sys_prompt = bool(specific_steering_config["sys_prompt"])
    enable_thinking = bool(specific_steering_config["enable_thinking"])

    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for messages in conversations
    ]

    answers = sample(
        model=model,
        tokenizer=tokenizer,
        conversations=conversations,
        prompts=prompts,
        bs=common_steering_config["batch_size"],
        top_p=common_steering_config["top_p"],
        max_tokens=common_steering_config["max_tokens"],
        temperature=common_steering_config["temperature"],
        min_tokens=common_steering_config["min_tokens"],
        enable_thinking=enable_thinking,
    )

    results = _rows_from_answers(conversations, answers, has_sys_prompt, scenario_meta)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
