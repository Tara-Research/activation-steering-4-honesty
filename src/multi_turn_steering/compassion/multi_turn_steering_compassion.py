import contextlib
import json
import os
from typing import List, Dict, Optional, Tuple

import torch
import yaml

from src.constants import constants
from src.constants.constants import DATA_FOLDER, PERSONA_VECTORS_FOLDER, VECTOR_TYPE_LABEL, get_config_for_trait
from src.multi_turn_steering.generate_answer import (
    generate_steered_answer,
    generate_plain,
    generate_plain_batch,
    generate_steered_batch,
)
from src.steering.sttp import SteeringToTargetProjection
from src.steering.swfc import SteeringWithFixedCoef
from src.steering.stmp import SteeringToMirrorProjection
from src.utils.exp_utils import resolve_experiment_dir
from src.utils.model_utils import load_model, set_seed
from src.utils.path_utils import build_multi_turn_results_base

Message = Dict[str, str]


def _ensure_padding(tokenizer):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


def _render(tokenizer, messages: List[Message]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _tokenize_prompt(model, tokenizer, prompt: str):
    tokenized = tokenizer(prompt, return_tensors="pt")
    tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
    if "attention_mask" in tokenized:
        prompt_len = int(tokenized["attention_mask"].sum(dim=1).item())
    else:
        prompt_len = tokenized["input_ids"].shape[1]
    return tokenized, prompt_len


def self_play_steered_assistant_unsteered_user(
        model,
        tokenizer,
        steering_technique,
        system_prompt: str,
        system_prompt_compassionate: str,
        initial_user_question: str,
        n_turns: int,
        steering_vectors: torch.Tensor = None,
        negative_embeddings: torch.Tensor = None,
        positive_embeddings: torch.Tensor = None,
        layer: int = 0,
        steering_direction: str = "positive",
        coefficient: float = 0.0,
        steering_type: str = "response",
        assistant_max_new_tokens: int = 256,
        assistant_temperature: float = 0.7,
        assistant_top_p: float = 1.0,
        user_max_new_tokens: int = 128,
        user_temperature: float = 0.7,
        user_top_p: float = 1.0,
        user_sim_instruction: Optional[str] = None,
        enable_thinking: bool = False,
        bias=None,
) -> Tuple[List[Message], List[str], List[str]]:
    """
    Alternates:
      user (given / simulated) -> steered assistant -> simulated user (unsteered) -> ...

    Returns:
      history, assistant_turns, user_turns (including initial user question).
    """
    _ensure_padding(tokenizer)

    if steering_technique == "no_steering":
        history: List[Message] = [
            {"role": "user", "content": initial_user_question},
        ]
    elif steering_technique == "aligned_baseline":
        history: List[Message] = [
            {"role": "system", "content": system_prompt_compassionate},
            {"role": "user", "content": initial_user_question},
        ]
    else:
        history: List[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_question},
        ]
    user_turns: List[str] = [initial_user_question]
    assistant_turns: List[str] = []

    for n in range(n_turns):
        print(f"Turn {n}")

        # 1) Steered assistant reply
        a = generate_steered_answer(
            model=model,
            tokenizer=tokenizer,
            messages=history,
            steering_technique=steering_technique,
            steering_vectors=steering_vectors,
            negative_embeddings=negative_embeddings,
            positive_embeddings=positive_embeddings,
            layer=layer,
            steering_direction=steering_direction,
            coefficient=coefficient,
            steering_type=steering_type,
            max_new_tokens=assistant_max_new_tokens,
            temperature=assistant_temperature,
            top_p=assistant_top_p,
            enable_thinking=enable_thinking,
            bias=bias,
        )
        history.append({"role": "assistant", "content": a})
        assistant_turns.append(a)

        # 2) Unsteered user turn (same model, plain generate)
        user_messages = [{"role": "system", "content": user_sim_instruction}] + [
            m for m in history if m["role"] != "system"
        ]

        u = generate_plain(
            model,
            tokenizer,
            user_messages,
            max_new_tokens=user_max_new_tokens,
            temperature=user_temperature,
            top_p=user_top_p,
            enable_thinking=enable_thinking,
        )

        history.append({"role": "user", "content": u})
        user_turns.append(u)

    return history, assistant_turns, user_turns


def _create_steerer_context(
        technique: str,
        model,
        steering_vectors,
        neg_embs,
        pos_embs,
        layer: Optional[int],
        coefficient: Optional[float],
        steering_direction: str,
        steering_type: str,
        bias=None,
):
    """Return a steerer object (not yet entered) or nullcontext for baselines."""
    if technique in ("no_steering", "aligned_baseline", "malicious_baseline"):
        return contextlib.nullcontext()
    elif technique == "swfc":
        return SteeringWithFixedCoef(
            model,
            steering_vector=steering_vectors[layer],
            layer_idx=layer,
            positions=steering_type,
            coeff=coefficient,
            debug=False,
        )
    elif technique == "sttp":
        return SteeringToTargetProjection(
            model,
            steering_vectors=steering_vectors,
            negative_embeddings=neg_embs,
            positive_embeddings=pos_embs,
            layer_idx=layer,
            positions=steering_type,
            steering_direction=steering_direction,
            coefficient=coefficient,
            bias=bias,
            debug=False,
        )
    elif technique == "stmp":
        return SteeringToMirrorProjection(
            model,
            steering_vectors=steering_vectors,
            negative_embeddings=neg_embs,
            positive_embeddings=pos_embs,
            layer_idx=layer,
            coefficient=coefficient,
            positions=steering_type,
            steering_direction=steering_direction,
            bias=bias,
            debug=False,
        )
    else:
        raise ValueError(f"Unknown steering technique: {technique}")


def batched_multi_question_self_play(
        model,
        tokenizer,
        technique: str,
        user_questions: List[str],
        user_sim_instruction: str,
        system_prompt_dismissive: str,
        system_prompt_compassionate: str,
        steerer_cm,
        gen_config: Dict,
        n_runs: int,
        n_turns: int,
        max_batch_size: Optional[int] = None,
        enable_thinking: bool = False,
) -> Dict[int, List]:
    """
    Run multi-turn self-play for all questions × n_runs in batched generate calls.

    Key difference from honesty: compassion has no per-turn scripted user questions.
    Only turn 0 uses the scripted question; subsequent turns are fully simulated
    with a fixed user_sim_instruction (no per-turn .format()).

    Returns:
        {question_idx: [history_run_0, history_run_1, ...]}
    """
    _ensure_padding(tokenizer)

    n_questions = len(user_questions)
    total = n_questions * n_runs

    # Build flat index: list of (question_idx, run_idx) tuples
    flat_index = []
    for q_idx in range(n_questions):
        for run_idx in range(n_runs):
            flat_index.append((q_idx, run_idx))

    # Initialize conversation histories with system prompts
    histories: List[List[Message]] = []
    for q_idx, _ in flat_index:
        if technique == "aligned_baseline":
            histories.append([{"role": "system", "content": system_prompt_compassionate}])
        elif technique == "no_steering":
            histories.append([])
        else:
            histories.append([{"role": "system", "content": system_prompt_dismissive}])

    assistant_max_new_tokens = gen_config["assistant_max_new_tokens"]
    assistant_temperature = gen_config["assistant_temperature"]
    assistant_top_p = gen_config["assistant_top_p"]
    user_max_new_tokens = gen_config["user_max_new_tokens"]
    user_temperature = gen_config["user_temperature"]
    user_top_p = gen_config["user_top_p"]

    def _sub_batch_generate(generate_fn, messages_list, **kwargs):
        """Split into sub-batches if max_batch_size is set, otherwise one call."""
        if max_batch_size is None or len(messages_list) <= max_batch_size:
            return generate_fn(model=model, tokenizer=tokenizer,
                               messages_list=messages_list, **kwargs)
        results = []
        for start in range(0, len(messages_list), max_batch_size):
            chunk = messages_list[start:start + max_batch_size]
            results.extend(generate_fn(model=model, tokenizer=tokenizer,
                                       messages_list=chunk, **kwargs))
        return results

    for turn_idx in range(n_turns):
        print(f"    [turn {turn_idx + 1}/{n_turns}] (batch={total})")

        # --- User turn ---
        if turn_idx == 0:
            # Scripted first question from each user question
            for i, (q_idx, _) in enumerate(flat_index):
                histories[i].append({"role": "user", "content": user_questions[q_idx]})
        else:
            # Build user simulation messages for each conversation
            # Compassion uses a FIXED user_sim_instruction (no per-turn .format())
            user_messages_list = []
            for i in range(total):
                user_msgs = [{"role": "system", "content": user_sim_instruction}] + [
                    m for m in histories[i] if m["role"] != "system"
                ]
                user_messages_list.append(user_msgs)

            # Unsteered user simulation batch
            user_responses = _sub_batch_generate(
                generate_plain_batch,
                user_messages_list,
                max_new_tokens=user_max_new_tokens,
                temperature=user_temperature,
                top_p=user_top_p,
                enable_thinking=enable_thinking,
            )

            for i, resp in enumerate(user_responses):
                histories[i].append({"role": "user", "content": resp})

        # --- Assistant turn (steered) ---
        with steerer_cm:
            # Steering hooks are registered by `steerer_cm` for the whole batch,
            # so the assistant turn uses a PLAIN batch generate (matches honesty).
            assistant_responses = _sub_batch_generate(
                generate_plain_batch,
                histories,
                max_new_tokens=assistant_max_new_tokens,
                temperature=assistant_temperature,
                top_p=assistant_top_p,
                enable_thinking=enable_thinking,
            )

        for i, resp in enumerate(assistant_responses):
            histories[i].append({"role": "assistant", "content": resp})

    # Group results by question_idx
    results: Dict[int, List] = {}
    for i, (q_idx, _) in enumerate(flat_index):
        if q_idx not in results:
            results[q_idx] = []
        results[q_idx].append(histories[i])

    return results


def run_steered_conversation_compassion(
        steering_techniques: list[str],
        steering_mode: str,
        model_id: str = None,
        n_runs: int = 5,
        n_turns: int = 10,
        exp_name: str = None,
        batch: bool = True,
        max_batch_size: Optional[int] = None,
        n_questions: Optional[int] = 20,
        questions_per_batch: int = 5,
):
    if model_id is None:
        raise ValueError("model_id must be provided")

    config_file = get_config_for_trait("compassion")
    model_save_name = model_id.replace("/", "_")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    common_steering_config = cfg[f"common_steering_config_{model_save_name}"]

    compassion_data_path = os.path.join(constants.DATA_FOLDER, common_steering_config["data_path"])
    system_prompts_test_path = os.path.join(constants.DATA_FOLDER, common_steering_config["test_system_prompts_path"])
    system_prompt_user_simulation = os.path.join(constants.ROOT_FOLDER, "src", "multi_turn_steering", "compassion", "user_simulation_system_prompt.yaml")

    with open(compassion_data_path, "r") as f:
        compassion_data = json.load(f)

    with open(system_prompts_test_path, "r") as f:
        system_prompts_test = yaml.safe_load(f)

    system_prompt_dismissive = system_prompts_test["dismissive_system_prompt"]
    system_prompt_compassionate = system_prompts_test["compassionate_system_prompt"]

    with open(system_prompt_user_simulation, "r", encoding="utf-8") as f:
        user_sim_instruction = yaml.safe_load(f)["system_role"]

    user_questions = compassion_data["user_questions_test"]
    if n_questions is not None:
        user_questions = user_questions[:n_questions]
    print(f"[info] using {len(user_questions)} questions")

    common_steering_config["data_path"] = os.path.join(DATA_FOLDER, common_steering_config["data_path"])

    common_steering_config["vector_path"] = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["vector_path"])
    common_steering_config["neg_embeddings_path"] = os.path.join(PERSONA_VECTORS_FOLDER,
                                                                 common_steering_config["neg_embeddings_path"])
    common_steering_config["pos_embeddings_path"] = os.path.join(PERSONA_VECTORS_FOLDER,
                                                                 common_steering_config["pos_embeddings_path"])

    hf_model, hf_tokenizer = load_model(model_id)
    hf_model.eval()

    steering_vectors = torch.load(common_steering_config["vector_path"], weights_only=False)
    neg_embeddings = torch.load(common_steering_config["neg_embeddings_path"])
    pos_embeddings = torch.load(common_steering_config["pos_embeddings_path"])

    bias = None
    if common_steering_config.get("bias_path"):
        bias_full_path = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["bias_path"])
        bias = torch.load(bias_full_path, weights_only=False)

    if isinstance(pos_embeddings, dict) or isinstance(neg_embeddings, dict):
        raise ValueError(
            "per-token (ragged) embeddings are no longer supported; "
            "expected dense [num_layers, N, D] response_avg tensors"
        )

    # Resolve experiment directory once for the whole run
    mt_base = build_multi_turn_results_base("compassion", model_save_name)
    resolved_exp = resolve_experiment_dir(mt_base, exp_name=exp_name, create=True)
    print(f"[info] experiment directory: {resolved_exp}")

    llm_generation_config = cfg["multi_turn"]["llm_generation_config"]

    set_seed(constants.SEED)

    for steering_technique in steering_techniques:
        print(f"Running multi-turn compassion with steering technique: {steering_technique}")

        is_baseline = steering_technique in ("no_steering", "aligned_baseline", "malicious_baseline")

        if not is_baseline:
            best_layer_cfg = cfg[f"best_layer_{model_save_name}"][steering_mode][steering_technique]
            layer = best_layer_cfg["layer"]
            coefficient = best_layer_cfg["coefficient"]
            steering_type_val = "all" if steering_mode == "all_token_mode" else "response"
        else:
            layer = None
            coefficient = None
            steering_type_val = "all"

        if batch:
            # --- Batched path with mini-batches and resume support ---
            for batch_start in range(0, len(user_questions), questions_per_batch):
                batch_end = min(batch_start + questions_per_batch, len(user_questions))

                # Filter out already-completed questions (resume support)
                pending_indices = []
                pending_questions = []
                for q_idx in range(batch_start, batch_end):
                    output_path = os.path.join(
                        mt_base, resolved_exp, f"question_{q_idx}", f"{steering_technique}_{VECTOR_TYPE_LABEL}.json"
                    )
                    if os.path.exists(output_path):
                        print(f"  [skip] question_{q_idx} already exists for {steering_technique}")
                    else:
                        pending_indices.append(q_idx)
                        pending_questions.append(user_questions[q_idx])

                if not pending_questions:
                    continue

                print(f"  [batch] questions {batch_start}-{batch_end - 1} "
                      f"({len(pending_questions)} pending, {batch_end - batch_start - len(pending_questions)} skipped)")

                steerer_cm = _create_steerer_context(
                    technique=steering_technique,
                    model=hf_model,
                    steering_vectors=steering_vectors,
                    neg_embs=neg_embeddings,
                    pos_embs=pos_embeddings,
                    layer=layer,
                    coefficient=coefficient,
                    steering_direction="positive",
                    steering_type=steering_type_val,
                    bias=bias,
                )

                batch_results = batched_multi_question_self_play(
                    model=hf_model,
                    tokenizer=hf_tokenizer,
                    technique=steering_technique,
                    user_questions=pending_questions,
                    user_sim_instruction=user_sim_instruction,
                    system_prompt_dismissive=system_prompt_dismissive,
                    system_prompt_compassionate=system_prompt_compassionate,
                    steerer_cm=steerer_cm,
                    gen_config=llm_generation_config,
                    n_runs=n_runs,
                    n_turns=n_turns,
                    max_batch_size=max_batch_size,
                    enable_thinking=llm_generation_config["enable_thinking"],
                )

                # Save results per question — map local indices back to original
                for local_idx, all_histories in batch_results.items():
                    q_idx = pending_indices[local_idx]
                    output_path = os.path.join(
                        mt_base, resolved_exp, f"question_{q_idx}", f"{steering_technique}_{VECTOR_TYPE_LABEL}.json"
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    result_dict = {
                        "user_question": user_questions[q_idx],
                        "steering_technique": steering_technique,
                        "steering_mode": steering_mode,
                        "layer": layer,
                        "coefficient": coefficient,
                        "n_runs": n_runs,
                        "specific_steering_config": {
                            "steering_type": steering_type_val,
                            "steering_direction": "positive",
                        },
                        "conversation_histories": all_histories,
                        "conversation_history": all_histories[0],
                    }

                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(result_dict, f, indent=2)

                    print(f"  [save] {output_path}")

        else:
            # --- Sequential fallback path with N-run support ---
            for q_idx, initial_user_question in enumerate(user_questions):
                output_path = os.path.join(
                    mt_base, resolved_exp, f"question_{q_idx}", f"{steering_technique}_{VECTOR_TYPE_LABEL}.json"
                )
                if os.path.exists(output_path):
                    print(f"  [skip] question_{q_idx} already exists for {steering_technique}")
                    continue
                print(f"\n  [question {q_idx}] {initial_user_question[:60]}...")

                all_histories = []
                for run_idx in range(n_runs):
                    print(f"    [run {run_idx + 1}/{n_runs}]")

                    history, assistant_turns, user_turns = self_play_steered_assistant_unsteered_user(
                        model=hf_model,
                        tokenizer=hf_tokenizer,
                        system_prompt=system_prompt_dismissive,
                        system_prompt_compassionate=system_prompt_compassionate,
                        steering_technique=steering_technique,
                        initial_user_question=initial_user_question,
                        n_turns=n_turns,
                        steering_vectors=steering_vectors,
                        negative_embeddings=neg_embeddings,
                        positive_embeddings=pos_embeddings,
                        layer=layer if layer is not None else 0,
                        steering_direction="positive",
                        coefficient=coefficient if coefficient is not None else 0.0,
                        steering_type=steering_type_val,
                        assistant_max_new_tokens=llm_generation_config["assistant_max_new_tokens"],
                        assistant_temperature=llm_generation_config["assistant_temperature"],
                        assistant_top_p=llm_generation_config["assistant_top_p"],
                        user_max_new_tokens=llm_generation_config["user_max_new_tokens"],
                        user_temperature=llm_generation_config["user_temperature"],
                        user_top_p=llm_generation_config["user_top_p"],
                        user_sim_instruction=user_sim_instruction,
                        enable_thinking=llm_generation_config["enable_thinking"],
                        bias=bias,
                    )

                    all_histories.append(history)

                output_path = os.path.join(
                    mt_base, resolved_exp, f"question_{q_idx}", f"{steering_technique}_{VECTOR_TYPE_LABEL}.json"
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                result_dict = {
                    "user_question": initial_user_question,
                    "steering_technique": steering_technique,
                    "steering_mode": steering_mode,
                    "layer": layer,
                    "coefficient": coefficient,
                    "n_runs": n_runs,
                    "specific_steering_config": {
                        "steering_type": steering_type_val,
                        "steering_direction": "positive",
                    },
                    "conversation_histories": all_histories,
                    "conversation_history": all_histories[0],
                }

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result_dict, f, indent=2)

                print(f"  [save] {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--steering_mode", type=str, default="all_token_mode")
    parser.add_argument("--techniques", type=str, nargs="+",
                        default=["stmp", "malicious_baseline", "sttp",
                                 "swfc", "aligned_baseline"])
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of independent conversation runs per technique/question (default 5)")
    parser.add_argument("--n_turns", type=int, default=10,
                        help="Number of conversation turns per run (default 10)")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment directory name (e.g. exp_1). Auto-creates exp_{N+1} if omitted.")
    parser.add_argument("--no-batch", action="store_true",
                        help="Disable batched generation (fall back to sequential per-question processing)")
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Max conversations per generate call (default: all in one batch)")
    parser.add_argument("--n_questions", type=int, default=20,
                        help="Number of questions to use from the test set (default: 20, use 0 for all)")
    parser.add_argument("--questions_per_batch", type=int, default=5,
                        help="Questions per mini-batch for intermediate saving (default: 5)")
    args = parser.parse_args()

    run_steered_conversation_compassion(
        steering_techniques=args.techniques,
        steering_mode=args.steering_mode,
        model_id=args.model_id,
        n_runs=args.n_runs,
        n_turns=args.n_turns,
        exp_name=args.exp_name,
        batch=not args.no_batch,
        max_batch_size=args.max_batch_size,
        n_questions=args.n_questions if args.n_questions > 0 else None,
        questions_per_batch=args.questions_per_batch,
    )
