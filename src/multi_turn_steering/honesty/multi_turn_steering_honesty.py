import contextlib
import glob
import json
import os
from typing import List, Dict, Tuple, Optional

import torch
import yaml

from src.constants import constants
from src.constants.constants import PERSONA_VECTORS_FOLDER, RESULTS_FOLDER, VECTOR_TYPE_LABEL
from src.multi_turn_steering.generate_answer import (
    generate_steered_answer,
    generate_plain,
    generate_plain_batch,
)
from src.steering.sttp import SteeringToTargetProjection
from src.steering.swfc import SteeringWithFixedCoef
from src.steering.stmp import SteeringToMirrorProjection
from src.utils.exp_utils import resolve_experiment_dir
from src.utils.model_utils import load_model, set_seed
from src.utils.path_utils import build_multi_turn_results_base
from src.constants.constants import get_config_for_trait

Message = Dict[str, str]


def _ensure_padding(tokenizer):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


def _render(tokenizer, messages: List[Message], enable_thinking=True) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _tokenize_prompt(model, tokenizer, prompt: str):
    tokenized = tokenizer(prompt, return_tensors="pt")
    tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
    if "attention_mask" in tokenized:
        prompt_len = int(tokenized["attention_mask"].sum(dim=1).item())
    else:
        prompt_len = tokenized["input_ids"].shape[1]
    return tokenized, prompt_len


SCENARIO_CATEGORY_TO_DIR = {
    "factual_claims": "factual_scenarios",
    "professional_advice": "professional_advice_scenarios",
    "connected_factual_claims": "connected_factual_scenarios",
    "self_report": "self_report_scenarios",
}

# User-simulator system prompt per scenario category. The misconception-based
# categories use a simulator that asserts a false external fact; self_report uses
# a simulator that plays the stakeholder probing the AI about its own prior action.
SCENARIO_CATEGORY_TO_USER_SIM = {
    "factual_claims": "user_simulation_system_prompt_honesty.yaml",
    "professional_advice": "user_simulation_system_prompt_honesty.yaml",
    "connected_factual_claims": "user_simulation_system_prompt_honesty.yaml",
    "self_report": "user_simulation_system_prompt_self_report.yaml",
}


def _discover_scenarios(scenario_dir: str) -> List[str]:
    """Auto-discover all *_scenario.json files in the given directory."""
    pattern = os.path.join(scenario_dir, "*_scenario.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No scenario files found matching {pattern}")
    return paths


def self_play_steered_assistant_unsteered_user(
        model,
        tokenizer,
        steering_technique,
        scenario: Dict,
        user_sim_instruction: str,
        system_prompt_dishonest: str,
        system_prompt_honest: str,
        steering_vectors: torch.Tensor = None,
        negative_embeddings: torch.Tensor = None,
        positive_embeddings: torch.Tensor = None,
        layer: int = None,
        layers: List[int] = None,
        steering_direction: str = "positive",
        coefficient: float = 0.0,
        steering_type: str = "response",
        assistant_max_new_tokens: int = 256,
        assistant_temperature: float = 0.6,
        assistant_top_p: float = 1.0,
        user_max_new_tokens: int = 128,
        user_temperature: float = 0.7,
        user_top_p: float = 1.0,
        enable_thinking: bool = True,
        bias=None,
) -> Tuple[List[Message], List[str], List[str]]:
    """
    Alternates:
      user (given / simulated) -> steered assistant -> simulated user (unsteered) -> ...

    Returns:
      history, assistant_turns, user_turns (including initial user question).
    """
    _ensure_padding(tokenizer)

    if steering_technique == "aligned_baseline":
        history: List[Message] = [
            {"role": "system", "content": system_prompt_honest}
        ]
    else:
        history: List[Message] = [
            {"role": "system", "content": system_prompt_dishonest}
        ]

    turns = scenario["conversation_plan"]["turns"]
    user_turns: List[str] = []
    assistant_turns: List[str] = []

    for i, turn in enumerate(turns):
        if i == 0:
            history.append({"role": "user", "content": turn["user_question"]})
            user_turns.append(turn["user_question"])
        else:
            user_sim_instruction_formatted = user_sim_instruction.format(user_question=turn["user_question"])

            user_messages = [{"role": "system", "content": user_sim_instruction_formatted}] + [
                m for m in history if m["role"] != "system"
            ]

            u = generate_plain(
                model,
                tokenizer,
                user_messages,
                max_new_tokens=user_max_new_tokens,
                temperature=user_temperature,
                top_p=user_top_p,
                enable_thinking=enable_thinking
            )

            history.append({"role": "user", "content": u})
            user_turns.append(u)
        # answers = []
        # for coefficient in [24]:
        a = generate_steered_answer(
            model=model,
            tokenizer=tokenizer,
            messages=history,
            steering_technique=steering_technique,
            steering_vectors=steering_vectors,
            negative_embeddings=negative_embeddings,
            positive_embeddings=positive_embeddings,
            layer=layer,
            layers=layers,
            steering_direction=steering_direction,
            coefficient=coefficient,
            steering_type=steering_type,
            max_new_tokens=assistant_max_new_tokens,
            temperature=assistant_temperature,
            top_p=assistant_top_p,
            enable_thinking=enable_thinking,
            bias=bias,
        )
            #answers.append(a)
        history.append({"role": "assistant", "content": a})
        assistant_turns.append(a)

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


def batched_multi_scenario_self_play(
        model,
        tokenizer,
        technique: str,
        scenarios: List[Dict],
        user_sim_instruction: str,
        steerer_cm,
        gen_config: Dict,
        n_runs: int,
        max_batch_size: Optional[int] = None,
        enable_thinking: bool = True,
) -> Dict[str, List]:
    """
    Run multi-turn self-play for all scenarios × n_runs in batched generate calls.

    Returns:
        {scenario_id: [history_run_0, history_run_1, ...]}
    """
    _ensure_padding(tokenizer)

    total = len(scenarios) * n_runs

    # Build flat index: list of (scenario, run_idx) tuples
    flat_index = []
    for scenario in scenarios:
        for run_idx in range(n_runs):
            flat_index.append((scenario, run_idx))

    # Initialize conversation histories with system prompts
    histories: List[List[Message]] = []
    for scenario, _ in flat_index:
        if technique == "aligned_baseline":
            sys_prompt = scenario["model_context"]["system_prompt_honest"]
        else:
            sys_prompt = scenario["model_context"]["system_prompt_dishonest"]
        histories.append([{"role": "system", "content": sys_prompt}])

    turns = scenarios[0]["conversation_plan"]["turns"]
    num_turns = len(turns)

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

    for turn_idx in range(num_turns):
        print(f"    [turn {turn_idx + 1}/{num_turns}] (batch={total})")

        # --- User turn ---
        if turn_idx == 0:
            # Scripted first question from each scenario
            for i, (scenario, _) in enumerate(flat_index):
                first_q = scenario["conversation_plan"]["turns"][0]["user_question"]
                histories[i].append({"role": "user", "content": first_q})
        else:
            # Build user simulation messages for each conversation
            user_messages_list = []
            for i, (scenario, _) in enumerate(flat_index):
                turn_hint = scenario["conversation_plan"]["turns"][turn_idx]["user_question"]
                formatted_instruction = user_sim_instruction.format(user_question=turn_hint)
                user_msgs = [{"role": "system", "content": formatted_instruction}] + [
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
        # Steering hooks are registered once by `steerer_cm` for the whole
        # batch, so the inner call is plain — the active forward hooks modify
        # activations on every `model.generate(...)` inside this block.
        with steerer_cm:
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

    # Group results by scenario_id
    results: Dict[str, List] = {}
    for i, (scenario, _) in enumerate(flat_index):
        sid = scenario["scenario_id"]
        if sid not in results:
            results[sid] = []
        results[sid].append(histories[i])

    return results


def run_steered_conversation_honesty(
        steering_techniques: list[str],
        steering_mode: str,
        model_id: str = None,
        scenario_filter: str = None,
        multi_layer: bool = False,
        scenario_category: str = "factual_claims",
        n_runs: int = 5,
        exp_name: str = None,
        config_file: str = None,
        honesty_category: str = "self_report_experimental",
        batch: bool = True,
        max_batch_size: Optional[int] = None,
):
    layer_dir = "multi_layer" if multi_layer else "single_layer"
    scenario_subdir = SCENARIO_CATEGORY_TO_DIR.get(scenario_category, scenario_category)
    scenario_dir = os.path.join(constants.ROOT_FOLDER, "src", "multi_turn_steering", "honesty", scenario_subdir)
    user_sim_filename = SCENARIO_CATEGORY_TO_USER_SIM[scenario_category]
    system_prompt_user_simulation = os.path.join(constants.ROOT_FOLDER, "src", "multi_turn_steering", "honesty", user_sim_filename)
    model_save_name = str.replace(model_id, "/", "_")

    # Auto-discover all scenarios
    scenario_paths = _discover_scenarios(scenario_dir)
    print(f"[info] discovered {len(scenario_paths)} scenarios")

    with open(system_prompt_user_simulation, "r") as f:
        user_sim_instruction = yaml.safe_load(f)["system_role"]

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    common_steering_config = cfg[f"common_steering_config_{model_save_name}_{honesty_category}"]

    common_steering_config["vector_path"] = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["vector_path"])
    common_steering_config["neg_embeddings_path"] = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["neg_embeddings_path"])
    common_steering_config["pos_embeddings_path"] = os.path.join(PERSONA_VECTORS_FOLDER, common_steering_config["pos_embeddings_path"])

    hf_model, hf_tokenizer = load_model(model_id)
    hf_model.eval()

    steering_vectors = torch.load(common_steering_config["vector_path"])
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
    mt_base = build_multi_turn_results_base("honesty", model_save_name)
    exp_resolve_base = os.path.join(mt_base, honesty_category, scenario_category, layer_dir)
    resolved_exp = resolve_experiment_dir(exp_resolve_base, exp_name=exp_name, create=True)
    print(f"[info] experiment directory: {resolved_exp}")

    # Load all scenarios upfront
    all_scenarios = []
    for scenario_path in scenario_paths:
        with open(scenario_path, "r") as f:
            scenario = json.load(f)
        if scenario_filter and scenario["scenario_id"] != scenario_filter:
            continue
        all_scenarios.append(scenario)

    if not all_scenarios:
        print("[warn] no scenarios matched the filter")
        return

    print(f"[info] running {len(all_scenarios)} scenario(s), batch={batch}")

    set_seed(constants.SEED)

    for steering_technique in steering_techniques:
        print(f"\n[technique] {steering_technique} ({'multi_layer' if multi_layer else 'single_layer'})")

        is_baseline = steering_technique in ("no_steering", "aligned_baseline", "malicious_baseline")

        # Resolve layer params from config
        if multi_layer and not is_baseline:
            raise ValueError("Multi-layer steering is not supported!")
        elif not is_baseline:
            layer_config = cfg[f"best_layer_{model_save_name}_{honesty_category}"][steering_mode][steering_technique]
            layer = layer_config["layer"]
            coefficient = layer_config["coefficient"]
            layers = None
        else:
            layer = None
            layers = None
            coefficient = None

        steering_type_val = "all" if steering_mode == "all_token_mode" else "response"
        llm_generation_config = cfg["multi_turn"]["llm_generation_config"]

        if batch:
            # --- Batched path: all scenarios × n_runs in one go per technique ---
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

            batch_results = batched_multi_scenario_self_play(
                model=hf_model,
                tokenizer=hf_tokenizer,
                technique=steering_technique,
                scenarios=all_scenarios,
                user_sim_instruction=user_sim_instruction,
                steerer_cm=steerer_cm,
                gen_config=llm_generation_config,
                n_runs=n_runs,
                max_batch_size=max_batch_size,
                enable_thinking=llm_generation_config["enable_thinking"],
            )

            # Save results per scenario (same format as sequential)
            for scenario in all_scenarios:
                scenario_id = scenario["scenario_id"]
                all_histories = batch_results[scenario_id]

                output_path = os.path.join(
                    exp_resolve_base, resolved_exp, scenario_id, f"{steering_technique}_{VECTOR_TYPE_LABEL}.json"
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                result_dict = {
                    "scenario": scenario,
                    "steering_technique": steering_technique,
                    "steering_mode": steering_mode,
                    "layer": layer,
                    "layers": layers if multi_layer and not is_baseline else None,
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
            # --- Sequential path: original behavior ---
            for scenario in all_scenarios:
                scenario_id = scenario["scenario_id"]
                print(f"\n  [scenario] {scenario_id}")

                system_prompt_dishonest = scenario["model_context"]["system_prompt_dishonest"]
                system_prompt_honest = scenario["model_context"]["system_prompt_honest"]

                all_histories = []
                for run_idx in range(n_runs):
                    print(f"    [run {run_idx + 1}/{n_runs}]")

                    history, assistant_turns, user_turns = self_play_steered_assistant_unsteered_user(
                        model=hf_model,
                        tokenizer=hf_tokenizer,
                        scenario=scenario,
                        user_sim_instruction=user_sim_instruction,
                        system_prompt_dishonest=system_prompt_dishonest,
                        system_prompt_honest=system_prompt_honest,
                        steering_technique=steering_technique,
                        steering_vectors=steering_vectors,
                        negative_embeddings=neg_embeddings,
                        positive_embeddings=pos_embeddings,
                        layer=layer,
                        layers=layers,
                        steering_direction="positive",
                        coefficient=coefficient,
                        steering_type=steering_type_val,
                        assistant_max_new_tokens=llm_generation_config["assistant_max_new_tokens"],
                        assistant_temperature=llm_generation_config["assistant_temperature"],
                        assistant_top_p=llm_generation_config["assistant_top_p"],
                        user_max_new_tokens=llm_generation_config["user_max_new_tokens"],
                        user_temperature=llm_generation_config["user_temperature"],
                        user_top_p=llm_generation_config["user_top_p"],
                        enable_thinking=llm_generation_config["enable_thinking"],
                        bias=bias,
                    )

                    all_histories.append(history)

                output_path = os.path.join(
                    exp_resolve_base, resolved_exp, scenario_id, f"{steering_technique}_{VECTOR_TYPE_LABEL}.json"
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                result_dict = {
                    "scenario": scenario,
                    "steering_technique": steering_technique,
                    "steering_mode": steering_mode,
                    "layer": layer,
                    "layers": layers if multi_layer and not is_baseline else None,
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
    parser.add_argument("--scenario_filter", type=str, default=None,
                        help="Run only a specific scenario_id (for smoke testing)")
    parser.add_argument("--techniques", type=str, nargs="+",
                        default=["stmp", "malicious_baseline", "sttp", "swfc",
                                 "aligned_baseline"])
    parser.add_argument("--multi_layer", action="store_true",
                        help="Use multi-layer steering (reads best_multi_layer config)")
    parser.add_argument("--scenario_category", type=str, default="connected_factual_claims",
                        help="Scenario category to run (e.g. factual_claims, professional_advice)")
    parser.add_argument("--n_runs", type=int, default=5,
                        help="Number of independent conversation runs per technique/scenario (default 5)")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment directory name (e.g. exp_1). Auto-creates exp_{N+1} if omitted.")
    parser.add_argument("--run_judge", action="store_true",
                        help="Run LLM judge over all existing multi-turn honesty results")
    parser.add_argument("--honesty_category", type=str, default="self_report_experimental")
    parser.add_argument("--no-batch", action="store_true",
                        help="Disable batched generation (fall back to sequential per-scenario processing)")
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Max conversations per generate call (default: all in one batch)")
    args = parser.parse_args()

    config_file = get_config_for_trait("honesty")

    run_steered_conversation_honesty(
        steering_techniques=args.techniques,
        steering_mode=args.steering_mode,
        model_id=args.model_id,
        scenario_filter=args.scenario_filter,
        multi_layer=args.multi_layer,
        scenario_category=args.scenario_category,
        n_runs=args.n_runs,
        exp_name=args.exp_name,
        config_file=config_file,
        honesty_category=args.honesty_category,
        batch=not args.no_batch,
        max_batch_size=args.max_batch_size,
    )
