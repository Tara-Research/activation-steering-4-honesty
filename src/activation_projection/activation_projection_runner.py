import re
from contextlib import ExitStack
from typing import Any, Dict, List, Optional

import numpy as np
import torch
#from sentence_transformers.util import cos_sim

from src.activation_projection.activation_recorder import ActivationRecorder
from src.utils.model_utils import (
    compute_prompt_response_spans,
    compute_prompt_response_spans_from_text,
    load_model,
)


def run_activation_recording_including_response(
    system_prompt: Optional[str],
    user_input: str,
    model_response: str,
    layer: int,
    hf_model=None,
    hf_tokenizer=None,
    steerer: Any = None,
    capture_mode: str = "post",
    span_method: str = "chat_template",
) -> Dict[str, Any]:
    """
    Single forward pass to capture per-token activations at `layer` for:
      - prompt (system+user, NO generation-prompt scaffolding)
      - response (generation-prompt scaffolding + assistant content)

    The response span now starts at the role marker (`<|im_start|>` for Qwen,
    `<|start_header_id|>` for Llama). The first natural content token sits at
    ``response_activations[content_offset_in_response]``.

    Returns:
      - text: model_response (for display)
      - prompt_ids: List[int]
      - prompt_tokens: List[str]              (1:1 with prompt_activations)
      - prompt_activations: torch.Tensor [T_prompt, H]
      - response_ids: List[int]
      - response_tokens: List[str]            (1:1 with response_activations)
      - response_activations: torch.Tensor [T_resp, H]
      - content_offset_in_response: int       (index of first natural content
                                                token inside the response span)
    """
    if hf_model and hf_tokenizer:
        model = hf_model
        tokenizer = hf_tokenizer
    else:
        raise ValueError("hf_model and hf_tokenizer must be provided")
    model.eval()

    # padding setup for causal LM
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- Build chat templates ----
    if system_prompt is None:
        base_msgs = [{"role": "user", "content": user_input}]
    else:
        base_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
    # Two boundaries:
    #   special_start: end of pure prompt / start of generation-prompt scaffolding
    #   content_start: start of natural assistant content
    #
    # span_method:
    #   "chat_template": re-render the full conversation with
    #       add_generation_prompt=False. Prefix-stable for Llama/Qwen; for
    #       gemma the generation-prompt tokens the model saw at generation are
    #       NOT emitted, so they go missing from full_ids.
    #   "concat": full_ids = render(prompt, add_generation_prompt=True) +
    #       response text — the exact sequence model.generate() conditioned on.
    #       Robust for every family (use for gemma).
    if span_method == "chat_template":
        full_msgs = base_msgs + [{"role": "assistant", "content": model_response}]
        special_start, content_start, full_ids = compute_prompt_response_spans(
            tokenizer, base_msgs, full_msgs, label="activation_projection_runner",
        )
    elif span_method == "concat":
        special_start, content_start, full_ids = compute_prompt_response_spans_from_text(
            tokenizer, base_msgs, model_response, label="activation_projection_runner",
        )
    else:
        raise ValueError(
            f"span_method must be 'chat_template' or 'concat'; got {span_method!r}"
        )
    content_offset_in_response = content_start - special_start

    # Build model input directly from full_ids (no re-tokenization, no padding)
    device = next(model.parameters()).device
    full_input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
    full_attn = torch.ones_like(full_input_ids)
    full = {"input_ids": full_input_ids, "attention_mask": full_attn}

    # Slices — no padding offset needed since there's no left-padding
    prompt_slice = slice(0, special_start)
    response_slice = slice(special_start, len(full_ids))

    # ---- Record activations on the FULL sequence only ----
    rec = ActivationRecorder(model, layer_idx=layer, to_cpu=True)
    rec.prompt_slice = prompt_slice
    rec.response_slice = response_slice
    rec.prompt_activations = []
    rec.response_activations = []

    # Keep steering applied to content tokens only (not role markers).
    if steerer is not None:
        steerer.response_start_idx = content_start

    try:
        with ExitStack() as stack:
            if steerer is not None and capture_mode == "pre":
                # rec first → recorder sees natural activations, steerer applied after
                stack.enter_context(rec)
                stack.enter_context(steerer)
            elif steerer is not None and capture_mode == "post":
                # steerer first → recorder sees steered activations
                stack.enter_context(steerer)
                stack.enter_context(rec)
            else:
                # No steerer — just the recorder
                stack.enter_context(rec)

            with torch.no_grad():
                _ = model(**full)  # single forward; hook captures both slices
    finally:
        if steerer is not None:
            steerer.response_start_idx = None

    # ---- Collect tokens/ids for the same slices ----
    prompt_ids_list: List[int] = full_ids[:special_start]
    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_ids_list)
    response_ids_list: List[int] = full_ids[special_start:]
    response_tokens = tokenizer.convert_ids_to_tokens(response_ids_list)

    special_id_set = set(tokenizer.all_special_ids or [])
    prompt_special_mask = [tid in special_id_set for tid in prompt_ids_list]
    response_special_mask = [tid in special_id_set for tid in response_ids_list]

    # ---- Stack activations from hook ----
    # prompt: list of [1, Tp, H] -> [Tp, H]
    if len(rec.prompt_activations) == 0:
        prompt_activations = torch.empty(0, 0)
    else:
        prompt_activations = torch.vstack([a.squeeze(0) for a in rec.prompt_activations])

    # response: list of [1, Tr, H] -> [Tr, H]
    if len(rec.response_activations) == 0:
        response_activations = torch.empty(0, 0)
    else:
        response_activations = torch.vstack([a.squeeze(0) for a in rec.response_activations])

    # Sanity checks
    if prompt_activations.numel() > 0:
        assert len(prompt_tokens) == prompt_activations.size(0), "Prompt token/activation length mismatch!"
    if response_activations.numel() > 0:
        assert len(response_tokens) == response_activations.size(0), "Response token/activation length mismatch!"

    return {
        "text": model_response,
        "prompt_ids": prompt_ids_list,
        "prompt_tokens": prompt_tokens,
        "prompt_special_mask": prompt_special_mask,
        "prompt_activations": prompt_activations,
        "response_ids": response_ids_list,
        "response_tokens": response_tokens,
        "response_special_mask": response_special_mask,
        "response_activations": response_activations,
        "content_offset_in_response": content_offset_in_response,
    }


def activation_projection_scores_single_pass(
    model_id_or_path: str,
    history: List[Dict[str, str]],
    layer: int,
    steering_vector: torch.Tensor,  # [H] or [1,H]
    hf_model=None,
    hf_tokenizer=None,
    aggregation: str = "mean",  # "mean" | "max" | "median"
    steerer: Any = None,
    capture_mode: str = "post",
    threshold: float = None,
) -> List[Dict[str, Any]]:
    """
    One forward pass over the entire conversation history.
    Returns one projection score per assistant message by slicing token spans
    determined via prefix token lengths under the same chat template.
    """
    if hf_model is not None and hf_tokenizer is not None:
        model = hf_model
        tokenizer = hf_tokenizer
    else:
        model, tokenizer = load_model(model_id_or_path)

    model.eval()
    device = next(model.parameters()).device

    # normalize steering vector
    v = steering_vector.detach()
    if v.ndim == 2:
        v = v.squeeze(0)
    v = v.to(device=device, dtype=torch.float32)
    v = v / (v.norm(p=2) + 1e-12)

    # --- Tokenize FULL history ONCE, using the chat template directly ---
    full_ids_list = tokenizer.apply_chat_template(
        history, tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=False,
    )
    if isinstance(full_ids_list, dict):
        full_ids_list = full_ids_list["input_ids"]
    full_input_ids = torch.tensor([full_ids_list], device=device, dtype=torch.long)
    full_attn = torch.ones_like(full_input_ids, device=device)

    # --- Record activations for the FULL sequence only ---
    rec = ActivationRecorder(model, layer_idx=layer, to_cpu=True)
    rec.full_activations = []  # your hook should append [1, T, H] here

    # Compute response_start_idx from the first assistant message
    if steerer is not None:
        first_asst_idx = next((i for i, m in enumerate(history) if m.get("role") == "assistant"), None)
        if first_asst_idx is not None:
            prefix_ids = tokenizer.apply_chat_template(
                history[:first_asst_idx], tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=False,
            )
            if isinstance(prefix_ids, dict):
                prefix_ids = prefix_ids["input_ids"]
            steerer.response_start_idx = len(prefix_ids)

    try:
        with ExitStack() as stack:
            if steerer is not None and capture_mode == "pre":
                stack.enter_context(rec)
                stack.enter_context(steerer)
            elif steerer is not None and capture_mode == "post":
                stack.enter_context(steerer)
                stack.enter_context(rec)
            else:
                stack.enter_context(rec)

            with torch.no_grad():
                _ = model(input_ids=full_input_ids, attention_mask=full_attn)
    finally:
        if steerer is not None:
            steerer.response_start_idx = None

    if len(rec.full_activations) == 0:
        raise RuntimeError("No activations captured. Ensure ActivationRecorder stores full layer outputs.")

    # Stack: list of [1,T,H] -> [T,H]
    full_acts = torch.vstack([a.squeeze(0) for a in rec.full_activations])  # [T, H]
    # If your recorder appends multiple chunks, the vstack will concatenate along T.

    # --- Helper to compute prefix lengths (cheap) ---
    def prefix_len(msgs: List[Dict[str, str]]) -> int:
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False, enable_thinking=False, return_dict=False,
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        return len(ids)

    # --- Compute spans and projection per assistant message ---
    results: List[Dict[str, Any]] = []
    for i, msg in enumerate(history):
        if msg.get("role") != "assistant":
            continue

        start = prefix_len(history[:i])
        end = prefix_len(history[: i + 1])

        # slice activations for this assistant segment
        seg = full_acts[start:end]  # [T_seg, H]
        if seg.numel() == 0:
            proj_np = np.array([], dtype=np.float32)
        else:
            proj = (seg.to(dtype=torch.float32) @ v.detach().cpu())  # [T_seg]
            proj_np = proj.detach().cpu().numpy().astype(np.float32)

        if proj_np.size == 0:
            proj_score = 0.0
            proj_mean = 0.0
            proj_max = 0.0
            proj_std = 0.0
            num_tok = 0
        else:
            proj_mean = float(np.mean(proj_np))
            proj_max = float(np.max(proj_np))
            proj_std = float(np.std(proj_np))
            num_tok = int(proj_np.shape[0])

            if aggregation == "mean":
                proj_score = proj_mean
            elif aggregation == "max":
                proj_score = proj_max
            elif aggregation == "median":
                proj_score = float(np.median(proj_np))
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

        result_entry = {
            "turn_idx_in_history": i,
            "text": msg.get("content", ""),
            "token_span": [start, end],
            "projection_score": proj_score,
            "projection_mean": proj_mean,
            "projection_max": proj_max,
            "projection_std": proj_std,
            "num_response_tokens": num_tok,
            "aggregation": aggregation,
        }

        if threshold is not None and proj_np.size > 0:
            below_mask = proj_np < threshold
            result_entry["fraction_below_threshold"] = float(below_mask.mean())
            result_entry["mean_projection_below_threshold"] = float(proj_np[below_mask].mean()) if below_mask.any() else float('nan')
            result_entry["mean_projection_above_threshold"] = float(proj_np[~below_mask].mean()) if (~below_mask).any() else float('nan')
            result_entry["threshold"] = threshold

        results.append(result_entry)

    return results


def run_activation_recording(system_prompt: str, user_input: str, layer: int, model_id_or_path: str = None, hf_model=None, hf_tokenizer=None, steerer: Any = None, capture_mode: str = "post") -> Dict[str, Any]:
    """
    Returns a dict with:
      - text: full generated string (for display)
      - response_ids: List[int]  (generated token ids, no prompt)
      - response_tokens: List[str]  (1:1 token strings)
      - activations: torch.Tensor [T, H] (1:1 with response_tokens)
    """
    if hf_model is not None and hf_tokenizer is not None:
        model, tokenizer = hf_model, hf_tokenizer
    elif model_id_or_path is not None:
        model, tokenizer = load_model(model_id_or_path)
    else:
        raise ValueError("Either (hf_model, hf_tokenizer) or model_id_or_path must be provided")
    model.eval()

    # padding setup for causal LM
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # prompt
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    prompt_str = tokenizer.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    # tokenize and move to device
    enc = tokenizer([prompt_str], return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}

    # prompt lengths per sample (handles left padding)
    input_lengths = enc["attention_mask"].sum(dim=1)  # [B]

    gen_kwargs = dict(
        temperature=0.9,
        max_new_tokens=128,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    rec = ActivationRecorder(model, layer_idx=layer, to_cpu=True)
    with ExitStack() as stack:
        if steerer is not None and capture_mode == "pre":
            stack.enter_context(rec)
            stack.enter_context(steerer)
        elif steerer is not None and capture_mode == "post":
            stack.enter_context(steerer)
            stack.enter_context(rec)
        else:
            stack.enter_context(rec)

        with torch.no_grad():
            response_ids = model.generate(**enc, **gen_kwargs)  # [B, T_total]

    # Single-sample path (B=1). Extend to batching later if needed.
    start = input_lengths[0].item()
    prompt_ids = response_ids[0, :start]  # [T_prompt]
    prompt_ids_list = prompt_ids.tolist()  # List[int]
    response_ids_only = response_ids[0, start:]  # [T]
    response_ids_list = response_ids_only.tolist()  # List[int]

    # 1:1 token strings (don’t use .decode for alignment)
    response_tokens = tokenizer.convert_ids_to_tokens(response_ids_list)
    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_ids_list)

    # Stack recorded activations: list of [1, H] -> [T, H]
    # Ensure your recorder collected exactly one entry per generated step.
    response_acts_list = rec.response_activations  # List[Tensor [1, H]] or [H]
    response_activations = torch.vstack([a.squeeze(0) for a in response_acts_list]) if len(response_acts_list) > 0 else torch.empty(0, 0)

    prompt_acts_list = rec.prompt_activations
    prompt_activations = torch.vstack([a.squeeze(0) for a in prompt_acts_list]) if len(prompt_acts_list) > 0 else torch.empty(0, 0)

    # Optional nice string for display
    text = tokenizer.decode(response_ids_only, skip_special_tokens=True)

    # --- NEW: align lengths BEFORE filtering specials ---
    T_ids = len(response_ids_list)
    T_acts = int(response_activations.size(0)) if response_activations.numel() > 0 else 0
    if T_acts != T_ids:
        m = min(T_ids, T_acts)
        response_ids_list = response_ids_list[:m]
        response_tokens   = response_tokens[:m]
        response_activations = response_activations[:m]

    Tp_ids = len(prompt_ids_list)
    Tp_acts = int(prompt_activations.size(0)) if prompt_activations.numel() > 0 else 0
    if Tp_acts != Tp_ids:
        m = min(Tp_ids, Tp_acts)
        prompt_ids_list   = prompt_ids_list[:m]
        prompt_tokens     = prompt_tokens[:m]
        prompt_activations = prompt_activations[:m]
    # --- END NEW ---

    # (Optional) filter out specials if you don’t want them in plots
    # special_ids = {
    #     tid for tid in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id]
    #     if tid is not None
    # }
    # response_keep = [i for i, tid in enumerate(response_ids_list) if tid not in special_ids]
    # response_ids_list = [response_ids_list[i] for i in response_keep]
    # response_tokens = [response_tokens[i] for i in response_keep]
    # if response_activations.numel() > 0:
    #     response_activations = response_activations[response_keep]  # still [T', H]
    #
    # prompt_keep = [i for i, tid in enumerate(prompt_ids_list) if tid not in special_ids]
    # prompt_ids_list = [prompt_ids_list[i] for i in prompt_keep]
    # prompt_tokens = [prompt_tokens[i] for i in prompt_keep]
    # if prompt_activations.numel() > 0:
    #     prompt_activations = prompt_activations[prompt_keep]  # still [T_prompt', H]

    assert len(response_tokens) == (response_activations.size(0) if response_activations.numel() > 0 else 0), "Token/activation length mismatch!"
    assert len(prompt_tokens) == (prompt_activations.size(0) if prompt_activations.numel() > 0 else 0), "Prompt token/activation length mismatch!"

    return {
        "text": text,
        "prompt_ids": prompt_ids_list,
        "prompt_tokens": prompt_tokens,  # 1:1 with prompt_activations
        "prompt_activations": prompt_activations,  # [T_prompt, H]
        "response_ids": response_ids_list,
        "response_tokens": response_tokens,  # 1:1 with activations
        "response_activations": response_activations,  # [T, H]
    }


# def compute_projection_scores(activations: torch.Tensor, direction: torch.Tensor, scale=1.0) -> torch.Tensor:
#     """
#     Compute cosine similarity between each activation and the direction vector
#     using sentence-transformers' cos_sim (range: [-1, 1]).
#
#     Args:
#         activations: [T, H]  token-level activations
#         direction:   [H]     steering or reference vector
#
#     Returns:
#         torch.Tensor [T] — cosine similarity per token (normalized)
#     """
#     # Align dtype/device
#     direction = direction.to(activations.device, dtype=activations.dtype)
#
#     # Ensure direction has batch dimension
#     direction = direction.unsqueeze(0)  # [1, H]
#
#     # cos_sim normalizes internally
#     scores = cos_sim(activations, direction).squeeze(1)  # [T, 1] → [T]
#
#     if isinstance(scores, torch.Tensor):
#         scores = scores.detach().cpu().to(torch.float32).numpy()
#     elif isinstance(scores, (list, tuple)) and any(isinstance(x, torch.Tensor) for x in scores):
#         scores = np.array([float(x.detach().cpu().to(torch.float32)) for x in scores], dtype=np.float32)
#     else:
#         scores = np.asarray(scores, dtype=np.float32)
#
#     return scores * scale

def compute_projection_scores(activations: torch.Tensor, direction: torch.Tensor) -> np.ndarray:
    """
    Compute scalar projection of each activation onto `direction`:
        score_i = <activations[i], dir_hat>
    where dir_hat is the L2-normalized `direction`.

    Args:
        activations: [T, H] token-level activations
        direction:   [H]    steering/reference vector (will be L2-normalized)
        scale:       float  optional multiplier for the returned scores

    Returns:
        np.ndarray [T] — projection per token (float32)
    """
    # align device/dtype
    direction = direction.to(activations.device, dtype=activations.dtype)
    # normalize direction to get true scalar projection
    direction = direction / (direction.norm(p=2) + 1e-12)  # [H]
    # dot product per token
    scores = activations @ direction  # [T]
    return scores.detach().cpu().to(torch.float32).numpy()


def clean_tokens(tokens, scores, special_mask=None):
    """Strip byte-prefix artifacts and drop tokens that become empty.

    If `special_mask` is provided, it is filtered in lockstep with the tokens
    and returned as the third value so it stays aligned with the cleaned
    tokens / scores. Note: cleaning still preserves chat-template special
    tokens like `<|im_start|>` because they have no leading Ġ/Ċ prefix.
    """
    cleaned_strs = [re.sub(r"^[ĠĊÂãĉâĢĶ]+", "", t).strip() for t in tokens]
    cleaned_tokens: List[str] = []
    cleaned_scores: List[Any] = []
    cleaned_mask: List[bool] = []
    iter_mask = special_mask if special_mask is not None else [False] * len(tokens)
    for t, s, m in zip(cleaned_strs, scores, iter_mask):
        if t == "":
            continue
        cleaned_tokens.append(t)
        cleaned_scores.append(s)
        cleaned_mask.append(bool(m))

    if special_mask is None:
        return cleaned_tokens, cleaned_scores
    return cleaned_tokens, cleaned_scores, cleaned_mask
