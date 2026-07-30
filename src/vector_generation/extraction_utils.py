"""
Shared helpers for response generation and steering-vector extraction.

The only supported activation feature is the response average: for each
sample, hidden states are averaged over all assistant-response tokens
(role marker + content) at every layer. Artifact filenames carry the
``response_avg`` infix via ``MODE``.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import yaml
from tqdm import tqdm

from src.utils.model_utils import (
    compute_prompt_response_spans_from_text,
    get_num_hidden_layers,
)

# The only supported activation aggregation; appears in artifact filenames.
MODE = "response_avg"


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "_")


def ensure_padding(tokenizer) -> None:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


# =====================================================================
# COMPASSION SYSTEM-PROMPT POOL (5 positive + 5 negative)
# =====================================================================

def load_compassion_system_prompts(
    yaml_path: Union[str, Path],
    *,
    pos_key: str = "compassionate_system_prompts",
    neg_key: str = "dismissive_system_prompts",
) -> Dict[str, Dict[str, str]]:
    """
    Load the compassion system-prompt pool YAML.

    Expected YAML structure:
      compassionate_system_prompts:
        - id: compassionate_*_v1
          prompt: | ...
      dismissive_system_prompts:
        - id: dismissive_*_v1
          prompt: | ...

    Returns:
      {
        "pos": {prompt_id: prompt_text, ...},   # insertion order = YAML order
        "neg": {prompt_id: prompt_text, ...},
      }
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    pos_list = cfg.get(pos_key, [])
    neg_list = cfg.get(neg_key, [])

    if not isinstance(pos_list, list) or not isinstance(neg_list, list):
        raise ValueError(f"YAML must contain list keys: {pos_key} and {neg_key}.")

    def _to_map(lst: List[Dict[str, Any]], name: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for i, item in enumerate(lst):
            if not isinstance(item, dict):
                raise ValueError(f"{name}[{i}] is not a dict.")
            pid = item.get("id")
            ptxt = item.get("prompt")
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError(f"{name}[{i}] missing valid 'id'.")
            if not isinstance(ptxt, str) or not ptxt.strip():
                raise ValueError(f"{name}[{i}] missing valid 'prompt'.")
            out[pid.strip()] = ptxt.strip()

        if len(out) != 5:
            raise ValueError(f"{name} must contain exactly 5 prompts; got {len(out)}.")
        return out

    return {
        "pos": _to_map(pos_list, pos_key),
        "neg": _to_map(neg_list, neg_key),
    }


# =====================================================================
# EXTRACTION (IN ORDER, RETURNS KEPT IDS)
# =====================================================================

def forward_pass_response_span(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    response: str,
) -> Optional[Tuple[Any, int, int]]:
    """
    One forward pass over the full (system, user, assistant) sequence.

    The response span is ``[special_start, end)``: role marker +
    generation-prompt scaffolding + assistant content (see
    ``compute_prompt_response_spans_from_text``).

    Returns ``(outputs, abs_start, abs_end)`` or ``None`` if the sample is
    skipped (empty response or degenerate span). Caller must ``del outputs``
    when done.
    """
    response = str(response).strip()
    if not response:
        return None

    model.eval()
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        raise ValueError(
            "Tokenizer has no chat template; response-span extraction requires one."
        )

    msgs_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    special_start, _content_start, full_ids = compute_prompt_response_spans_from_text(
        tokenizer, msgs_prompt, response, label="vector_extraction",
    )
    if special_start <= 0 or special_start > len(full_ids):
        return None

    full_input_ids = torch.tensor([full_ids], device=model.device, dtype=torch.long)
    inputs = {
        "input_ids": full_input_ids,
        "attention_mask": torch.ones_like(full_input_ids),
    }

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    abs_start = special_start
    abs_end = len(full_ids)
    T = outputs.hidden_states[0].shape[1]
    if abs_end > T or abs_end <= abs_start:
        del outputs
        return None

    return outputs, abs_start, abs_end


def extract_response_embeddings_in_order(
    model,
    tokenizer,
    samples: List[Dict[str, Any]],
    per_token_layers: Optional[List[int]] = None,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, List[torch.Tensor]], List[str], int]:
    """
    Extract per-sample response-average hidden states in sample order.

    Each sample dict must carry ``pair_id``, ``system_prompt``,
    ``user_prompt``, and ``response``.

    Layer convention: 0-indexed transformer block, matching
    ``model.layers[layer]`` and steering's ``layer_idx``. We access
    ``outputs.hidden_states[layer + 1]`` so that ``layer=0`` is the output of
    the first transformer block.

    Returns:
      avg_by_layer[layer] -> Tensor [N_used, D] (mean over response tokens)
      per_token[layer]    -> List[Tensor [T_i, D]] (only for per_token_layers)
      kept_ids            -> List[str] (pair_id aligned to tensor rows)
      skipped             -> int
    """
    model.eval()
    layer_list = list(range(get_num_hidden_layers(model)))
    per_token_set = set(per_token_layers or [])

    avg_acc: Dict[int, List[torch.Tensor]] = {l: [] for l in layer_list}
    per_token_acc: Dict[int, List[torch.Tensor]] = {l: [] for l in per_token_set}
    kept_ids: List[str] = []
    skipped = 0

    for s in tqdm(samples, desc="Extracting response-avg hidden states (in-order)"):
        pair_id = str(s.get("pair_id", "")).strip()
        if not pair_id:
            skipped += 1
            continue

        result = forward_pass_response_span(
            model, tokenizer, s["system_prompt"], s["user_prompt"], s["response"],
        )
        if result is None:
            skipped += 1
            continue

        outputs, abs_start, abs_end = result
        kept_ids.append(pair_id)
        for l in layer_list:
            hs = outputs.hidden_states[l + 1]      # [1, T, D]
            slice_ = hs[:, abs_start:abs_end, :]   # [1, T_span, D]
            avg_acc[l].append(slice_.mean(dim=1).detach().cpu())  # [1, D]
            if l in per_token_set:
                per_token_acc[l].append(slice_[0].detach().cpu())  # [T_span, D]
        del outputs

    avg_by_layer = {l: torch.cat(rows, dim=0) for l, rows in avg_acc.items() if rows}
    return avg_by_layer, per_token_acc, kept_ids, skipped


# =====================================================================
# CAA VECTOR
# =====================================================================

def compute_paired_mean_diff_vector(
    pos_avg: Dict[int, torch.Tensor],
    pos_ids: List[str],
    neg_avg: Dict[int, torch.Tensor],
    neg_ids: List[str],
) -> torch.Tensor:
    """
    Computes per-layer mean of (pos - neg) over matched pair_ids.
    Returns a tensor [L, D] where L is the number of layers present in both.
    """
    p_map = {pid: i for i, pid in enumerate(pos_ids)}
    n_map = {pid: i for i, pid in enumerate(neg_ids)}
    common = sorted(set(p_map.keys()) & set(n_map.keys()))
    if not common:
        raise RuntimeError("No overlapping pair_ids between pos and neg after skipping.")

    layers = sorted(set(pos_avg.keys()) & set(neg_avg.keys()))
    if not layers:
        raise RuntimeError("No overlapping layers between pos and neg embeddings.")

    diffs_per_layer: List[torch.Tensor] = []
    for l in layers:
        P = pos_avg[l].float()
        N = neg_avg[l].float()

        p_rows = torch.stack([P[p_map[pid]] for pid in common], dim=0)  # [N, D]
        n_rows = torch.stack([N[n_map[pid]] for pid in common], dim=0)  # [N, D]
        diffs_per_layer.append((p_rows - n_rows).mean(dim=0))           # [D]

    return torch.stack(diffs_per_layer, dim=0)  # [L, D]


# =====================================================================
# SAVING
# =====================================================================

def _stack_layers(avg_by_layer: Dict[int, torch.Tensor]) -> torch.Tensor:
    layers = sorted(avg_by_layer.keys())
    return torch.stack([avg_by_layer[l].float() for l in layers], dim=0)  # [L, N, D]


def save_steering_vector(
    vec: torch.Tensor,
    save_dir: Union[str, Path],
    trait: str,
    dataset: str,
    model_id: str,
) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out_path = save_dir / f"{trait}_{dataset}_steering_vector_{MODE}_{model_slug(model_id)}.pt"
    torch.save(vec, out_path)
    print(f"Saved steering vector to: {out_path}")
    return out_path


def save_side_embeddings(
    pos_avg: Dict[int, torch.Tensor],
    neg_avg: Dict[int, torch.Tensor],
    save_dir: Union[str, Path],
    trait: str,
    dataset: str,
    model_id: str,
    side_names: Tuple[str, str],
) -> None:
    """
    Save per-side response-average embeddings as [L, N, D] tensors:
      {trait}_{dataset}_{side}_embeddings_response_avg_{model_slug}.pt
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for side_name, avg_by_layer in zip(side_names, (pos_avg, neg_avg)):
        out_path = save_dir / f"{trait}_{dataset}_{side_name}_embeddings_{MODE}_{model_slug(model_id)}.pt"
        torch.save(_stack_layers(avg_by_layer), out_path)
        print(f"Saved {side_name} embeddings to: {out_path}")


def save_per_token_embeddings(
    pos_per_token: Dict[int, List[torch.Tensor]],
    neg_per_token: Dict[int, List[torch.Tensor]],
    save_dir: Union[str, Path],
    trait: str,
    dataset: str,
    model_id: str,
    side_names: Tuple[str, str],
) -> None:
    """
    Save ragged per-token response embeddings (dict[layer] -> List[Tensor[T_i, D]]):
      per_token/{trait}_{dataset}_{side}_per_token_embeddings_{model_slug}.pt
    """
    save_dir = Path(save_dir) / "per_token"
    save_dir.mkdir(parents=True, exist_ok=True)

    for side_name, per_token in zip(side_names, (pos_per_token, neg_per_token)):
        out_path = save_dir / f"{trait}_{dataset}_{side_name}_per_token_embeddings_{model_slug(model_id)}.pt"
        torch.save(per_token, out_path)
        print(f"Saved {side_name} per-token embeddings to: {out_path}")
