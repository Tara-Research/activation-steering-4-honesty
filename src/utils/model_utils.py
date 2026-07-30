import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _resolve_local_repo(repo_id_or_path: str) -> str:
    if os.path.isdir(repo_id_or_path):
        return repo_id_or_path

    if "/" in repo_id_or_path:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        slug = repo_id_or_path.replace("/", "--")
        repo_dir = os.path.join(hf_home, "hub", f"models--{slug}")
        refs_main = os.path.join(repo_dir, "refs", "main")
        if os.path.isfile(refs_main):
            with open(refs_main, "r") as f:
                commit = f.read().strip()
            snapshot = os.path.join(repo_dir, "snapshots", commit)
            if os.path.isdir(snapshot):
                return snapshot

    return snapshot_download(
        repo_id=repo_id_or_path,
        local_files_only=True,
    )

def _load_tokenizer(path_or_id: str):
    local_path = _resolve_local_repo(path_or_id)
    tok = AutoTokenizer.from_pretrained(local_path, local_files_only=True, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    return tok

def prompt_response_boundary(base_ids, full_ids, *, label: str = None) -> int:
    """Return the prompt/response boundary as the longest common prefix length
    of `base_ids` (prompt-only chat-template tokenization, typically with
    `add_generation_prompt=True`) and `full_ids` (full-conversation chat-template
    tokenization with the assistant turn included, `add_generation_prompt=False`).

    For prefix-stable templates (Llama, Qwen) this equals `len(base_ids)`. For
    templates that emit different scaffolding when the assistant turn is/isn't
    present (notably gemma's `<|channel>` block under `add_generation_prompt=True`),
    LCP is strictly less than `len(base_ids)` and points to the first assistant
    content token in `full_ids`.

    If `label` is provided and divergence is detected, prints a one-line warning
    showing the index and the diverging token ids. Pass `label=None` to stay silent.
    """
    n = min(len(base_ids), len(full_ids))
    lcp = next((i for i in range(n) if base_ids[i] != full_ids[i]), n)
    if label is not None and lcp < len(base_ids):
        print(
            f"[{label}] chat template not strictly prefix-stable; using "
            f"prompt_len=LCP={lcp} instead of len(base)={len(base_ids)}. "
            f"diverging tokens: base[{lcp}]={base_ids[lcp]} vs "
            f"full[{lcp}]={full_ids[lcp]}"
        )
    return lcp


def _apply_chat_template_ids(tokenizer, msgs, *, add_generation_prompt: bool):
    """Tokenize messages via chat template, returning a flat List[int].

    Defends against the BatchEncoding/dict return-type drift seen with some
    `trust_remote_code=True` tokenizers (e.g. Qwen3.6) where `return_dict=False`
    is silently ignored.
    """
    ids = tokenizer.apply_chat_template(
        msgs,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
        return_dict=False,
    )
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    # Some templates wrap single-sequence output in an extra list.
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def compute_prompt_response_spans(
    tokenizer,
    msgs_prompt,    # system + user (no assistant role)
    msgs_full,      # system + user + assistant
    *,
    label: str = None,
):
    """Compute the two token-span boundaries needed to separate prompt, generation-
    prompt scaffolding, and assistant content within `full_ids`.

    Performs three chat-template tokenizations:
        prompt_only_ids = apply_chat_template(msgs_prompt, add_generation_prompt=False)
        base_ids        = apply_chat_template(msgs_prompt, add_generation_prompt=True)
        full_ids        = apply_chat_template(msgs_full,   add_generation_prompt=False)

    Returns ``(special_start, content_start, full_ids)`` where:

      - ``special_start = LCP(prompt_only_ids, full_ids)`` — index in ``full_ids``
        where the generation-prompt scaffolding begins. Equivalent to
        ``len(prompt_only_ids)`` for prefix-stable templates. End of the prompt
        span / start of the response span under the new definition.
      - ``content_start = LCP(base_ids, full_ids)`` (= legacy
        ``prompt_response_boundary``) — index where the first natural assistant
        content token sits.

    Always passes ``enable_thinking=False`` (matches existing call sites).
    Raises ``ValueError`` if ``special_start > content_start`` (template
    inconsistency — should never happen on a sane chat template).
    """
    prompt_only_ids = _apply_chat_template_ids(tokenizer, msgs_prompt, add_generation_prompt=False)
    base_ids = _apply_chat_template_ids(tokenizer, msgs_prompt, add_generation_prompt=True)
    full_ids = _apply_chat_template_ids(tokenizer, msgs_full, add_generation_prompt=False)

    special_start = prompt_response_boundary(prompt_only_ids, full_ids, label=label)
    content_start = prompt_response_boundary(base_ids, full_ids, label=label)

    if special_start > content_start:
        raise ValueError(
            f"compute_prompt_response_spans: special_start ({special_start}) > "
            f"content_start ({content_start}); chat template is inconsistent."
        )

    return special_start, content_start, full_ids


def compute_prompt_response_spans_from_text(
    tokenizer,
    msgs_prompt,        # system + user (no assistant role)
    response_text: str,
    *,
    label: str = None,
    append_eot: bool = False,
):
    """Template-agnostic drop-in replacement for ``compute_prompt_response_spans``.

    ``compute_prompt_response_spans`` reconstructs the generation-time sequence
    by re-rendering the *full* conversation with ``add_generation_prompt=False``.
    That is only correct for prefix-stable templates (Llama, Qwen). For gemma
    the assistant-turn render differs from the ``add_generation_prompt=True``
    scaffolding, so the generation-prompt tokens the model actually conditioned
    on are missing from ``full_ids`` entirely.

    This variant instead builds the exact sequence ``model.generate()`` saw:

        full_str = render(msgs_prompt, add_generation_prompt=True) + response_text

    ``base_str`` is *literally* the generation prefix, so the scaffolding is
    guaranteed present in ``full_ids`` for every model family. The response is
    tokenized in-context (joined string, not separately) so the prompt/response
    boundary token matches what autoregressive generation produced.

    Returns ``(special_start, content_start, full_ids)`` with the same meaning
    as ``compute_prompt_response_spans``. Raises ``ValueError`` (no silent
    fallback) if the tokenizer re-segments across the prompt/response join.
    """
    prompt_only_str = tokenizer.apply_chat_template(
        msgs_prompt, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    base_str = tokenizer.apply_chat_template(
        msgs_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    full_str = base_str + response_text
    if append_eot:
        if tokenizer.eos_token is None:
            raise ValueError("append_eot=True but tokenizer has no eos_token")
        full_str = full_str + tokenizer.eos_token

    def _enc(s: str):
        # The rendered string already contains BOS / role markers as literal
        # special-token text; do not let the tokenizer add specials again.
        return list(tokenizer(s, add_special_tokens=False)["input_ids"])

    prompt_only_ids = _enc(prompt_only_str)
    base_ids = _enc(base_str)
    full_ids = _enc(full_str)

    # base_str is a string prefix of full_str by construction, so base_ids must
    # be a token-prefix of full_ids. If not, the tokenizer re-segmented across
    # the join (rare) — fail loud rather than slice a misaligned span.
    if full_ids[: len(base_ids)] != base_ids:
        raise ValueError(
            f"compute_prompt_response_spans_from_text[{label}]: tokenizer is not "
            f"prefix-stable across the prompt/response join "
            f"(len(base)={len(base_ids)}); cannot derive a clean response span."
        )

    special_start = prompt_response_boundary(prompt_only_ids, full_ids, label=label)
    content_start = len(base_ids)  # exact: base_ids is a verified prefix of full_ids

    if special_start > content_start:
        raise ValueError(
            f"compute_prompt_response_spans_from_text: special_start "
            f"({special_start}) > content_start ({content_start}); "
            f"chat template is inconsistent."
        )

    return special_start, content_start, full_ids


def get_num_hidden_layers(model) -> int:
    cfg = model.config
    if hasattr(cfg, "num_hidden_layers"):
        return cfg.num_hidden_layers
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
        return cfg.text_config.num_hidden_layers
    raise AttributeError(
        f"Cannot resolve num_hidden_layers from {type(cfg).__name__}"
    )


def get_hidden_size(model) -> int:
    cfg = model.config
    if hasattr(cfg, "hidden_size"):
        return cfg.hidden_size
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return cfg.text_config.hidden_size
    raise AttributeError(
        f"Cannot resolve hidden_size from {type(cfg).__name__}"
    )


LAYER_ATTR_PATHS = (
    "transformer.h",                # GPT-2/Neo, Bloom
    "encoder.layer",                # BERT/RoBERTa
    "model.layers",                 # Llama/Mistral/Qwen/Gemma text-only
    "gpt_neox.layers",              # GPT-NeoX
    "block",                        # Flan-T5
    "model.language_model.layers",  # Gemma-4 ForConditionalGeneration (and similar)
)


def locate_layer(model, layer_idx: int, *, debug: bool = False, label: str = "locate_layer"):
    """
    Walk ``LAYER_ATTR_PATHS`` to find the transformer block list on ``model``
    and return the block at ``layer_idx`` (supports negative indexing).

    Raises:
        IndexError: layer_idx out of range for the located layer list.
        ValueError: no known layer-attribute path matched on this model.
    """
    for path in LAYER_ATTR_PATHS:
        cur = model
        for part in path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                break
        else:
            if not hasattr(cur, "__getitem__"):
                continue
            if not (-len(cur) <= layer_idx < len(cur)):
                raise IndexError("layer_idx out of range")
            if debug:
                print(f"[{label}] hooking {path}[{layer_idx}]")
            return cur[layer_idx]
    raise ValueError(
        "Could not find layer list on the model. "
        "Add the attribute name to LAYER_ATTR_PATHS in src/utils/model_utils.py if needed."
    )


def load_model(model_id_or_path: str, dtype=None):
    local_path = _resolve_local_repo(model_id_or_path)

    # Auto-detect dtype based on device
    if dtype is None:
        if torch.backends.mps.is_available():
            dtype = torch.float16  # MPS doesn't support bfloat16
        elif torch.cuda.is_available():
            dtype = torch.bfloat16  # CUDA supports bfloat16
        else:
            dtype = torch.float32  # CPU fallback

    # Set device_map based on available hardware
    if torch.backends.mps.is_available():
        device_map = "mps"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        torch_dtype=dtype,
        device_map=device_map,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tok = _load_tokenizer(local_path)
    return model, tok


def load_model_with_lora(
    base_model_id_or_path: str,
    lora_adapter_repo_or_path: str,
    dtype=None,
):
    """Load a base model and merge a LoRA adapter into it.

    Returns a model whose architecture matches the base, so ``locate_layer``
    and the steering forward hooks operate without modification. The merged
    model is read-only on its adapter (no swap path).

    Args:
        base_model_id_or_path: HuggingFace repo or local path for the base model.
        lora_adapter_repo_or_path: HuggingFace repo or local path for the LoRA.
        dtype: Forwarded to ``load_model``.

    Returns:
        (model, tokenizer) tuple. The tokenizer is the base model's tokenizer.
    """
    from peft import PeftModel

    model, tok = load_model(base_model_id_or_path, dtype=dtype)
    adapter_path = _resolve_local_repo(lora_adapter_repo_or_path)
    peft_model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    merged = peft_model.merge_and_unload()
    return merged, tok
