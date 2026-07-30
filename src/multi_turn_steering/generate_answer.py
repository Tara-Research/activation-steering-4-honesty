from typing import List, Dict

import torch

from src.steering.sttp import (
    SteeringToTargetProjection,
    SteeringToTargetProjectionMultipleLayers,
)
from src.steering.swfc import SteeringWithFixedCoef, SteeringWithFixedCoefMultiple
from src.steering.stmp import (
    SteeringToMirrorProjection,
    SteeringToMirrorProjectionMultipleLayers,
)

Message = Dict[str, str]

def _ensure_padding(tokenizer):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id


def _render(tokenizer, messages: List[Message], enable_thinking=False) -> str:
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

def _generate_plain(
        model,
        tokenizer,
        messages: List[Message],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        min_new_tokens: int = 1,
        enable_thinking: bool = False,
) -> str:
    prompt = _render(tokenizer, messages, enable_thinking=enable_thinking)
    tokenized, prompt_len = _tokenize_prompt(model, tokenizer, prompt)

    with torch.no_grad():
        out = model.generate(
            **tokenized,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            use_cache=True,
        )

    new_tokens = out[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _generate_steered_assistant_sttp(
        model,
        tokenizer,
        messages: List[Message],
        *,
        steering_vectors,
        negative_embeddings,
        positive_embeddings,
        layer: int,
        steering_direction: str,
        coefficient: float,
        steering_type: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        min_new_tokens: int = 1,
        enable_thinking: bool = False,
        bias=None,
) -> str:
    prompt = _render(tokenizer, messages, enable_thinking=enable_thinking)
    tokenized, prompt_len = _tokenize_prompt(model, tokenizer, prompt)

    with SteeringToTargetProjection(
            model,
            steering_vectors=steering_vectors,
            negative_embeddings=negative_embeddings,
            positive_embeddings=positive_embeddings,
            layer_idx=layer,
            positions=steering_type,
            steering_direction=steering_direction,
            coefficient=coefficient,
            bias=bias,
            debug=False,
    ):
        with torch.no_grad():
            out = model.generate(
                **tokenized,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                use_cache=True,
            )

    new_tokens = out[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _generate_steered_assistant_stmp(
        model,
        tokenizer,
        messages: List[Message],
        *,
        steering_vectors,
        negative_embeddings,
        positive_embeddings,
        layer: int,
        coefficient: float,
        steering_direction: str,
        steering_type: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        min_new_tokens: int = 1,
        enable_thinking: bool = False,
        bias=None,
) -> str:
    prompt = _render(tokenizer, messages, enable_thinking=enable_thinking)
    tokenized, prompt_len = _tokenize_prompt(model, tokenizer, prompt)

    with SteeringToMirrorProjection(
            model,
            steering_vectors=steering_vectors,
            negative_embeddings=negative_embeddings,
            positive_embeddings=positive_embeddings,
            layer_idx=layer,
            coefficient=coefficient,
            positions=steering_type,
            steering_direction=steering_direction,
            bias=bias,
            debug=False,
    ):
        with torch.no_grad():
            out = model.generate(
                **tokenized,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                use_cache=True,
            )

    new_tokens = out[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _generate_steered_assistant_swfc(
        model,
        tokenizer,
        messages: List[Message],
        *,
        steering_vectors,
        layer: int,
        coefficient: float,
        steering_type: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        min_new_tokens: int = 1,
        enable_thinking: bool = False,
) -> str:
    prompt = _render(tokenizer, messages, enable_thinking=enable_thinking)
    tokenized, prompt_len = _tokenize_prompt(model, tokenizer, prompt)

    with SteeringWithFixedCoef(
            model,
            steering_vector=steering_vectors[layer],
            layer_idx=layer,
            positions=steering_type,
            coeff=coefficient,
            debug=False,
    ):
        with torch.no_grad():
            out = model.generate(
                **tokenized,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                use_cache=True,
            )

    new_tokens = out[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def _generate_steered_assistant_multi_layer_sttp(
        model,
        tokenizer,
        messages: List[Message],
        *,
        steering_vectors,
        negative_embeddings,
        positive_embeddings,
        layers: List[int],
        steering_direction: str,
        coefficient: float,
        steering_type: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        min_new_tokens: int = 1,
        enable_thinking: bool = False,
        bias=None,
) -> str:
    prompt = _render(tokenizer, messages, enable_thinking=enable_thinking)
    tokenized, prompt_len = _tokenize_prompt(model, tokenizer, prompt)

    instructions = [
        {
            "steering_vectors": steering_vectors,
            "negative_embeddings": negative_embeddings,
            "positive_embeddings": positive_embeddings,
            "layer_idx": layer,
            "coefficient": coefficient,
            "steering_direction": steering_direction,
            "positions": steering_type,
            "bias": bias,
        }
        for layer in layers
    ]

    with SteeringToTargetProjectionMultipleLayers(model, instructions=instructions):
        with torch.no_grad():
            out = model.generate(
                **tokenized,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                use_cache=True,
            )

    new_tokens = out[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _generate_steered_assistant_multi_layer_stmp(
        model,
        tokenizer,
        messages: List[Message],
        *,
        steering_vectors,
        negative_embeddings,
        positive_embeddings,
        layers: List[int],
        coefficient: float,
        steering_direction: str,
        steering_type: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        min_new_tokens: int = 1,
        enable_thinking: bool = False,
        bias=None,
) -> str:
    prompt = _render(tokenizer, messages, enable_thinking=enable_thinking)
    tokenized, prompt_len = _tokenize_prompt(model, tokenizer, prompt)

    instructions = [
        {
            "steering_vectors": steering_vectors,
            "negative_embeddings": negative_embeddings,
            "positive_embeddings": positive_embeddings,
            "layer_idx": layer,
            "coefficient": coefficient,
            "steering_direction": steering_direction,
            "positions": steering_type,
            "bias": bias,
        }
        for layer in layers
    ]

    with SteeringToMirrorProjectionMultipleLayers(model, instructions=instructions):
        with torch.no_grad():
            out = model.generate(
                **tokenized,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                use_cache=True,
            )

    new_tokens = out[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _generate_steered_assistant_multi_layer_swfc(
        model,
        tokenizer,
        messages: List[Message],
        *,
        steering_vectors,
        layers: List[int],
        coefficient: float,
        steering_type: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        min_new_tokens: int = 1,
        enable_thinking: bool = False,
) -> str:
    prompt = _render(tokenizer, messages, enable_thinking=enable_thinking)
    tokenized, prompt_len = _tokenize_prompt(model, tokenizer, prompt)

    instructions = [
        {
            "steering_vector": steering_vectors[layer],
            "coeff": coefficient,
            "layer_idx": layer,
            "positions": steering_type,
        }
        for layer in layers
    ]

    with SteeringWithFixedCoefMultiple(model, instructions=instructions):
        with torch.no_grad():
            out = model.generate(
                **tokenized,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                use_cache=True,
            )

    new_tokens = out[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def generate_steered_answer(
        model,
        tokenizer,
        steering_technique,
        messages: List[Message],
        *,
        steering_vectors,
        negative_embeddings,
        positive_embeddings,
        layer: int = None,
        layers: List[int] = None,
        steering_direction: str,
        coefficient: float,
        steering_type: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        min_new_tokens: int = 1,
        enable_thinking: bool = False,
        bias=None,
):
    multi_layer = layers is not None

    if steering_technique in ("no_steering", "aligned_baseline", "malicious_baseline"):
        return _generate_plain(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
    elif steering_technique == "swfc":
        if multi_layer:
            return _generate_steered_assistant_multi_layer_swfc(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                steering_vectors=steering_vectors,
                layers=layers,
                coefficient=coefficient,
                steering_type=steering_type,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                min_new_tokens=min_new_tokens,
                enable_thinking=enable_thinking,
            )
        return _generate_steered_assistant_swfc(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            steering_vectors=steering_vectors,
            layer=layer,
            coefficient=coefficient,
            steering_type=steering_type,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            min_new_tokens=min_new_tokens,
            enable_thinking=enable_thinking,
        )
    elif steering_technique == "sttp":
        if multi_layer:
            return _generate_steered_assistant_multi_layer_sttp(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                steering_vectors=steering_vectors,
                negative_embeddings=negative_embeddings,
                positive_embeddings=positive_embeddings,
                layers=layers,
                steering_direction=steering_direction,
                coefficient=coefficient,
                steering_type=steering_type,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=enable_thinking,
                bias=bias,
            )
        return _generate_steered_assistant_sttp(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            steering_vectors=steering_vectors,
            negative_embeddings=negative_embeddings,
            positive_embeddings=positive_embeddings,
            layer=layer,
            steering_direction=steering_direction,
            coefficient=coefficient,
            steering_type=steering_type,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
            bias=bias,
        )
    elif steering_technique == "stmp":
        if multi_layer:
            return _generate_steered_assistant_multi_layer_stmp(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                steering_vectors=steering_vectors,
                negative_embeddings=negative_embeddings,
                positive_embeddings=positive_embeddings,
                layers=layers,
                coefficient=coefficient,
                steering_direction=steering_direction,
                steering_type=steering_type,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=enable_thinking,
                bias=bias,
            )
        return _generate_steered_assistant_stmp(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            steering_vectors=steering_vectors,
            negative_embeddings=negative_embeddings,
            positive_embeddings=positive_embeddings,
            layer=layer,
            coefficient=coefficient,
            steering_direction=steering_direction,
            steering_type=steering_type,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
            bias=bias,
        )
    else:
        raise ValueError(f"steering technique {steering_technique} not implemented yet for multi turn conversations.")

def generate_plain(
        model,
        tokenizer,
        messages: List[Message],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        min_new_tokens: int = 1,
        enable_thinking: bool = False,
) -> str:
    prompt = _render(tokenizer, messages, enable_thinking=enable_thinking)
    tokenized, prompt_len = _tokenize_prompt(model, tokenizer, prompt)

    with torch.no_grad():
        out = model.generate(
            **tokenized,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            use_cache=True,
        )

    new_tokens = out[0, prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Batch generation helpers
# ---------------------------------------------------------------------------

def _render_batch(tokenizer, messages_list: List[List[Message]], enable_thinking: bool = False) -> List[str]:
    return [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for msgs in messages_list
    ]


def _tokenize_batch(model, tokenizer, prompts: List[str]):
    _ensure_padding(tokenizer)
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True)
    tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
    prompt_len = tokenized["input_ids"].shape[1]
    return tokenized, prompt_len


def _decode_batch(tokenizer, generated: torch.Tensor, prompt_len: int) -> List[str]:
    return [
        tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
        for seq in generated
    ]


def _generate_batch_inner(
        model,
        tokenizer,
        messages_list: List[List[Message]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        min_new_tokens: int = 1,
        enable_thinking: bool = False,
) -> List[str]:
    prompts = _render_batch(tokenizer, messages_list, enable_thinking=enable_thinking)
    tokenized, prompt_len = _tokenize_batch(model, tokenizer, prompts)

    with torch.no_grad():
        generated = model.generate(
            **tokenized,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            use_cache=True,
        )

    results = _decode_batch(tokenizer, generated, prompt_len)
    del generated, tokenized
    return results


def generate_plain_batch(
        model,
        tokenizer,
        messages_list: List[List[Message]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        min_new_tokens: int = 1,
        enable_thinking: bool = False,
) -> List[str]:
    return _generate_batch_inner(
        model=model,
        tokenizer=tokenizer,
        messages_list=messages_list,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        min_new_tokens=min_new_tokens,
        enable_thinking=enable_thinking,
    )


def generate_steered_batch(
        model,
        tokenizer,
        steering_technique: str,
        messages_list: List[List[Message]],
        *,
        steering_vectors,
        negative_embeddings=None,
        positive_embeddings=None,
        layer: int,
        steering_direction: str = "positive",
        coefficient: float,
        steering_type: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        min_new_tokens: int = 1,
        enable_thinking: bool = False,
        bias=None,
) -> List[str]:
    """Batched generation with steering hooks applied uniformly to all items.

    All items in *messages_list* must share the same steering configuration
    (technique, layer, coefficient, etc.) so that a single set of hooks can
    be registered on the model for the entire batch.
    """
    prompts = _render_batch(tokenizer, messages_list, enable_thinking=enable_thinking)
    tokenized, prompt_len = _tokenize_batch(model, tokenizer, prompts)

    if steering_technique == "swfc":
        ctx = SteeringWithFixedCoef(
            model,
            steering_vector=steering_vectors[layer],
            layer_idx=layer,
            positions=steering_type,
            coeff=coefficient,
            debug=False,
        )
    elif steering_technique == "sttp":
        ctx = SteeringToTargetProjection(
            model,
            steering_vectors=steering_vectors,
            negative_embeddings=negative_embeddings,
            positive_embeddings=positive_embeddings,
            layer_idx=layer,
            positions=steering_type,
            steering_direction=steering_direction,
            coefficient=coefficient,
            bias=bias,
            debug=False,
        )
    elif steering_technique == "stmp":
        ctx = SteeringToMirrorProjection(
            model,
            steering_vectors=steering_vectors,
            negative_embeddings=negative_embeddings,
            positive_embeddings=positive_embeddings,
            layer_idx=layer,
            coefficient=coefficient,
            positions=steering_type,
            steering_direction=steering_direction,
            bias=bias,
            debug=False,
        )
    else:
        raise ValueError(
            f"generate_steered_batch: unknown technique '{steering_technique}'. "
            f"Must be one of: swfc, sttp, stmp"
        )

    with ctx:
        with torch.no_grad():
            generated = model.generate(
                **tokenized,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                use_cache=True,
            )

    results = _decode_batch(tokenizer, generated, prompt_len)
    del generated, tokenized
    return results

