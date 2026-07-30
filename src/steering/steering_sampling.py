import gc
from typing import Sequence, Dict, Any, Optional, List

import torch
from tqdm import trange
from transformers import StoppingCriteria

from src.steering.sttp import (
    SteeringToTargetProjection,
    SteeringToTargetProjectionMultipleLayers,
)
from src.steering.swfc import SteeringWithFixedCoef, SteeringWithFixedCoefMultiple
from src.steering.stmp import (
    SteeringToMirrorProjection,
    SteeringToMirrorProjectionMultipleLayers,
)


# class StopOnEOT(StoppingCriteria):
#     def __init__(self, eot_id: int):
#         self.eot_id = int(eot_id)
#
#     def __call__(self, input_ids, scores, **kwargs) -> bool:
#         # stop only once all batch items ended with EOT
#         return torch.all(input_ids[:, -1] == self.eot_id).item()


def sample_swfc(model, tokenizer, conversations, prompts, vector, layer, coef, bs=20, top_p=1, max_tokens=512,
                    temperature=1, min_tokens=1, steering_type="response", enable_thinking=False):
    """Generate model responses with activation steering applied.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        conversations: List of conversation messages
        vector: Steering vector to apply
        layer: 0-based layer index where steering is applied
        coef: Coefficient for steering strength
        bs: Batch size for generation
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        min_tokens: Minimum tokens to generate
        steering_type: Type of steering ("response" or other)

    Returns:
        generated outputs
    """
    # Configure tokenizer padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Convert conversations to prompts using chat template
    if prompts is None:
        prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for messages in conversations
        ]

    # Generate responses in batches
    outputs = []
    for i in trange(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        input_device = model.get_input_embeddings().weight.device
        tokenized_batch = {k: v.to(input_device, non_blocking=True) for k, v in tokenized_batch.items()}

        # Apply activation steering during generation
        with SteeringWithFixedCoef(model, vector, coeff=coef, layer_idx=layer, positions=steering_type):
            with torch.no_grad():
                generated = model.generate(
                    **tokenized_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    min_new_tokens=min_tokens,
                )

        # Decode only the generated part (excluding prompt)
        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in generated]
        outputs.extend(output)
        del output, tokenized_batch

    return outputs

def sample_multi_layer_swfc(model, tokenizer, conversations, prompts, vectors, layers: list[int],
                                          coefficient: float=1.0, bs=20, top_p=1, max_tokens=512,
                                          temperature=1, min_tokens=1, steering_type="response", enable_thinking=False):
    """Generate model responses with projection steering applied.

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        conversations: List of conversation messages
        vectors: Steering vector to apply
        layers: Layer indices where steering is applied
        bs: Batch size for generation
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        min_tokens: Minimum tokens to generate
        steering_type: Type of steering ("response" or other)

    Returns:
        generated outputs
    """
    # Configure tokenizer padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Convert conversations to prompts using chat template
    if prompts is None:
        prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for messages in conversations
        ]

    instructions = []
    for layer in layers:
        instructions.append({
            "layer_idx": layer,
            "steering_vector": vectors[layer],
            "coeff": coefficient,
            "positions": steering_type
        })

    # Generate responses in batches
    outputs = []
    for i in trange(0, len(prompts), bs):
        batch = prompts[i:i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        input_device = model.get_input_embeddings().weight.device
        tokenized_batch = {k: v.to(input_device, non_blocking=True) for k, v in tokenized_batch.items()}

        # Apply activation steering during generation
        with SteeringWithFixedCoefMultiple(model=model, instructions=instructions):
            with torch.no_grad():
                generated = model.generate(
                    **tokenized_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    min_new_tokens=min_tokens,
                )

        # Decode only the generated part (excluding prompt)
        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in generated]
        outputs.extend(output)
        del output, tokenized_batch

    return outputs

def sample_sttp(
    model,
    tokenizer,
    conversations: Sequence[Sequence[dict]],
    prompts: Optional[List],
    steering_vectors: torch.Tensor,
    negative_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    layer: int,
    steering_direction: str = "positive",
    coefficient: float = 0.0,
    bs: int = 20,
    top_p: float = 1.0,
    max_tokens: int = 512,
    temperature: float = 1.0,
    min_tokens: int = 1,
    steering_type: str = "response",
    enable_thinking=False,
    bias=None,
):
    """
    Generate model responses with *Steering to Target Projection (StTP)* applied at a given layer.

    Args:
        model:
            The language model (HF-style) to steer.
        tokenizer:
            Tokenizer corresponding to the model.
        conversations:
            List of conversations, where each conversation is a list of
            chat messages (dicts) compatible with `apply_chat_template`.
        steering_vectors:
            Tensor of shape [L, H]; one steering direction per layer.
        negative_embeddings:
            Tensor of shape [L, N_neg, H]; activations for the "low trait" class.
        positive_embeddings:
            Tensor of shape [L, N_pos, H]; activations for the "high trait" class.
        layer:
            0-based index of the transformer block at which to apply steering.
        steering_direction:
            "positive" – push projections up toward μ_pos + coefficient·σ_pos
            "negative" – push projections down toward μ_neg + coefficient·σ_neg
        coefficient:
            Multiplier for the std term in the target projection s.
        bs:
            Batch size for generation.
        top_p:
            Nucleus sampling parameter.
        max_tokens:
            Maximum number of new tokens to generate.
        temperature:
            Sampling temperature; if 0, greedy decoding.
        min_tokens:
            Minimum number of new tokens to generate.
        steering_type:
            Where to apply steering in the sequence:
              - "all"      → steer all time steps at that layer
              - "response" → steer only during decoding, skip prefill

    Returns:
        List of decoded model responses (one string per conversation).
    """
    # Configure tokenizer padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Convert conversations to prompts using chat template
    if prompts is None:
        prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for messages in conversations
        ]

    outputs = []

    for i in trange(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        input_device = model.get_input_embeddings().weight.device
        tokenized_batch = {k: v.to(input_device, non_blocking=True) for k, v in tokenized_batch.items()}

        # Apply StTP steering during generation
        with SteeringToTargetProjection(
            model,
            steering_vectors=steering_vectors,
            negative_embeddings=negative_embeddings,
            positive_embeddings=positive_embeddings,
            bias=bias,
            layer_idx=layer,
            positions=steering_type,          # "all" | "response"
            steering_direction=steering_direction,
            coefficient=coefficient,
            debug=False,
        ):
            with torch.no_grad():
                generated = model.generate(
                    **tokenized_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    min_new_tokens=min_tokens,
                )

        # Decode only the generated part (exclude prompt tokens)
        prompt_len = tokenized_batch["input_ids"].shape[1]
        decoded = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
            for o in generated
        ]
        outputs.extend(decoded)
        del generated, tokenized_batch

    return outputs


def sample_multi_layer_sttp(
    model,
    tokenizer,
    conversations: Sequence[Sequence[dict]],
    prompts: Optional[List],
    instructions: Sequence[Dict[str, Any]],
    bs: int = 20,
    top_p: float = 1.0,
    max_tokens: int = 512,
    temperature: float = 1.0,
    min_tokens: int = 1,
    enable_thinking=False
):
    """
    Generate model responses with *multi-layer Steering to Target Projection (StTP)* applied.

    This uses `SteeringToTargetProjectionMultipleLayers`, i.e. several
    SteeringToTargetProjection instances active at once (possibly at different
    layers, with different steering vectors / distributions).

    Args:
        model:
            HF-style language model to steer.
        tokenizer:
            Corresponding tokenizer.
        conversations:
            List of conversations, where each conversation is a list of
            chat messages (dicts) compatible with `tokenizer.apply_chat_template`.
        instructions:
            Sequence of dicts, one per steered configuration, each of the form:

                {
                    "steering_vectors": torch.Tensor [L, H],
                    "negative_embeddings": torch.Tensor [L, N_neg, H],
                    "positive_embeddings": torch.Tensor [L, N_pos, H],
                    "layer_idx": int,                     # 0-based index
                    "coefficient": float,                 # optional, default 0.0
                    "steering_direction": "positive"
                                          or "negative",   # optional, default "positive"
                    "positions": "all" | "response",  # optional, default "all"
                }

            These are passed directly to `SteeringToTargetProjectionMultipleLayers`.

        bs:
            Batch size for generation.
        top_p:
            Nucleus sampling parameter.
        max_tokens:
            Maximum number of new tokens to generate.
        temperature:
            Sampling temperature; if 0, greedy decoding.
        min_tokens:
            Minimum number of new tokens to generate.

    Returns:
        List of decoded model responses (one string per conversation).
    """
    # Configure tokenizer padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Convert conversations to prompts using chat template
    if prompts is None:
        prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for messages in conversations
        ]

    outputs = []

    for i in trange(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        input_device = model.get_input_embeddings().weight.device
        tokenized_batch = {k: v.to(input_device, non_blocking=True) for k, v in tokenized_batch.items()}

        # Apply *multi-layer* StTP steering during generation
        with SteeringToTargetProjectionMultipleLayers(
            model=model,
            instructions=instructions,
            debug=False,
        ):
            with torch.no_grad():
                generated = model.generate(
                    **tokenized_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    min_new_tokens=min_tokens
                )

        # Decode only the generated part (exclude prompt tokens)
        prompt_len = tokenized_batch["input_ids"].shape[1]
        decoded = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
            for o in generated
        ]
        outputs.extend(decoded)
        del generated, tokenized_batch

    return outputs


def sample_stmp(
    model,
    tokenizer,
    conversations: Sequence[Sequence[dict]],
    prompts: Optional[List],
    steering_vectors: torch.Tensor,
    negative_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    layer: int,
    steering_direction: str = "positive",
    coefficient: float = 0.0,
    bs: int = 20,
    top_p: float = 1.0,
    max_tokens: int = 512,
    temperature: float = 1.0,
    min_tokens: int = 1,
    steering_type: str = "response",
    enable_thinking=False,
    bias=None,
):
    """
    Generate model responses with *Steering to Mirror Projection (StMP)* applied at a given layer.

    Args:
        model:
            The language model (HF-style) to steer.
        tokenizer:
            Tokenizer corresponding to the model.
        conversations:
            List of conversations, where each conversation is a list of
            chat messages (dicts) compatible with `apply_chat_template`.
        steering_vectors:
            Tensor of shape [L, H]; one steering direction per layer.
        negative_embeddings:
            Tensor of shape [L, N_neg, H]; activations for the "low trait" class.
        positive_embeddings:
            Tensor of shape [L, N_pos, H]; activations for the "high trait" class.
        layer:
            0-based index of the transformer block at which to apply steering.
            (Internally passed as `layer_idx=layer-1` to the steerer.)
        steering_direction:
            "positive" – push projections up toward μ_pos + coefficient·σ_pos
            "negative" – push projections down toward μ_neg + coefficient·σ_neg
        coefficient:
            Multiplier for the std term in the target projection s.
        bs:
            Batch size for generation.
        top_p:
            Nucleus sampling parameter.
        max_tokens:
            Maximum number of new tokens to generate.
        temperature:
            Sampling temperature; if 0, greedy decoding.
        min_tokens:
            Minimum number of new tokens to generate.
        steering_type:
            Where to apply steering in the sequence:
              - "all"      → steer all time steps at that layer
              - "response" → steer only during decoding, skip prefill

    Returns:
        List of decoded model responses (one string per conversation).
    """
    # Configure tokenizer padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Convert conversations to prompts using chat template
    if prompts is None:
        prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for messages in conversations
        ]

    outputs = []

    for i in trange(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, max_length=2048)

        input_device = model.get_input_embeddings().weight.device
        tokenized_batch = {k: v.to(input_device, non_blocking=True) for k, v in tokenized_batch.items()}

        # Apply StMP steering during generation
        with SteeringToMirrorProjection(
            model,
            steering_vectors=steering_vectors,
            negative_embeddings=negative_embeddings,
            positive_embeddings=positive_embeddings,
            bias=bias,
            layer_idx=layer,
            coefficient=coefficient,
            positions=steering_type,          # "all" | "response"
            steering_direction=steering_direction,
            debug=False,
        ):
            with torch.inference_mode():
                generated = model.generate(
                    **tokenized_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    min_new_tokens=min_tokens,
                )


        # Decode only the generated part (exclude prompt tokens)
        prompt_len = tokenized_batch["input_ids"].shape[1]
        decoded = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
            for o in generated
        ]

        gen_ids = generated[0, prompt_len:]
        outputs.extend(decoded)
        del generated, tokenized_batch
        
        gc.collect()
        torch.cuda.empty_cache()
        
    return outputs


def sample_multi_layer_stmp(
    model,
    tokenizer,
    conversations: Sequence[Sequence[dict]],
    prompts: Optional[List],
    instructions: Sequence[Dict[str, Any]],
    bs: int = 20,
    top_p: float = 1.0,
    max_tokens: int = 512,
    temperature: float = 1.0,
    min_tokens: int = 1,
    enable_thinking=False,
):
    """
    Generate model responses with *multi-layer Steering to Mirror Projection (StMP)* applied.

    This uses `SteeringToMirrorProjectionMultipleLayers`, i.e. several
    SteeringToMirrorProjection instances active at once (possibly at different
    layers, with different steering vectors / distributions).

    Args:
        model:
            HF-style language model to steer.
        tokenizer:
            Corresponding tokenizer.
        conversations:
            List of conversations, where each conversation is a list of
            chat messages (dicts) compatible with `tokenizer.apply_chat_template`.
        instructions:
            Sequence of dicts, one per steered configuration, each of the form:

                {
                    "steering_vectors": torch.Tensor [L, H],
                    "negative_embeddings": torch.Tensor [L, N_neg, H],
                    "positive_embeddings": torch.Tensor [L, N_pos, H],
                    "layer_idx": int,                     # 0-based index
                    "coefficient": float,                 # optional, default 0.0
                    "steering_direction": "positive"
                                          or "negative",   # optional, default "positive"
                    "positions": "all" | "response",  # optional, default "all"
                }

            These are passed directly to `SteeringToMirrorProjectionMultipleLayers`.

        bs:
            Batch size for generation.
        top_p:
            Nucleus sampling parameter.
        max_tokens:
            Maximum number of new tokens to generate.
        temperature:
            Sampling temperature; if 0, greedy decoding.
        min_tokens:
            Minimum number of new tokens to generate.

    Returns:
        List of decoded model responses (one string per conversation).
    """
    # Configure tokenizer padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Convert conversations to prompts using chat template
    if prompts is None:
        prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for messages in conversations
        ]

    outputs = []

    for i in trange(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        input_device = model.get_input_embeddings().weight.device
        tokenized_batch = {k: v.to(input_device, non_blocking=True) for k, v in tokenized_batch.items()}

        # Apply *multi-layer* StMP steering during generation
        with SteeringToMirrorProjectionMultipleLayers(
            model=model,
            instructions=instructions,
            debug=False,
        ):
            with torch.no_grad():
                generated = model.generate(
                    **tokenized_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    min_new_tokens=min_tokens
                )

        # Decode only the generated part (exclude prompt tokens)
        prompt_len = tokenized_batch["input_ids"].shape[1]
        decoded = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True)
            for o in generated
        ]
        outputs.extend(decoded)
        del generated, tokenized_batch

    return outputs


def sample(model, tokenizer, conversations, prompts, top_p=1.0, max_tokens=512,
           temperature=1.0, min_tokens=1, bs=20, enable_thinking=False):
    """
    Generate model responses without steering (using standard Hugging Face transformers).

    Args:
        model: AutoModelForCausalLM (already loaded on device)
        tokenizer: corresponding tokenizer
        conversations: list of messages [{"role": "...", "content": "..."}]
        top_p: nucleus sampling parameter
        max_tokens: maximum tokens to generate
        temperature: sampling temperature
        min_tokens: minimum tokens to generate
        bs: batch size

    Returns:
        generated_texts
    """
    # Make sure tokenizer and padding are configured
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Convert chat messages into text prompts
    if prompts is None:
        prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for messages in conversations
        ]

    outputs = []
    for i in trange(0, len(prompts), bs, desc="Generating (HF)"):
        batch = prompts[i:i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        input_device = model.get_input_embeddings().weight.device
        tokenized_batch = {k: v.to(input_device, non_blocking=True) for k, v in tokenized_batch.items()}

        with torch.no_grad():
            generated = model.generate(
                **tokenized_batch,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                use_cache=True
            )

        # Decode only the newly generated part
        prompt_len = tokenized_batch["input_ids"].shape[1]
        texts = [tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                 for seq in generated]
        outputs.extend(texts)
        del generated, tokenized_batch

    return outputs
