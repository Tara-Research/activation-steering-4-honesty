"""The judge inference engine.

This is the ONLY module in the repository that imports vLLM. Everything else in
``src/llm_judge`` (config, prompts, parsing) is importable without a GPU, which
is what lets the runners and their callers be exercised on a laptop.

Deliberately stateless with respect to prompts: callers pass fully-formed chat
conversations. Judges routinely swap system prompts mid-run (trait phase then
coherence phase; one rubric per role) and some callers build their own messages
entirely, so a mutable ``system_prompt`` attribute on the engine would just be a
shared variable everyone has to remember to set.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams

from src.constants.constants import SEED
from src.llm_judge.config import JudgeConfig
from src.utils.model_utils import set_seed

Conversation = Sequence[Dict[str, str]]


def _resolve_local_repo(repo_id_or_path: str) -> str:
    if os.path.isdir(repo_id_or_path):
        return repo_id_or_path
    return snapshot_download(repo_id=repo_id_or_path, local_files_only=True)


class VLLMJudge:
    """Batched chat inference against a locally cached judge model."""

    def __init__(self, config: JudgeConfig) -> None:
        set_seed(SEED)
        self.config = config
        self.chat_kwargs = {
            "chat_template_kwargs": {"reasoning_effort": config.reasoning_effort}
        }

        self.llm = LLM(
            model=_resolve_local_repo(config.model),
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            disable_custom_all_reduce=True,
            enforce_eager=True,
            seed=SEED,
        )
        self.tokenizer = self.llm.get_tokenizer()

        stop_tokens: List[str] = []
        if getattr(self.tokenizer, "eos_token", None):
            stop_tokens.append(self.tokenizer.eos_token)

        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            min_tokens=config.min_tokens,
            stop=stop_tokens or None,
            skip_special_tokens=True,
            seed=SEED,
        )

    def generate(
        self,
        conversations: Sequence[Conversation],
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """One output text per conversation, in input order.

        ``batch_size`` of None or <= 0 submits everything in a single call and
        lets vLLM schedule it -- that is the mega-batching the sweep depends on.
        A positive value chunks the submission, for callers that want to bound
        peak memory.
        """
        if not conversations:
            return []

        if batch_size is None or batch_size <= 0:
            outputs = self.llm.chat(
                messages=list(conversations),
                sampling_params=self.sampling_params,
                use_tqdm=False,
                **self.chat_kwargs,
            )
            return [o.outputs[0].text for o in outputs]

        results: List[str] = []
        for start in range(0, len(conversations), batch_size):
            chunk = list(conversations[start:start + batch_size])
            outputs = self.llm.chat(
                messages=chunk,
                sampling_params=self.sampling_params,
                use_tqdm=False,
                **self.chat_kwargs,
            )
            results.extend(o.outputs[0].text for o in outputs)
        return results
