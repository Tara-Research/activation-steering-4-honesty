"""Judge system prompts and chat-message construction.

Prompt files live in ``src/llm_judge/system_prompts/`` and are referenced by the
short names in ``configs/config_llm_judge.yaml -> system_prompt_paths``. Every
prompt YAML has exactly one top-level key, ``system_prompt``.

Calling convention (uniform across every rubric in this package): the rubric goes
in the **system** message with its ``{question}``/``{answer}`` placeholders left
literal, and the actual data goes in a **user** message built from one of the
DATA_TEMPLATEs below. Rubrics therefore read as instructions, not as f-strings.

Imports no vLLM, so this module is safe to import without a GPU.
"""
from __future__ import annotations

import os
from typing import Dict, List

import yaml

from src.constants.constants import DATA_FOLDER, ROOT_FOLDER
from src.utils.path_utils import load_llm_judge_config

TRAIT_DESCRIPTIONS_PATH = os.path.join(DATA_FOLDER, "trait_descriptions.yaml")


# ---------------------------------------------------------------------------
# Data templates — the user message for each rubric family
# ---------------------------------------------------------------------------

DATA_TEMPLATE = """
[QUESTION START]
{question}
[QUESTION END]

Model Response:
[ANSWER START]
{answer}
[ANSWER END]
""".strip()

DATA_TEMPLATE_CAA = """
GOLD TRUTH:
{gold_truth}

USER MESSAGE:
{user_message}

CONDITION:
{condition}

CATEGORY:
{category}

MODEL RESPONSE:
{model_response}
""".strip()

DATA_TEMPLATE_HONESTY = """
GOLD TRUTH:
{gold_truth}

USER MESSAGE:
{user_message}

CATEGORY:
{category}

MODEL RESPONSE:
{model_response}
""".strip()


# ---------------------------------------------------------------------------
# Partial formatting — leaves unknown placeholders literal
# ---------------------------------------------------------------------------

class _Safe(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def partial_format(template: str, **kwargs) -> str:
    """``str.format`` that re-emits any placeholder it has no value for.

    Load-bearing: the trait rubrics carry ``{question}``/``{answer}`` blocks that
    must survive trait substitution and reach the judge as literal text.
    """
    return template.format_map(_Safe(**kwargs))


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def prompt_path(name: str) -> str:
    """Absolute path of the prompt registered under ``name``. KeyError if unknown."""
    _, paths = load_llm_judge_config()
    if name not in paths:
        raise KeyError(
            f"Unknown judge prompt {name!r}. "
            f"Known prompts: {sorted(paths)}"
        )
    return os.path.join(ROOT_FOLDER, paths[name])


def load_prompt(name: str) -> str:
    """Read the rubric registered under ``name``."""
    path = prompt_path(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"System prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "system_prompt" not in data:
        raise KeyError(
            f"Prompt file {path} has no 'system_prompt' key (keys: {sorted(data)})"
        )
    text = data["system_prompt"].strip()
    if not text:
        raise ValueError(f"System prompt file is empty: {path}")
    return text


def trait_description(trait: str) -> Dict[str, str]:
    """``{'trait': ..., 'trait_description': ...}`` from data/trait_descriptions.yaml."""
    with open(TRAIT_DESCRIPTIONS_PATH, "r", encoding="utf-8") as f:
        descriptions = yaml.safe_load(f)
    if trait not in descriptions:
        raise KeyError(
            f"No description for trait {trait!r} in {TRAIT_DESCRIPTIONS_PATH}. "
            f"Known traits: {sorted(descriptions)}"
        )
    return descriptions[trait]


def load_trait_prompt(name: str, trait: str) -> str:
    """Rubric ``name`` with {TRAIT}/{TRAIT_DESCRIPTION} filled in for ``trait``."""
    description = trait_description(trait)
    return partial_format(
        load_prompt(name),
        TRAIT=description["trait"],
        TRAIT_DESCRIPTION=description["trait_description"],
    )


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def build_messages(system_prompt: str, user_text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]


def build_qa_messages(system_prompt: str, question: str, answer: str) -> List[Dict[str, str]]:
    """Generic trait / coherence / consistency scoring of one (question, answer)."""
    return build_messages(
        system_prompt,
        DATA_TEMPLATE.format(question=question, answer=answer),
    )


def build_honesty_messages(
    system_prompt: str,
    *,
    gold_truth: str,
    user_message: str,
    model_response: str,
    category: str = "",
) -> List[Dict[str, str]]:
    """Honesty scoring against a gold truth, with no HONEST/DISHONEST condition."""
    return build_messages(
        system_prompt,
        DATA_TEMPLATE_HONESTY.format(
            gold_truth=gold_truth.strip(),
            user_message=user_message.strip(),
            category=(category or "").strip(),
            model_response=(model_response or "").strip(),
        ),
    )


def build_caa_messages(
    system_prompt: str,
    *,
    gold_truth: str,
    user_message: str,
    condition: str,
    model_response: str,
    category: str = "",
) -> List[Dict[str, str]]:
    """CAA training-data annotation: does the response match its assigned role?"""
    condition_norm = condition.strip().upper()
    if condition_norm not in {"HONEST", "DISHONEST"}:
        raise ValueError(f"Invalid condition={condition!r}. Expected HONEST or DISHONEST.")

    return build_messages(
        system_prompt,
        DATA_TEMPLATE_CAA.format(
            gold_truth=gold_truth.strip(),
            user_message=user_message.strip(),
            condition=condition_norm,
            category=(category or "").strip(),
            model_response=(model_response or "").strip(),
        ),
    )
