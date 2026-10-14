"""Adapters for the preference pairs published with the official StructRAG code."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


OFFICIAL_LABELS = ("chunk", "table", "graph")


def load_official_pairs(path: Path) -> list[dict[str, Any]]:
    """Load and validate the JSON array used by the upstream DPO script."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty JSON array at {path}")

    for index, row in enumerate(rows):
        for key in ("chosen", "rejected"):
            messages = row.get(key)
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"Invalid {key} messages at row {index}")
            answer = str(messages[-1].get("content", "")).strip().lower()
            if answer not in OFFICIAL_LABELS:
                raise ValueError(
                    f"Unsupported label {answer!r} at row {index}; "
                    f"expected one of {OFFICIAL_LABELS}"
                )
    return rows


def make_dpo_rows(rows: list[dict[str, Any]], tokenizer, prompt_format: str) -> list[dict[str, str]]:
    """Convert upstream chat-message pairs into DPOTrainer's prompt/completion format."""
    converted: list[dict[str, str]] = []
    for row in rows:
        chosen_messages = row["chosen"]
        rejected_messages = row["rejected"]
        prompt_messages = chosen_messages[:-1]
        chosen_label = str(chosen_messages[-1]["content"]).strip().lower()
        rejected_label = str(rejected_messages[-1]["content"]).strip().lower()

        if prompt_format == "qwen_chat":
            # The official pairs are preserved; only their serialization is adapted
            # to Qwen2.5-Instruct's native chat template.
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            chosen = chosen_label + tokenizer.eos_token
            rejected = rejected_label + tokenizer.eos_token
        elif prompt_format == "upstream":
            # This follows train_router/dpo.py in the official repository exactly.
            prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            chosen = tokenizer.apply_chat_template([chosen_messages[-1]], tokenize=False)
            rejected = tokenizer.apply_chat_template([rejected_messages[-1]], tokenize=False)
        else:
            raise ValueError(f"Unknown prompt format: {prompt_format}")

        converted.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return converted


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["chosen"][-1]["content"]).strip().lower() for row in rows).items()))


def router_prompt(tokenizer, source_row: dict[str, Any], prompt_format: str) -> str:
    """Build the inference prompt using the same format as DPO training."""
    messages = source_row["chosen"][:-1]
    if prompt_format == "qwen_chat":
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt_format == "upstream":
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        prefix = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": ""}], tokenize=False
        )
        return prompt + prefix.rsplit("<|im_end|>", 1)[0]
    raise ValueError(f"Unknown prompt format: {prompt_format}")
