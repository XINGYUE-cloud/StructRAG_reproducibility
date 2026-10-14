"""Shared five-type router data helpers for the synthetic extension experiment.

The public StructRAG router release contains only chunk/table/graph. This module
defines an explicitly separate, paper-semantic five-type protocol for synthetic
data generated from the supplied person-profile world.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


FIVE_LABELS = ("chunk", "table", "graph", "algorithm", "catalogue")

ROUTER_INSTRUCTION = """Instruction:
To answer a question from a collection of documents, select the most suitable
knowledge structure. Use chunk for a direct single-hop lookup, table for
multi-record comparison, counting, or filtering, graph for relationship-chain
reasoning, algorithm for a multi-condition configuration or decision procedure,
and catalogue for a hierarchical summary of broad document content.
Output exactly one word from: chunk, table, graph, algorithm, catalogue.
Do not provide any explanation.
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def router_user_prompt(query: str, titles: list[str]) -> str:
    quoted_titles = "\n".join(f'"{title}"' for title in titles)
    return f"{ROUTER_INSTRUCTION}\nDoc Info:\n{quoted_titles}\n\nQuery:\n{query}\n\nDecision:"


def router_messages(query: str, titles: list[str]) -> list[dict[str, str]]:
    return [{"role": "user", "content": router_user_prompt(query, titles)}]


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["label"]) for row in rows).items()))


def validate_rows(rows: list[dict[str, Any]], source: Path) -> None:
    for index, row in enumerate(rows):
        label = str(row.get("label", ""))
        if label not in FIVE_LABELS:
            raise ValueError(f"{source}:{index} has invalid label {label!r}")
        for side in ("chosen", "rejected"):
            messages = row.get(side)
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"{source}:{index} has invalid {side} messages")
        chosen = str(row["chosen"][-1].get("content", "")).strip().lower()
        rejected = str(row["rejected"][-1].get("content", "")).strip().lower()
        if chosen != label or chosen not in FIVE_LABELS or rejected not in FIVE_LABELS:
            raise ValueError(f"{source}:{index} has invalid DPO completion labels")
        if chosen == rejected:
            raise ValueError(f"{source}:{index} uses identical chosen and rejected labels")


def load_router_rows(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path} is empty")
    validate_rows(rows, path)
    return rows
