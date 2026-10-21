"""Merge LLM-ranked five-structure preference batches into DPO files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} is not an object")
        rows.append(row)
    return rows


def validate(row: dict[str, Any]) -> None:
    labels = {"chunk", "table", "graph", "algorithm", "catalogue"}
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    if not isinstance(chosen, list) or not isinstance(rejected, list):
        raise ValueError("preference row must contain chosen and rejected messages")
    chosen_label = str(chosen[-1].get("content", "")).strip().lower()
    rejected_label = str(rejected[-1].get("content", "")).strip().lower()
    if chosen_label not in labels or rejected_label not in labels or chosen_label == rejected_label:
        raise ValueError(f"invalid preference labels: {chosen_label!r}, {rejected_label!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    merged: dict[str, dict[str, Any]] = {}
    for batch_path in args.batch:
        for row in read_jsonl(batch_path):
            validate(row)
            source_id = str(row.get("source_id") or row.get("id"))
            if not source_id:
                raise ValueError(f"missing source_id in {batch_path}")
            merged.setdefault(source_id, row)

    rows = list(merged.values())
    rows.sort(key=lambda row: (str(row.get("unit_id", "")), str(row.get("source_id", ""))))
    split_at = max(1, int(len(rows) * 0.8))
    validation_at = max(split_at + 1, int(len(rows) * 0.9))
    splits = {
        "train": rows[:split_at],
        "validation": rows[split_at:validation_at],
        "test": rows[validation_at:],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        (args.output_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows),
            encoding="utf-8",
        )
        # The local DPO trainer consumes a JSON array of chat-message pairs.
        dpo_rows = [{"chosen": row["chosen"], "rejected": row["rejected"]} for row in split_rows]
        (args.output_dir / f"{name}.json").write_text(
            json.dumps(dpo_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    metadata = {
        "source_batches": [str(path) for path in args.batch],
        "deduplicated_rows": len(rows),
        "split_counts": {name: len(split_rows) for name, split_rows in splits.items()},
        "label_counts": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
        "task_family_counts": dict(sorted(Counter(str(row.get("task_family", "unknown")) for row in rows).items())),
        "preference_source": "LLM comparison of five structure-specific solving plans",
        "training_scope": "router outputs only the selected structure label",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
