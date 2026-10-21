"""Select a paired subset of predictions for cache-first semantic judging."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-dataset", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    if args.per_dataset < 1:
        raise ValueError("--per-dataset must be positive")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in args.predictions:
        for row in read_jsonl(path):
            mode = str(row.get("mode", "run"))
            if mode == "fixed" and row.get("selected_structure"):
                mode = f"fixed_{row['selected_structure']}"
            groups[(mode, str(row["dataset"]))].append(row)
    normalized = []
    for (mode, dataset), rows in sorted(groups.items()):
        rows.sort(key=lambda row: str(row["id"]))
        if len(rows) > args.per_dataset:
            rng = random.Random(f"{args.seed}:{mode}:{dataset}")
            rows = rng.sample(rows, args.per_dataset)
        for row in rows:
            normalized.append(
                {
                    "id": f"{mode}::{dataset}::{row['id']}",
                    "query": row["query"],
                    "gold_answer": row["gold"],
                    "prediction": row["answer"],
                    "type": row.get("task_type", row.get("task")),
                    "selected_structure": row.get("selected_structure"),
                    "mode": mode,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized), encoding="utf-8")
    print(json.dumps({"examples": len(normalized), "per_dataset_cap": args.per_dataset, "modes": sorted({row['mode'] for row in normalized})}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
