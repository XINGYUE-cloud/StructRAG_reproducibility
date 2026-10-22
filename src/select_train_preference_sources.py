"""Select unseen source rows from already assigned training units only."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--locked-train", type=Path, required=True)
    parser.add_argument("--used-preferences", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()

    train_units = {str(row["unit_id"]) for row in read_jsonl(args.locked_train)}
    used_ids = {
        str(row.get("source_id", ""))
        for path in args.used_preferences
        for row in read_jsonl(path)
    }
    candidates = [
        row for row in read_jsonl(args.source)
        if str(row.get("source_root", row.get("unit_id", ""))) in train_units
        and str(row.get("id", "")) not in used_ids
    ]
    if len(candidates) < args.limit:
        raise ValueError(f"Only {len(candidates)} eligible rows for requested limit {args.limit}")
    selected = random.Random(args.seed).sample(candidates, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    print(json.dumps({
        "eligible_sources": len(candidates),
        "selected_sources": len(selected),
        "train_units": sorted(train_units),
        "seed": args.seed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
