"""Aggregate auditable metrics from resumable official-Loong predictions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    recalls = [float(row["gold_leaf_recall"]) for row in values if row.get("gold_leaf_recall") is not None]
    return {
        "examples": len(values),
        "mean_gold_leaf_recall": sum(recalls) / len(recalls) if recalls else None,
        "exact_canonical_match": (
            sum(bool(row.get("exact_canonical_match")) for row in values) / len(values)
            if values else None
        ),
    }


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        groups[str(value if value is not None else "unknown")].append(row)
    return {name: _summary(group) for name, group in sorted(groups.items())}


def aggregate_predictions(
    predictions: list[dict[str, Any]],
    errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    errors = errors or []
    preparation = [row for row in errors if str(row.get("stage", "")).lower() in {"preparation", "prepare"}]
    model_errors = [row for row in errors if row not in preparation]
    return {
        "overall": _summary(predictions),
        "by_structure": _group(predictions, "selected_structure"),
        "by_type": _group(predictions, "type"),
        "by_level": _group(predictions, "level"),
        "by_coverage_status": _group(predictions, "coverage_status"),
        "completed_examples": len(predictions),
        "preparation_exclusions": len(preparation),
        "model_errors": len(model_errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--errors", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    predictions = load_jsonl(args.predictions)
    errors = load_jsonl(args.errors) if args.errors and args.errors.exists() else []
    metrics = aggregate_predictions(predictions, errors)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "official_structure_metrics.json"
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
