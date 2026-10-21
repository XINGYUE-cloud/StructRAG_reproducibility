"""Score router predictions against the blinded human annotation sheets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.five_type_router_data_v1 import FIVE_LABELS


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_annotations(paths: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def labels_in(value: str) -> list[str]:
    labels = [part.strip().lower() for part in value.replace(";", ",").replace("|", ",").split(",") if part.strip()]
    invalid = sorted(set(labels) - set(FIVE_LABELS))
    if invalid:
        raise ValueError(f"Invalid five-type annotation(s): {invalid}")
    return labels


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {gold: {pred: 0 for pred in FIVE_LABELS} for gold in FIVE_LABELS}
    correct = top2 = tie_accepted = 0
    by_class: dict[str, dict[str, int]] = {label: {"tp": 0, "fp": 0, "fn": 0} for label in FIVE_LABELS}
    for row in rows:
        gold = row["preferred_structure"]
        pred = row["predicted_structure"]
        confusion[gold][pred] += 1
        correct += pred == gold
        top2 += pred == gold or pred == row.get("second_choice")
        tie_accepted += pred == gold or pred in row.get("acceptable_ties", [])
        for label in FIVE_LABELS:
            if gold == label and pred == label:
                by_class[label]["tp"] += 1
            elif gold != label and pred == label:
                by_class[label]["fp"] += 1
            elif gold == label and pred != label:
                by_class[label]["fn"] += 1
    per_label = {}
    f1_values = []
    for label, counts in by_class.items():
        precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": counts["tp"] + counts["fn"]}
        f1_values.append(f1)
    total = len(rows)
    return {
        "examples": total,
        "accuracy": correct / total if total else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values and total else None,
        "top2_accuracy": top2 / total if total else None,
        "tie_aware_accuracy": tie_accepted / total if total else None,
        "per_label": per_label,
        "confusion_matrix": confusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = {(str(row["dataset"]), str(row["id"])): row for row in read_jsonl(args.predictions)}
    joined = []
    unannotated = []
    for annotation in read_annotations(args.annotations):
        primary = annotation.get("preferred_structure", "").strip().lower()
        if not primary:
            unannotated.append({"dataset": annotation.get("dataset"), "id": annotation.get("id")})
            continue
        if primary not in FIVE_LABELS:
            raise ValueError(f"Invalid preferred_structure for {annotation.get('id')}: {primary}")
        key = (annotation["dataset"], annotation["id"])
        if key not in predictions:
            raise KeyError(f"No router prediction for annotation {key}")
        acceptable_ties = labels_in(annotation.get("acceptable_ties", ""))
        second = annotation.get("second_choice", "").strip().lower()
        if second and second not in FIVE_LABELS:
            raise ValueError(f"Invalid second_choice for {key}: {second}")
        joined.append(
            {
                "dataset": key[0],
                "id": key[1],
                "task_type": annotation.get("task_type", ""),
                "preferred_structure": primary,
                "second_choice": second,
                "acceptable_ties": acceptable_ties,
                "predicted_structure": predictions[key]["predicted_structure"],
            }
        )

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_dataset[row["dataset"]].append(row)
        by_task[f"{row['dataset']}::{row['task_type']}"].append(row)
    result = {
        "overall": classification_metrics(joined),
        "by_dataset": {name: classification_metrics(group) for name, group in sorted(by_dataset.items())},
        "by_task_type": {name: classification_metrics(group) for name, group in sorted(by_task.items())},
        "annotated_examples": len(joined),
        "pending_annotations": len(unannotated),
        "annotation_source": "human-filled blinded CSV sheets",
        "unannotated_ids": unannotated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"annotated_examples": len(joined), "pending_annotations": len(unannotated), "overall": result["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
