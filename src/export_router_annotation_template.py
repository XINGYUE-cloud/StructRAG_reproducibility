"""Export blinded human route-ranking sheets for the supplied test bundles."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from src.evaluate_router_external_datasets import personwork_rows, rcaeval_rows


FIELDS = [
    "dataset",
    "id",
    "task_type",
    "question",
    "document_overview",
    "preferred_structure",
    "second_choice",
    "acceptable_ties",
    "rationale",
    "annotator_id",
]


def sample_stratified(rows: list[dict], count: int, seed: int) -> list[dict]:
    if count <= 0 or count >= len(rows):
        return rows
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["task_type"])].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < count:
        progressed = False
        for key in sorted(groups):
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def write_sheet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "id": row["id"],
                    "task_type": row["task_type"],
                    "question": row["query"],
                    "document_overview": row["titles"][0],
                    "preferred_structure": "",
                    "second_choice": "",
                    "acceptable_ties": "",
                    "rationale": "",
                    "annotator_id": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personwork-zip", type=Path, required=True)
    parser.add_argument("--rcaeval-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()

    rows_by_dataset = {
        "personwork": personwork_rows(args.personwork_zip),
        "rcaeval": rcaeval_rows(args.rcaeval_zip),
    }
    for offset, (name, rows) in enumerate(rows_by_dataset.items()):
        sample = sample_stratified(rows, args.samples_per_dataset, args.seed + offset)
        write_sheet(sample, args.output_dir / f"{name}_router_annotations.csv")
        print(f"{name}: exported {len(sample)} blinded rows")


if __name__ == "__main__":
    main()
