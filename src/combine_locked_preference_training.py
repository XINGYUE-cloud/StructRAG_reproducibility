"""Append added training preferences while preserving locked validation/test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--added-train", type=Path, required=True)
    parser.add_argument("--locked-validation", type=Path, required=True)
    parser.add_argument("--locked-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-existing-overlap",
        action="store_true",
        help="Preserve a pre-existing overlap among locked splits and record it in metadata.",
    )
    args = parser.parse_args()

    base_train = read_jsonl(args.base_train)
    added_train = read_jsonl(args.added_train)
    validation = read_jsonl(args.locked_validation)
    test = read_jsonl(args.locked_test)
    train_units = {str(row["unit_id"]) for row in base_train}
    added_units = {str(row["unit_id"]) for row in added_train}
    validation_units = {str(row["unit_id"]) for row in validation}
    test_units = {str(row["unit_id"]) for row in test}
    if not added_units <= train_units:
        raise ValueError("Added preferences contain units outside the locked training split")
    existing_overlap = {
        "train_validation": sorted(train_units & validation_units),
        "train_test": sorted(train_units & test_units),
        "validation_test": sorted(validation_units & test_units),
    }
    if any(existing_overlap.values()) and not args.allow_existing_overlap:
        raise ValueError("Locked train/validation/test units overlap")
    existing_ids = {str(row.get("source_id", row["id"])) for row in base_train}
    duplicate_ids = existing_ids & {str(row.get("source_id", row["id"])) for row in added_train}
    if duplicate_ids:
        raise ValueError(f"Added preferences duplicate {len(duplicate_ids)} source IDs")

    output_train = base_train + added_train
    write_jsonl(args.output_dir / "train.jsonl", output_train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    write_jsonl(args.output_dir / "test.jsonl", test)
    metadata = {
        "base_train_rows": len(base_train),
        "added_train_rows": len(added_train),
        "combined_train_rows": len(output_train),
        "validation_rows_preserved": len(validation),
        "test_rows_preserved": len(test),
        "added_units_subset_of_locked_train": True,
        "locked_validation_and_test_unchanged": True,
        "pre_existing_unit_overlap": existing_overlap,
        "overlap_explicitly_allowed": args.allow_existing_overlap,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
