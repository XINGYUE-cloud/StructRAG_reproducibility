"""Resumeable official-Loong StructRAG execution with JSONL checkpoints.

This entry point keeps the module order of
``src.run_official_loong_structrag_qwen`` while making long runs restartable.
Routes, predictions, and errors are appended and flushed after every record.
When the process is restarted, an existing prediction is complete and an
existing route is reused.  The output directory may already exist.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from src.run_official_loong_structrag_qwen import (
    canonical,
    construct,
    decompose,
    extract,
    gold_leaves,
    load_model,
    load_rows,
    merge,
    normalize,
    parse_documents,
    route,
    select_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume official Loong StructRAG execution from JSONL checkpoints."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--router-adapter", type=Path, required=True)
    parser.add_argument("--prompts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-limit", type=int, default=32768)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--max-subqueries", type=int, default=4)
    parser.add_argument("--max-new-tokens-structure", type=int, default=512)
    parser.add_argument("--max-new-tokens-answer", type=int, default=384)
    return parser.parse_args()


def sample_id(row: dict[str, Any]) -> str:
    if "id" not in row:
        raise KeyError("Prepared Loong row has no id")
    return str(row["id"])


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append and flush one checkpoint record before continuing."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read valid records and tolerate a torn final line after a hard kill."""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # Leave the torn line untouched for forensic inspection.  New
                # valid records can still be appended after it.
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_progress(
    path: Path,
    *,
    phase: str,
    selected_examples: int,
    route_ids: set[str],
    prediction_ids: set[str],
    errors: list[dict[str, Any]],
    last_id: str | None,
) -> None:
    atomic_json(
        path,
        {
            "phase": phase,
            "updated_at": now_utc(),
            "selected_examples": selected_examples,
            "routes_completed": len(route_ids),
            "predictions_completed": len(prediction_ids),
            "error_records": len(errors),
            "error_ids": len({str(row.get("id")) for row in errors if row.get("id") is not None}),
            "last_id": last_id,
        },
    )


def route_record(row: dict[str, Any], query: str, chosen: str, raw: str) -> dict[str, Any]:
    return {
        "id": sample_id(row),
        "type": row.get("type"),
        "level": row.get("level"),
        "set": row.get("set"),
        "query": query,
        "selected_structure": chosen,
        "raw_route": raw,
        "recorded_at": now_utc(),
    }


def error_record(row_id: str, stage: str, exc: BaseException) -> dict[str, Any]:
    return {
        "id": row_id,
        "stage": stage,
        "error": repr(exc),
        "recorded_at": now_utc(),
    }


def prediction_record(
    row: dict[str, Any],
    route_info: dict[str, Any],
    query: str,
    raw_decomposition: str,
    subqueries: list[str],
    structured: list[str],
    evidence: list[str],
    answer: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    leaves = gold_leaves(row["answer"])
    normalized_answer = normalize(answer)
    hits = [leaf for leaf in leaves if normalize(leaf) in normalized_answer]
    return {
        "id": sample_id(row),
        "type": row.get("type"),
        "level": row.get("level"),
        "set": row.get("set"),
        "coverage_status": row.get("coverage_status"),
        "source_doc_names": row.get("source_doc_names"),
        "included_doc_names": row.get("included_doc_names"),
        "omitted_doc_names": row.get("omitted_doc_names"),
        "qwen_input_tokens": row.get("qwen_input_tokens"),
        "selected_structure": route_info.get("selected_structure", "chunk"),
        "raw_route": route_info.get("raw_route", ""),
        "question": row.get("question"),
        "gold_answer": row.get("answer"),
        "answer": answer,
        "gold_leaf_recall": len(hits) / len(leaves) if leaves else None,
        "exact_canonical_match": normalized_answer == normalize(canonical(row["answer"])),
        "raw_decomposition": raw_decomposition,
        "subqueries": subqueries,
        "structured_documents": structured,
        "evidence": evidence,
        "elapsed_seconds": elapsed_seconds,
        "recorded_at": now_utc(),
    }


def write_metrics(
    path: Path,
    selected: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    by_type: dict[str, dict[str, Any]] = {}
    for doc_type in sorted({str(row.get("type")) for row in predictions}):
        group = [row for row in predictions if str(row.get("type")) == doc_type]
        recalls = [row["gold_leaf_recall"] for row in group if row.get("gold_leaf_recall") is not None]
        by_type[doc_type] = {
            "examples": len(group),
            "mean_gold_leaf_recall": sum(recalls) / len(recalls) if recalls else None,
            "exact_canonical_match": sum(bool(row.get("exact_canonical_match")) for row in group) / len(group),
        }
    recalls = [row["gold_leaf_recall"] for row in predictions if row.get("gold_leaf_recall") is not None]
    metrics = {
        "selected_official_examples": len(selected),
        "completed_examples": len(predictions),
        "errors": errors,
        "mean_gold_leaf_recall": sum(recalls) / len(recalls) if recalls else None,
        "exact_canonical_match": (
            sum(bool(row.get("exact_canonical_match")) for row in predictions) / len(predictions)
            if predictions
            else None
        ),
        "route_counts": dict(sorted(Counter(str(row.get("selected_structure")) for row in predictions).items())),
        "per_type": by_type,
        "base_model": str(args.base_model),
        "router_adapter": str(args.router_adapter),
        "context_limit": args.context_limit,
        "protocol": "Official Loong source set, public StructRAG module order, local Qwen2.5-1.5B; resumable whole-document preparation without tokenizer truncation.",
        "evaluation_caveat": "These auxiliary string metrics are not the authors' GPT-4 score.",
    }
    atomic_json(path, metrics)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run.")
    if args.context_limit <= 0:
        raise ValueError("--context-limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    routes_path = args.output_dir / "routes.jsonl"
    predictions_path = args.output_dir / "predictions.jsonl"
    errors_path = args.output_dir / "errors.jsonl"
    progress_path = args.output_dir / "progress.json"
    metrics_path = args.output_dir / "metrics.json"

    rows = select_rows(load_rows(args.dataset), args.max_samples, args.stratified, args.seed)
    if not rows:
        raise ValueError("No official Loong rows selected.")
    selected_ids = {sample_id(row) for row in rows}

    route_rows = read_jsonl(routes_path)
    prediction_rows = read_jsonl(predictions_path)
    error_rows = [
        row for row in read_jsonl(errors_path) if str(row.get("id")) in selected_ids
    ]
    route_map = {
        str(row["id"]): row
        for row in route_rows
        if str(row.get("id")) in selected_ids and row.get("selected_structure")
    }
    prediction_map = {
        str(row["id"]): row
        for row in prediction_rows
        if str(row.get("id")) in selected_ids
    }
    completed_ids = set(prediction_map)
    write_progress(
        progress_path,
        phase="loaded",
        selected_examples=len(rows),
        route_ids=set(route_map),
        prediction_ids=completed_ids,
        errors=error_rows,
        last_id=None,
    )

    tokenizer = None
    missing_routes = [
        row
        for row in rows
        if sample_id(row) not in route_map and sample_id(row) not in completed_ids
    ]
    if missing_routes:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        router_model = load_model(args.base_model, args.router_adapter)
        try:
            for row in missing_routes:
                row_id = sample_id(row)
                try:
                    documents = parse_documents(row["docs"])
                    query = row["prompt_template"].format(
                        instruction=row["instruction"], question=row["question"], docs="......"
                    )
                    chosen, raw = route(
                        tokenizer,
                        router_model,
                        query,
                        [document["title"] for document in documents],
                        args.context_limit,
                    )
                    record = route_record(row, query, chosen, raw)
                    append_jsonl(routes_path, record)
                    route_map[row_id] = record
                    print(f"routed {len(route_map)}/{len(rows)} {row_id} as {chosen}", flush=True)
                except Exception as exc:
                    failure = error_record(row_id, "route", exc)
                    append_jsonl(errors_path, failure)
                    error_rows.append(failure)
                    print(f"route failed {row_id}: {exc!r}", flush=True)
                write_progress(
                    progress_path,
                    phase="routing",
                    selected_examples=len(rows),
                    route_ids=set(route_map),
                    prediction_ids=completed_ids,
                    errors=error_rows,
                    last_id=row_id,
                )
        finally:
            del router_model
            gc.collect()
            torch.cuda.empty_cache()
        write_metrics(metrics_path, rows, list(prediction_map.values()), error_rows, args)

    pending_rows = [
        row
        for row in rows
        if sample_id(row) not in completed_ids and sample_id(row) in route_map
    ]
    if pending_rows:
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        model = load_model(args.base_model)
        try:
            for index, row in enumerate(pending_rows, start=1):
                row_id = sample_id(row)
                started = time.time()
                try:
                    route_info = route_map[row_id]
                    documents = parse_documents(row["docs"])
                    query = str(
                        route_info.get("query")
                        or row["prompt_template"].format(
                            instruction=row["instruction"], question=row["question"], docs="......"
                        )
                    )
                    structure = str(route_info.get("selected_structure", "chunk"))
                    kb_info, structured = construct(
                        tokenizer,
                        model,
                        args.prompts_dir,
                        query,
                        structure,
                        documents,
                        args.context_limit,
                        args.max_new_tokens_structure,
                    )
                    raw_decomposition, subqueries = decompose(
                        tokenizer,
                        model,
                        args.prompts_dir,
                        query,
                        kb_info,
                        args.context_limit,
                        args.max_subqueries,
                        args.max_new_tokens_answer,
                    )
                    evidence = extract(
                        tokenizer,
                        model,
                        structure,
                        query,
                        subqueries,
                        structured,
                        args.context_limit,
                        args.max_new_tokens_answer,
                    )
                    answer = merge(
                        tokenizer,
                        model,
                        query,
                        subqueries,
                        evidence,
                        structure,
                        args.context_limit,
                        args.max_new_tokens_answer,
                    )
                    prediction = prediction_record(
                        row,
                        route_info,
                        query,
                        raw_decomposition,
                        subqueries,
                        structured,
                        evidence,
                        answer,
                        time.time() - started,
                    )
                    append_jsonl(predictions_path, prediction)
                    prediction_map[row_id] = prediction
                    completed_ids.add(row_id)
                    print(f"answered {index}/{len(pending_rows)} {row_id}", flush=True)
                except Exception as exc:
                    failure = error_record(row_id, "pipeline", exc)
                    append_jsonl(errors_path, failure)
                    error_rows.append(failure)
                    print(f"failed {index}/{len(pending_rows)} {row_id}: {exc!r}", flush=True)
                write_progress(
                    progress_path,
                    phase="execution",
                    selected_examples=len(rows),
                    route_ids=set(route_map),
                    prediction_ids=completed_ids,
                    errors=error_rows,
                    last_id=row_id,
                )
                write_metrics(metrics_path, rows, list(prediction_map.values()), error_rows, args)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    write_metrics(metrics_path, rows, list(prediction_map.values()), error_rows, args)
    phase = "complete" if len(completed_ids) == len(rows) else "incomplete"
    write_progress(
        progress_path,
        phase=phase,
        selected_examples=len(rows),
        route_ids=set(route_map),
        prediction_ids=completed_ids,
        errors=error_rows,
        last_id=None,
    )
    print(metrics_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
