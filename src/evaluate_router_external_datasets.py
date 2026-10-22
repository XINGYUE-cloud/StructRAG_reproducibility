"""Evaluate the five-type StructRAG router on supplied external test bundles.

The model receives only the question and document titles. Oracle labels are
joined after generation and never appear in the router prompt.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.five_type_router_data_v1 import FIVE_LABELS, router_messages


PERSONWORK_LABELS = {
    "hard_attribute_filter": "table",
    "hard_common_relative": "graph",
    "hard_company_department_stat": "table",
    "hard_cross_department_collaboration": "graph",
    "hard_derived_kinship": "graph",
    "hard_event_exception": "algorithm",
    "hard_family_member": "graph",
    "hard_long_chain_kinship": "graph",
    "hard_multi_attribute_candidate": "table",
    "hard_multi_hop_kinship": "graph",
    "hard_org_colleague": "graph",
    "hard_personality_fit": "algorithm",
    "hard_ranking_comparison": "table",
    "hard_task_assignment": "algorithm",
    "hard_time_capacity": "algorithm",
    "hard_trusted_helper": "algorithm",
}
LABELS = (*FIVE_LABELS, "catalogue")


def jsonl(archive: zipfile.ZipFile, suffix: str) -> list[dict[str, Any]]:
    path = next(name for name in archive.namelist() if name.endswith(suffix))
    return [json.loads(line) for line in archive.read(path).decode("utf-8").splitlines() if line.strip()]


def personwork_rows(path: Path) -> list[dict[str, Any]]:
    prefix = "VALUE-KAO-PersonWork-EN-v2-Experiment-1k"
    with zipfile.ZipFile(path) as archive:
        runtime = jsonl(archive, f"{prefix}/runtime.jsonl")
        question_map = {row["request_id"]: row for row in jsonl(archive, f"{prefix}/question_id_map.jsonl")}
        oracle = {row["request_id"]: row for row in jsonl(archive, f"{prefix}/oracle_private.jsonl")}
        rows = []
        for item in runtime:
            mapped = question_map[item["request_id"]]
            expected = PERSONWORK_LABELS[mapped["task_type"]]
            store_name = f"{prefix}/stores/{item['store_key']}.json"
            store = json.loads(archive.read(store_name).decode("utf-8"))
            records = store.get("records", {})
            fields = sorted({field for record in records.values() for field in record})
            overview = (
                f"Person/work-profile collection with {len(records)} people; "
                f"fields include {', '.join(fields)}."
            )
            rows.append({
                "dataset": "VALUE-KAO-PersonWork-EN-v2",
                "id": item["request_id"],
                "query": item["question"],
                "titles": [overview],
                "expected_structure": expected,
                "label_source": "task_type mapping",
                "task_type": mapped["task_type"],
                "oracle_split": oracle[item["request_id"]]["split"],
            })
        return rows


def rcaeval_rows(path: Path) -> list[dict[str, Any]]:
    prefix = "RCAEval-RE2-Microservice-v1/RCALoop-Full-v1"
    with zipfile.ZipFile(path) as archive:
        runtime = jsonl(archive, f"{prefix}/runtime.jsonl")
        oracle = {row["request_id"]: row for row in jsonl(archive, f"{prefix}/oracle_private.jsonl")}
        rows = []
        for item in runtime:
            gold = oracle[item["request_id"]]
            if gold.get("split") != "test":
                continue
            prep_name = f"{prefix}/preparation_private/{item['scope']}.json"
            prepared = json.loads(archive.read(prep_name).decode("utf-8"))
            records = prepared.get("native", {}).get("records", {})
            kinds = Counter(record.get("_kind", "unknown") for record in records.values())
            overview = (
                f"Microservice incident telemetry with {len(records)} records; "
                f"record types: {', '.join(f'{kind}={count}' for kind, count in sorted(kinds.items()))}. "
                "Sources include metrics, logs, and traces where available."
            )
            rows.append({
                "dataset": "RCAEval-RE2-Microservice-v1",
                "id": item["request_id"],
                "query": item["question"],
                "titles": [overview],
                "expected_structure": "graph" if item["task"] == "root_cause" else "table",
                "label_source": "task mapping: root-cause ranking -> graph; fault classification -> table",
                "task_type": item["task"],
                "condition": gold.get("condition"),
            })
        return rows


def predict(tokenizer: Any, model: Any, query: str, titles: list[str]) -> dict[str, Any]:
    import torch

    messages = router_messages(query, titles)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started
    raw = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    match = re.search(r"\b(chunk|table|graph|algorithm|catalogue)\b", raw.lower())
    return {
        "predicted_structure": match.group(1) if match else "invalid",
        "raw_output": raw,
        "input_tokens": int(inputs["input_ids"].shape[-1]),
        "output_tokens": max(0, int(output[0].shape[-1]) - int(inputs["input_ids"].shape[-1])),
        "latency_seconds": elapsed,
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {label: {pred: 0 for pred in (*FIVE_LABELS, "invalid")} for label in FIVE_LABELS}
    for row in rows:
        confusion[row["expected_structure"]][row["predicted_structure"]] += 1
    total = len(rows)
    correct = sum(confusion[label][label] for label in FIVE_LABELS)
    per_class = {}
    f1_values = []
    for label in FIVE_LABELS:
        support = sum(confusion[label].values())
        predicted = sum(confusion[gold][label] for gold in FIVE_LABELS)
        tp = confusion[label][label]
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
        if support:
            f1_values.append(f1)
    return {
        "examples": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1_on_supported_labels": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "supported_labels": [label for label in FIVE_LABELS if per_class[label]["support"]],
        "per_class": per_class,
        "confusion_matrix": confusion,
        "route_counts": dict(Counter(row["predicted_structure"] for row in rows)),
        "label_counts": dict(Counter(row["expected_structure"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personwork-zip", type=Path, required=True)
    parser.add_argument("--rcaeval-zip", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--router-adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    wall_started = time.perf_counter()

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    rows = personwork_rows(args.personwork_zip) + rcaeval_rows(args.rcaeval_zip)
    if args.max_samples:
        rows = rows[:args.max_samples]
    print(json.dumps({"rows": len(rows), "datasets": dict(Counter(row["dataset"] for row in rows))}, indent=2))
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for router evaluation.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "router_predictions.jsonl"
    previous = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()] if prediction_path.exists() else []
    done = {(row["dataset"], row["id"]) for row in previous}
    tokenizer = AutoTokenizer.from_pretrained(args.router_adapter, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).to("cuda")
    model = PeftModel.from_pretrained(model, args.router_adapter)
    model.eval()

    predictions = list(previous)
    with prediction_path.open("a", encoding="utf-8") as output:
        pending = [row for row in rows if (row["dataset"], row["id"]) not in done]
        for index, row in enumerate(pending, start=1):
            prediction = predict(tokenizer, model, row["query"], row["titles"])
            item = {**row, **prediction}
            output.write(json.dumps(item, ensure_ascii=False) + "\n")
            output.flush()
            predictions.append(item)
            print(f"routed {index}/{len(pending)}: {row['dataset']} {row['task_type']} -> {predicted}", flush=True)

    by_dataset = {
        dataset: metrics([row for row in predictions if row["dataset"] == dataset])
        for dataset in sorted({row["dataset"] for row in predictions})
    }
    result = {
        "model": str(args.base_model),
        "router_adapter": str(args.router_adapter),
        "task": "question plus data-derived document overview -> one structure label",
        "structures": list(FIVE_LABELS),
        "downstream_generation": False,
        "semantic_judge": False,
        "datasets": by_dataset,
        "cost_and_speed": {
            "total_input_tokens": sum(int(row.get("input_tokens", 0)) for row in predictions),
            "total_output_tokens": sum(int(row.get("output_tokens", 0)) for row in predictions),
            "total_tokens": sum(int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0)) for row in predictions),
            "generation_calls": len(predictions),
            "generation_seconds": sum(float(row.get("latency_seconds", 0.0)) for row in predictions),
            "wall_clock_seconds": time.perf_counter() - wall_started,
            "timing_scope": "process wall-clock including archive parsing, model load, inference, and metric writing",
        },
        "labeling_note": "Neither supplied test package contains human-annotated StructRAG five-structure targets. Scores are accuracy and macro-F1 on task-derived proxy labels, not official five-class routing accuracy. PersonWork maps task types to graph/table/algorithm; RCAEval maps root-cause ranking to graph and fault classification to table. Neither dataset evaluates chunk or catalogue.",
    }
    (args.output_dir / "router_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
