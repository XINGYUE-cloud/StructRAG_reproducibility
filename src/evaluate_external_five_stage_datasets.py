"""Evaluate the five-stage pipeline on the supplied PersonWork and RCAEval bundles.

Model input is built only from public runtime questions and evidence stores.
Oracle records are joined after generation for metrics and never enter prompts.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

torch = None
PeftModel = AutoModelForCausalLM = AutoTokenizer = None
construct = decompose = extract = generate = merge = route = None


def read_jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def get_zip_jsonl(archive: zipfile.ZipFile, suffix: str) -> list[dict[str, Any]]:
    match = next(name for name in archive.namelist() if name.endswith(suffix))
    return read_jsonl(archive.read(match))


def get_store(archive: zipfile.ZipFile, prefix: str, store_key: str) -> dict[str, Any]:
    name = f"{prefix}/stores/{store_key}.json"
    return json.loads(archive.read(name))


def person_document(store: dict[str, Any]) -> dict[str, str]:
    records = store.get("records", {})
    lines = [
        json.dumps({"entity_id": entity_id, **record}, ensure_ascii=False, sort_keys=True)
        for entity_id, record in sorted(records.items())
        if isinstance(record, dict)
    ]
    return {
        "title": f"People, work profiles, and relationships for {store.get('scope', 'this unit')}",
        "content": "\n".join(lines),
    }


def rca_document(store: dict[str, Any]) -> dict[str, str]:
    records = store.get("records", {})
    lines = [
        json.dumps({"record_id": record_id, **record}, ensure_ascii=False, sort_keys=True)
        for record_id, record in sorted(records.items())
        if isinstance(record, dict)
    ]
    return {
        "title": f"Alert-window telemetry for {store.get('scope', 'this incident')}",
        "content": "\n".join(lines),
    }


def load_personwork(path: Path, max_samples: int) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        prefix = "VALUE-KAO-PersonWork-EN-v2-Experiment-1k"
        runtime = get_zip_jsonl(archive, f"{prefix}/runtime.jsonl")
        question_map = get_zip_jsonl(archive, f"{prefix}/question_id_map.jsonl")
        oracle = get_zip_jsonl(archive, f"{prefix}/oracle_private.jsonl")
        by_request = {row["request_id"]: row for row in oracle}
        by_question = {row["request_id"]: row for row in question_map}
        rows = []
        for item in runtime:
            gold = by_request[item["request_id"]]
            task_type = by_question[item["request_id"]]["task_type"]
            store = get_store(archive, prefix, item["store_key"])
            rows.append({
                "dataset": "VALUE-KAO-PersonWork-EN-v2",
                "id": item["request_id"],
                "task": "person",
                "task_type": task_type,
                "query": item["question"],
                "documents": [person_document(store)],
                "gold": gold["answer"],
                "gold_names": [name.strip() for name in gold["answer"].split(",") if name.strip()],
                "unit_id": item["scope"],
                "proxy_structure": personwork_proxy_structure(task_type),
            })
        return rows[:max_samples] if max_samples else rows


def load_rcaeval(path: Path, max_samples: int) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        prefix = "RCAEval-RE2-Microservice-v1/RCALoop-Full-v1"
        runtime = get_zip_jsonl(archive, f"{prefix}/runtime.jsonl")
        oracle = get_zip_jsonl(archive, f"{prefix}/oracle_private.jsonl")
        by_request = {row["request_id"]: row for row in oracle}
        rows = []
        for item in runtime:
            gold = by_request.get(item["request_id"])
            if not gold or gold.get("split") != "test":
                continue
            store = get_store(archive, prefix, item["store_key"])
            rows.append({
                "dataset": "RCAEval-RE2-Microservice-v1",
                "id": item["request_id"],
                "task": item["task"],
                "condition": gold.get("condition"),
                "query": item["question"],
                "documents": [rca_document(store)],
                "gold": gold.get("root_cause_service") if item["task"] == "root_cause" else gold.get("fault"),
                "candidates": re.findall(r"svc-\d+", item["question"]),
                "unit_id": item["scope"],
                "proxy_structure": "graph" if item["task"] == "root_cause" else "table",
            })
        return rows[:max_samples] if max_samples else rows


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def structure_format_valid(structure: str, outputs: list[str]) -> bool:
    text = "\n".join(outputs).strip()
    if not text:
        return False
    if structure == "chunk":
        return True
    if structure == "table":
        return "|" in text or "\t" in text
    if structure == "graph":
        return bool(re.search(r"\b(head|relation|tail)\b|->|--", text, re.IGNORECASE))
    if structure == "algorithm":
        return bool(re.search(r"(?m)^\s*(?:\d+[.)]|[-*]|if\b|for\b|while\b|step\s+\d+)", text, re.IGNORECASE))
    if structure == "catalogue":
        return bool(re.search(r"(?m)^\s*(?:#{1,6}\s|\d+(?:\.\d+)*[.)]?\s|[-*]\s)", text))
    return False


def token_support_precision(outputs: list[str], documents: list[dict[str, str]]) -> float | None:
    generated_tokens = re.findall(r"[a-z0-9_-]+", " ".join(outputs).lower())
    if not generated_tokens:
        return None
    source_tokens = set(re.findall(r"[a-z0-9_-]+", " ".join(doc["content"] for doc in documents).lower()))
    return sum(token in source_tokens for token in generated_tokens) / len(generated_tokens)


def personwork_proxy_structure(task_type: str) -> str:
    if "kinship" in task_type or "relative" in task_type or "family_member" in task_type or "colleague" in task_type or "collaboration" in task_type:
        return "graph"
    if "filter" in task_type or "stat" in task_type or "ranking" in task_type or "candidate" in task_type:
        return "table"
    if "event" in task_type or "personality" in task_type or "assignment" in task_type or "capacity" in task_type or "trusted_helper" in task_type:
        return "algorithm"
    return "chunk"


def choose_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if not count or count >= len(rows):
        return rows
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("task_type", row.get("task", "other")))].append(row)
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


def run_pipeline(
    rows: list[dict[str, Any]], tokenizer: Any, model: Any, prompts_dir: Path,
    max_new_tokens_structure: int, max_new_tokens_answer: int,
    predictions_path: Path, completed_ids: set[tuple[str, str]],
    mode: str = "full", fixed_structure: str = "table", seed: int = 20261021,
    human_routes: dict[tuple[str, str], str] | None = None,
) -> list[dict[str, Any]]:
    predictions = []
    pending = [row for row in rows if (row["dataset"], row["id"]) not in completed_ids]
    token_usage = {"input_tokens": 0, "output_tokens": 0, "generation_calls": 0}
    original_generate = model.generate

    def tracked_generate(*args: Any, **kwargs: Any) -> Any:
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        input_length = int(input_ids.shape[-1]) if input_ids is not None else 0
        generated = original_generate(*args, **kwargs)
        output_length = int(generated[0].shape[-1]) if input_length else 0
        token_usage["input_tokens"] += input_length
        token_usage["output_tokens"] += max(0, output_length - input_length)
        token_usage["generation_calls"] += 1
        return generated

    model.generate = tracked_generate
    try:
      with predictions_path.open("a", encoding="utf-8") as output:
       for index, row in enumerate(pending, start=1):
        token_usage.update(input_tokens=0, output_tokens=0, generation_calls=0)
        stage_times = {"routing": 0.0, "structuring": 0.0, "decomposition": 0.0, "extraction": 0.0, "answer": 0.0}
        started = time.perf_counter()
        raw_route = ""
        if mode in {"full", "no_structuring", "no_decomposition"}:
            stage_start = time.perf_counter()
            selected, raw_route = route(tokenizer, model, row["query"], [doc["title"] for doc in row["documents"]])
            stage_times["routing"] = time.perf_counter() - stage_start
        elif mode == "fixed":
            selected = fixed_structure
        elif mode == "random":
            selected = random.Random(f"{seed}:{row['dataset']}:{row['id']}").choice(["chunk", "graph", "table", "algorithm", "catalogue"])
        elif mode == "proxy_oracle":
            selected = row["proxy_structure"]
        elif mode == "human_oracle":
            if human_routes is None or (row["dataset"], row["id"]) not in human_routes:
                raise ValueError(f"Missing human route annotation for {(row['dataset'], row['id'])}")
            selected = human_routes[(row["dataset"], row["id"])]
        elif mode in {"direct", "lexical_rag"}:
            selected = "none"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        structured: list[str] = []
        subquestions: list[str] = []
        evidence: list[str] = []
        if mode in {"direct", "lexical_rag"}:
            if mode == "direct":
                contexts = [f"{doc['title']}\n{doc['content']}" for doc in row["documents"]]
            else:
                terms = set(re.findall(r"[a-z0-9_-]+", row["query"].lower()))
                chunks = [line for doc in row["documents"] for line in doc["content"].splitlines() if line.strip()]
                ranked = sorted(
                    enumerate(chunks),
                    key=lambda pair: (len(terms & set(re.findall(r"[a-z0-9_-]+", pair[1].lower()))), -pair[0]),
                    reverse=True,
                )
                contexts = [chunk for _, chunk in ranked[:8]]
            prompt = (
                "Answer the question using only the provided context. Preserve requested output format and abstain when evidence is insufficient.\n\n"
                f"Question:\n{row['query']}\n\nContext:\n" + "\n\n".join(contexts)
            )
            stage_start = time.perf_counter()
            with model.disable_adapter():
                answer = generate(tokenizer, model, prompt, max_new_tokens_answer)
            stage_times["answer"] = time.perf_counter() - stage_start
            retrieved_context = contexts
        else:
            with model.disable_adapter():
                if mode == "no_structuring":
                    info = " ".join(doc["title"] for doc in row["documents"])
                    structured = [f"{doc['title']}: {doc['content']}" for doc in row["documents"]]
                else:
                    stage_start = time.perf_counter()
                    info, structured = construct(
                        tokenizer, model, prompts_dir, row["query"], selected, row["documents"], max_new_tokens_structure
                    )
                    stage_times["structuring"] = time.perf_counter() - stage_start
                if mode == "no_decomposition":
                    subquestions = [row["query"]]
                else:
                    stage_start = time.perf_counter()
                    subquestions = decompose(tokenizer, model, prompts_dir, row["query"], info, 4, max_new_tokens_answer)
                    stage_times["decomposition"] = time.perf_counter() - stage_start
                stage_start = time.perf_counter()
                evidence = extract(tokenizer, model, selected if mode != "no_structuring" else "chunk", row["query"], subquestions, structured, max_new_tokens_answer)
                stage_times["extraction"] = time.perf_counter() - stage_start
                stage_start = time.perf_counter()
                answer = merge(tokenizer, model, row["query"], subquestions, evidence, max_new_tokens_answer)
                stage_times["answer"] = time.perf_counter() - stage_start
            retrieved_context = []
        prediction = {
            "mode": mode,
            "dataset": row["dataset"],
            "id": row["id"],
            "task": row["task"],
            "task_type": row.get("task_type", row["task"]),
            "condition": row.get("condition"),
            "unit_id": row["unit_id"],
            "query": row["query"],
            "gold": row["gold"],
            "answer": answer,
            "selected_structure": selected,
            "proxy_structure": row["proxy_structure"],
            "human_structure": human_routes.get((row["dataset"], row["id"])) if human_routes else None,
            "raw_route": raw_route,
            "candidates": row.get("candidates", []),
            "gold_names": row.get("gold_names", []),
            "subquestions": subquestions,
            "structure_output": structured,
            "structure_format_valid": structure_format_valid(selected, structured) if structured else None,
            "structure_token_support_precision_proxy": token_support_precision(structured, row["documents"]) if structured else None,
            "retrieved_context": retrieved_context,
            "evidence": evidence,
            "document_count": len(row["documents"]),
            "document_chars": sum(len(doc["content"]) for doc in row["documents"]),
            "stage_latency_seconds": stage_times,
            "latency_seconds": time.perf_counter() - started,
            **token_usage,
        }
        output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        output.flush()
        predictions.append(prediction)
        completed_ids.add((row["dataset"], row["id"]))
        print(f"answered {index}/{len(pending)}: {row['dataset']} {row['task']} [{mode}]", flush=True)
    finally:
        model.generate = original_generate
    return predictions


def person_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = 0
    recalls = []
    for row in rows:
        predicted = {normalized(name) for name in row.get("gold_names", []) if normalized(name) and normalized(name) in normalized(row["answer"])}
        gold = {normalized(name) for name in row.get("gold_names", []) if normalized(name)}
        hit = len(predicted & gold)
        exact += normalized(row["gold"]) == normalized(row["answer"])
        recalls.append(hit / len(gold) if gold else 0.0)
    return {
        "examples": len(rows),
        "exact_match": exact / len(rows) if rows else 0.0,
        "mean_name_recall": sum(recalls) / len(recalls) if rows else 0.0,
    }


def rca_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    metrics: dict[str, Any] = {"examples": len(rows)}
    for task, group in sorted(by_task.items()):
        if task == "root_cause":
            hits1 = hits3 = reciprocal_rank = 0.0
            for row in group:
                candidates = list(dict.fromkeys(re.findall(r"svc-\d+", row["answer"].lower())))
                gold = str(row["gold"]).lower()
                hits1 += bool(candidates and candidates[0] == gold)
                hits3 += gold in candidates[:3]
                reciprocal_rank += 1 / (candidates.index(gold) + 1) if gold in candidates else 0.0
            metrics[task] = {
                "examples": len(group),
                "hit_at_1": hits1 / len(group),
                "hit_at_3": hits3 / len(group),
                "mrr": reciprocal_rank / len(group),
            }
        else:
            correct = 0
            for row in group:
                found = re.findall(r"\b(cpu|mem|disk|delay|loss|socket)\b", row["answer"].lower())
                correct += bool(found and found[0] == str(row["gold"]).lower())
            metrics[task] = {"examples": len(group), "accuracy": correct / len(group)}
    metrics["by_condition"] = {
        condition: len([row for row in rows if row.get("condition") == condition])
        for condition in sorted({str(row.get("condition")) for row in rows})
    }
    return metrics


def run_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answer_metrics = {
        "personwork": person_metrics([row for row in rows if row["dataset"].startswith("VALUE-KAO")]),
        "rcaeval": rca_metrics([row for row in rows if row["dataset"].startswith("RCAEval")]),
    }
    total = len(rows)
    stage_names = ("routing", "structuring", "decomposition", "extraction", "answer")
    stages = {}
    for stage in stage_names:
        values = sorted(float(row.get("stage_latency_seconds", {}).get(stage, 0)) for row in rows)
        stages[stage] = {
            "total_seconds": sum(values),
            "mean_seconds": sum(values) / len(values) if values else None,
            "p50_seconds": values[int(0.5 * (len(values) - 1))] if values else None,
            "p95_seconds": values[int(0.95 * (len(values) - 1))] if values else None,
        }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[f"{row['dataset']}::{row.get('task_type', row['task'])}"].append(row)
    by_task_type = {
        task: {
            "examples": len(group),
            "answer_metrics": person_metrics(group) if group[0]["dataset"].startswith("VALUE-KAO") else rca_metrics(group),
            "mean_latency_seconds": sum(float(row.get("latency_seconds", 0)) for row in group) / len(group),
        }
        for task, group in sorted(grouped.items())
    }
    char_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        chars = int(row.get("document_chars", 0))
        bucket = "small:<4000" if chars < 4000 else "medium:4000-16000" if chars <= 16000 else "large:>16000"
        char_buckets[bucket].append(row)
    proxy_correct = sum(row.get("selected_structure") == row.get("proxy_structure") for row in rows)
    proxy_accuracy = proxy_correct / total if total and rows[0].get("mode") not in {"direct", "lexical_rag", "human_oracle"} else None
    human_scored = [row for row in rows if row.get("human_structure") in {"chunk", "graph", "table", "algorithm", "catalogue"}]
    human_confusion = {label: {pred: 0 for pred in ("chunk", "graph", "table", "algorithm", "catalogue")} for label in ("chunk", "graph", "table", "algorithm", "catalogue")}
    for row in human_scored:
        human_confusion[row["human_structure"]][row["selected_structure"]] += 1
    structured_rows = [row for row in rows if row.get("structure_output")]
    valid_format_rows = [row for row in structured_rows if row.get("structure_format_valid") is not None]
    support_precision_rows = [row["structure_token_support_precision_proxy"] for row in structured_rows if row.get("structure_token_support_precision_proxy") is not None]
    supported = 0
    checked = 0
    for row in structured_rows:
        output = " ".join(row["structure_output"])
        gold_facts = row.get("gold_names") or [str(row.get("gold", ""))]
        for fact in gold_facts:
            normalized_fact = normalized(str(fact))
            if normalized_fact:
                checked += 1
                supported += normalized_fact in normalized(output)
    return {
        "examples": total,
        "answer_metrics": answer_metrics,
        "router_proxy_accuracy": proxy_accuracy,
        "router_human_accuracy": sum(row["selected_structure"] == row["human_structure"] for row in human_scored) / len(human_scored) if human_scored and rows[0].get("mode") != "human_oracle" else None,
        "human_labelled_router_examples": len(human_scored) if rows[0].get("mode") != "human_oracle" else 0,
        "human_labelled_confusion_matrix": human_confusion if rows[0].get("mode") != "human_oracle" else None,
        "route_counts": dict(Counter(row["selected_structure"] for row in rows)),
        "task_type_counts": dict(Counter(str(row.get("task_type", row["task"])) for row in rows)),
        "by_task_type": by_task_type,
        "robustness_by_document_characters": {
            bucket: {"examples": len(group), "mean_latency_seconds": sum(float(row.get("latency_seconds", 0)) for row in group) / len(group)}
            for bucket, group in sorted(char_buckets.items())
        },
        "robustness_by_document_count": {
            str(count): {"examples": sum(int(row.get("document_count", 0)) == count for row in rows)}
            for count in sorted({int(row.get("document_count", 0)) for row in rows})
        },
        "structure_extraction": {
            "structured_examples": len(structured_rows),
            "format_validity_rate": sum(bool(row["structure_format_valid"]) for row in valid_format_rows) / len(valid_format_rows) if valid_format_rows else None,
            "token_support_precision_proxy": sum(support_precision_rows) / len(support_precision_rows) if support_precision_rows else None,
            "gold_fact_recall_proxy": supported / checked if checked else None,
            "gold_facts_checked": checked,
        },
        "cost_and_speed": {
            "stage_latency": stages,
            "total_input_tokens": sum(int(row.get("input_tokens", 0)) for row in rows),
            "total_output_tokens": sum(int(row.get("output_tokens", 0)) for row in rows),
            "generation_calls": sum(int(row.get("generation_calls", 0)) for row in rows),
            "mean_end_to_end_seconds": sum(float(row.get("latency_seconds", 0)) for row in rows) / total if total else None,
            "server_api_cost_usd": None,
        },
        "by_condition": {
            condition: len([row for row in rows if str(row.get("condition")) == condition])
            for condition in sorted({str(row.get("condition")) for row in rows})
        },
    }


def main() -> None:
    global torch, PeftModel, AutoModelForCausalLM, AutoTokenizer
    global construct, decompose, extract, generate, merge, route
    parser = argparse.ArgumentParser()
    parser.add_argument("--personwork-zip", type=Path, required=True)
    parser.add_argument("--rcaeval-zip", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--router-adapter", type=Path, required=True)
    parser.add_argument("--prompts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "direct", "lexical_rag", "fixed", "random", "proxy_oracle", "human_oracle", "no_structuring", "no_decomposition"), default="full")
    parser.add_argument("--fixed-structure", choices=("chunk", "graph", "table", "algorithm", "catalogue"), default="table")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--human-annotations", type=Path, nargs="*")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-new-tokens-structure", type=int, default=256)
    parser.add_argument("--max-new-tokens-answer", type=int, default=192)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch as torch_runtime
    from peft import PeftModel as PeftModelRuntime
    from transformers import AutoModelForCausalLM as AutoModelForCausalLMRuntime, AutoTokenizer as AutoTokenizerRuntime
    from src.run_paper_aligned_five_type_structrag_v1 import (
        construct as construct_runtime,
        decompose as decompose_runtime,
        extract as extract_runtime,
        generate as generate_runtime,
        merge as merge_runtime,
        route as route_runtime,
    )

    torch = torch_runtime
    PeftModel = PeftModelRuntime
    AutoModelForCausalLM = AutoModelForCausalLMRuntime
    AutoTokenizer = AutoTokenizerRuntime
    construct = construct_runtime
    decompose = decompose_runtime
    extract = extract_runtime
    generate = generate_runtime
    merge = merge_runtime
    route = route_runtime

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    person_rows = choose_rows(load_personwork(args.personwork_zip, 0), args.max_samples, args.seed)
    rca_rows = choose_rows(load_rcaeval(args.rcaeval_zip, 0), args.max_samples, args.seed + 1)
    human_routes: dict[tuple[str, str], str] = {}
    if args.human_annotations:
        for path in args.human_annotations:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for annotation in csv.DictReader(handle):
                    label = annotation.get("preferred_structure", "").strip().lower()
                    if label:
                        human_routes[(annotation["dataset"], annotation["id"])] = label
    if args.mode == "human_oracle":
        if not human_routes:
            raise ValueError("human_oracle mode requires filled --human-annotations")
        person_rows = [row for row in person_rows if (row["dataset"], row["id"]) in human_routes]
        rca_rows = [row for row in rca_rows if (row["dataset"], row["id"]) in human_routes]
    person_rows = person_rows[args.shard_index::args.num_shards]
    rca_rows = rca_rows[args.shard_index::args.num_shards]
    print(json.dumps({"personwork_rows": len(person_rows), "rcaeval_test_rows": len(rca_rows)}, indent=2))
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for five-stage evaluation.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    previous = read_jsonl(predictions_path.read_bytes()) if predictions_path.exists() else []
    completed_ids = {(row["dataset"], row["id"]) for row in previous}
    tokenizer = AutoTokenizer.from_pretrained(args.router_adapter, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).to("cuda")
    model = PeftModel.from_pretrained(model, args.router_adapter)
    model.eval()
    predictions = run_pipeline(
        person_rows, tokenizer, model, args.prompts_dir,
        args.max_new_tokens_structure, args.max_new_tokens_answer,
        predictions_path, completed_ids, args.mode, args.fixed_structure, args.seed, human_routes,
    )
    predictions.extend(run_pipeline(
        rca_rows, tokenizer, model, args.prompts_dir,
        args.max_new_tokens_structure, args.max_new_tokens_answer,
        predictions_path, completed_ids, args.mode, args.fixed_structure, args.seed, human_routes,
    ))
    predictions = previous + predictions

    result = {
        "model": str(args.base_model),
        "router_adapter": str(args.router_adapter),
        "mode": args.mode,
        "fixed_structure": args.fixed_structure if args.mode == "fixed" else None,
        "pipeline": ["router", "structurizer", "decomposer", "precise_extractor", "answer_merger"],
        "semantic_judge": "not included; run cache-first judge as a separate stage",
        "labeling_note": "proxy_oracle uses task-derived labels, not human-annotated ideal routes. Router accuracy in this run is agreement with those proxy labels only.",
        **run_metrics(predictions),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
