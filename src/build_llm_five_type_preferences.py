"""Synthesize five-type router DPO preferences from document-backed tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib import error, request

from src.evaluate_loong_llm_judge import load_config
from src.five_type_router_data_v1 import FIVE_LABELS, router_messages, write_jsonl


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Preference response must be a JSON object")
    return value


def source_prompt(row: dict[str, Any], content_limit: int, target_family: str | None = None) -> list[dict[str, str]]:
    documents = row.get("documents") or []
    sources = []
    for document in documents:
        title = str(document.get("title", "Document"))
        content = str(document.get("content", ""))[:content_limit]
        sources.append(f"[{title}]\n{content}")
    if not sources:
        raise ValueError(f"Source row {row.get('id')} has no document content")
    user = {
        "source_question": str(row.get("query", row.get("question", ""))),
        "source_documents": "\n\n".join(sources),
    }
    return [
        {
            "role": "system",
            "content": (
                "You create router preference data for knowledge-intensive QA. "
                "Use only the supplied source documents. First synthesize one new, answerable question that requires their facts. "
                + (f"The question must primarily be a {target_family} task. " if target_family else "")
                + "Return its short gold answer and one exact supporting quotation copied from the source text. "
                "Then write one concise solving plan for each structure: chunk, graph, table, algorithm, catalogue. "
                "Compare all five plans for evidence fit, completeness, and unnecessary complexity. "
                "Rank all five from most suitable to least suitable. The plans must describe how that structure would organize/use evidence; "
                "do not put the answer in any plan. Return one JSON object with keys question, answer, evidence_quote, task_family, overview, plans, ranking. "
                "task_family must be one of comparison, relation, planning, summary, single_hop, other. "
                "plans must map all five structure names to short strings. ranking must contain each structure exactly once."
            ),
        },
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def validate_candidate(value: dict[str, Any]) -> dict[str, Any]:
    question = value.get("question")
    answer = value.get("answer")
    evidence_quote = value.get("evidence_quote")
    family = value.get("task_family")
    overview = value.get("overview")
    plans = value.get("plans")
    ranking = value.get("ranking")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Response has no synthesized question")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Response has no short gold answer")
    if not isinstance(evidence_quote, str) or not evidence_quote.strip():
        raise ValueError("Response has no supporting quotation")
    if family not in {"comparison", "relation", "planning", "summary", "single_hop", "other"}:
        raise ValueError("Response has an unsupported task_family")
    if not isinstance(overview, str) or not overview.strip():
        raise ValueError("Response has no document overview")
    if not isinstance(plans, dict) or set(plans) != set(FIVE_LABELS):
        raise ValueError("Response plans must contain all five structures")
    if any(not isinstance(plans[label], str) or not plans[label].strip() for label in FIVE_LABELS):
        raise ValueError("Every structure plan must be non-empty text")
    if not isinstance(ranking, list) or len(ranking) != len(FIVE_LABELS) or set(ranking) != set(FIVE_LABELS):
        raise ValueError("Ranking must be a permutation of the five structures")
    return {"question": question.strip(), "answer": answer.strip(), "evidence_quote": evidence_quote.strip(), "task_family": family, "overview": overview.strip(), "plans": plans, "ranking": ranking}


def api_request(base_url: str, api_key: str, model: str, messages: list[dict[str, str]], timeout: float, retries: int = 2) -> dict[str, Any]:
    payload = json.dumps({"model": model, "messages": messages, "temperature": 0.2}, ensure_ascii=False).encode("utf-8")
    for attempt in range(retries + 1):
        req = request.Request(
            base_url.rstrip("/") + "/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
            return parse_json_object(value["choices"][0]["message"]["content"])
        except error.HTTPError as exc:
            if exc.code not in {408, 409, 429, *range(500, 600)} or attempt >= retries:
                raise
        except (error.URLError, TimeoutError, OSError):
            if attempt >= retries:
                raise
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            if attempt >= retries:
                raise
        time.sleep(min(8.0, 1.5 * (2**attempt)))
    raise RuntimeError("unreachable")


def preference_row(source: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    ranked = validate_candidate(candidate)
    source_text = "\n".join(str(document.get("content", "")) for document in source.get("documents", []))
    normalize = lambda text: " ".join(text.casefold().split())
    if normalize(ranked["evidence_quote"]) not in normalize(source_text):
        raise ValueError("Supporting quotation is not an exact match in source documents")
    messages = router_messages(ranked["question"], [ranked["overview"]])
    source_id = str(source.get("source_root", source.get("unit_id", source.get("id", "unknown"))))
    row_id = hashlib.sha256(f"{source_id}\0{ranked['question']}".encode("utf-8")).hexdigest()[:16]
    return {
        "id": row_id,
        "unit_id": source_id,
        "label": ranked["ranking"][0],
        "task_family": ranked["task_family"],
        "query": ranked["question"],
        "gold_answer": ranked["answer"],
        "evidence_quote": ranked["evidence_quote"],
        "document_overview": ranked["overview"],
        "candidate_plans": ranked["plans"],
        "preference_ranking": ranked["ranking"],
        "chosen": messages + [{"role": "assistant", "content": ranked["ranking"][0]}],
        "rejected": messages + [{"role": "assistant", "content": ranked["ranking"][1]}],
        "preference_source": "LLM comparison of five structure-specific solving plans",
        "source_id": str(source.get("id", "")),
    }


def split_rows(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    units: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        units.setdefault(row["unit_id"], []).append(row)
    keys = sorted(units)
    random.Random(seed).shuffle(keys)
    train_end = max(1, int(len(keys) * 0.8))
    validation_end = max(train_end + 1, int(len(keys) * 0.9))
    validation_end = min(validation_end, len(keys))
    assignment = {
        "train": set(keys[:train_end]),
        "validation": set(keys[train_end:validation_end]),
        "test": set(keys[validation_end:]),
    }
    return {name: [row for row in rows if row["unit_id"] in selected] for name, selected in assignment.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="JSONL with id, query, documents[{title,content}], and unit_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(".env.judge.local"))
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--content-limit", type=int, default=5000)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--balance-task-families", action="store_true")
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.limit < 1 or args.offset < 0:
        raise ValueError("--limit must be positive and --offset non-negative")
    if args.workers < 1 or args.workers > 8:
        raise ValueError("--workers must be between 1 and 8")

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")
    config = load_config(args.config)
    base_url = config.get("OPENAI_BASE_URL", "")
    api_key = config.get("OPENAI_API_KEY", "")
    model = config.get("OPENAI_MODEL", "gpt-5.6terra")
    if not base_url or not api_key:
        raise ValueError("OPENAI_BASE_URL and OPENAI_API_KEY must be configured")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "preferences.jsonl"
    cached = read_jsonl(output_path) if output_path.exists() else []
    completed = {row["source_id"] for row in cached}
    sources = read_jsonl(args.source)[args.offset : args.offset + args.limit]
    task_families = ("comparison", "relation", "planning", "summary", "single_hop")
    source_family = {
        str(row.get("id", "")): task_families[index % len(task_families)]
        for index, row in enumerate(sources)
    } if args.balance_task_families else {}
    pending = [row for row in sources if str(row.get("id", "")) not in completed]
    errors: list[dict[str, str]] = []
    def create_row(source: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                target_family = source_family.get(str(source.get("id", "")))
                candidate = api_request(base_url, api_key, model, source_prompt(source, args.content_limit, target_family), args.timeout)
                return preference_row(source, candidate)
            except (error.URLError, error.HTTPError, TimeoutError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
        assert last_error is not None
        raise last_error

    with output_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(create_row, source): source for source in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            source = futures[future]
            source_id = str(source.get("id", ""))
            try:
                row = future.result()
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                cached.append(row)
            except (error.URLError, error.HTTPError, TimeoutError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                errors.append({"source_id": source_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"processed {index}/{len(pending)}; accepted={len(cached)}; errors={len(errors)}", flush=True)

    cached.sort(key=lambda row: row["source_id"])
    write_jsonl(output_path, cached)
    splits = split_rows(cached, args.seed)
    for name, rows in splits.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)
    write_jsonl(args.output_dir / "generation_errors.jsonl", errors)
    metadata = {
        "model": model,
        "accepted_preferences": len(cached),
        "source_offset": args.offset,
        "source_limit": args.limit,
        "balanced_task_families_requested": args.balance_task_families,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "split_units": {name: sorted({row["unit_id"] for row in rows}) for name, rows in splits.items()},
        "task_family_counts": {name: {family: sum(row["task_family"] == family for row in rows) for family in sorted({row["task_family"] for row in rows})} for name, rows in splits.items()},
        "error_count": len(errors),
        "protocol": "LLM synthesizes a document-grounded task, writes plans for all five structures, ranks plan suitability, and yields top-vs-second DPO pair",
        "training_scope": "router outputs only the selected structure label",
        "api_key_written": False,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
