"""Prepare the official Loong source set for bounded StructRAG inference.

The upstream StructRAG code expects documents wrapped in Chinese title markers,
whereas the current public Loong loader emits plain text.  This utility reads
the official ``loong.jsonl`` and official ``doc`` package, restores that
StructRAG-compatible representation, and selects only whole documents that
fit a Qwen input budget.  It never cuts a document mid-stream.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--doc-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=18000)
    parser.add_argument("--seed", type=int, default=1000000007)
    return parser.parse_args()


def marker_document(title: str, content: str) -> str:
    return f"<标题起始符>{title}<标题终止符>\n{content}\n"


def resolve_document(doc_root: Path, item: dict[str, Any], name: str, index: int, legal: dict[str, Any]) -> tuple[str, str]:
    doc_type = item["type"]
    level = item["level"]
    if doc_type == "financial":
        pattern = f"*{name}*.txt" if str(level).strip() == "4" else f"*2024-{name}*.txt"
        matches = sorted((doc_root / "financial").glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No financial document matching {pattern}")
        path = matches[0]
        title = path.stem.split("-")[-1]
        return name, marker_document(title, path.read_text(encoding="utf-8"))
    if doc_type == "paper":
        path = doc_root / "paper" / name
        content = path.read_text(encoding="utf-8")
        title = content.split("\n", 1)[0].strip("#").strip()
        return name, marker_document(title, content)
    if doc_type == "legal":
        record = legal[name]
        content = record["content"]
        if not (level == 4 and "阅读以上判决文书，我将给你若干份判决结果：" in item["instruction"]):
            content += record["result"]
        return name, marker_document(f"判决文书{index + 1}", content)
    raise ValueError(f"Unsupported Loong document type: {doc_type}")


def render_prompt(item: dict[str, Any], docs: str) -> str:
    return item["prompt_template"].format(
        instruction=item["instruction"], question=item["question"], docs=docs
    )


def chat_tokens(tokenizer, text: str) -> int:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )
    return len(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Refusing to overwrite an existing prepared set or report.")
    if args.max_input_tokens <= 0:
        raise ValueError("--max-input-tokens must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    items = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    legal_path = args.doc_root / "legal" / "legal.json"
    legal = json.loads(legal_path.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)

    status_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    included_counts: list[int] = []
    output_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    document_token_cache: dict[str, int] = {}

    for item in items:
        try:
            source_docs = [resolve_document(args.doc_root, item, name, index, legal) for index, name in enumerate(item["doc"])]
            # This matches the public Loong loader: financial contents are first
            # shuffled, then every selected document is shuffled before joining.
            if item.get("shuffle_doc", True) and item["type"] == "financial":
                rng.shuffle(source_docs)

            base_tokens = chat_tokens(tokenizer, render_prompt(item, ""))
            selected: list[tuple[str, str]] = []
            omitted: list[str] = []
            estimated_tokens = base_tokens
            for name, document in source_docs:
                # Cache full-document lengths.  A document can occur in many
                # Loong items, and obviously over-budget documents need not be
                # concatenated with the full prompt just to be rejected.
                document_tokens = document_token_cache.get(document)
                if document_tokens is None:
                    document_tokens = len(tokenizer(document, add_special_tokens=False)["input_ids"])
                    document_token_cache[document] = document_tokens
                # The four-token cushion protects against BPE boundary effects;
                # an exact count is made once after the final deterministic shuffle.
                if estimated_tokens + document_tokens + 4 <= args.max_input_tokens:
                    selected.append((name, document))
                    estimated_tokens += document_tokens
                else:
                    omitted.append(name)
            if item.get("shuffle_doc", True):
                rng.shuffle(selected)

            docs = "".join(document for _, document in selected)
            input_tokens = chat_tokens(tokenizer, render_prompt(item, docs))
            # Never truncate.  Boundary effects can only make this small
            # adjustment necessary, and the removed whole document is logged.
            while selected and input_tokens > args.max_input_tokens:
                name, _ = selected.pop()
                omitted.append(name)
                docs = "".join(document for _, document in selected)
                input_tokens = chat_tokens(tokenizer, render_prompt(item, docs))
            if not selected:
                status = "no_document_fits"
            elif omitted:
                status = "partial_whole_document_coverage"
            else:
                status = "complete_whole_document_coverage"
            row = dict(item)
            row.update(
                {
                    "docs": docs,
                    "source_doc_names": list(item["doc"]),
                    "included_doc_names": [name for name, _ in selected],
                    "omitted_doc_names": omitted,
                    "qwen_input_tokens": input_tokens,
                    "coverage_status": status,
                    "preparation_max_input_tokens": args.max_input_tokens,
                }
            )
            output_rows.append(row)
            status_counts[status] += 1
            type_counts[str(item["type"])] += 1
            level_counts[str(item["level"])] += 1
            included_counts.append(len(selected))
        except Exception as exc:  # Keep an auditable record; do not fabricate a row.
            errors.append({"id": str(item.get("id", "")), "error": repr(exc)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "source": str(args.input),
        "doc_root": str(args.doc_root),
        "tokenizer": str(args.tokenizer),
        "seed": args.seed,
        "max_input_tokens": args.max_input_tokens,
        "source_examples": len(items),
        "prepared_examples": len(output_rows),
        "preparation_errors": errors,
        "coverage_status_counts": dict(sorted(status_counts.items())),
        "type_counts": dict(sorted(type_counts.items())),
        "level_counts": dict(sorted(level_counts.items())),
        "mean_included_documents": sum(included_counts) / len(included_counts) if included_counts else 0,
        "selection_policy": "Official source-document content; deterministic upstream-style shuffling; whole-document inclusion only; no token truncation.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
