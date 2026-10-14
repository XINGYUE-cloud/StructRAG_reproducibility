"""Run a bounded, one-GPU StructRAG adaptation on the supplied person-QA data.

The router is trained on the released official StructRAG DPO pairs. At test
time this module uses raw person-profile text as input, dynamically constructs
the routed representation, decomposes the question, extracts evidence, and
answers. The `provided` mode is a separately labelled control that reads the
pre-rendered matching view from the supplied generator.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROUTER_LABELS = ("chunk", "table", "graph")
VIEW_FILES = {
    "chunk": "kb_text.jsonl",
    "table": "kb_table.jsonl",
    # The supplied triples view is the closest one-to-one match for StructRAG's graph triples.
    "graph": "kb_triples.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb-dir", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--router-adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-docs", type=int, default=6)
    parser.add_argument("--max-subquestions", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--structurization-mode", choices=("dynamic", "provided"), default="dynamic")
    parser.add_argument("--stratified", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return re.sub(r"\s+|[，。；：、,.!?！？]", "", text).lower()


def chinese_bigrams(text: str) -> set[str]:
    chars = "".join(re.findall(r"[\u4e00-\u9fff]", text))
    return {chars[index : index + 2] for index in range(max(0, len(chars) - 1))}


def rank_documents(question: dict[str, Any], records: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    query = question["question"]
    question_terms = chinese_bigrams(query)

    def score(record: dict[str, Any]) -> tuple[int, int, str]:
        name = str(record.get("entity_name", ""))
        content = str(record["content"])
        name_match = 1000 if name and name in query else 0
        overlap = len(question_terms & chinese_bigrams(content))
        return (name_match + overlap, overlap, str(record["chunk_id"]))

    return sorted(records, key=score, reverse=True)[:maximum]


def select_questions(rows: list[dict[str, Any]], maximum: int, seed: int, stratified: bool) -> list[dict[str, Any]]:
    if maximum <= 0:
        return rows
    if not stratified:
        return rows[:maximum]
    rng = random.Random(seed)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_type"]].append(row)
    for group in by_task.values():
        rng.shuffle(group)
    task_types = sorted(by_task)
    base, remainder = divmod(maximum, len(task_types))
    selected: list[dict[str, Any]] = []
    for index, task_type in enumerate(task_types):
        selected.extend(by_task[task_type][: base + (index < remainder)])
    return selected


def load_model(base_model: Path, adapter: Path | None = None):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def generate(tokenizer, model, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()


def route_question(tokenizer, router_model, question: str, titles: list[str]) -> tuple[str, str]:
    prompt = (
        "Instruction:\n"
        "To answer the question from a document collection, select one knowledge structure. "
        "Statistical/filtering tasks prefer table, relationship-chain reasoning prefers graph, "
        "and simple single-hop QA prefers chunk. Output exactly one word: table, graph, or chunk.\n\n"
        f"Doc Info:\n{' '.join('《' + title + '》' for title in titles)}\n\n"
        f"Query:\n{question}\n\nDecision:"
    )
    raw = generate(
        tokenizer,
        router_model,
        [{"role": "user", "content": prompt}],
        max_new_tokens=8,
    )
    match = re.search(r"\b(chunk|table|graph)\b", raw.lower())
    return (match.group(1) if match else "chunk", raw)


def construct_dynamic(
    tokenizer,
    model,
    question: str,
    structure: str,
    raw_records: list[dict[str, Any]],
) -> list[str]:
    if structure == "chunk":
        return [str(record["content"]) for record in raw_records]

    if structure == "table":
        instruction = (
            "Convert the raw profile into a compact Markdown table containing only facts useful for "
            "the question. Preserve names, relationships, attributes, and values exactly. Do not infer facts."
        )
    else:
        instruction = (
            "Extract compact head-relation-tail triples from the raw profile that are useful for the "
            "question. Preserve names and values exactly. Do not infer facts."
        )

    structured: list[str] = []
    for index, record in enumerate(raw_records, start=1):
        output = generate(
            tokenizer,
            model,
            [
                {"role": "system", "content": "You perform faithful question-conditioned knowledge structurization."},
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\nRaw profile {index}:\n{record['content']}\n\n"
                        f"Task:\n{instruction}"
                    ),
                },
            ],
            max_new_tokens=192,
        )
        structured.append(output)
    return structured


def parse_subquestions(text: str, original_question: str, maximum: int) -> list[str]:
    candidates = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:\d+[.)、]|[-*])\s*", "", line).strip()
        if cleaned and len(cleaned) >= 4:
            candidates.append(cleaned)
    return candidates[:maximum] or [original_question]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this experiment.")
    if args.max_docs <= 0:
        raise ValueError("--max-docs must be positive")

    raw_records = load_jsonl(args.kb_dir / VIEW_FILES["chunk"])
    records_by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in raw_records:
        records_by_unit[str(record["unit_id"])].append(record)
    provided_by_structure: dict[str, dict[str, dict[str, Any]]] = {}
    if args.structurization_mode == "provided":
        for structure, filename in VIEW_FILES.items():
            provided_by_structure[structure] = {
                str(record["entity_id"]): record for record in load_jsonl(args.kb_dir / filename)
            }

    questions = select_questions(load_jsonl(args.questions), args.max_samples, args.seed, args.stratified)
    if not questions:
        raise ValueError("No questions selected")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    router_model = load_model(args.base_model, args.router_adapter)
    routed: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        unit_records = records_by_unit[str(question["unit_id"])]
        titles = [str(record["entity_name"]) for record in unit_records]
        selected, raw_route = route_question(tokenizer, router_model, question["question"], titles)
        routed.append({"question": question, "structure": selected, "raw_route": raw_route})
        print(f"routed {index}/{len(questions)} as {selected}")

    del router_model
    torch.cuda.empty_cache()
    model = load_model(args.base_model)
    predictions: list[dict[str, Any]] = []
    for index, item in enumerate(routed, start=1):
        question = item["question"]
        structure = item["structure"]
        selected_records = rank_documents(
            question, records_by_unit[str(question["unit_id"])], args.max_docs
        )
        if args.structurization_mode == "dynamic":
            structured_docs = construct_dynamic(
                tokenizer, model, question["question"], structure, selected_records
            )
        else:
            structured_docs = [
                str(provided_by_structure[structure][str(record["entity_id"])]["content"])
                for record in selected_records
            ]

        knowledge = "\n\n".join(
            f"Document {doc_index}:\n{content}"
            for doc_index, content in enumerate(structured_docs, start=1)
        )
        decomposition = generate(
            tokenizer,
            model,
            [
                {"role": "system", "content": "Split a reasoning question into at most two necessary factual subquestions."},
                {
                    "role": "user",
                    "content": f"Question: {question['question']}\n\nKnowledge summary: {knowledge[:2500]}",
                },
            ],
            max_new_tokens=96,
        )
        subquestions = parse_subquestions(decomposition, question["question"], args.max_subquestions)
        evidence: list[str] = []
        for subquestion in subquestions:
            evidence.append(
                generate(
                    tokenizer,
                    model,
                    [
                        {"role": "system", "content": "Extract only facts supported by the supplied knowledge. Do not answer from outside knowledge."},
                        {
                            "role": "user",
                            "content": f"Subquestion: {subquestion}\n\nStructured knowledge:\n{knowledge}",
                        },
                    ],
                    max_new_tokens=128,
                )
            )
        answer = generate(
            tokenizer,
            model,
            [
                {"role": "system", "content": "Answer only from the supplied evidence. Return the concise answer without explanation."},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question['question']}\n\nEvidence:\n"
                        + "\n\n".join(evidence)
                    ),
                },
            ],
            max_new_tokens=48,
        )
        target = str(question["answer"])
        normalized_answer = normalize(answer)
        normalized_target = normalize(target)
        retrieved_entity_ids = [str(record["entity_id"]) for record in selected_records]
        gold_entity_ids = [str(entity_id) for entity_id in question.get("gold_entity_ids", [])]
        predictions.append(
            {
                "question_id": question["question_id"],
                "unit_id": question["unit_id"],
                "task_type": question["task_type"],
                "question": question["question"],
                "expected": target,
                "predicted": answer,
                "exact_match": normalized_answer == normalized_target,
                "answer_contains": normalized_target in normalized_answer,
                "selected_structure": structure,
                "raw_route": item["raw_route"],
                "retrieved_entity_ids": retrieved_entity_ids,
                "gold_entity_ids": gold_entity_ids,
                "gold_entity_recall": (
                    len(set(gold_entity_ids) & set(retrieved_entity_ids)) / len(gold_entity_ids)
                    if gold_entity_ids
                    else None
                ),
                "decomposition": decomposition,
                "subquestions": subquestions,
                "structured_documents": structured_docs,
                "evidence": evidence,
            }
        )
        print(f"answered {index}/{len(routed)}")

    def rate(rows: list[dict[str, Any]], key: str) -> float:
        return sum(bool(row[key]) for row in rows) / len(rows)

    by_task = {}
    for task_type in sorted({row["task_type"] for row in predictions}):
        rows = [row for row in predictions if row["task_type"] == task_type]
        recalls = [row["gold_entity_recall"] for row in rows if row["gold_entity_recall"] is not None]
        by_task[task_type] = {
            "examples": len(rows),
            "exact_match": rate(rows, "exact_match"),
            "answer_contains": rate(rows, "answer_contains"),
            "gold_entity_recall": sum(recalls) / len(recalls) if recalls else None,
        }
    metrics = {
        "examples": len(predictions),
        "exact_match": rate(predictions, "exact_match"),
        "answer_contains": rate(predictions, "answer_contains"),
        "route_counts": dict(sorted(Counter(row["selected_structure"] for row in predictions).items())),
        "task_type_metrics": by_task,
        "base_model": str(args.base_model),
        "router_adapter": str(args.router_adapter) if args.router_adapter else None,
        "structurization_mode": args.structurization_mode,
        "max_docs": args.max_docs,
        "max_subquestions": args.max_subquestions,
        "source_mapping": {
            "chunk": "kb_text.jsonl",
            "table": "kb_table.jsonl",
            "graph": "kb_triples.jsonl",
        },
        "note": (
            "One-GPU Qwen2.5-1.5B resource-constrained adaptation. Dynamic mode has the full "
            "route -> question-conditioned structurization -> decomposition -> extraction -> answer chain. "
            "Provided mode is a pre-rendered-view control, not dynamic StructRAG."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "end_to_end_predictions.jsonl", predictions)
    (args.output_dir / "end_to_end_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
