"""Run the five-stage StructRAG flow on synthetic held-out five-type samples.

The module order and structure-specific prompts follow the public StructRAG
implementation: router -> structurizer -> decomposer -> precise extractor ->
answer merger. The only intentional model substitution is Qwen2.5-1.5B.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.five_type_router_data_v1 import FIVE_LABELS, load_router_rows, router_messages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--router-adapter", type=Path, required=True)
    parser.add_argument("--prompts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-subquestions", type=int, default=4)
    parser.add_argument("--max-new-tokens-structure", type=int, default=512)
    parser.add_argument("--max-new-tokens-answer", type=int, default=384)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--stratified", action="store_true")
    return parser.parse_args()


def generate(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=24576).to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def route(tokenizer, model, query: str, titles: list[str]) -> tuple[str, str]:
    messages = router_messages(query, titles)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    match = re.search(r"\b(chunk|table|graph|algorithm|catalogue)\b", text.lower())
    return (match.group(1) if match else "chunk"), text


def prompt_text(prompts_dir: Path, filename: str) -> str:
    return (prompts_dir / filename).read_text(encoding="utf-8")


def construct(
    tokenizer,
    model,
    prompts_dir: Path,
    query: str,
    structure: str,
    documents: list[dict[str, str]],
    max_new_tokens: int,
) -> tuple[str, list[str]]:
    titles = [document["title"] for document in documents]
    if structure == "chunk":
        chunks = [f"{document['title']}: {document['content']}" for document in documents]
        return " ".join(titles), chunks

    if structure == "graph":
        requirement = (
            "Based on the given document, construct a graph where entities are document subjects and "
            "relations are explicit relations that help answer the query."
        )
        template = prompt_text(prompts_dir, "construct_graph.txt")
        outputs = [
            generate(
                tokenizer,
                model,
                template.format(requirement=requirement, raw_content=document["content"], titles="\n".join(titles)),
                max_new_tokens,
            )
            for document in documents
        ]
    elif structure == "table":
        instruction = f"Query is {query}, please extract relevant complete tables from the document based on attributes and keywords in the Query. Retain table titles and source information."
        template = prompt_text(prompts_dir, "construct_table.txt")
        outputs = [
            generate(tokenizer, model, template.format(instruction=instruction, content=document["content"]), max_new_tokens)
            for document in documents
        ]
    elif structure == "algorithm":
        instruction = f"Query is {query}, please extract relevant algorithms from the document based on the Query."
        template = prompt_text(prompts_dir, "construct_algorithm.txt")
        outputs = [
            generate(
                tokenizer,
                model,
                template.format(requirement=instruction, raw_content=document["content"]),
                max_new_tokens,
            )
            for document in documents
        ]
    elif structure == "catalogue":
        instruction = f"Query:\n{query}\nPlease extract relevant catalogues from the document based on the Query."
        template = prompt_text(prompts_dir, "construct_catalogue.txt")
        outputs = [
            generate(
                tokenizer,
                model,
                template.format(requirement=instruction, raw_content=document["content"]),
                max_new_tokens,
            )
            for document in documents
        ]
    else:
        raise ValueError(structure)
    structured = [f"{document['title']}: {output}" for document, output in zip(documents, outputs)]
    kb_info = "".join(output.split("\n", 1)[0][:128] for output in outputs)
    return kb_info, structured


def decompose(tokenizer, model, prompts_dir: Path, query: str, kb_info: str, max_subquestions: int, max_new_tokens: int) -> list[str]:
    template = prompt_text(prompts_dir, "decompose.txt")
    output = generate(tokenizer, model, template.format(query=query, kb_info=kb_info), max_new_tokens)
    subquestions = [line.strip(" -0123456789.\t") for line in output.splitlines() if line.strip()]
    return subquestions[:max_subquestions] or [query]


def extract(
    tokenizer,
    model,
    structure: str,
    query: str,
    subquestions: list[str],
    structured: list[str],
    max_new_tokens: int,
) -> list[str]:
    if structure == "chunk":
        composed_query = "\n".join(subquestions)
        return [
            generate(
                tokenizer,
                model,
                f"Instruction:\nAnswer the Query based on the given Document.\n\nQuery:\n{composed_query}\n\nDocument:\n{chunk}\n\nOutput:",
                max_new_tokens,
            )
            for chunk in structured
        ]

    contents = "\n\n".join(structured)
    if structure == "table":
        instruction = (
            "The following Tables show multiple independent tables built from multiple documents. "
            "Filter them according to the query, retaining only information that helps answer it."
        )
        label = "Tables"
    elif structure == "graph":
        instruction = "According to the query, filter out triples from all triples in the graph that can help answer the query."
        label = "Graphs"
    elif structure == "algorithm":
        instruction = "According to the query, filter out information from algorithm descriptions that can help answer the query."
        label = "Algorithms"
    elif structure == "catalogue":
        instruction = "According to the query, filter out information from the catalogue that can help answer the query."
        label = "Catalogues"
    else:
        raise ValueError(structure)
    return [
        generate(
            tokenizer,
            model,
            f"Instruction:\n{instruction}\n\n{label}:\n{contents}\n\nQuery:\n{subquestion}\n\nOutput:",
            max_new_tokens,
        )
        for subquestion in subquestions
    ]


def merge(tokenizer, model, query: str, subquestions: list[str], evidence: list[str], max_new_tokens: int) -> str:
    pairs = "\n\n".join(
        f"Subquery: {subquestion}\nRetrieval results:\n{result}"
        for subquestion, result in zip(subquestions, evidence)
    )
    instruction = (
        "1. Answer the Question based on retrieval results. "
        "2. Find relevant information from retrieval results and output detailed, specific information. "
        "3. The output must be coherent and smooth text."
    )
    return generate(
        tokenizer,
        model,
        f"Instruction:\n{instruction}\n\nQuestion:\n{query}\n\nRetrieval:\n{pairs}",
        max_new_tokens,
    )


def normalized(text: str) -> str:
    return re.sub(r"[\s，。；、,.;:：!！?？'\"<>]", "", text).lower()


def choose_rows(rows: list[dict[str, Any]], max_samples: int, stratified: bool, seed: int) -> list[dict[str, Any]]:
    if not max_samples or max_samples >= len(rows):
        return rows
    rng = random.Random(seed)
    if not stratified:
        copied = list(rows)
        rng.shuffle(copied)
        return copied[:max_samples]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    selected: list[dict[str, Any]] = []
    remaining = max_samples
    for label in FIVE_LABELS:
        candidates = list(grouped[label])
        rng.shuffle(candidates)
        take = min(len(candidates), max_samples // len(FIVE_LABELS))
        selected.extend(candidates[:take])
        remaining -= take
    if remaining:
        leftovers = [row for row in rows if row not in selected]
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining])
    return selected


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this experiment.")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    rows = choose_rows(load_router_rows(args.dataset), args.max_samples, args.stratified, args.seed)
    args.output_dir.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(args.router_adapter, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    router_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).to("cuda")
    router_model = PeftModel.from_pretrained(router_model, args.router_adapter)
    router_model.eval()
    routed: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        selected, raw_route = route(tokenizer, router_model, row["query"], [doc["title"] for doc in row["documents"]])
        routed.append({"row": row, "selected_structure": selected, "raw_route": raw_route})
        print(f"routed {index}/{len(rows)} as {selected}")
    del router_model
    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).to("cuda")
    model.eval()
    predictions: list[dict[str, Any]] = []
    for index, item in enumerate(routed, start=1):
        row = item["row"]
        structure = item["selected_structure"]
        kb_info, structured = construct(
            tokenizer, model, args.prompts_dir, row["query"], structure, row["documents"], args.max_new_tokens_structure
        )
        subquestions = decompose(
            tokenizer, model, args.prompts_dir, row["query"], kb_info, args.max_subquestions, args.max_new_tokens_answer
        )
        evidence = extract(tokenizer, model, structure, row["query"], subquestions, structured, args.max_new_tokens_answer)
        answer = merge(tokenizer, model, row["query"], subquestions, evidence, args.max_new_tokens_answer)
        gold = normalized(row["answer"])
        answer_normalized = normalized(answer)
        keyword_hits = [keyword for keyword in row["answer_keywords"] if normalized(keyword) in answer_normalized]
        predictions.append({
            "id": row["id"],
            "unit_id": row["unit_id"],
            "expected_structure": row["label"],
            "selected_structure": structure,
            "raw_route": item["raw_route"],
            "query": row["query"],
            "gold_answer": row["answer"],
            "answer_keywords": row["answer_keywords"],
            "answer": answer,
            "exact_match": gold == answer_normalized,
            "answer_contains": gold in answer_normalized,
            "keyword_recall": len(keyword_hits) / len(row["answer_keywords"]),
            "subquestions": subquestions,
            "evidence": evidence,
            "structured_documents": structured,
        })
        print(f"answered {index}/{len(routed)}")

    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_label[row["expected_structure"]].append(row)
    metrics = {
        "examples": len(predictions),
        "router_accuracy": sum(row["expected_structure"] == row["selected_structure"] for row in predictions) / len(predictions),
        "exact_match": sum(row["exact_match"] for row in predictions) / len(predictions),
        "answer_contains": sum(row["answer_contains"] for row in predictions) / len(predictions),
        "mean_keyword_recall": sum(row["keyword_recall"] for row in predictions) / len(predictions),
        "route_counts": dict(sorted(Counter(row["selected_structure"] for row in predictions).items())),
        "per_expected_structure": {
            label: {
                "examples": len(group),
                "router_accuracy": sum(row["expected_structure"] == row["selected_structure"] for row in group) / len(group),
                "exact_match": sum(row["exact_match"] for row in group) / len(group),
                "answer_contains": sum(row["answer_contains"] for row in group) / len(group),
                "mean_keyword_recall": sum(row["keyword_recall"] for row in group) / len(group),
            }
            for label, group in sorted(by_label.items())
        },
        "base_model": str(args.base_model),
        "router_adapter": str(args.router_adapter),
        "prompts_dir": str(args.prompts_dir),
        "protocol": "paper-module-order five-type synthetic evaluation",
        "caveat": "The test data and the two missing router classes are synthetic. This is a protocol-aligned adaptation, not a numerical replication of the paper's Loong evaluation.",
    }
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
