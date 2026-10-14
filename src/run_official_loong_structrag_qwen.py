"""Run the public StructRAG module order on prepared official Loong items.

This is a Qwen2.5-1.5B single-GPU adaptation of the released inference code.
It keeps the released route, structurization, decomposition, extraction, and
merging prompts, but uses a local Transformers model instead of the authors'
four-GPU Qwen2-72B vLLM endpoint.  Context overflow is an explicit per-sample
failure; no call uses tokenizer truncation.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROUTER_LABELS = ("table", "graph", "chunk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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


class ContextLimitError(RuntimeError):
    pass


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def select_rows(rows: list[dict[str, Any]], maximum: int, stratified: bool, seed: int) -> list[dict[str, Any]]:
    if maximum <= 0 or maximum >= len(rows):
        return rows
    rng = random.Random(seed)
    if not stratified:
        return list(rows[:maximum])
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["type"]), int(row["level"]))].append(row)
    for group in groups.values():
        rng.shuffle(group)
    keys = sorted(groups)
    base, remainder = divmod(maximum, len(keys))
    selected: list[dict[str, Any]] = []
    for index, key in enumerate(keys):
        selected.extend(groups[key][: base + int(index < remainder)])
    return selected


def read_prompt(prompts_dir: Path, filename: str) -> str:
    return (prompts_dir / filename).read_text(encoding="utf-8")


def parse_documents(docs: str) -> list[dict[str, str]]:
    pattern = re.compile(r"<标题起始符>(.*?)<标题终止符>\n?(.*?)(?=<标题起始符>|\Z)", re.S)
    parsed = [
        {"title": title.strip(), "document": content.strip()}
        for title, content in pattern.findall(docs)
        if title.strip() and content.strip()
    ]
    if not parsed:
        raise ValueError("Prepared item does not contain any StructRAG title-marked documents.")
    return parsed


def render(tokenizer, messages: list[dict[str, str]]) -> tuple[str, int]:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    return prompt, token_count


def generate(tokenizer, model, messages: list[dict[str, str]], max_new_tokens: int, context_limit: int) -> str:
    prompt, token_count = render(tokenizer, messages)
    if token_count + max_new_tokens > context_limit:
        raise ContextLimitError(
            f"input_tokens={token_count}, requested_new_tokens={max_new_tokens}, context_limit={context_limit}"
        )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def router_prompt(query: str, titles: list[str]) -> str:
    return (
        "Instruction:\n"
        "To generate answers to questions based on documents, I need to structure the documents as a table, graph, or chunk. \n"
        "Generally speaking, statistical questions prefer tables, chain reasoning questions prefer graphs, and single-hop QA questions prefer chunks. \n"
        "Now, given the following document information and question, please determine which type of knowledge structure I should use. \n"
        "Simply output one of the three words: table, graph, or chunk, without providing any further explanation.\n\n"
        f"Doc Info:\n {' '.join('《' + title + '》' for title in titles)}\n\n"
        f"Query:\n {query}\n\nDecision:"
    )


def route(tokenizer, model, query: str, titles: list[str], context_limit: int) -> tuple[str, str]:
    raw = generate(
        tokenizer, model, [{"role": "user", "content": router_prompt(query, titles)}], 8, context_limit
    )
    match = re.search(r"\b(table|graph|chunk)\b", raw.lower())
    return (match.group(1) if match else "chunk"), raw


def construct(tokenizer, model, prompts_dir: Path, query: str, structure: str, documents: list[dict[str, str]], context_limit: int, max_new_tokens: int) -> tuple[str, list[str]]:
    titles = [document["title"] for document in documents]
    if structure == "chunk":
        return " ".join(titles), [f"{document['title']}: {document['document']}" for document in documents]
    if structure == "graph":
        template = read_prompt(prompts_dir, "construct_graph.txt")
        requirement = "Based on the given document, construct a graph where entities are the titles of papers and the relation is 'reference', using the given document title as the head and other paper titles as tails."
        prompts = [template.format(requirement=requirement, raw_content=document["document"], titles="\n".join(titles)) for document in documents]
    elif structure == "table":
        template = read_prompt(prompts_dir, "construct_table.txt")
        instruction = f"Query is {query}, please extract relevant complete tables from the document based on the attributes and keywords mentioned in the Query. Note: retain table titles and source information."
        prompts = [template.format(instruction=instruction, content=document["document"]) for document in documents]
    else:
        raise ValueError(f"Unsupported released router output: {structure}")
    outputs = [
        generate(tokenizer, model, [{"role": "user", "content": prompt}], max_new_tokens, context_limit)
        for prompt in prompts
    ]
    return "".join(output.split("\n", 1)[0][:128] for output in outputs), [
        f"{document['title']}: {output}" for document, output in zip(documents, outputs)
    ]


def decompose(tokenizer, model, prompts_dir: Path, query: str, kb_info: str, context_limit: int, maximum: int, max_new_tokens: int) -> tuple[str, list[str]]:
    template = read_prompt(prompts_dir, "decompose.txt")
    raw = generate(tokenizer, model, [{"role": "user", "content": template.format(query=query, kb_info=kb_info)}], max_new_tokens, context_limit)
    questions = [re.sub(r"^\s*(?:\d+[.)、]|[-*])\s*", "", line).strip() for line in raw.splitlines()]
    questions = [question for question in questions if question]
    return raw, questions[:maximum] or [query]


def extract(tokenizer, model, structure: str, query: str, subqueries: list[str], structured: list[str], context_limit: int, max_new_tokens: int) -> list[str]:
    if structure == "chunk":
        composed = "\n".join(subqueries)
        return [
            generate(tokenizer, model, [{"role": "user", "content": f"Instruction:\nAnswer the Query based on the given Document.\n\nQuery:\n{composed}\n\nDocument:\n{chunk}\n\nOutput:"}], max_new_tokens, context_limit)
            for chunk in structured
        ]
    if structure == "table":
        instruction = "The following Tables show multiple independent tables built from multiple documents.\nFilter these tables according to the query, retaining only the table information that helps answer the query.\nNote that you need to analyze the attributes and entities mentioned in the query and filter accordingly.\nThe information needed to answer the query must exist in one or several tables, and you need to check these tables one by one."
        title = "Tables"
        contents = "".join(f"Table {index}:\n{table}\n\n" for index, table in enumerate(structured, start=1))
    elif structure == "graph":
        instruction = "According to the query, filter out the triples from all triples in the graph that can help answer the query.\nNote, carefully analyze the entities and relationships mentioned in the query and filter based on this information."
        title = "Graphs"
        contents = "\n\n".join(structured)
    else:
        raise ValueError(structure)
    return [
        generate(tokenizer, model, [{"role": "user", "content": f"Instruction:\n{instruction}\n\n{title}:{contents}\n\nQuery:{subquery}\n\nOutput:"}], max_new_tokens, context_limit)
        for subquery in subqueries
    ]


def merge(tokenizer, model, query: str, subqueries: list[str], evidence: list[str], structure: str, context_limit: int, max_new_tokens: int) -> str:
    if structure == "chunk":
        retrieval = f"Subquery: {query}\nRetrieval results:\n{' '.join(evidence)}\n\n"
    else:
        retrieval = "".join(f"Subquery: {question}\nRetrieval results:\n{result}\n\n" for question, result in zip(subqueries, evidence))
    instruction = "1. Answer the Question based on retrieval results. \n2. Find the relevant information from given retrieval results and output as detailed, specific, and lengthy as possible. \n3. The output must be a coherent and smooth piece of text."
    return generate(tokenizer, model, [{"role": "user", "content": f"Instruction:\n{instruction}\n\nQuestion:\n{query}\n\nRetrieval:\n{retrieval}"}], max_new_tokens, context_limit)


def canonical(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def normalize(text: str) -> str:
    return re.sub(r"[\s，。；、,.;:：!！?？'\"<>]", "", text).lower()


def gold_leaves(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [leaf for nested in value.values() for leaf in gold_leaves(nested)]
    if isinstance(value, list):
        return [leaf for nested in value for leaf in gold_leaves(nested)]
    return [str(value)]


def load_model(path: Path, adapter: Path | None = None):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True).to("cuda")
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run.")
    rows = select_rows(load_rows(args.dataset), args.max_samples, args.stratified, args.seed)
    if not rows:
        raise ValueError("No official Loong rows selected.")
    args.output_dir.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    router_model = load_model(args.base_model, args.router_adapter)
    routed: list[dict[str, Any]] = []
    route_errors: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        try:
            documents = parse_documents(row["docs"])
            query = row["prompt_template"].format(instruction=row["instruction"], question=row["question"], docs="......")
            chosen, raw = route(tokenizer, router_model, query, [document["title"] for document in documents], args.context_limit)
            routed.append({"row": row, "query": query, "documents": documents, "chosen": chosen, "raw_route": raw})
            print(f"routed {index}/{len(rows)} {row['id']} as {chosen}", flush=True)
        except Exception as exc:
            route_errors.append({"id": row["id"], "stage": "route", "error": repr(exc)})
    del router_model
    gc.collect()
    torch.cuda.empty_cache()

    model = load_model(args.base_model)
    predictions: list[dict[str, Any]] = []
    execution_errors = list(route_errors)
    for index, item in enumerate(routed, start=1):
        row = item["row"]
        started = time.time()
        try:
            kb_info, structured = construct(tokenizer, model, args.prompts_dir, item["query"], item["chosen"], item["documents"], args.context_limit, args.max_new_tokens_structure)
            raw_decomposition, subqueries = decompose(tokenizer, model, args.prompts_dir, item["query"], kb_info, args.context_limit, args.max_subqueries, args.max_new_tokens_answer)
            evidence = extract(tokenizer, model, item["chosen"], item["query"], subqueries, structured, args.context_limit, args.max_new_tokens_answer)
            answer = merge(tokenizer, model, item["query"], subqueries, evidence, item["chosen"], args.context_limit, args.max_new_tokens_answer)
            leaves = gold_leaves(row["answer"])
            normalized_answer = normalize(answer)
            hits = [leaf for leaf in leaves if normalize(leaf) in normalized_answer]
            predictions.append({
                "id": row["id"], "type": row["type"], "level": row["level"], "set": row["set"],
                "coverage_status": row["coverage_status"], "source_doc_names": row["source_doc_names"], "included_doc_names": row["included_doc_names"], "omitted_doc_names": row["omitted_doc_names"], "qwen_input_tokens": row["qwen_input_tokens"],
                "selected_structure": item["chosen"], "raw_route": item["raw_route"], "question": row["question"], "gold_answer": row["answer"], "answer": answer,
                "gold_leaf_recall": len(hits) / len(leaves) if leaves else None,
                "exact_canonical_match": normalize(answer) == normalize(canonical(row["answer"])),
                "raw_decomposition": raw_decomposition, "subqueries": subqueries, "structured_documents": structured, "evidence": evidence,
                "elapsed_seconds": time.time() - started,
            })
            print(f"answered {index}/{len(routed)} {row['id']}", flush=True)
        except Exception as exc:
            execution_errors.append({"id": row["id"], "stage": "pipeline", "error": repr(exc)})
            print(f"failed {index}/{len(routed)} {row['id']}: {exc!r}", flush=True)

    (args.output_dir / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
    by_type: dict[str, dict[str, Any]] = {}
    for doc_type in sorted({row["type"] for row in predictions}):
        group = [row for row in predictions if row["type"] == doc_type]
        recalls = [row["gold_leaf_recall"] for row in group if row["gold_leaf_recall"] is not None]
        by_type[doc_type] = {"examples": len(group), "mean_gold_leaf_recall": sum(recalls) / len(recalls) if recalls else None, "exact_canonical_match": sum(row["exact_canonical_match"] for row in group) / len(group)}
    all_recalls = [row["gold_leaf_recall"] for row in predictions if row["gold_leaf_recall"] is not None]
    metrics = {
        "selected_official_examples": len(rows), "completed_examples": len(predictions), "errors": execution_errors,
        "mean_gold_leaf_recall": sum(all_recalls) / len(all_recalls) if all_recalls else None,
        "exact_canonical_match": sum(row["exact_canonical_match"] for row in predictions) / len(predictions) if predictions else None,
        "route_counts": dict(sorted(Counter(row["selected_structure"] for row in predictions).items())), "per_type": by_type,
        "base_model": str(args.base_model), "router_adapter": str(args.router_adapter), "context_limit": args.context_limit,
        "protocol": "Official Loong source set, public StructRAG module order, local Qwen2.5-1.5B; whole-document preparation without tokenizer truncation.",
        "evaluation_caveat": "These auxiliary string metrics are not the authors' GPT-4 score. Run the official Loong step-3 GPT-4 evaluator with a user-provided API configuration for comparable paper scores.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
