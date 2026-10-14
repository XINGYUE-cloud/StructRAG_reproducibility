"""Evaluate a Qwen router on the held-out synthetic five-type test split."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.five_type_router_data_v1 import FIVE_LABELS, load_router_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_label(text: str) -> str:
    match = re.search(r"\b(chunk|table|graph|algorithm|catalogue)\b", text.lower())
    return match.group(1) if match else "invalid"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for evaluation.")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    rows = load_router_rows(args.test)
    args.output_dir.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).to("cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    predictions: list[dict] = []
    for index, row in enumerate(rows, start=1):
        messages = row["chosen"][:-1]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        predicted = parse_label(completion)
        predictions.append({
            "id": row["id"],
            "unit_id": row["unit_id"],
            "expected": row["label"],
            "predicted": predicted,
            "raw_completion": completion,
            "correct": predicted == row["label"],
        })
        if index % 25 == 0 or index == len(rows):
            print(f"evaluated {index}/{len(rows)}")
    expected = [row["expected"] for row in predictions]
    predicted = [row["predicted"] for row in predictions]
    metrics = {
        "examples": len(predictions),
        "accuracy": accuracy_score(expected, predicted),
        "macro_f1": f1_score(expected, predicted, labels=list(FIVE_LABELS), average="macro", zero_division=0),
        "labels": list(FIVE_LABELS),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=list(FIVE_LABELS)).tolist(),
        "per_label": classification_report(expected, predicted, labels=list(FIVE_LABELS), target_names=list(FIVE_LABELS), output_dict=True, zero_division=0),
        "prediction_counts": dict(sorted(Counter(predicted).items())),
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "caveat": "Synthetic unit-disjoint five-type evaluation. It is not the paper's unreleased five-type router test set.",
    }
    with (args.output_dir / "router_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "router_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
