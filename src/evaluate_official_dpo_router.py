"""Evaluate a Qwen router on the released StructRAG router test pairs."""

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

from src.official_router_data import OFFICIAL_LABELS, load_official_pairs, router_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--prompt-format", choices=("qwen_chat", "upstream"), default="qwen_chat")
    return parser.parse_args()


def parse_label(text: str) -> str:
    match = re.search(r"\b(chunk|table|graph)\b", text.lower())
    return match.group(1) if match else "invalid"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for router evaluation.")
    rows = load_official_pairs(args.test)
    if args.max_samples:
        rows = rows[: args.max_samples]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_source = args.adapter if args.adapter else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    predictions: list[dict] = []
    for index, row in enumerate(rows, start=1):
        prompt = router_prompt(tokenizer, row, args.prompt_format)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        expected = str(row["chosen"][-1]["content"]).strip().lower()
        predicted = parse_label(completion)
        predictions.append(
            {
                "index": index - 1,
                "expected": expected,
                "predicted": predicted,
                "raw_completion": completion,
                "correct": expected == predicted,
            }
        )
        if index % 25 == 0 or index == len(rows):
            print(f"evaluated {index}/{len(rows)}")

    expected = [row["expected"] for row in predictions]
    predicted = [row["predicted"] for row in predictions]
    metrics = {
        "examples": len(predictions),
        "accuracy": accuracy_score(expected, predicted),
        "macro_f1": f1_score(
            expected, predicted, labels=list(OFFICIAL_LABELS), average="macro", zero_division=0
        ),
        "labels": list(OFFICIAL_LABELS),
        "confusion_matrix": confusion_matrix(
            expected, predicted, labels=list(OFFICIAL_LABELS)
        ).tolist(),
        "per_label": classification_report(
            expected,
            predicted,
            labels=list(OFFICIAL_LABELS),
            target_names=list(OFFICIAL_LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "prediction_counts": dict(sorted(Counter(predicted).items())),
        "base_model": str(args.base_model),
        "adapter": str(args.adapter) if args.adapter else None,
        "prompt_format": args.prompt_format,
        "note": (
            "Evaluation on the official released three-class router test set. It is not directly "
            "comparable to the paper's reported five-type router result."
        ),
    }
    (args.output_dir / "router_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8"
    )
    (args.output_dir / "router_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
