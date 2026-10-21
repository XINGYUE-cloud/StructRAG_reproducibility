"""Evaluate the DPO router on held-out LLM-ranked five-type preferences."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.five_type_router_data_v1 import FIVE_LABELS, load_router_rows


def parse_label(text: str) -> str:
    match = re.search(r"\b(chunk|table|graph|algorithm|catalogue)\b", text.lower())
    return match.group(1) if match else "invalid"


def main() -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for router evaluation")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {args.output_dir}")

    rows = load_router_rows(args.test)
    args.output_dir.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True
    ).to("cuda")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    predictions = []
    for index, row in enumerate(rows, 1):
        prompt = tokenizer.apply_chat_template(row["chosen"][:-1], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=8, do_sample=False,
                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        output = tokenizer.decode(generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        prediction = parse_label(output)
        predictions.append({
            "id": row["id"], "unit_id": row["unit_id"], "task_family": row.get("task_family"),
            "expected": row["label"], "predicted": prediction, "raw_output": output,
            "correct": prediction == row["label"],
        })
        if index % 50 == 0 or index == len(rows):
            print(f"evaluated {index}/{len(rows)}", flush=True)

    confusion = {label: {pred: 0 for pred in (*FIVE_LABELS, "invalid")} for label in FIVE_LABELS}
    for row in predictions:
        confusion[row["expected"]][row["predicted"]] += 1
    label_metrics = {}
    f1_values = []
    for label in FIVE_LABELS:
        tp = confusion[label][label]
        predicted_count = sum(confusion[gold][label] for gold in FIVE_LABELS)
        support = sum(confusion[label].values())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        label_metrics[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    total = len(predictions)
    metrics = {
        "examples": total,
        "accuracy": sum(row["correct"] for row in predictions) / total if total else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values and total else None,
        "labels": list(FIVE_LABELS),
        "per_label": label_metrics,
        "confusion_matrix": confusion,
        "prediction_counts": dict(Counter(row["predicted"] for row in predictions)),
        "base_model": str(args.base_model),
        "adapter": str(args.adapter),
        "label_source": "held-out LLM rankings of five structure-specific solving plans",
        "human_gold": False,
    }
    (args.output_dir / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
