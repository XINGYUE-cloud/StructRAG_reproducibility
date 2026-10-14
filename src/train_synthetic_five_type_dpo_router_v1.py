"""Train a Qwen2.5 LoRA DPO router on the synthetic five-type extension data."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

from src.five_type_router_data_v1 import FIVE_LABELS, label_counts, load_router_rows
from src.train_official_dpo_router_v3 import batch_loss, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=576)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261013)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for DPO router training.")
    if args.batch_size <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("Batch size and gradient accumulation must be positive.")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    set_seed(args.seed)
    raw_train = load_router_rows(args.train)
    raw_validation = load_router_rows(args.validation)
    args.output_dir.mkdir(parents=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    ).to("cuda")
    initial_adapter_l2 = sum(
        parameter.detach().float().square().sum()
        for parameter in model.parameters()
        if parameter.requires_grad
    ).sqrt().item()

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    batches_per_epoch = math.ceil(len(raw_train) / args.batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation)
    total_updates = math.ceil(args.epochs * updates_per_epoch)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * 0.05)),
        num_training_steps=total_updates,
    )

    best_validation_loss = float("inf")
    history: list[dict[str, float]] = []
    epoch_count = math.ceil(args.epochs)
    for epoch in range(epoch_count):
        epoch_rows = list(raw_train)
        random.Random(args.seed + epoch).shuffle(epoch_rows)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        rows_seen = 0
        for batch_index, start in enumerate(range(0, len(epoch_rows), args.batch_size), start=1):
            batch = epoch_rows[start : start + args.batch_size]
            loss = batch_loss(model, tokenizer, batch, args, training=True)
            (loss / args.gradient_accumulation).backward()
            running_loss += loss.detach().item() * len(batch)
            rows_seen += len(batch)
            if batch_index % args.gradient_accumulation == 0 or batch_index == batches_per_epoch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if batch_index % 25 == 0 or batch_index == batches_per_epoch:
                print(f"epoch {epoch + 1}/{epoch_count}, row {rows_seen}/{len(epoch_rows)}, mean_dpo_loss={running_loss / rows_seen:.6f}")
        validation_loss = evaluate(model, tokenizer, raw_validation, args)
        record = {
            "epoch": float(epoch + 1),
            "train_dpo_loss": running_loss / rows_seen,
            "validation_dpo_loss": validation_loss,
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    final_adapter_l2 = sum(
        parameter.detach().float().square().sum()
        for parameter in model.parameters()
        if parameter.requires_grad
    ).sqrt().item()
    metadata = {
        "experiment": "synthetic five-type router extension",
        "source": "Synthetic five-type dataset generated from supplied person-profile generator",
        "base_model": str(args.base_model),
        "candidate_structures": list(FIVE_LABELS),
        "train_examples": len(raw_train),
        "validation_examples": len(raw_validation),
        "train_label_counts": label_counts(raw_train),
        "validation_label_counts": label_counts(raw_validation),
        "training_objective": "DPO with LoRA",
        "actual_dtype": "bfloat16",
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "lora_rank": args.lora_rank,
            "beta": args.beta,
        },
        "best_validation_dpo_loss": best_validation_loss,
        "adapter_l2_norm": {
            "before_training": initial_adapter_l2,
            "after_training": final_adapter_l2,
            "changed": initial_adapter_l2 != final_adapter_l2,
        },
        "history": history,
        "caveat": "Algorithm and catalogue examples are synthetic because those labels are absent from the public StructRAG router release.",
    }
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
