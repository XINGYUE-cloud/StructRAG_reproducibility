"""Train a Qwen2.5 router on the official StructRAG DPO preference pairs.

The upstream public data contains three candidate structures: chunk, table, and
graph. This script deliberately does not invent labels for catalogue or
algorithm, which are absent from those released preference pairs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as functional
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

from src.official_router_data import label_counts, load_official_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=576)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-validation-samples", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-format",
        choices=("qwen_chat", "upstream"),
        default="qwen_chat",
        help="Use qwen_chat for Qwen2.5's native template; upstream reproduces the original script serialization.",
    )
    return parser.parse_args()


def tokenize_pair(tokenizer, prompt: str, completion: str, max_prompt_length: int, max_length: int) -> tuple[list[int], int]:
    """Tokenize a DPO prompt/completion pair and retain labels only in completion."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][-max_prompt_length:]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if not completion_ids:
        raise ValueError("DPO completion tokenized to zero tokens")
    input_ids = (prompt_ids + completion_ids)[:max_length]
    if len(input_ids) <= len(prompt_ids):
        raise ValueError("Completion was truncated out of the DPO sequence")
    return input_ids, len(prompt_ids)


def completion_logprobs(
    model,
    sequences: list[tuple[list[int], int]],
    pad_token_id: int,
) -> torch.Tensor:
    """Return summed completion log probabilities for a padded batch of sequences."""
    maximum_length = max(len(input_ids) for input_ids, _ in sequences)
    input_ids = torch.full(
        (len(sequences), maximum_length), pad_token_id, dtype=torch.long, device="cuda"
    )
    attention_mask = torch.zeros_like(input_ids)
    prompt_lengths = torch.empty(len(sequences), dtype=torch.long, device="cuda")
    lengths = torch.empty(len(sequences), dtype=torch.long, device="cuda")
    for index, (sequence, prompt_length) in enumerate(sequences):
        length = len(sequence)
        input_ids[index, :length] = torch.tensor(sequence, dtype=torch.long, device="cuda")
        attention_mask[index, :length] = 1
        prompt_lengths[index] = prompt_length
        lengths[index] = length

    logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, :-1]
    targets = input_ids[:, 1:]
    token_logprobs = functional.log_softmax(logits.float(), dim=-1).gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)
    positions = torch.arange(1, maximum_length, device=input_ids.device).unsqueeze(0)
    completion_mask = (positions >= prompt_lengths.unsqueeze(1)) & (positions < lengths.unsqueeze(1))
    return (token_logprobs * completion_mask).sum(dim=1)


def batch_loss(model, tokenizer, rows: list[dict], args: argparse.Namespace, training: bool) -> torch.Tensor:
    """Compute standard DPO loss using the base model with adapters disabled as reference."""
    chosen_sequences: list[tuple[list[int], int]] = []
    rejected_sequences: list[tuple[list[int], int]] = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            row["chosen"][:-1], tokenize=False, add_generation_prompt=True
        )
        chosen = str(row["chosen"][-1]["content"]).strip().lower() + tokenizer.eos_token
        rejected = str(row["rejected"][-1]["content"]).strip().lower() + tokenizer.eos_token
        chosen_sequences.append(
            tokenize_pair(tokenizer, prompt, chosen, args.max_prompt_length, args.max_length)
        )
        rejected_sequences.append(
            tokenize_pair(tokenizer, prompt, rejected, args.max_prompt_length, args.max_length)
        )
    sequences = chosen_sequences + rejected_sequences

    if training:
        policy_logprobs = completion_logprobs(model, sequences, tokenizer.pad_token_id)
    else:
        with torch.no_grad():
            policy_logprobs = completion_logprobs(model, sequences, tokenizer.pad_token_id)
    with torch.no_grad(), model.disable_adapter():
        reference_logprobs = completion_logprobs(model, sequences, tokenizer.pad_token_id)
    split = len(rows)
    policy_chosen, policy_rejected = policy_logprobs[:split], policy_logprobs[split:]
    reference_chosen, reference_rejected = reference_logprobs[:split], reference_logprobs[split:]
    logits = args.beta * (
        (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)
    )
    return -functional.logsigmoid(logits).mean()


def evaluate(model, tokenizer, rows: list[dict], args: argparse.Namespace) -> float:
    model.eval()
    losses = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        losses.append(batch_loss(model, tokenizer, batch, args, training=False).item() * len(batch))
    return sum(losses) / len(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for DPO router training.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.prompt_format != "qwen_chat":
        raise ValueError("This Qwen2.5 adaptation requires --prompt-format qwen_chat.")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_train = load_official_pairs(args.train)
    raw_validation = load_official_pairs(args.validation)
    if args.max_train_samples:
        raw_train = raw_train[: args.max_train_samples]
    if args.max_validation_samples:
        raw_validation = raw_validation[: args.max_validation_samples]
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    # The 1.5B model fits comfortably on one 48GB A6000. Avoid activation
    # checkpointing here because frozen embeddings can detach checkpoint inputs
    # and prevent LoRA weights from receiving gradients.
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
    global_update = 0
    epoch_count = math.ceil(args.epochs)
    for epoch in range(epoch_count):
        rng = random.Random(args.seed + epoch)
        epoch_rows = list(raw_train)
        rng.shuffle(epoch_rows)
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
                global_update += 1
            if batch_index % 25 == 0 or batch_index == batches_per_epoch:
                print(
                    f"epoch {epoch + 1}/{epoch_count}, row {rows_seen}/{len(epoch_rows)}, "
                    f"mean_dpo_loss={running_loss / rows_seen:.6f}"
                )
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
        "source": "Official StructRAG train_router/data preference pairs",
        "base_model": str(args.base_model),
        "candidate_structures": ["chunk", "table", "graph"],
        "train_examples": len(raw_train),
        "validation_examples": len(raw_validation),
        "train_label_counts": label_counts(raw_train),
        "validation_label_counts": label_counts(raw_validation),
        "training_objective": "DPO with LoRA",
        "actual_dtype": "bfloat16",
        "prompt_format": "qwen_chat",
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "lora_rank": args.lora_rank,
            "beta": args.beta,
        },
        "caveat": (
            "The released upstream router data has three labels, not the five candidate "
            "types described in the paper. This run reproduces the released data scope."
        ),
        "best_validation_dpo_loss": best_validation_loss,
        "adapter_l2_norm": {
            "before_training": initial_adapter_l2,
            "after_training": final_adapter_l2,
            "changed": initial_adapter_l2 != final_adapter_l2,
        },
        "history": history,
        "implementation": (
            "Explicit LoRA DPO loss loop. The reference log probabilities are computed with the "
            "same base model and LoRA adapter disabled, equivalent to a frozen base reference."
        ),
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
