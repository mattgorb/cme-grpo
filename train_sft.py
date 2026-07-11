#!/usr/bin/env python3
"""Exp 3 (forward-KD): SFT a student on teacher generations.

Reads the JSONL produced by scripts/gen_teacher_data.py and fine-tunes the
student (base) model to imitate the teacher's responses. This is the simple
forward-distillation baseline reviewers asked for ("why not just distill?").

Prompt formatting matches train_quality.format_prompt (the student's own chat
template when present, else the Alpaca template), so the SFT'd model is prompted
identically to how it is evaluated in eval_quality / eval_head2head. Loss is
computed on response tokens only (prompt tokens are masked to -100).

Usage (GPU box):
    python train_sft.py \
        --data data/teacher_sft_qwenprompts.jsonl \
        --student Qwen/Qwen2.5-0.5B \
        --output-dir ./outputs/exp3-sft-kd-qwen \
        --epochs 1 --lr 1e-5
"""
from __future__ import annotations

import argparse
import json

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from train_quality import format_prompt

IGNORE = -100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="teacher JSONL (instruction, response)")
    ap.add_argument("--student", required=True, help="base model to SFT (e.g. Qwen/Qwen2.5-0.5B)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Cap steps to match the CME-GRPO token/update budget if desired.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    rows = [r for r in rows if r.get("response", "").strip()]
    print(f"[sft] {len(rows)} teacher examples", flush=True)

    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def encode(ex):
        prompt = format_prompt(ex["instruction"], tok)
        response = ex["response"].strip() + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(response, add_special_tokens=False)["input_ids"]
        input_ids = (p_ids + r_ids)[: args.max_len]
        labels = ([IGNORE] * len(p_ids) + r_ids)[: args.max_len]
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": [1] * len(input_ids)}

    ds = Dataset.from_list(rows).map(
        encode, remove_columns=Dataset.from_list(rows).column_names
    )
    # Drop examples where the prompt filled the whole window (no supervised tokens).
    ds = ds.filter(lambda ex: any(x != IGNORE for x in ex["labels"]))
    print(f"[sft] {len(ds)} usable examples after masking", flush=True)

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad_id = tok.pad_token_id
        input_ids, labels, attn = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * n)
            labels.append(b["labels"] + [IGNORE] * n)
            attn.append(b["attention_mask"] + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }

    model = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=args.seed,
        report_to=["wandb"],
        run_name=f"sft-kd-{args.student.split('/')[-1]}",
    )

    trainer = Trainer(
        model=model, args=targs, train_dataset=ds, data_collator=collate,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"[sft] saved -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
