#!/usr/bin/env python3
"""Exp 3 (forward-KD): generate teacher responses for SFT.

Produces one teacher response per prompt over the SAME 5k UltraFeedback subset
used for CME-GRPO training, so the forward-KD baseline sees the identical prompt
distribution. Output is a JSONL of {"instruction", "response"} consumed by
train_sft.py.

Teacher = the CME verifier (google/gemma-4-E4B-it) so this is "distill from the
same model that provides the CME reward" — exactly the baseline VZUe asked for.

Usage (on the GPU box):
    python scripts/gen_teacher_data.py \
        --config configs/config_exp1c_qwen_revkl.yaml \
        --teacher google/gemma-4-E4B-it \
        --out data/teacher_sft_qwenprompts.jsonl \
        --temperature 0.7 --max-new-tokens 1024
"""
from __future__ import annotations

import argparse
import json
import os
import time

import sys

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow running as `python scripts/gen_teacher_data.py` from the repo root:
# put the repo root (parent of scripts/) on the path so repo modules import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse the EXACT training-prompt selection so KD prompts == GRPO prompts.
from train_quality import build_train_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="A config whose data:/model.generator define the prompt subset.")
    ap.add_argument("--teacher", default="google/gemma-4-E4B-it")
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="0.0 -> greedy. Logged into the output for reproducibility.")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(args.config))

    # Same subset as training: filter uses the STUDENT/generator tokenizer.
    gen_tok = AutoTokenizer.from_pretrained(cfg["model"]["generator"])
    ds = build_train_dataset(cfg, gen_tok)
    instructions = [ex["instruction"] for ex in ds]
    print(f"[teacher] {len(instructions)} prompts (matched to training subset)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.teacher)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    def build_prompt(instruction: str) -> str:
        if tok.chat_template is not None:
            return tok.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=False, add_generation_prompt=True,
            )
        return instruction

    do_sample = args.temperature > 0.0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    n = len(instructions)
    written = 0
    with open(args.out, "w") as f:
        for i in range(0, n, args.batch_size):
            batch = instructions[i : i + args.batch_size]
            prompts = [build_prompt(x) for x in batch]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else None,
                    top_p=0.95 if do_sample else None,
                    pad_token_id=tok.pad_token_id,
                )
            for j, inst in enumerate(batch):
                text = tok.decode(out[j, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                f.write(json.dumps({
                    "instruction": inst,
                    "response": text,
                    "teacher": args.teacher,
                    "temperature": args.temperature,
                }) + "\n")
                written += 1
            if (i // args.batch_size) % 10 == 0:
                el = time.time() - t0
                print(f"[teacher] {written}/{n} | {el:.0f}s | eta={ (n-written)/max(written/el,1e-6):.0f}s", flush=True)
    print(f"[teacher] wrote {written} examples -> {args.out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
