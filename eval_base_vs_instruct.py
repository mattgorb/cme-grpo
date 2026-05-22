"""Pairwise base-vs-instruct evaluation on AlpacaEval 2.0 (805 prompts).

For each (base, instruct) model pair listed below, generates greedy responses
from both, then runs pairwise GPT-5.2 + Claude Sonnet 4.6 judgments and
reports tie-adjusted win rates.

Outputs are cached per pair under outputs/base_vs_instruct/<slug>/.

Setup:
    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...

Usage:
    python eval_base_vs_instruct.py
"""
from __future__ import annotations

import csv
import json
import os
import random

import torch

from eval_quality import (
    load_alpacaeval_prompts,
    generate_for_model,
    save_outputs_json,
    judge_pairwise,
)

# (base_model, instruct_model) pairs.
PAIRS = [
    ("Qwen/Qwen2.5-0.5B",           "Qwen/Qwen2.5-0.5B-Instruct"),
    ("meta-llama/Llama-3.2-1B",     "meta-llama/Llama-3.2-1B-Instruct"),
    ("google/gemma-3-1b-pt",        "google/gemma-3-1b-it"),
    ("allenai/OLMo-2-0425-1B-SFT",  "allenai/OLMo-2-0425-1B-DPO"),
]

JUDGES = ["gpt-5.2"]
NUM_SAMPLES = 200
OUTPUT_ROOT = "outputs/base_vs_instruct"
MAX_NEW_TOKENS = 2048
BATCH_SIZE = 8
SEED = 42

random.seed(SEED)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
instructions = load_alpacaeval_prompts()
print(f"Loaded {len(instructions)} AlpacaEval prompts")
if NUM_SAMPLES is not None and NUM_SAMPLES < len(instructions):
    sub_rng = random.Random(SEED)
    idx = sub_rng.sample(range(len(instructions)), k=NUM_SAMPLES)
    instructions = [instructions[i] for i in idx]
    print(f"Subsampled {len(instructions)} prompts (seed={SEED})")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

summary_rows = []
for base_model, instruct_model in PAIRS:
    pair_slug = f"{base_model.split('/')[-1]}__vs__{instruct_model.split('/')[-1]}"
    out_dir = os.path.join(OUTPUT_ROOT, pair_slug)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 60}\nPair: {pair_slug}\n{'=' * 60}", flush=True)

    responses = {}
    for label, model_name in [("base", base_model), ("instruct", instruct_model)]:
        out_path = os.path.join(out_dir, f"{label}_outputs.json")
        if os.path.exists(out_path):
            with open(out_path) as f:
                responses[label] = [d["output"] for d in json.load(f)]
            print(f"[{label}] loaded {len(responses[label])} cached -> {out_path}", flush=True)
            continue
        responses[label] = generate_for_model(
            model_name, instructions, device,
            max_new_tokens=MAX_NEW_TOKENS, batch_size=BATCH_SIZE, label=label,
        )
        save_outputs_json(out_path, model_name, instructions, responses[label])
        print(f"[{label}] saved -> {out_path}", flush=True)

    for judge_model in JUDGES:
        judge_slug = judge_model.replace("/", "_").replace(".", "_")
        result_path = os.path.join(out_dir, f"judge_base_vs_instruct__{judge_slug}.json")
        if os.path.exists(result_path):
            with open(result_path) as f:
                result = json.load(f)
            print(f"[{judge_model}] cached -> {result_path}", flush=True)
        else:
            print(f"\nJudging ({judge_model}): base vs instruct", flush=True)
            result = judge_pairwise(
                instructions, responses["base"], responses["instruct"],
                judge_model=judge_model, label_a="base", label_b="instruct",
            )
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[{judge_model}] saved -> {result_path}", flush=True)
        summary_rows.append({
            "pair": pair_slug,
            "judge": judge_model,
            "wins_base": result["wins_a"],
            "wins_instruct": result["wins_b"],
            "ties": result["ties"],
            "total": result["total"],
            "winrate_base": result["winrate_a"],
            "winrate_instruct": result["winrate_b"],
        })

print(f"\n{'=' * 80}\nSUMMARY — tie-adjusted win rates (base vs instruct)\n{'=' * 80}")
print(f"{'pair':<55} {'judge':<22} {'WR-base':>8} {'WR-instruct':>12}")
for r in summary_rows:
    print(f"{r['pair']:<55} {r['judge']:<22} {r['winrate_base']:>7.1%} {r['winrate_instruct']:>11.1%}")

summary_path = os.path.join(OUTPUT_ROOT, "summary.csv")
with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    w.writerows(summary_rows)
print(f"\nwrote {summary_path}")
