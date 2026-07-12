#!/usr/bin/env python3
"""Generate greedy responses for one model on AlpacaEval prompts -> outputs.json.

API-free (no judge). Used to feed drift_metrics.py so the decisive collapse
comparison costs zero judge calls.

Usage:
    python scripts/gen_outputs.py --model ./outputs/drift-qwenbase-groupmean \
        --out results/drift/groupmean_outputs.json --num 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval_quality import generate_for_model, load_alpacaeval_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local checkpoint dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--label", default="model")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prompts = load_alpacaeval_prompts()[: args.num]
    resp = generate_for_model(args.model, prompts, device,
                              args.max_new_tokens, args.batch_size, args.label)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump([{"instruction": i, "output": o} for i, o in zip(prompts, resp)],
              open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"[gen_outputs] wrote {len(resp)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
