#!/usr/bin/env python3
"""Arbitrary A-vs-B pairwise LLM-judge eval (rebuttal Exp 1/2/3).

eval_quality.py is hardwired to base/finetuned/instruct/reference. This script
compares any TWO systems head-to-head, where each side is EITHER:
  * a cached outputs JSON  (list of {"instruction","output"}), e.g. condition A's
    finetuned_outputs.json  -> reused, no regeneration, or
  * a model name / checkpoint path              -> greedy-generated here.

When one side is cached, its instruction list is the canonical prompt set and the
other side is generated on exactly those prompts (guarantees alignment). Runs
every judge in --judges and reports tie-adjusted win rate for A,
(wins_A + 0.5*ties)/N, mean across judges.

Examples:
  # A (reuse cached) vs C (new checkpoint) — the core Exp-1 exhibit
  python eval_head2head.py \
    --outputs-a /path/cme-grpo-quality-config6/quality_eval/finetuned_outputs.json \
    --label-a CME_group_std \
    --model-b ./outputs/exp1c-qwen-revkl --label-b revKL_group_mean \
    --out-dir results/exp1_qwen_AvsC

  # C vs base
  python eval_head2head.py --model-a ./outputs/exp1c-qwen-revkl --label-a revKL \
    --model-b Qwen/Qwen2.5-0.5B --label-b base --out-dir results/exp1_qwen_Cvsbase
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from eval_quality import generate_for_model, judge_pairwise, load_alpacaeval_prompts


def _load_cached(path: str):
    data = json.load(open(path))
    instr = [d["instruction"] for d in data]
    resp = [d.get("output", d.get("response", "")) for d in data]
    return instr, resp


def _resolve_side(model, outputs, label, canonical_instr, device, max_new_tokens, batch_size):
    """Return responses aligned to canonical_instr for one side."""
    if outputs:
        instr, resp = _load_cached(outputs)
        by_instr = {i: r for i, r in zip(instr, resp)}
        missing = [i for i in canonical_instr if i not in by_instr]
        if missing:
            raise SystemExit(
                f"[{label}] cached outputs missing {len(missing)} of the canonical "
                f"prompts; cannot align. Regenerate this side as a model instead.")
        return [by_instr[i] for i in canonical_instr]
    return generate_for_model(model, canonical_instr, device, max_new_tokens, batch_size, label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a"); ap.add_argument("--outputs-a")
    ap.add_argument("--model-b"); ap.add_argument("--outputs-b")
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--judges", default="gpt-5.2,claude-sonnet-4-6")
    ap.add_argument("--num-samples", type=int, default=None,
                    help="Limit prompts (default: all cached, else 805).")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not (args.model_a or args.outputs_a) or not (args.model_b or args.outputs_b):
        raise SystemExit("Each side needs --model-X or --outputs-X.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Canonical prompt set: prefer a cached side's instructions (guarantees the
    # reused generations align); else fall back to the AlpacaEval 805.
    if args.outputs_a:
        canonical, _ = _load_cached(args.outputs_a)
    elif args.outputs_b:
        canonical, _ = _load_cached(args.outputs_b)
    else:
        canonical = load_alpacaeval_prompts()
    if args.num_samples:
        canonical = canonical[: args.num_samples]
    print(f"[h2h] {len(canonical)} prompts | {args.label_a} vs {args.label_b}", flush=True)

    resp_a = _resolve_side(args.model_a, args.outputs_a, args.label_a, canonical,
                           device, args.max_new_tokens, args.batch_size)
    resp_b = _resolve_side(args.model_b, args.outputs_b, args.label_b, canonical,
                           device, args.max_new_tokens, args.batch_size)

    # Persist generations for reuse / drift metrics.
    for lbl, r in [(args.label_a, resp_a), (args.label_b, resp_b)]:
        json.dump([{"instruction": i, "output": o} for i, o in zip(canonical, r)],
                  open(os.path.join(args.out_dir, f"{lbl}_outputs.json"), "w"),
                  indent=2, ensure_ascii=False)

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    summary = {}
    for judge in judges:
        print(f"\n[h2h] judge={judge}", flush=True)
        res = judge_pairwise(canonical, resp_a, resp_b, judge, args.label_a, args.label_b)
        wa, wb, ties = res["wins_a"], res["wins_b"], res["ties"]
        n = wa + wb + ties
        wr_a = (wa + 0.5 * ties) / max(n, 1)
        summary[judge] = {"wins_a": wa, "wins_b": wb, "ties": ties, "n": n, "wr_a": wr_a}
        json.dump(res, open(os.path.join(args.out_dir, f"judge_{judge.replace('.','_')}.json"), "w"),
                  indent=2, ensure_ascii=False)
        print(f"[h2h] {judge}: WR({args.label_a})={wr_a:.1%}  "
              f"({wa}/{wb}/{ties} win/loss/tie)", flush=True)

    mean_wr = sum(s["wr_a"] for s in summary.values()) / len(summary)
    summary["mean_wr_a"] = mean_wr
    summary["label_a"], summary["label_b"] = args.label_a, args.label_b
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=2)
    print(f"\n[h2h] MEAN tie-adjusted WR({args.label_a}) vs {args.label_b} "
          f"across {len(judges)} judges = {mean_wr:.1%}", flush=True)


if __name__ == "__main__":
    main()
