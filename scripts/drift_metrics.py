#!/usr/bin/env python3
"""Exp 4: drift / collapse diagnostics over generated outputs.

Consumes one or more *_outputs.json files (list of {"instruction","output"}) —
e.g. condition A's cached finetuned_outputs.json, condition C's h2h outputs, the
beta-sweep outputs, and the teacher's own generations — and reports, per
condition:
  * response length (tokens): mean / median / std
  * distinct-1, distinct-2 (corpus lexical diversity; low => collapse)
  * self-BLEU (higher => more repetitive across the corpus; sampled for cost)
  * similarity-to-teacher: mean cosine of sentence embeddings between each output
    and the teacher's output for the same prompt (requires --teacher-outputs and
    sentence-transformers; skipped with a note otherwise)

Prediction to test: the no-sigma runs (Exp 1 C, Exp 2 beta=0) are shorter, less
diverse, and more teacher-similar than group_std (condition A) — the collapse
that group standardization mitigates. This is the figure backing the corrected
Appendix C.

Usage:
  python scripts/drift_metrics.py \
    --cond A_group_std=/path/config6/quality_eval/finetuned_outputs.json \
    --cond C_group_mean=results/exp1_qwen_AvsC/revKL_group_mean_outputs.json \
    --cond beta0=results/exp2_beta0/.../outputs.json \
    --teacher-outputs data/teacher_outputs_on_alpaca.json \
    --tokenizer Qwen/Qwen2.5-0.5B \
    --out results/drift/qwen_drift.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter


def load(path):
    data = json.load(open(path))
    return {d["instruction"]: d.get("output", d.get("response", "")) for d in data}


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(all_tokens, n):
    total, uniq = 0, set()
    for toks in all_tokens:
        gs = ngrams(toks, n)
        total += len(gs)
        uniq.update(gs)
    return len(uniq) / max(total, 1)


def bleu_against(cand, refs, max_n=4):
    """Light BLEU: geometric mean of clipped n-gram precisions + brevity penalty."""
    import math
    if not cand:
        return 0.0
    weights = []
    for n in range(1, max_n + 1):
        cg = Counter(ngrams(cand, n))
        if not cg:
            weights.append(1e-9)
            continue
        max_ref = Counter()
        for r in refs:
            rg = Counter(ngrams(r, n))
            for k, v in rg.items():
                if v > max_ref[k]:
                    max_ref[k] = v
        clipped = sum(min(c, max_ref[k]) for k, c in cg.items())
        weights.append(clipped / max(sum(cg.values()), 1) or 1e-9)
    gm = math.exp(sum(math.log(max(w, 1e-9)) for w in weights) / max_n)
    ref_len = min((len(r) for r in refs), default=len(cand))
    bp = 1.0 if len(cand) > ref_len else math.exp(1 - ref_len / max(len(cand), 1))
    return bp * gm


def self_bleu(all_tokens, sample=80, seed=42):
    rng = random.Random(seed)
    idx = list(range(len(all_tokens)))
    picks = idx if len(idx) <= sample else rng.sample(idx, sample)
    refpool = [all_tokens[i] for i in idx]
    scores = []
    for i in picks:
        refs = rng.sample([refpool[j] for j in idx if j != i],
                          k=min(15, len(idx) - 1))
        scores.append(bleu_against(all_tokens[i], refs))
    return sum(scores) / max(len(scores), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", action="append", required=True,
                    help="label=path.json (repeatable)")
    ap.add_argument("--teacher-outputs", default=None)
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer for token-length; falls back to whitespace.")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tokenize = str.split
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer
            _tk = AutoTokenizer.from_pretrained(args.tokenizer)
            tokenize = lambda s: _tk.tokenize(s)
        except Exception as e:
            print(f"[drift] tokenizer load failed ({e}); using whitespace.", flush=True)

    teacher = load(args.teacher_outputs) if args.teacher_outputs else None
    embedder = None
    if teacher is not None:
        try:
            from sentence_transformers import SentenceTransformer, util
            embedder = (SentenceTransformer(args.embed_model), util)
        except Exception as e:
            print(f"[drift] sentence-transformers unavailable ({e}); "
                  f"skipping teacher-similarity.", flush=True)

    report = {}
    for spec in args.cond:
        label, path = spec.split("=", 1)
        m = load(path)
        instrs = list(m.keys())
        texts = [m[i] for i in instrs]
        toks = [tokenize(t) for t in texts]
        lens = sorted(len(t) for t in toks)
        n = len(lens)
        mean_len = sum(lens) / max(n, 1)
        median_len = lens[n // 2] if n else 0
        var = sum((x - mean_len) ** 2 for x in lens) / max(n, 1)

        row = {
            "n": n,
            "len_mean": round(mean_len, 1),
            "len_median": median_len,
            "len_std": round(var ** 0.5, 1),
            "distinct_1": round(distinct_n(toks, 1), 4),
            "distinct_2": round(distinct_n(toks, 2), 4),
            "self_bleu": round(self_bleu(toks), 4),
        }

        if embedder is not None:
            model, util = embedder
            paired = [(m[i], teacher[i]) for i in instrs if i in teacher]
            if paired:
                ga = model.encode([p[0] for p in paired], convert_to_tensor=True,
                                  normalize_embeddings=True)
                gt = model.encode([p[1] for p in paired], convert_to_tensor=True,
                                  normalize_embeddings=True)
                cos = util.cos_sim(ga, gt).diagonal()
                row["teacher_cos_sim_mean"] = round(float(cos.mean()), 4)
                row["teacher_cos_sim_n"] = len(paired)

        report[label] = row
        print(f"[drift] {label}: {row}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"[drift] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
