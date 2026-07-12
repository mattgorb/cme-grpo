#!/usr/bin/env python3
"""Extract per-step collapse curves from training logs -> long-format CSV.

train_quality.py prints one dict per logged step (logging_steps=1), containing
'entropy' and 'completions/mean_length' — the two live collapse signals. This
pulls them into a CSV you can plot (entropy / length vs step), overlaying
group_std vs group_mean to show HOW FAST each collapses.

Usage:
    python scripts/extract_curves.py \
        group_std=logs/drift_groupstd.log \
        group_mean=logs/drift_groupmean.log \
        --out results/drift/curves.csv
"""
from __future__ import annotations

import argparse
import csv
import re

FIELDS = {
    "entropy": r"'entropy':\s*'?([-\d.eE]+)'?",
    "mean_length": r"'completions/mean_length':\s*'?([-\d.eE]+)'?",
    "reward": r"'reward':\s*'?([-\d.eE]+)'?",
    "kl": r"'kl':\s*'?([-\d.eE]+)'?",
}


def parse(path):
    rows = []
    step = 0
    for line in open(path, errors="ignore"):
        if "'entropy'" not in line:
            continue
        rec = {}
        for name, pat in FIELDS.items():
            m = re.search(pat, line)
            rec[name] = float(m.group(1)) if m else ""
        step += 1                       # logging_steps=1 -> one dict per step
        rec["step"] = step
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="label=logfile ...")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_rows = []
    for spec in args.specs:
        label, path = spec.split("=", 1)
        rows = parse(path)
        for r in rows:
            r["condition"] = label
        out_rows.extend(rows)
        if rows:
            print(f"[curves] {label}: {len(rows)} steps  "
                  f"entropy {rows[0]['entropy']:.3f} -> {rows[-1]['entropy']:.3f}  "
                  f"len {rows[0]['mean_length']:.0f} -> {rows[-1]['mean_length']:.0f}")
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "step", "entropy", "mean_length", "reward", "kl"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"[curves] wrote {len(out_rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
