"""For each MATH-500 problem, generate N samples with the generator and score each
under BOTH the generator (self-perplexity) and the verifier (cross-model perplexity).

Per sample we record:
  ppl_gen_full    — generator perplexity over full response
  ppl_ver_full    — verifier perplexity over full response
  ppl_gen_answer  — generator perplexity over \\boxed{} span only
  ppl_ver_answer  — verifier perplexity over \\boxed{} span only
  correct         — does extract_boxed match gold?

Aggregates correct vs wrong with AUROC(wrong>correct) per metric.

Usage:
    python analyze_cme_correctness.py --config config1.yaml --max-samples 100 --num-generations 8
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from typing import List, Optional, Tuple

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval import extract_boxed, format_prompt, is_correct
from reward import _find_boxed_span


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def generate_responses(
    model, tokenizer, problem: str, n: int, max_new_tokens: int,
    temperature: float, device: str,
) -> List[str]:
    prompt = format_prompt(problem, tokenizer)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    do_sample = n > 1 and temperature > 0
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        num_return_sequences=n,
        pad_token_id=tokenizer.pad_token_id,
    )
    return [
        tokenizer.decode(out[i, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for i in range(out.shape[0])
    ]


@torch.no_grad()
def score_response(
    model, tokenizer, prompt: str, response: str, max_length: int, device: str,
) -> Tuple[float, float, float, float]:
    """Return (ce_full, ce_answer, entropy_full, entropy_answer).

    Entropy is the per-step predictive entropy -sum p*log p over the vocab,
    averaged over response positions. NaN for answer if no \\boxed{}.
    """
    prompt_enc = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    resp_enc = tokenizer(
        response, add_special_tokens=False, return_tensors="pt",
        return_offsets_mapping=True,
    )
    prompt_ids = prompt_enc.input_ids[0]
    resp_ids = resp_enc.input_ids[0]
    offsets = resp_enc.offset_mapping[0].tolist()
    if resp_ids.numel() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    full = torch.cat([prompt_ids, resp_ids], dim=0)
    if full.shape[0] > max_length:
        overflow = full.shape[0] - max_length
        full = full[overflow:]
        prompt_len = max(1, prompt_ids.shape[0] - overflow)
    else:
        prompt_len = prompt_ids.shape[0]

    input_ids = full.unsqueeze(0).to(device)
    labels = input_ids.clone()
    labels[0, :prompt_len] = -100

    logits = model(input_ids=input_ids).logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    per_tok_ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(shift_labels.shape)

    # Predictive entropy per position: -sum(p * log p)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    per_tok_ent = -(probs * log_probs).sum(dim=-1)  # [1, T]

    resp_mask = shift_labels[0] != -100
    resp_ce = per_tok_ce[0][resp_mask].cpu()
    resp_ent = per_tok_ent[0][resp_mask].cpu()
    if resp_ce.numel() == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    ce_full = float(resp_ce.mean().item())
    ent_full = float(resp_ent.mean().item())

    span = _find_boxed_span(response)
    if span is None:
        return ce_full, float("nan"), ent_full, float("nan")
    a, b = span
    v_off = offsets[: resp_ce.numel()]
    keep_idx = [
        j for j, (va, vb) in enumerate(v_off)
        if va != vb and vb > a and va < b
    ]
    if not keep_idx:
        return ce_full, float("nan"), ent_full, float("nan")
    ce_ans = sum(resp_ce[j].item() for j in keep_idx) / len(keep_idx)
    ent_ans = sum(resp_ent[j].item() for j in keep_idx) / len(keep_idx)
    return ce_full, ce_ans, ent_full, ent_ans


def auroc(scores: List[float], labels: List[int]) -> Optional[float]:
    paired = [(s, l) for s, l in zip(scores, labels) if not math.isnan(s)]
    pos = [s for s, l in paired if l == 1]
    neg = [s for s, l in paired if l == 0]
    if not pos or not neg:
        return None
    wins = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(
        1 for p in pos for n in neg if p == n
    )
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config1.yaml")
    ap.add_argument("--benchmark", default="math500")
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--output", default="cme_correctness.csv")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    bench = next(b for b in cfg["benchmarks"] if b["name"] == args.benchmark)

    gen_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ver_device = "cuda:1" if torch.cuda.device_count() > 1 else gen_device
    gen_max_len = cfg["reward"]["max_verifier_length"]
    ver_max_len = cfg["reward"]["max_verifier_length"]

    print(f"Loading generator {cfg['model']['generator']} on {gen_device}")
    gen_tok = AutoTokenizer.from_pretrained(cfg["model"]["generator"])
    if gen_tok.pad_token is None:
        gen_tok.pad_token = gen_tok.eos_token
    gen_tok.padding_side = "left"
    gen_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["generator"], torch_dtype=torch.bfloat16,
    ).to(gen_device)
    gen_model.eval()

    print(f"Loading verifier {cfg['model']['verifier']} on {ver_device}")
    ver_tok = AutoTokenizer.from_pretrained(cfg["model"]["verifier"])
    if ver_tok.pad_token is None:
        ver_tok.pad_token = ver_tok.eos_token
    ver_model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["verifier"], torch_dtype=torch.bfloat16,
    ).to(ver_device)
    ver_model.eval()

    ds = load_dataset(bench["dataset"], split=bench["split"])
    if bench["dataset"] == "HuggingFaceH4/MATH-500" and len(ds) > args.max_samples:
        ds = ds.shuffle(seed=42).select(range(args.max_samples))
    else:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    rows = []
    for i, ex in enumerate(ds):
        problem = ex[bench["problem_key"]]
        gold = ex[bench["answer_key"]]

        gen_prompt = format_prompt(problem, gen_tok)
        ver_prompt = format_prompt(problem, ver_tok)

        responses = generate_responses(
            gen_model, gen_tok, problem,
            n=args.num_generations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=gen_device,
        )

        for k, response in enumerate(responses):
            pred = extract_boxed(response)
            correct = is_correct(pred, gold)

            ce_g_full, ce_g_ans, ent_g_full, ent_g_ans = score_response(
                gen_model, gen_tok, gen_prompt, response, gen_max_len, gen_device
            )
            ce_v_full, ce_v_ans, ent_v_full, ent_v_ans = score_response(
                ver_model, ver_tok, ver_prompt, response, ver_max_len, ver_device
            )

            def _exp(x):
                return math.exp(x) if not math.isnan(x) else float("nan")

            if not pred:
                status_label = "none"
            elif correct:
                status_label = "correct"
            else:
                status_label = "incorrect"

            row = {
                "idx": i,
                "gen": k,
                "gold": gold,
                "pred": pred or "",
                "correct": int(correct),
                "status": status_label,
                "ppl_gen_full": _exp(ce_g_full),
                "ppl_ver_full": _exp(ce_v_full),
                "ppl_gen_answer": _exp(ce_g_ans),
                "ppl_ver_answer": _exp(ce_v_ans),
                "entropy_gen_full": ent_g_full,
                "entropy_ver_full": ent_v_full,
                "entropy_gen_answer": ent_g_ans,
                "entropy_ver_answer": ent_v_ans,
                "ce_gen_full": ce_g_full,
                "ce_ver_full": ce_v_full,
                "ce_gen_answer": ce_g_ans,
                "ce_ver_answer": ce_v_ans,
                "response_chars": len(response),
            }
            rows.append(row)
            status = "✓" if correct else "✗"
            ans_g = f"{row['ppl_gen_answer']:8.2f}" if not math.isnan(row["ppl_gen_answer"]) else "     NaN"
            ans_v = f"{row['ppl_ver_answer']:8.2f}" if not math.isnan(row["ppl_ver_answer"]) else "     NaN"
            eg = f"{ent_g_full:.2f}" if not math.isnan(ent_g_full) else "NaN"
            ev = f"{ent_v_full:.2f}" if not math.isnan(ent_v_full) else "NaN"
            print(
                f"[{i+1}/{len(ds)} g{k}] {status} pred={pred!r:16.16} gold={gold!r:10.10} | "
                f"PPL-gen full={row['ppl_gen_full']:7.2f} ans={ans_g} entropy={eg} | "
                f"PPL-ver full={row['ppl_ver_full']:7.2f} ans={ans_v} entropy={ev}"
            )

        # Running averages over all generations seen so far, split into 3 buckets:
        # correct / incorrect (has boxed answer but wrong) / none (no boxed answer).
        metric_keys = [
            "ppl_gen_full", "ppl_gen_answer",
            "ppl_ver_full", "ppl_ver_answer",
            "entropy_gen_full", "entropy_gen_answer",
            "entropy_ver_full", "entropy_ver_answer",
        ]
        def _mean(key, label):
            vals = [
                r[key] for r in rows
                if r["status"] == label and not math.isnan(r[key])
            ]
            return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)

        n_c = sum(1 for r in rows if r["status"] == "correct")
        n_i = sum(1 for r in rows if r["status"] == "incorrect")
        n_n = sum(1 for r in rows if r["status"] == "none")
        print(f"  [running after problem {i+1}] correct={n_c} incorrect={n_i} none={n_n}")
        for key in metric_keys:
            mc, cc = _mean(key, "correct")
            mi, ci = _mean(key, "incorrect")
            mn, cn = _mean(key, "none")
            def _fmt(v):
                return f"{v:8.3f}" if not math.isnan(v) else "     NaN"
            print(
                f"    {key:22s}  "
                f"correct={_fmt(mc)} (n={cc})  "
                f"incorrect={_fmt(mi)} (n={ci})  "
                f"none={_fmt(mn)} (n={cn})"
            )

    if not rows:
        print("No rows produced.")
        return

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.output}")

    n_correct = sum(r["correct"] for r in rows)
    total = len(rows)
    print(f"\nTotal generations: {total}  correct: {n_correct}  ({n_correct/total:.3f})")

    def summarize(label: str, key: str):
        cor = [r[key] for r in rows if r["correct"] and not math.isnan(r[key])]
        wro = [r[key] for r in rows if not r["correct"] and not math.isnan(r[key])]
        if not cor or not wro:
            print(f"{label}: insufficient data (cor={len(cor)} wro={len(wro)})")
            return
        au = auroc([r[key] for r in rows], [1 - r["correct"] for r in rows])
        au_str = f"{au:.3f}" if au is not None else "NA"
        print(
            f"{label}: correct mean={statistics.mean(cor):7.3f} med={statistics.median(cor):7.3f} | "
            f"wrong mean={statistics.mean(wro):7.3f} med={statistics.median(wro):7.3f} | "
            f"AUROC(wrong>correct)={au_str}"
        )

    print()
    summarize("PPL-gen full   ", "ppl_gen_full")
    summarize("PPL-gen answer ", "ppl_gen_answer")
    summarize("PPL-ver full   ", "ppl_ver_full")
    summarize("PPL-ver answer ", "ppl_ver_answer")
    summarize("H-gen full     ", "entropy_gen_full")
    summarize("H-gen answer   ", "entropy_gen_answer")
    summarize("H-ver full     ", "entropy_ver_full")
    summarize("H-ver answer   ", "entropy_ver_answer")


if __name__ == "__main__":
    main()
