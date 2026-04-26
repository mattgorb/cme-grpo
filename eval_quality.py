"""Pairwise quality evaluation on the AlpacaEval 2.0 prompt set (805 prompts).

Generates responses for FOUR models — base, finetuned, instruct, and a small
reference model (default: google/gemma-3n-E2B-it) — then runs FIVE pairwise
GPT-5.2 judge comparisons:

  1. base       vs  finetuned
  2. finetuned  vs  instruct
  3. base       vs  reference
  4. finetuned  vs  reference
  5. instruct   vs  reference

Each pair is judged by GPT-5.2 with randomized presentation order to mitigate
position bias. Results are reported as tie-adjusted win rates
((wins + 0.5 * ties) / total). Per-sample verdicts and reasons are saved.

Generation outputs and judge results are cached on disk; rerunning skips
already-completed steps.

Setup:
    pip install openai
    export OPENAI_API_KEY=sk-...

Usage:
    python eval_quality.py \\
        --config config_quality1_token_level.yaml \\
        --checkpoint ./outputs/.../best-vs-base

    # Custom reference model:
    python eval_quality.py \\
        --config config_quality1_token_level.yaml \\
        --checkpoint ./outputs/.../best-vs-base \\
        --reference-model google/gemma-3n-E4B-it
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from typing import Optional

import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def format_prompt(instruction: str, tokenizer) -> str:
    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False, add_generation_prompt=True,
        )
    return PROMPT_TEMPLATE.format(instruction=instruction)


def load_alpacaeval_prompts() -> list[str]:
    """Download and parse the AlpacaEval 2.0 evaluation set (805 prompts)."""
    path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="alpaca_eval.json",
        repo_type="dataset",
    )
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [r["instruction"] for r in records]


def _resolve_path(p: str) -> str:
    """Normalize local paths so transformers doesn't misread them as repo IDs."""
    if (("/" in p and p.count("/") != 1)
            or p.startswith(".")
            or os.path.isdir(p)):
        abs_p = os.path.abspath(p)
        if not os.path.isdir(abs_p):
            raise FileNotFoundError(f"Local checkpoint not found: {abs_p}")
        return abs_p
    return p


def generate_for_model(
    model_name_or_path: str,
    instructions: list[str],
    device: str,
    max_new_tokens: int,
    batch_size: int,
    label: str,
) -> list[str]:
    """Load model, greedy-decode responses for all instructions, unload."""
    model_name_or_path = _resolve_path(model_name_or_path)
    print(f"\n[{label}] Loading {model_name_or_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    prompts = [format_prompt(inst, tokenizer) for inst in instructions]
    responses: list[str] = []
    n_total = len(prompts)
    t_start = time.time()
    print(f"[{label}] generating {n_total} responses (batch_size={batch_size}, max_new_tokens={max_new_tokens})...", flush=True)

    for i in range(0, n_total, batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=2048,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j in range(len(batch)):
            gen_ids = out[j, enc.input_ids.shape[1]:]
            responses.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
        done = i + len(batch)
        elapsed = time.time() - t_start
        rate = done / max(elapsed, 1e-6)
        eta = (n_total - done) / max(rate, 1e-6)
        if (i // batch_size) % 10 == 0:
            print(f"[{label}]   {done}/{n_total} | elapsed={elapsed:.0f}s | eta={eta:.0f}s", flush=True)

    print(f"[{label}] done: {n_total} responses in {time.time()-t_start:.0f}s", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return responses


def save_outputs_json(path: str, generator_name: str,
                     instructions: list[str], responses: list[str]) -> None:
    """Save responses in AlpacaEval-compatible format."""
    data = [
        {"instruction": inst, "output": resp, "generator": generator_name}
        for inst, resp in zip(instructions, responses)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def judge_pairwise(
    instructions: list[str],
    responses_a: list[str],
    responses_b: list[str],
    judge_model: str,
    label_a: str,
    label_b: str,
    log_every: int = 25,
) -> dict:
    """LLM-as-judge pairwise comparison.

    Each prompt is shown to the judge with randomized A/B presentation order
    to mitigate position bias. Returns aggregate stats plus per-sample
    verdicts and reasons.
    """
    from openai import OpenAI
    client = OpenAI()

    wins_a, wins_b, ties = 0, 0, 0
    per_sample: list[dict] = []
    n = len(instructions)
    t_start = time.time()

    for i, (instruction, resp_a, resp_b) in enumerate(zip(instructions, responses_a, responses_b)):
        # Randomize order to mitigate position bias.
        if random.random() < 0.5:
            first, second = resp_a, resp_b
            order = "ab"
        else:
            first, second = resp_b, resp_a
            order = "ba"

        judge_prompt = (
            "You are evaluating two AI assistant responses to the same instruction.\n\n"
            f"INSTRUCTION:\n{instruction}\n\n"
            f"RESPONSE A:\n{first[:2500]}\n\n"
            f"RESPONSE B:\n{second[:2500]}\n\n"
            "Which response is better? Consider helpfulness, accuracy, depth, and clarity.\n"
            "Respond in EXACTLY this format (one line each, no extra text):\n"
            "WINNER: <A | B | TIE>\n"
            "REASON: <one or two sentences>\n"
        )

        raw, verdict, reason = "", "TIE", ""
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_completion_tokens=200,
                temperature=0,
            )
            raw = (response.choices[0].message.content or "").strip()
            for line in raw.splitlines():
                s = line.strip()
                if s.upper().startswith("WINNER:"):
                    tok = s.split(":", 1)[1].strip().upper()
                    if tok.startswith("A"):
                        verdict = "A"
                    elif tok.startswith("B"):
                        verdict = "B"
                    else:
                        verdict = "TIE"
                elif s.upper().startswith("REASON:"):
                    reason = s.split(":", 1)[1].strip()
            if verdict == "TIE" and not reason:
                head = raw.strip().upper()[:3]
                if head.startswith("A"):
                    verdict = "A"
                elif head.startswith("B"):
                    verdict = "B"
        except Exception as e:
            print(f"  [judge error sample {i}]: {e}", flush=True)
            reason = f"(judge error: {type(e).__name__}: {e})"

        if verdict == "A":
            winner = "a" if order == "ab" else "b"
        elif verdict == "B":
            winner = "b" if order == "ab" else "a"
        else:
            winner = "tie"

        winner_label = {"a": label_a, "b": label_b, "tie": "TIE"}[winner]

        if winner == "a":
            wins_a += 1
        elif winner == "b":
            wins_b += 1
        else:
            ties += 1

        per_sample.append({
            "index": i,
            "instruction": instruction[:200],
            "winner": winner,
            "winner_label": winner_label,
            "order_shown": order,
            "verdict": verdict,
            "reason": reason,
        })

        if (i + 1) % log_every == 0 or (i + 1) == n:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1e-6)
            eta = (n - i - 1) / max(rate, 1e-6)
            wr_a = (wins_a + 0.5 * ties) / (i + 1)
            print(
                f"  [{i+1}/{n}] {label_a}: {wins_a}  {label_b}: {wins_b}  "
                f"ties: {ties}  WR-{label_a}: {wr_a:.1%}  "
                f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )

    total = wins_a + wins_b + ties
    return {
        "label_a": label_a,
        "label_b": label_b,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "total": total,
        "winrate_a": (wins_a + 0.5 * ties) / total if total else 0,
        "winrate_b": (wins_b + 0.5 * ties) / total if total else 0,
        "strict_winrate_a": wins_a / total if total else 0,
        "strict_winrate_b": wins_b / total if total else 0,
        "per_sample": per_sample,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="Path to the trained CME-GRPO checkpoint")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write outputs (default: <checkpoint>/quality_eval)")
    ap.add_argument("--reference-model", default="google/gemma-3-1b-it",
                    help="Small reference model for cross-comparison")
    ap.add_argument("--num-samples", type=int, default=None,
                    help="If set, evaluate on a random sample of N prompts "
                         "(seeded by --seed). Default uses all 805 prompts.")
    ap.add_argument("--judge-model", default="gpt-5.2",
                    help="OpenAI model used as the LLM judge")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.checkpoint.rstrip("/")), "quality_eval"
    )
    os.makedirs(output_dir, exist_ok=True)

    base_name = cfg["model"]["base"]
    instruct_name = cfg["model"]["instruct"]
    ref_name = args.reference_model

    print(f"{'=' * 60}")
    print("Pairwise quality evaluation on AlpacaEval 2.0 prompts")
    print(f"{'=' * 60}")
    print(f"  base:       {base_name}")
    print(f"  finetuned:  {args.checkpoint}")
    print(f"  instruct:   {instruct_name}")
    print(f"  reference:  {ref_name}")
    print(f"  judge:      {args.judge_model}")
    print(f"  output:     {output_dir}")
    print()

    # ── Load prompts ──
    instructions = load_alpacaeval_prompts()
    print(f"Loaded {len(instructions)} AlpacaEval prompts")

    # Optional random subsample (seeded for reproducibility).
    if args.num_samples is not None and args.num_samples < len(instructions):
        # Use a separate Random instance so this doesn't drain the global one
        # used for judge order randomization.
        sub_rng = random.Random(args.seed)
        idx = sub_rng.sample(range(len(instructions)), k=args.num_samples)
        instructions = [instructions[i] for i in idx]
        print(f"Subsampled {len(instructions)} prompts (seed={args.seed})")

    # ── Generate responses for all four models ──
    models_to_generate = [
        ("base", base_name),
        ("finetuned", args.checkpoint),
        ("instruct", instruct_name),
        ("reference", ref_name),
    ]

    responses_by_label: dict[str, list[str]] = {}
    for label, path in models_to_generate:
        out_path = os.path.join(output_dir, f"{label}_outputs.json")
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            responses_by_label[label] = [d["output"] for d in data]
            print(f"[{label}] loaded {len(responses_by_label[label])} cached responses from {out_path}", flush=True)
            continue

        responses = generate_for_model(
            path, instructions, device,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            label=label,
        )
        responses_by_label[label] = responses
        save_outputs_json(out_path, path, instructions, responses)
        print(f"[{label}] saved {len(responses)} outputs -> {out_path}", flush=True)

    # ── Five pairwise comparisons ──
    comparisons = [
        ("base", "finetuned"),
        ("finetuned", "instruct"),
        ("base", "reference"),
        ("finetuned", "reference"),
        ("instruct", "reference"),
    ]

    all_results: dict[str, dict] = {}
    for label_a, label_b in comparisons:
        cmp_name = f"{label_a}_vs_{label_b}"
        result_path = os.path.join(output_dir, f"judge_{cmp_name}.json")
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                all_results[cmp_name] = json.load(f)
            print(f"\n[{cmp_name}] loaded cached judge results from {result_path}", flush=True)
            continue

        print(f"\n{'=' * 60}")
        print(f"Judging: {cmp_name}")
        print(f"{'=' * 60}", flush=True)
        result = judge_pairwise(
            instructions,
            responses_by_label[label_a],
            responses_by_label[label_b],
            judge_model=args.judge_model,
            label_a=label_a,
            label_b=label_b,
        )
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[{cmp_name}] saved judge results -> {result_path}", flush=True)
        all_results[cmp_name] = result

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY (tie-adjusted win rate, judge = " + args.judge_model + ")")
    print(f"{'=' * 70}")
    print(f"{'Comparison (A vs B)':<35} {'A wins':>8} {'B wins':>8} {'Ties':>6} {'WR-A':>8} {'WR-B':>8}")
    print("-" * 70)
    for cmp_name, r in all_results.items():
        print(
            f"{cmp_name:<35} {r['wins_a']:>8} {r['wins_b']:>8} {r['ties']:>6} "
            f"{r['winrate_a']:>7.1%} {r['winrate_b']:>7.1%}"
        )

    # ── Write summary CSV ──
    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "comparison", "model_a", "model_b",
            "wins_a", "wins_b", "ties", "total",
            "winrate_a", "winrate_b",
            "strict_winrate_a", "strict_winrate_b",
        ])
        for cmp_name, r in all_results.items():
            w.writerow([
                cmp_name, r["label_a"], r["label_b"],
                r["wins_a"], r["wins_b"], r["ties"], r["total"],
                f"{r['winrate_a']:.4f}", f"{r['winrate_b']:.4f}",
                f"{r['strict_winrate_a']:.4f}", f"{r['strict_winrate_b']:.4f}",
            ])
    print(f"\nWrote summary -> {summary_path}")

    # ── Sample-level markdown for manual inspection ──
    md_path = os.path.join(output_dir, "samples.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Quality eval — judge = {args.judge_model}\n\n")
        for cmp_name, r in all_results.items():
            f.write(f"## {cmp_name}\n\n")
            f.write(
                f"- **A ({r['label_a']})**: {r['wins_a']} wins  · "
                f"**B ({r['label_b']})**: {r['wins_b']} wins  · "
                f"**Ties**: {r['ties']}  · "
                f"**WR-A (tie-adjusted)**: {r['winrate_a']:.1%}\n\n"
            )
            for s in r.get("per_sample", [])[:10]:
                f.write(f"### Sample {s['index']}: winner = {s['winner_label']}\n\n")
                f.write(f"- **Instruction**: {s['instruction']}\n")
                f.write(f"- **Reason**: {s['reason']}\n\n")
            f.write("---\n\n")
    print(f"Wrote samples markdown -> {md_path}")


if __name__ == "__main__":
    main()
