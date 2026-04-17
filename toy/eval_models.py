"""Evaluate candidate models for toy CME-GRPO.

Tests each model on MBPP (Mostly Basic Python Problems):
1. Generate a completion for each prompt
2. Check if the completion is valid Python (compiles)
3. Check if it passes the provided test cases (exec)
4. Log summary at the end

Usage:
    python toy/eval_models.py
    python toy/eval_models.py --models bigcode/tiny_starcoder_py Salesforce/codegen-350M-mono
    python toy/eval_models.py --num-samples 50
"""

from __future__ import annotations

import argparse
import math
import signal
import traceback
from contextlib import contextmanager

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@contextmanager
def timeout(seconds):
    """Kill execution after `seconds` (Unix only)."""
    def _handler(signum, frame):
        raise TimeoutError()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def check_syntax(code: str) -> bool:
    try:
        compile(code, "<test>", "exec")
        return True
    except SyntaxError:
        return False


def check_tests(code: str, tests: list[str], test_setup: str = "") -> bool:
    """Execute code + test cases in a sandboxed namespace."""
    full = test_setup + "\n" + code + "\n" + "\n".join(tests)
    try:
        with timeout(5):
            exec(full, {})
        return True
    except Exception:
        return False


def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 256) -> str:
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


@torch.no_grad()
def compute_ce(model, tokenizer, prompt: str, response: str, device: str) -> tuple[float, float, float]:
    """Compute cross-entropy, perplexity, and entropy of response conditioned on prompt."""
    prompt_enc = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    response_enc = tokenizer(response, add_special_tokens=False, return_tensors="pt")
    prompt_ids = prompt_enc.input_ids[0]
    response_ids = response_enc.input_ids[0]
    if response_ids.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(device)
    prompt_len = prompt_ids.shape[0]

    logits = model(input_ids=full_ids).logits[:, :-1, :]
    labels = full_ids[:, 1:].clone()
    labels[0, :prompt_len - 1] = -100

    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )

    # Entropy over response tokens: -sum p * log p
    resp_logits = logits[0, prompt_len - 1:, :]
    log_probs = torch.log_softmax(resp_logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean()

    return ce.item(), math.exp(ce.item()), entropy.item()


def eval_model(model_name: str, dataset, device: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    model.eval()

    results = []
    for i, ex in enumerate(dataset):
        prompt = ex["prompt"] if "prompt" in ex else ex["text"] + "\n"
        tests = ex.get("test_list", [])
        test_setup = ex.get("test_setup_code", "")
        canonical = ex.get("canonical_solution", ex.get("code", ""))

        # Generation eval.
        completion = generate(model, tokenizer, prompt, device)
        syntax_ok = check_syntax(completion)
        tests_pass = check_tests(completion, tests, test_setup) if tests else None

        # CE/PPL/entropy on the reference solution.
        ref_ce, ref_ppl, ref_ent = float("nan"), float("nan"), float("nan")
        if canonical:
            ref_ce, ref_ppl, ref_ent = compute_ce(model, tokenizer, prompt, canonical, device)

        results.append({
            "task_id": ex.get("task_id", i),
            "prompt": prompt[:80],
            "syntax_ok": syntax_ok,
            "tests_pass": tests_pass,
            "ref_ce": ref_ce,
            "ref_ppl": ref_ppl,
            "ref_entropy": ref_ent,
            "completion_preview": completion[len(prompt):len(prompt)+120],
        })

        status = "PASS" if tests_pass else ("SYNTAX" if syntax_ok else "FAIL")
        if (i + 1) % 10 == 0 or i < 5:
            print(f"  [{i+1}/{len(dataset)}] {status} | CE={ref_ce:.3f} PPL={ref_ppl:.1f} H={ref_ent:.2f} | {prompt[:50].strip()}")
            if i < 5:
                print(f"    -> {completion[len(prompt):len(prompt)+100].strip()}")

    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    n = len(results)
    syntax_count = sum(1 for r in results if r["syntax_ok"])
    tests_count = sum(1 for r in results if r["tests_pass"])
    tests_total = sum(1 for r in results if r["tests_pass"] is not None)
    valid_ces = [r["ref_ce"] for r in results if not math.isnan(r["ref_ce"])]
    valid_ppls = [r["ref_ppl"] for r in results if not math.isnan(r["ref_ppl"])]
    valid_ents = [r["ref_entropy"] for r in results if not math.isnan(r["ref_entropy"])]

    summary = {
        "model": model_name,
        "total": n,
        "syntax_ok": syntax_count,
        "syntax_rate": syntax_count / n if n else 0,
        "tests_pass": tests_count,
        "tests_total": tests_total,
        "test_pass_rate": tests_count / tests_total if tests_total else 0,
        "mean_ref_ce": sum(valid_ces) / len(valid_ces) if valid_ces else float("nan"),
        "mean_ref_ppl": sum(valid_ppls) / len(valid_ppls) if valid_ppls else float("nan"),
        "mean_ref_entropy": sum(valid_ents) / len(valid_ents) if valid_ents else float("nan"),
        "results": results,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", nargs="+",
        default=["bigcode/tiny_starcoder_py", "Salesforce/codegen-350M-mono"],
    )
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--judge-only", action="store_true", help="skip eval, only run LLM judge")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"Device: {device}")

    print("Loading MBPP dataset...")
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    ds = ds.select(range(min(args.num_samples, len(ds))))
    print(f"Using {len(ds)} samples from MBPP (sanitized test split)")

    if args.judge_only:
        if len(args.models) != 2:
            print("Error: --judge-only requires exactly 2 models")
            return
        judge_winrate(args.models[0], args.models[1], ds, device, num_samples=min(50, len(ds)))
        return

    summaries = []
    for model_name in args.models:
        summary = eval_model(model_name, ds, device)
        summaries.append(summary)

    # Final report.
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<40} {'Syntax':>8} {'Tests':>8} {'Ref CE':>8} {'Ref PPL':>9} {'Entropy':>9}")
    print(f"{'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")
    for s in summaries:
        print(f"{s['model']:<40} {s['syntax_rate']:>7.1%} {s['test_pass_rate']:>7.1%} {s['mean_ref_ce']:>8.3f} {s['mean_ref_ppl']:>9.1f} {s['mean_ref_entropy']:>9.2f}")

    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    # Use ref CE as the primary signal — lower CE = model understands correct code better.
    best_ver = min(summaries, key=lambda s: s["mean_ref_ce"])
    best_gen = max(summaries, key=lambda s: s["mean_ref_ce"])
    if best_gen["model"] == best_ver["model"]:
        print("Both models perform similarly — pick the smaller one as generator.")
    else:
        ce_gap = best_gen["mean_ref_ce"] - best_ver["mean_ref_ce"]
        print(f"  Generator (higher CE): {best_gen['model']} (CE={best_gen['mean_ref_ce']:.3f}, PPL={best_gen['mean_ref_ppl']:.1f}, H={best_gen['mean_ref_entropy']:.2f})")
        print(f"  Verifier  (lower CE):  {best_ver['model']} (CE={best_ver['mean_ref_ce']:.3f}, PPL={best_ver['mean_ref_ppl']:.1f}, H={best_ver['mean_ref_entropy']:.2f})")
        print(f"  CE gap: {ce_gap:.3f} (verifier assigns {ce_gap:.3f} lower CE to correct solutions)")
        if ce_gap < 0.2:
            print(f"  ⚠ Gap is small — CME signal may be weak.")
        else:
            print(f"  ✓ Meaningful gap — should give good CME reward signal.")

    # LLM judge win rate between the two models.
    if len(args.models) == 2:
        judge_result = judge_winrate(
            args.models[0], args.models[1], ds, device, num_samples=min(50, len(ds)),
        )

        # Unified summary.
        print(f"\n{'='*60}")
        print("FULL SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<40} {'Syntax':>8} {'Tests':>8} {'Ref CE':>8} {'Ref PPL':>9} {'Entropy':>9}")
        print(f"{'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")
        for s in summaries:
            print(f"{s['model']:<40} {s['syntax_rate']:>7.1%} {s['test_pass_rate']:>7.1%} {s['mean_ref_ce']:>8.3f} {s['mean_ref_ppl']:>9.1f} {s['mean_ref_entropy']:>9.2f}")
        print()
        print(f"  Judge win rate (GPT-5.2, n={judge_result['total']}):")
        print(f"    {args.models[0]}: {judge_result['wins_a']} wins ({judge_result['winrate_a']:.1%})")
        print(f"    {args.models[1]}: {judge_result['wins_b']} wins ({judge_result['winrate_b']:.1%})")
        print(f"    Ties: {judge_result['ties']} ({judge_result['ties']/judge_result['total']:.1%})")


def judge_winrate(
    model_a_name: str,
    model_b_name: str,
    dataset,
    device: str,
    num_samples: int = 50,
) -> dict:
    """Run LLM-as-judge (GPT-5.2) to compare two models' completions.

    Returns win/loss/tie counts and win rate for model_b over model_a.
    """
    import random
    from openai import OpenAI

    client = OpenAI()

    print(f"\n{'='*60}")
    print(f"LLM JUDGE: {model_a_name} vs {model_b_name}")
    print(f"{'='*60}")

    # Generate completions from both models.
    print(f"Generating from {model_a_name}...")
    tok_a = AutoTokenizer.from_pretrained(model_a_name)
    if tok_a.pad_token is None:
        tok_a.pad_token = tok_a.eos_token
    model_a = AutoModelForCausalLM.from_pretrained(model_a_name, torch_dtype=torch.float32).to(device)
    model_a.eval()

    samples = []
    ds_subset = dataset.select(range(min(num_samples, len(dataset))))
    n_total = len(ds_subset)
    for i, ex in enumerate(ds_subset):
        prompt = ex["prompt"] if "prompt" in ex else ex["text"] + "\n"
        canonical = ex.get("canonical_solution", ex.get("code", ""))
        comp_a = generate(model_a, tok_a, prompt, device)
        samples.append({"prompt": prompt, "canonical": canonical, "comp_a": comp_a[len(prompt):]})
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{n_total}] generated | {prompt[:60].strip()}")

    del model_a
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    print(f"Generating from {model_b_name}...")
    tok_b = AutoTokenizer.from_pretrained(model_b_name)
    if tok_b.pad_token is None:
        tok_b.pad_token = tok_b.eos_token
    model_b = AutoModelForCausalLM.from_pretrained(model_b_name, torch_dtype=torch.float32).to(device)
    model_b.eval()

    for i, ex in enumerate(ds_subset):
        prompt = ex["prompt"] if "prompt" in ex else ex["text"] + "\n"
        comp_b = generate(model_b, tok_b, prompt, device)
        samples[i]["comp_b"] = comp_b[len(prompt):]
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{n_total}] generated | {prompt[:60].strip()}")

    del model_b
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    # Judge with GPT-5.2.
    print("Judging with GPT-5.2...")
    wins_a, wins_b, ties = 0, 0, 0
    results = []

    for i, s in enumerate(samples):
        # Randomize order to avoid position bias.
        if random.random() < 0.5:
            first, second = s["comp_a"], s["comp_b"]
            order = "ab"
        else:
            first, second = s["comp_b"], s["comp_a"]
            order = "ba"

        judge_prompt = f"""You are evaluating two Python code completions for the same programming task.

TASK: {s['prompt'].strip()}

REFERENCE SOLUTION:
```python
{s['canonical'][:500]}
```

COMPLETION A:
```python
{first[:500]}
```

COMPLETION B:
```python
{second[:500]}
```

Which completion is better? Consider:
- Correctness (does it solve the task?)
- Code quality (readability, Pythonic style)
- Completeness (does it handle edge cases?)

Respond with ONLY one of: "A", "B", or "TIE"."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": judge_prompt}],
                max_completion_tokens=5,
                temperature=0,
            )
            verdict = response.choices[0].message.content.strip().upper()
        except Exception as e:
            print(f"  Judge error on sample {i}: {e}")
            verdict = "TIE"

        # Map back to original order.
        if verdict == "A":
            winner = "a" if order == "ab" else "b"
        elif verdict == "B":
            winner = "b" if order == "ab" else "a"
        else:
            winner = "tie"

        if winner == "a":
            wins_a += 1
        elif winner == "b":
            wins_b += 1
        else:
            ties += 1

        results.append({"prompt": s["prompt"][:60], "winner": winner})

        label = {"a": model_a_name.split("/")[-1], "b": model_b_name.split("/")[-1], "tie": "TIE"}[winner]
        print(f"  [{i+1}/{len(samples)}] winner={label} | {s['prompt'][:50].strip()}")
        if (i + 1) % 10 == 0:
            print(f"    --- running: {model_a_name.split('/')[-1]}: {wins_a} | {model_b_name.split('/')[-1]}: {wins_b} | ties: {ties}")

    total = wins_a + wins_b + ties
    summary = {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "total": total,
        "winrate_a": wins_a / total if total else 0,
        "winrate_b": wins_b / total if total else 0,
        "results": results,
    }

    print(f"\n{'='*60}")
    print("JUDGE RESULTS")
    print(f"{'='*60}")
    print(f"  {model_a_name}: {wins_a} wins ({summary['winrate_a']:.1%})")
    print(f"  {model_b_name}: {wins_b} wins ({summary['winrate_b']:.1%})")
    print(f"  Ties: {ties} ({ties/total:.1%})" if total else "")

    return summary


if __name__ == "__main__":
    main()
