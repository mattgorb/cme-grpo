"""Generate side-by-side outputs from base and finetuned models on same problems."""

from __future__ import annotations

import argparse
import gc
import json

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval import format_prompt, extract_boxed, is_correct


@torch.no_grad()
def generate_all(model, tokenizer, problems, max_new_tokens, device, temperature=0.0):
    model.eval()
    outputs = []
    for i, p in enumerate(problems):
        prompt = format_prompt(p, tokenizer)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        outputs.append(text)
        print(f"  [{i+1}/{len(problems)}]", flush=True)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-Math-1.5B")
    ap.add_argument("--finetuned", required=True, help="Path to checkpoint or model name")
    ap.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    ap.add_argument("--split", default="test")
    ap.add_argument("--problem-key", default="problem")
    ap.add_argument("--answer-key", default="answer")
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--only-base-wrong", action="store_true", help="Filter to problems base gets wrong")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--output", default="comparison.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = load_dataset(args.dataset, split=args.split)
    if args.only_base_wrong:
        ds = ds.shuffle(seed=42).select(range(args.num_samples * 6))
    else:
        ds = ds.shuffle(seed=42).select(range(args.num_samples))
    problems = ds[args.problem_key]
    answers = ds[args.answer_key]

    # Base model
    print(f"\nLoading base: {args.base}")
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map=device)
    print("Generating base outputs...")
    base_outs = generate_all(base, tok, problems, args.max_new_tokens, device)
    del base

    if args.only_base_wrong:
        keep_idx = [i for i, (g, b) in enumerate(zip(answers, base_outs)) if not is_correct(extract_boxed(b), g)][: args.num_samples]
        if len(keep_idx) == 0:
            print("Base got all correct — nothing to compare. Exiting.")
            return
        problems = [problems[i] for i in keep_idx]
        answers = [answers[i] for i in keep_idx]
        base_outs = [base_outs[i] for i in keep_idx]
        print(f"Kept {len(keep_idx)} problems where base was wrong")
    gc.collect()
    torch.cuda.empty_cache()

    # Finetuned model
    print(f"\nLoading finetuned: {args.finetuned}")
    ft_tok = AutoTokenizer.from_pretrained(args.finetuned)
    if ft_tok.pad_token is None:
        ft_tok.pad_token = ft_tok.eos_token
    ft_tok.padding_side = "left"
    ft = AutoModelForCausalLM.from_pretrained(args.finetuned, torch_dtype=torch.bfloat16, device_map=device)
    print("Generating finetuned outputs...")
    ft_outs = generate_all(ft, ft_tok, problems, args.max_new_tokens, device)
    del ft
    gc.collect()
    torch.cuda.empty_cache()

    # Write markdown
    lines = [f"# Model Comparison: {args.base} vs {args.finetuned}\n"]
    for i, (prob, gold, b, f) in enumerate(zip(problems, answers, base_outs, ft_outs)):
        b_pred = extract_boxed(b)
        f_pred = extract_boxed(f)
        b_ok = "✓" if is_correct(b_pred, gold) else "✗"
        f_ok = "✓" if is_correct(f_pred, gold) else "✗"
        lines.append(f"## Problem {i+1}\n")
        lines.append(f"**Problem:** {prob}\n")
        lines.append(f"**Gold answer:** `{gold}`\n")
        lines.append(f"### Base ({b_ok} pred=`{b_pred}`)\n")
        lines.append(f"```\n{b}\n```\n")
        lines.append(f"### Finetuned ({f_ok} pred=`{f_pred}`)\n")
        lines.append(f"```\n{f}\n```\n")
        lines.append("---\n")

    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved to {args.output}")

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISONS")
    print("=" * 80)
    for i, (prob, gold, b, ft_out) in enumerate(zip(problems, answers, base_outs, ft_outs)):
        b_pred = extract_boxed(b)
        f_pred = extract_boxed(ft_out)
        b_ok = "CORRECT" if is_correct(b_pred, gold) else "WRONG"
        f_ok = "CORRECT" if is_correct(f_pred, gold) else "WRONG"
        print(f"\n{'#' * 80}")
        print(f"PROBLEM {i+1}: {prob}")
        print(f"GOLD: {gold}")
        print(f"\n--- BASE [{b_ok}, pred={b_pred}] ---")
        print(b)
        print(f"\n--- FINETUNED [{f_ok}, pred={f_pred}] ---")
        print(ft_out)

    # Also save JSON
    json_path = args.output.replace(".md", ".json")
    with open(json_path, "w") as f:
        json.dump([
            {"problem": p, "gold": g, "base": b, "finetuned": ft}
            for p, g, b, ft in zip(problems, answers, base_outs, ft_outs)
        ], f, indent=2)


if __name__ == "__main__":
    main()
