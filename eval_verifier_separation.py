"""Measure CME separation: which verifier's perplexity best separates correct vs incorrect solutions?

Generates solutions from the base generator, labels them, then computes CME
from each candidate verifier. Reports AUROC and mean CME gap.

Usage:
    python eval_verifier_separation.py
    python eval_verifier_separation.py --verifiers "deepseek-ai/deepseek-math-7b-instruct,Qwen/Qwen2.5-Math-7B-Instruct"
    python eval_verifier_separation.py --num-samples 100 --num-generations 4
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval import extract_boxed, normalize, is_correct, PROMPT_TEMPLATE, format_prompt

GENERATOR = "Qwen/Qwen2.5-Math-1.5B"


DEFAULT_VERIFIERS = [
    "deepseek-ai/deepseek-math-7b-instruct",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "microsoft/Phi-4-mini-reasoning",
    "nvidia/Nemotron-Cascade-8B",
    "google/gemma-4-E4B-it",
    #"internlm/internlm2-math-plus-7b",
]


def auroc(scores: list[float], labels: list[int]) -> float:
    """Compute AUROC. Higher score should correspond to label=1 (correct)."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = 0
    auc = 0.0
    for score, label in pairs:
        if label == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg)


@torch.no_grad()
def generate_solutions(
    model, tokenizer, problems: list[str], num_generations: int, device: str
) -> list[list[str]]:
    """Generate multiple solutions per problem. Returns list of list of strings."""
    model.eval()
    all_solutions = []
    for idx, problem in enumerate(problems):
        prompt = format_prompt(problem, tokenizer)
        enc = tokenizer(
            [prompt] * num_generations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=1024,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, enc.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
        all_solutions.append(decoded)
        if (idx + 1) % 10 == 0:
            print(f"  generated {idx + 1}/{len(problems)} problems", flush=True)
    return all_solutions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default=GENERATOR)
    ap.add_argument("--verifiers", default=None, help="Comma-separated verifier model names")
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--output", default="verifier_separation.csv")
    ap.add_argument("--save-generations", default="generations.json", help="Save/load generator outputs")
    args = ap.parse_args()

    verifiers = args.verifiers.split(",") if args.verifiers else DEFAULT_VERIFIERS
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds = ds.shuffle(seed=42).select(range(args.num_samples))
    problems = ds["problem"]
    gold_answers = ds["answer"]

    # Step 1: Generate solutions (or load cached)
    gen_path = args.save_generations
    if os.path.exists(gen_path):
        print(f"Loading cached generations from {gen_path}")
        with open(gen_path) as f:
            cached = json.load(f)
        all_solutions = cached["solutions"]
        print(f"  loaded {len(all_solutions)} problems x {len(all_solutions[0])} generations")
    else:
        print(f"Generating {args.num_generations} solutions per problem from {args.generator}")
        gen_tokenizer = AutoTokenizer.from_pretrained(args.generator, trust_remote_code=True)
        if gen_tokenizer.pad_token is None:
            gen_tokenizer.pad_token = gen_tokenizer.eos_token
        gen_tokenizer.padding_side = "left"

        gen_model = AutoModelForCausalLM.from_pretrained(
            args.generator,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

        all_solutions = generate_solutions(
            gen_model, gen_tokenizer, problems, args.num_generations, device
        )

        with open(gen_path, "w") as f:
            json.dump({"generator": args.generator, "solutions": all_solutions}, f)
        print(f"  saved generations to {gen_path}")

        del gen_model, gen_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Step 2: Label correct/incorrect
    prompts_flat = []
    responses_flat = []
    labels_flat = []
    for i, (solutions, gold) in enumerate(zip(all_solutions, gold_answers)):
        prompt = PROMPT_TEMPLATE.format(problem=problems[i])
        for sol in solutions:
            pred = extract_boxed(sol)
            label = 1 if is_correct(pred, gold) else 0
            prompts_flat.append(prompt)
            responses_flat.append(sol)
            labels_flat.append(label)

    n_correct = sum(labels_flat)
    n_total = len(labels_flat)
    print(f"\n{n_correct}/{n_total} solutions correct ({n_correct/n_total:.1%})")

    if n_correct == 0 or n_correct == n_total:
        print("All solutions same label — can't compute AUROC. Try more samples.")
        return

    # Step 3: Score with each verifier
    results = []
    for verifier_name in verifiers:
        print(f"\nScoring with {verifier_name}")
        v_tokenizer = AutoTokenizer.from_pretrained(verifier_name, trust_remote_code=True)
        if v_tokenizer.pad_token is None:
            v_tokenizer.pad_token = v_tokenizer.eos_token

        v_model = AutoModelForCausalLM.from_pretrained(
            verifier_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        v_model.eval()

        cme_scores = []
        for prompt, response in zip(prompts_flat, responses_flat):
            if not response:
                cme_scores.append(-10.0)
                continue
            prompt_ids = v_tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids[0]
            response_ids = v_tokenizer(response, add_special_tokens=False, return_tensors="pt").input_ids[0]
            if response_ids.numel() == 0:
                cme_scores.append(-10.0)
                continue
            full_ids = torch.cat([prompt_ids, response_ids], dim=0)[-2048:]
            prompt_len = max(1, full_ids.shape[0] - response_ids.shape[0])
            input_ids = full_ids.unsqueeze(0).to(device)
            labels = input_ids.clone()
            labels[0, :prompt_len] = -100
            with torch.no_grad():
                logits = v_model(input_ids=input_ids).logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )
            cme_scores.append(-loss.item())

        correct_cmes = [s for s, l in zip(cme_scores, labels_flat) if l == 1]
        incorrect_cmes = [s for s, l in zip(cme_scores, labels_flat) if l == 0]

        mean_correct = np.mean(correct_cmes)
        mean_incorrect = np.mean(incorrect_cmes)
        gap = mean_correct - mean_incorrect
        auc = auroc(cme_scores, labels_flat)

        print(f"  mean CME (correct):   {mean_correct:.4f}")
        print(f"  mean CME (incorrect): {mean_incorrect:.4f}")
        print(f"  gap (correct - incorrect): {gap:.4f}")
        print(f"  AUROC: {auc:.4f}")

        results.append({
            "verifier": verifier_name,
            "auroc": auc,
            "gap": gap,
            "mean_correct": mean_correct,
            "mean_incorrect": mean_incorrect,
        })

        del v_model, v_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("VERIFIER SEPARATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Verifier':<45} {'AUROC':>7} {'Gap':>8}")
    print("-" * 62)
    for r in sorted(results, key=lambda x: -x["auroc"]):
        print(f"{r['verifier']:<45} {r['auroc']:>7.4f} {r['gap']:>+8.4f}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["verifier", "auroc", "gap", "mean_correct", "mean_incorrect"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
