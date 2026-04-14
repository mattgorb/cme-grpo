"""Evaluate candidate verifier models on MATH-500, AMC 2023, AIME 2024 (pass@1 greedy).

Usage:
    python eval_verifier_candidates.py
    python eval_verifier_candidates.py --models "Qwen/Qwen3.5-4B-Instruct,google/gemma-4-E4B-it"
    python eval_verifier_candidates.py --benchmarks math500,aime24
"""

from __future__ import annotations

import argparse
import csv
import gc
import time

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval import evaluate, PROMPT_TEMPLATE

DEFAULT_MODELS = [
    "nvidia/Nemotron-Cascade-8B",
    "Qwen/Qwen2.5-Math-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "microsoft/Phi-4-mini-reasoning",
    "internlm/internlm2-math-plus-7b",
    "deepseek-ai/deepseek-math-7b-instruct",
    "google/gemma-4-E4B-it",
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
]

BENCHMARKS = {
    "math500": {
        "dataset": "HuggingFaceH4/MATH-500",
        "split": "test",
        "problem_key": "problem",
        "answer_key": "answer",
    },
    "amc23": {
        "dataset": "math-ai/amc23",
        "split": "test",
        "problem_key": "question",
        "answer_key": "answer",
    },
    "aime24": {
        "dataset": "Maxwell-Jia/AIME_2024",
        "split": "train",
        "problem_key": "Problem",
        "answer_key": "Answer",
    },
}


def load_benchmark(bench: dict, max_samples: int = 50):
    ds = load_dataset(bench["dataset"], split=bench["split"])
    if bench["dataset"] == "HuggingFaceH4/MATH-500":
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))
    return ds


def eval_model(model_name: str, benchmarks: dict, max_new_tokens: int, batch_size: int, max_samples: int = 50):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Loading {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e9

    results = {"model": model_name, "params_B": round(param_count, 2)}
    for bench_name, bench_cfg in benchmarks.items():
        ds = load_benchmark(bench_cfg, max_samples=max_samples)
        t0 = time.time()
        res = evaluate(
            model,
            tokenizer,
            ds,
            problem_key=bench_cfg["problem_key"],
            answer_key=bench_cfg["answer_key"],
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            device=device,
        )
        elapsed = time.time() - t0
        results[bench_name] = res["pass@1"]
        print(
            f"  {bench_name}: {res['pass@1']:.4f} "
            f"({res['correct']}/{res['total']}) [{elapsed:.0f}s]"
        )

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


def print_summary(all_results: list[dict], bench_names: list[str]):
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    header = f"{'Model':<35} {'Params':>6}"
    for b in bench_names:
        header += f" {b:>8}"
    header += f" {'  Avg':>8}"
    print(header)
    print("-" * len(header))

    for r in sorted(all_results, key=lambda x: -sum(x.get(b, 0) for b in bench_names)):
        line = f"{r['model']:<35} {r['params_B']:>5.1f}B"
        scores = []
        for b in bench_names:
            s = r.get(b, 0)
            scores.append(s)
            line += f" {s:>7.1%}"
        avg = sum(scores) / len(scores) if scores else 0
        line += f" {avg:>7.1%}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="Comma-separated model names")
    ap.add_argument("--benchmarks", default=None, help="Comma-separated: math500,amc23,aime24")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-samples", type=int, default=50, help="Max samples for MATH-500")
    ap.add_argument("--output", default="verifier_candidates.csv", help="CSV output path")
    args = ap.parse_args()

    models = args.models.split(",") if args.models else DEFAULT_MODELS

    if args.benchmarks:
        bench_names = args.benchmarks.split(",")
        benchmarks = {k: BENCHMARKS[k] for k in bench_names}
    else:
        bench_names = list(BENCHMARKS.keys())
        benchmarks = BENCHMARKS

    csv_path = args.output
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "params_B"] + bench_names + ["avg"])

    all_results = []
    for model_name in models:
        model_name = model_name.strip()
        try:
            res = eval_model(model_name, benchmarks, args.max_new_tokens, args.batch_size, args.max_samples)
        except Exception as e:
            print(f"  FAILED: {e}")
            res = {"model": model_name, "params_B": 0}
        all_results.append(res)
        scores = [res.get(b, 0) for b in bench_names]
        avg = sum(scores) / len(scores) if scores else 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([res["model"], res["params_B"]] + [f"{s:.4f}" for s in scores] + [f"{avg:.4f}"])
        print(f"  >> appended to {csv_path}")

    print_summary(all_results, bench_names)


if __name__ == "__main__":
    main()
