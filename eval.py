"""Standalone pass@1 greedy evaluation on MATH-500, AMC 2023, AIME 2024."""

from __future__ import annotations

import argparse
import re
from typing import Optional

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = (
    "Solve the following math problem. Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}\n\nSolution:"
)

CHAT_INSTRUCTION = (
    "Solve the following math problem. Put your final answer in \\boxed{}."
)


def format_prompt(problem: str, tokenizer) -> str:
    if tokenizer.chat_template is not None:
        messages = [
            {"role": "user", "content": f"{CHAT_INSTRUCTION}\n\nProblem: {problem}"},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return PROMPT_TEMPLATE.format(problem=problem)


def extract_boxed(text: str) -> Optional[str]:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None


def normalize(ans) -> str:
    if ans is None:
        return ""
    ans = str(ans).strip().replace(" ", "").replace("\\!", "")
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = re.sub(r"\\text\{[^}]*\}", "", ans)
    # Drop trailing zeros on decimals and canonicalize integers.
    try:
        f = float(ans)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except (ValueError, TypeError):
        return ans


def is_correct(pred, gold) -> bool:
    return bool(pred is not None) and normalize(pred) == normalize(gold)


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    dataset,
    problem_key: str = "problem",
    answer_key: str = "answer",
    max_new_tokens: int = 2048,
    batch_size: int = 8,
    device: str = "cuda",
    num_samples: int = 1,
    temperature: float = 0.0,
    debug: bool = False,
) -> dict:
    model.eval()
    correct = 0
    total = 0

    problems = dataset[problem_key]
    answers = dataset[answer_key]

    do_sample = temperature > 0 and num_samples > 1

    n_total = len(problems)
    for i in range(0, n_total, batch_size):
        if i > 0:
            acc_so_far = correct / total if total else 0
            print(f"    [{total}/{n_total}] running acc: {acc_so_far:.3f}", flush=True)
        batch_problems = problems[i : i + batch_size]
        batch_answers = answers[i : i + batch_size]

        prompts = [format_prompt(p, tokenizer) for p in batch_problems]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)

        if num_samples > 1 and do_sample:
            all_correct_flags = [[] for _ in batch_problems]
            for _ in range(num_samples):
                out = model.generate(**enc, **gen_kwargs)
                gen = out[:, enc.input_ids.shape[1]:]
                decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
                for j, (text, gold) in enumerate(zip(decoded, batch_answers)):
                    pred = extract_boxed(text)
                    all_correct_flags[j].append(1 if is_correct(pred, gold) else 0)
                    if debug and i == 0 and j == 0:
                        print(f"    [DEBUG] gold={gold} pred={pred} text[-200:]={repr(text[-200:])}", flush=True)
            for flags in all_correct_flags:
                if sum(flags) / len(flags) > 0:
                    correct += sum(flags) / len(flags)
                total += 1
        else:
            out = model.generate(**enc, **gen_kwargs)
            gen = out[:, enc.input_ids.shape[1]:]
            decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for j, (text, gold) in enumerate(zip(decoded, batch_answers)):
                pred = extract_boxed(text)
                if debug and i == 0:
                    print(f"    [DEBUG] gold={gold} pred={pred} text[-200:]={repr(text[-200:])}", flush=True)
                if is_correct(pred, gold):
                    correct += 1
                total += 1

    acc = correct / total if total else 0.0
    return {"pass@1": acc, "correct": correct, "total": total}


def evaluate_benchmark(model, tokenizer, bench: dict, cfg: dict, device: str, max_samples: int = 0, debug: bool = False) -> dict:
    ds = load_dataset(bench["dataset"], split=bench["split"])
    # Per-benchmark num_test_samples in config overrides the caller's max_samples.
    n = int(bench.get("num_test_samples", 0)) or max_samples
    if n > 0 and len(ds) > n:
        ds = ds.shuffle(seed=42).select(range(n))
    return evaluate(
        model,
        tokenizer,
        ds,
        problem_key=bench["problem_key"],
        answer_key=bench["answer_key"],
        max_new_tokens=cfg["eval"]["max_new_tokens"],
        batch_size=cfg["eval"]["batch_size"],
        device=device,
        debug=debug,
    )


def evaluate_all(model, tokenizer, cfg: dict, device: str, max_samples: int = 0, debug: bool = False) -> dict:
    results = {}
    for bench in cfg["benchmarks"]:
        res = evaluate_benchmark(model, tokenizer, bench, cfg, device, max_samples=max_samples, debug=debug)
        results[bench["name"]] = res
        print(
            f"  {bench['name']}: pass@1 = {res['pass@1']:.4f} "
            f"({res['correct']}/{res['total']})"
        )
    return results


def run_eval(model_name_or_path: str, cfg: dict) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    return evaluate_all(model, tokenizer, cfg, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_path = args.model or cfg["model"]["generator"]
    print(f"Evaluating {model_path}")
    run_eval(model_path, cfg)


if __name__ == "__main__":
    main()
