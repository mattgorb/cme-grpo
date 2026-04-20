"""Generate AlpacaEval 2.0 responses and run LLM-judge three-way comparison.

For a given checkpoint, generates responses from:
  1. Finetuned model (checkpoint)
  2. Base model (from config)
  3. Instruct model (from config)

Then runs pairwise LLM-judge comparisons and outputs AlpacaEval-format JSON.

Usage:
    python eval_alpacaeval.py --config config_quality1.yaml --checkpoint ./outputs/cme-grpo-quality-qwen0.5b/checkpoint-best
    python eval_alpacaeval.py --config config_quality2.yaml --checkpoint ./outputs/cme-grpo-quality-llama1b/checkpoint-best
    python eval_alpacaeval.py --config config_quality1.yaml --checkpoint ./outputs/cme-grpo-quality-qwen0.5b --num-samples 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def format_prompt(instruction: str, tokenizer) -> str:
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return PROMPT_TEMPLATE.format(instruction=instruction)


def generate_all(
    model_name: str,
    instructions: list[str],
    device: str,
    max_new_tokens: int = 2048,
    batch_size: int = 4,
) -> tuple[list[str], AutoTokenizer]:
    """Load model, generate responses for all instructions, unload."""
    print(f"\nGenerating from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    prompts = [format_prompt(inst, tokenizer) for inst in instructions]
    responses = []

    for i in range(0, len(prompts), batch_size):
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
        if (i // batch_size) % 5 == 0:
            print(f"  [{len(responses)}/{len(prompts)}] generated")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return responses, tokenizer


def judge_pairwise(
    instructions: list[str],
    responses_a: list[str],
    responses_b: list[str],
    judge_model: str = "gpt-4o-mini",
    label_a: str = "Model A",
    label_b: str = "Model B",
) -> dict:
    """LLM-judge pairwise comparison with position-bias mitigation."""
    from openai import OpenAI
    client = OpenAI()

    wins_a, wins_b, ties = 0, 0, 0
    details = []

    for i, (instruction, resp_a, resp_b) in enumerate(zip(instructions, responses_a, responses_b)):
        if random.random() < 0.5:
            first, second = resp_a, resp_b
            order = "ab"
        else:
            first, second = resp_b, resp_a
            order = "ba"

        judge_prompt = (
            "You are evaluating two AI assistant responses to the same instruction.\n\n"
            f"INSTRUCTION:\n{instruction}\n\n"
            f"RESPONSE A:\n{first[:2000]}\n\n"
            f"RESPONSE B:\n{second[:2000]}\n\n"
            "Which response is better? Consider helpfulness, accuracy, depth, and clarity.\n"
            "Respond with ONLY one of: \"A\", \"B\", or \"TIE\"."
        )

        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_completion_tokens=5,
                temperature=0,
            )
            verdict = response.choices[0].message.content.strip().upper()
        except Exception as e:
            print(f"  Judge error on sample {i}: {e}")
            verdict = "TIE"

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

        details.append({"instruction": instruction[:80], "winner": winner, "verdict": verdict})

        winner_label = {
            "a": label_a, "b": label_b, "tie": "TIE",
        }[winner]
        print(f"  [{i+1}/{len(instructions)}] winner={winner_label} | {instruction[:70]}")
        if (i + 1) % 10 == 0:
            print(f"    --- running: {label_a}: {wins_a} | {label_b}: {wins_b} | ties: {ties}")

    total = wins_a + wins_b + ties
    return {
        "label_a": label_a, "label_b": label_b,
        "wins_a": wins_a, "wins_b": wins_b, "ties": ties, "total": total,
        "winrate_a": wins_a / total if total else 0,
        "winrate_b": wins_b / total if total else 0,
        "details": details,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True, help="Path to finetuned checkpoint")
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--judge-model", default=None, help="Override judge model from config")
    ap.add_argument("--output-dir", default=None, help="Directory for output JSON files")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    judge_model = args.judge_model or cfg.get("eval", {}).get("judge_model", "gpt-5.2")
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "alpacaeval")
    os.makedirs(output_dir, exist_ok=True)

    base_name = cfg["model"]["base"]
    instruct_name = cfg["model"]["instruct"]
    checkpoint = args.checkpoint
    max_new_tokens = cfg.get("eval", {}).get("max_new_tokens", 2048)
    batch_size = cfg.get("eval", {}).get("batch_size", 4)

    print(f"{'=' * 60}")
    print(f"AlpacaEval — Three-Way Comparison")
    print(f"{'=' * 60}")
    print(f"  Finetuned: {checkpoint}")
    print(f"  Base:      {base_name}")
    print(f"  Instruct:  {instruct_name}")
    print(f"  Judge:     {judge_model}")
    print(f"  Samples:   {args.num_samples}")
    print()

    # ── Load eval instructions ──
    print("Loading instructions from UltraFeedback...")
    ds = load_dataset(cfg["data"]["train_dataset"], split="train")
    ds = ds.shuffle(seed=args.seed).select(range(args.num_samples))
    instructions = [ex.get("instruction", ex.get("prompt", "")) for ex in ds]
    print(f"  Loaded {len(instructions)} instructions")

    # ── Generate from all three models ──
    t0 = time.time()
    ft_responses, _ = generate_all(checkpoint, instructions, device, max_new_tokens, batch_size)
    base_responses, _ = generate_all(base_name, instructions, device, max_new_tokens, batch_size)
    instruct_responses, _ = generate_all(instruct_name, instructions, device, max_new_tokens, batch_size)
    gen_time = time.time() - t0
    print(f"\nAll generations complete in {gen_time:.0f}s")

    # ── Save AlpacaEval-format JSON ──
    def _to_alpacaeval(model_name, responses):
        return [
            {"instruction": inst, "output": resp, "generator": model_name}
            for inst, resp in zip(instructions, responses)
        ]

    ft_label = os.path.basename(checkpoint) or "finetuned"
    for label, name, resps in [
        (ft_label, checkpoint, ft_responses),
        ("base", base_name, base_responses),
        ("instruct", instruct_name, instruct_responses),
    ]:
        path = os.path.join(output_dir, f"{label}.json")
        with open(path, "w") as f:
            json.dump(_to_alpacaeval(name, resps), f, indent=2)
        print(f"  Saved {path}")

    # ── LLM Judge: three pairwise comparisons ──
    print(f"\n{'=' * 60}")
    print("LLM JUDGE EVALUATION")
    print(f"{'=' * 60}")

    print(f"\n[1/3] Finetuned vs Base")
    ft_vs_base = judge_pairwise(
        instructions, ft_responses, base_responses,
        judge_model=judge_model, label_a="finetuned", label_b="base",
    )

    print(f"\n[2/3] Finetuned vs Instruct")
    ft_vs_instruct = judge_pairwise(
        instructions, ft_responses, instruct_responses,
        judge_model=judge_model, label_a="finetuned", label_b="instruct",
    )

    print(f"\n[3/3] Instruct vs Base")
    instruct_vs_base = judge_pairwise(
        instructions, instruct_responses, base_responses,
        judge_model=judge_model, label_a="instruct", label_b="base",
    )

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Comparison':<30} {'Win A':>7} {'Win B':>7} {'Tie':>5} {'Winrate A':>10}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*5} {'-'*10}")
    for r, desc in [
        (ft_vs_base, "Finetuned vs Base"),
        (ft_vs_instruct, "Finetuned vs Instruct"),
        (instruct_vs_base, "Instruct vs Base"),
    ]:
        print(f"  {desc:<30} {r['wins_a']:>7} {r['wins_b']:>7} {r['ties']:>5} {r['winrate_a']:>9.1%}")

    # Save judge results.
    results = {
        "config": args.config,
        "checkpoint": checkpoint,
        "base": base_name,
        "instruct": instruct_name,
        "judge_model": judge_model,
        "num_samples": args.num_samples,
        "finetuned_vs_base": {k: v for k, v in ft_vs_base.items() if k != "details"},
        "finetuned_vs_instruct": {k: v for k, v in ft_vs_instruct.items() if k != "details"},
        "instruct_vs_base": {k: v for k, v in instruct_vs_base.items() if k != "details"},
    }
    results_path = os.path.join(output_dir, "judge_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved judge results to {results_path}")

    # Save detailed per-sample judgments.
    details_path = os.path.join(output_dir, "judge_details.json")
    with open(details_path, "w") as f:
        json.dump({
            "finetuned_vs_base": ft_vs_base["details"],
            "finetuned_vs_instruct": ft_vs_instruct["details"],
            "instruct_vs_base": instruct_vs_base["details"],
        }, f, indent=2)
    print(f"  Saved per-sample details to {details_path}")


if __name__ == "__main__":
    main()
