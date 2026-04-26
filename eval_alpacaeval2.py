"""Run AlpacaEval 2.0 on a trained checkpoint (publication-grade eval).

Generates model responses on the 805 AlpacaEval 2.0 prompts, saves them in
AlpacaEval's expected JSON format, and (optionally) invokes the AlpacaEval
package to compute the GPT-4-Turbo-judged length-controlled winrate against
the standard GPT-4-Turbo baseline.

Install:
    pip install alpaca-eval

Setup (once):
    export OPENAI_API_KEY=sk-...

Usage:
    # Step 1: generate only (free, local)
    python eval_alpacaeval2.py --config config_quality1_token_level.yaml \\
        --checkpoint ./outputs/cme-grpo-quality-qwen0.5b-token-level/checkpoint-best

    # Also generate for base + instruct so you can compare later
    python eval_alpacaeval2.py --config config_quality1_token_level.yaml \\
        --checkpoint ./outputs/.../checkpoint-best --also-baselines

    # Step 2: run the judge (costs ~$10-15 per model in OpenAI credits)
    python eval_alpacaeval2.py --config config_quality1_token_level.yaml \\
        --checkpoint ./outputs/.../checkpoint-best --also-baselines --run-eval
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import torch
import yaml
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


def generate_for_model(
    model_name_or_path: str,
    instructions: list[str],
    device: str,
    max_new_tokens: int,
    batch_size: int,
    label: str,
) -> list[str]:
    """Load a model, generate greedy responses on all instructions, unload, return."""
    # If it looks like a local path (contains "/" or starts with "." or exists
    # on disk), normalize to an absolute path so newer transformers versions
    # don't misinterpret it as a HuggingFace repo ID.
    if ("/" in model_name_or_path and not model_name_or_path.count("/") == 1) \
            or model_name_or_path.startswith(".") \
            or os.path.isdir(model_name_or_path):
        model_name_or_path = os.path.abspath(model_name_or_path)
        if not os.path.isdir(model_name_or_path):
            raise FileNotFoundError(
                f"Local checkpoint path not found: {model_name_or_path}\n"
                "Check the directory exists and contains config.json + model weights."
            )
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


def save_alpacaeval_json(path: str, generator_name: str,
                         instructions: list[str], responses: list[str]) -> None:
    """Save outputs in AlpacaEval 2.0 expected format:
        [{"instruction": ..., "output": ..., "generator": ...}, ...]
    """
    data = [
        {"instruction": inst, "output": resp, "generator": generator_name}
        for inst, resp in zip(instructions, responses)
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  saved {len(data)} outputs -> {path}", flush=True)


def run_alpacaeval(outputs_path: str, name: str, output_dir: str,
                   annotators_config: str | None = None) -> None:
    """Invoke `alpaca_eval evaluate` on a JSON file and save leaderboard.

    `annotators_config` controls which judge model AlpacaEval uses. Common values:
      - None (default): alpaca_eval_gpt4_turbo_fn (GPT-4-Turbo, ~$12/model, leaderboard-comparable)
      - 'alpaca_eval_gpt4o_fn': GPT-4o
      - 'alpaca_eval_gpt4o_mini_fn': GPT-4o-mini (~$1-2/model, less standard)
      - 'claude_3_opus_evaluator': Claude 3 Opus
    See `alpaca_eval/evaluators_configs/` in the package for all options.
    """
    print(f"\n{'=' * 60}", flush=True)
    print(f"Running AlpacaEval 2.0 on: {outputs_path}", flush=True)
    if annotators_config:
        print(f"  annotators_config: {annotators_config}", flush=True)
    print(f"{'=' * 60}", flush=True)
    cmd = [
        "alpaca_eval", "evaluate",
        "--model_outputs", outputs_path,
        "--output_path", os.path.join(output_dir, f"{name}_leaderboard"),
    ]
    if annotators_config:
        cmd += ["--annotators_config", annotators_config]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("ERROR: `alpaca_eval` CLI not found. Install with:")
        print("  pip install alpaca-eval")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: alpaca_eval failed with exit code {e.returncode}")
        print("Make sure OPENAI_API_KEY is set and has credits.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="YAML config (reads model.base, model.instruct from it)")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to trained checkpoint")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write outputs (default: <checkpoint>/alpaca_eval_2)")
    ap.add_argument("--also-baselines", action="store_true",
                    help="Also generate for base + instruct models from config")
    ap.add_argument("--run-eval", action="store_true",
                    help="After generating, invoke `alpaca_eval evaluate` (costs $10-15/model)")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--annotators-config", default=None,
                    help="AlpacaEval annotator name (judge model). "
                         "Examples: alpaca_eval_gpt4_turbo_fn (default), "
                         "alpaca_eval_gpt4o_fn, alpaca_eval_gpt4o_mini_fn")
    ap.add_argument("--name-prefix", default="",
                    help="Optional prefix for generator name in JSON (e.g. 'cme-grpo')")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.checkpoint.rstrip("/")), "alpaca_eval_2"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Load AlpacaEval 2.0 prompts (805 instructions). Download the JSON file
    # directly from HF — the dataset has a loading script that newer `datasets`
    # versions reject.
    print("Loading AlpacaEval 2.0 prompts...", flush=True)
    from huggingface_hub import hf_hub_download
    data_path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="alpaca_eval.json",
        repo_type="dataset",
    )
    with open(data_path, "r", encoding="utf-8") as f:
        eval_records = json.load(f)
    instructions = [ex["instruction"] for ex in eval_records]
    print(f"  loaded {len(instructions)} prompts", flush=True)

    # Models to generate for.
    ckpt_name = args.name_prefix + (args.name_prefix and "-" or "") + \
                os.path.basename(args.checkpoint.rstrip("/"))
    runs = [("finetuned", args.checkpoint, ckpt_name or "finetuned")]
    if args.also_baselines:
        runs.append(("base", cfg["model"]["base"],
                     "base-" + cfg["model"]["base"].replace("/", "_")))
        runs.append(("instruct", cfg["model"]["instruct"],
                     "instruct-" + cfg["model"]["instruct"].replace("/", "_")))

    generated_paths: list[tuple[str, str]] = []
    for label, path, name in runs:
        out_path = os.path.join(output_dir, f"{name}.json")
        if os.path.exists(out_path):
            print(f"\n[{label}] already exists at {out_path}, skipping generation", flush=True)
            generated_paths.append((name, out_path))
            continue

        responses = generate_for_model(
            path, instructions, device,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            label=label,
        )
        save_alpacaeval_json(out_path, name, instructions, responses)
        generated_paths.append((name, out_path))

    print("\n" + "=" * 60)
    print(f"Generated {len(generated_paths)} model output file(s) in {output_dir}")
    print("=" * 60)
    for name, path in generated_paths:
        print(f"  {name}: {path}")

    if args.run_eval:
        print("\n" + "=" * 60)
        print("Running AlpacaEval 2.0 judge")
        print("This uses GPT-4-Turbo and costs ~$10-15 per model (805 judgments each).")
        print("Set OPENAI_API_KEY to your key with credits.")
        print("=" * 60)
        for name, path in generated_paths:
            run_alpacaeval(path, name, output_dir, annotators_config=args.annotators_config)
        print("\nDone. Leaderboards written to", output_dir)
    else:
        print("\n[skip] --run-eval not passed. To evaluate manually:")
        for _, path in generated_paths:
            print(f"  alpaca_eval evaluate --model_outputs {path}")
        print("\nOr pass --run-eval to run it automatically.")


if __name__ == "__main__":
    main()
