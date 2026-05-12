"""DPO baseline for the quality experiments.

Uses TRL's DPOTrainer with UltraFeedback preference pairs (chosen/rejected
extracted from the dataset's per-completion overall_score). Same generator,
same eval pipeline as train_quality.py — only the training objective changes.

Reuses LLMJudgeEvalCallback so DPO and CME-GRPO are evaluated identically.
"""

from __future__ import annotations

import os
from typing import List

import torch
import wandb
import yaml
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from train_quality import (
    PROMPT_TEMPLATE,
    format_prompt,
    LLMJudgeEvalCallback,
    _cache_model_responses,
)


def load_config(path: str = "config_quality_dpo.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _coerce_score(v) -> float | None:
    """UltraFeedback's overall_score is sometimes a string ('3', '5.0') and
    sometimes null. Coerce to float or return None to drop the completion."""
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def build_dpo_dataset(cfg: dict, tokenizer) -> Dataset:
    """Build (prompt, chosen, rejected) triples from UltraFeedback.

    For each prompt, pick the completion with the highest overall_score as
    'chosen' and the lowest as 'rejected'. Drop prompts where we can't form
    a meaningful pair (single completion, all equal scores, or empty texts).
    """
    raw = load_dataset(cfg["data"]["train_dataset"], split="train")
    max_samples = cfg["data"].get("max_train_samples", 5000)
    max_prompt_length = cfg["data"].get("max_prompt_length", 512)

    rows: list[dict] = []
    dropped_no_pair = 0
    dropped_too_long = 0

    for ex in raw:
        instruction = ex.get("instruction", ex.get("prompt", ""))
        if not instruction:
            continue

        comps = ex.get("completions", [])
        scored = []
        for c in comps:
            s = _coerce_score(c.get("overall_score"))
            r = c.get("response", "")
            if s is None or not r or not r.strip():
                continue
            scored.append((s, r))
        if len(scored) < 2:
            dropped_no_pair += 1
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        chosen = scored[0][1]
        rejected = scored[-1][1]
        if chosen == rejected or scored[0][0] == scored[-1][0]:
            dropped_no_pair += 1
            continue

        prompt = format_prompt(instruction, tokenizer)
        n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if n_prompt > max_prompt_length:
            dropped_too_long += 1
            continue

        rows.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "instruction": instruction,  # kept for eval callback parity
        })

    if len(rows) > max_samples:
        ds = Dataset.from_list(rows).shuffle(seed=42).select(range(max_samples))
    else:
        ds = Dataset.from_list(rows)

    print(
        f"[build_dpo_dataset] kept {len(ds)} pairs "
        f"(dropped {dropped_no_pair} no-pair, {dropped_too_long} too-long, "
        f"max_prompt_length={max_prompt_length})",
        flush=True,
    )
    return ds


def _generate_responses_for_eval(model, tokenizer, prompts: List[str], device: str,
                                  max_new_tokens: int) -> List[str]:
    """Greedy generation, matching the eval callback's expectations.

    Local reimplementation to avoid the train_quality.py version's bound coupling
    to the GRPO trainer. Identical decoding settings.
    """
    import time
    model.eval()
    responses: list[str] = []
    batch_size = 4
    n_total = len(prompts)
    start = time.time()
    for i in range(0, n_total, batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=2048,
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[:, enc.input_ids.shape[1]:]
        responses.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        done = min(i + batch_size, n_total)
        elapsed = time.time() - start
        eta = elapsed / max(done, 1) * (n_total - done)
        if done % 20 == 0 or done == n_total:
            print(f"    [{done}/{n_total}] elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)
    return responses


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_quality_dpo.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    os.environ.setdefault("WANDB_PROJECT", cfg["wandb"]["project"])
    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"]["run_name"],
        config=cfg,
    )

    gen_name = cfg["model"]["generator"]
    tokenizer = AutoTokenizer.from_pretrained(gen_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(gen_name, torch_dtype=torch.bfloat16)

    train_ds = build_dpo_dataset(cfg, tokenizer)

    # ── Eval prompts cache (mirrors train_quality.py exactly) ────────────────
    judge_num_samples = cfg.get("eval", {}).get("judge_num_samples", 50)
    eval_max_tokens = cfg.get("eval", {}).get("max_new_tokens", 2048)
    base_name = cfg["model"].get("base", gen_name)
    instruct_name = cfg["model"]["instruct"]
    cache_path = os.path.join(cfg["training"]["output_dir"], "eval_cache.json")
    os.makedirs(cfg["training"]["output_dir"], exist_ok=True)
    cache_key = {
        "dataset": cfg["data"]["train_dataset"],
        "n_samples": judge_num_samples,
        "base": base_name,
        "instruct": instruct_name,
        "max_new_tokens": eval_max_tokens,
    }
    gen_device = "cuda" if torch.cuda.is_available() else "cpu"

    import json as _json
    cached = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = _json.load(f)
            if cached.get("key") != cache_key:
                print(f"\n[eval cache] key mismatch — regenerating", flush=True)
                cached = None
            else:
                print(f"\n[eval cache] loaded {cache_path}", flush=True)
        except Exception as e:
            print(f"\n[eval cache] failed to load ({e}) — regenerating", flush=True)
            cached = None

    if cached is not None:
        eval_instructions = cached["instructions"]
        eval_prompts = cached["prompts"]
        base_responses = cached["base_responses"]
        instruct_responses = cached["instruct_responses"]
    else:
        print("\nLoading eval prompts from UltraFeedback...", flush=True)
        eval_ds = load_dataset(cfg["data"]["train_dataset"], split="train")
        max_prompt_length = cfg["data"].get("max_prompt_length", 512)

        def _short_enough(ex):
            inst = ex.get("instruction", ex.get("prompt", ""))
            templated = format_prompt(inst, tokenizer)
            n = len(tokenizer(templated, add_special_tokens=False)["input_ids"])
            return n <= max_prompt_length

        eval_ds = eval_ds.filter(_short_enough)
        eval_ds = eval_ds.shuffle(seed=99).select(range(judge_num_samples))
        eval_instructions = [ex.get("instruction", ex.get("prompt", "")) for ex in eval_ds]
        eval_prompts = [format_prompt(inst, tokenizer) for inst in eval_instructions]

        print("Caching base model responses for LLM judge eval...", flush=True)
        model.to(gen_device)
        base_responses = _generate_responses_for_eval(
            model, tokenizer, eval_prompts, gen_device, eval_max_tokens,
        )

        print(f"\nCaching responses from {instruct_name}...", flush=True)
        instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_name)
        if instruct_tokenizer.pad_token is None:
            instruct_tokenizer.pad_token = instruct_tokenizer.eos_token
        instruct_tokenizer.padding_side = "left"
        instruct_prompts = [format_prompt(inst, instruct_tokenizer) for inst in eval_instructions]
        instruct_responses = _cache_model_responses(
            instruct_name, instruct_prompts, gen_device, eval_max_tokens,
        )

        with open(cache_path, "w", encoding="utf-8") as f:
            _json.dump({
                "key": cache_key,
                "instructions": eval_instructions,
                "prompts": eval_prompts,
                "base_responses": base_responses,
                "instruct_responses": instruct_responses,
            }, f, ensure_ascii=False)
        print(f"[eval cache] saved to {cache_path}\n", flush=True)

    # ── DPO config ───────────────────────────────────────────────────────────
    # NOTE: TRL 1.2.0 removed max_length/max_prompt_length from DPOConfig.
    # build_dpo_dataset already filters by max_prompt_length, and DPOTrainer
    # falls back to tokenizer.model_max_length for total seq length.
    dpo_cfg = DPOConfig(
        output_dir=cfg["training"]["output_dir"],
        learning_rate=cfg["training"]["learning_rate"],
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        max_steps=cfg["training"]["max_steps"],
        warmup_steps=cfg["training"].get("warmup_steps", 0),
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=cfg["training"].get("save_total_limit", 2),
        bf16=cfg["training"]["bf16"],
        seed=cfg["training"]["seed"],
        beta=cfg["training"].get("dpo_beta", 0.1),
        max_grad_norm=cfg["training"].get("max_grad_norm", 1.0),
        gradient_checkpointing=True,
        report_to=["wandb"],
    )

    eval_callback = LLMJudgeEvalCallback(
        cfg=cfg,
        eval_steps=cfg["training"]["eval_steps"],
        eval_instructions=eval_instructions,
        eval_prompts=eval_prompts,
        base_responses=base_responses,
        instruct_responses=instruct_responses,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,                  # DPO uses copy of model as reference
        args=dpo_cfg,
        processing_class=tokenizer,
        train_dataset=train_ds,
        callbacks=[eval_callback],
    )
    eval_callback.trainer = trainer

    import glob
    ckpts = sorted(glob.glob(f"{cfg['training']['output_dir']}/checkpoint-*"))
    resume_ckpt = ckpts[-1] if ckpts else None
    if resume_ckpt:
        print(f"Resuming from {resume_ckpt}", flush=True)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(cfg["training"]["output_dir"])

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
