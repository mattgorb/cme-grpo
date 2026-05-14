"""Vanilla GRPO training with binary correctness reward.

Mirrors train.py but:
  - skips loading any verifier model
  - reward = 1.0 if extract_boxed(completion) matches gold answer, else 0.0
  - rewrites output_dir / wandb run_name from "cme-grpo-*" → "vanilla-grpo-*"
    so vanilla runs don't clobber CME-GRPO checkpoints
"""

from __future__ import annotations

import glob
import os

import torch
import wandb
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import GRPOConfig, GRPOTrainer

from eval import extract_boxed, format_prompt, is_correct, run_eval


PROMPT_TEMPLATE = (
    "Solve the following math problem. Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}\n\nSolution:"
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def rewrite_paths(cfg: dict) -> dict:
    """Redirect output_dir and wandb run_name from cme-grpo → vanilla-grpo,
    and cap max_steps to 600 for the vanilla baseline."""
    cfg["training"]["output_dir"] = cfg["training"]["output_dir"].replace(
        "cme-grpo", "vanilla-grpo"
    )
    cfg["wandb"]["run_name"] = cfg["wandb"]["run_name"].replace(
        "cme-grpo", "vanilla-grpo"
    )
    if not cfg["wandb"]["run_name"].startswith("vanilla"):
        cfg["wandb"]["run_name"] = "vanilla-" + cfg["wandb"]["run_name"]
    cfg["training"]["max_steps"] = 600
    return cfg


def build_train_dataset(cfg: dict, tokenizer):
    ds = load_dataset(cfg["data"]["train_dataset"], split="train")

    def _map(ex):
        gold = extract_boxed(ex.get("solution", "")) or ""
        prompt = format_prompt(ex["problem"], tokenizer, fallback_tokenizer=None)
        return {"prompt": prompt, "gold_answer": gold}

    keep = {"problem", "gold_answer"}
    return ds.map(_map, remove_columns=[c for c in ds.column_names if c not in keep])


def build_correctness_reward_fn():
    """TRL-compatible reward_funcs callable.

    TRL passes `prompts`, `completions`, and any extra dataset columns as kwargs.
    We need `gold_answer` (column from our train dataset) per row.
    """
    def reward_fn(completions, gold_answer, **_kwargs):
        rewards = []
        for completion, gold in zip(completions, gold_answer):
            pred = extract_boxed(completion)
            rewards.append(1.0 if is_correct(pred, gold) else 0.0)
        return rewards

    return reward_fn


class GradientStepPrintCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        total = args.per_device_train_batch_size * args.gradient_accumulation_steps
        unique = total // args.num_generations
        print(
            f"[grad-update] step={state.global_step} "
            f"unique_prompts={unique} total_completions={total}",
            flush=True,
        )
        return control


class PeriodicEvalCallback(TrainerCallback):
    def __init__(self, cfg: dict, eval_steps: int):
        self.cfg = cfg
        self.eval_steps = eval_steps
        self.best_accuracy = -1.0
        self.best_dir = os.path.join(cfg["training"]["output_dir"], "checkpoint-best")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return control
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
        if model is None or tokenizer is None:
            return control

        was_training = model.training
        model.eval()
        try:
            from eval import evaluate_all
            tokenizer.padding_side = "left"
            device = next(model.parameters()).device
            print(f"[step {state.global_step}] running benchmarks")
            results = evaluate_all(model, tokenizer, self.cfg, device, max_samples=50)
            if wandb.run is not None:
                wandb.define_metric("eval/*", step_metric="train/global_step", step_sync=False)
                wandb.log({
                    "train/global_step": state.global_step,
                    **{f"eval/{name}_pass@1": r["pass@1"] for name, r in results.items()},
                })

            mean_acc = sum(r["pass@1"] for r in results.values()) / max(len(results), 1)
            if mean_acc > self.best_accuracy:
                self.best_accuracy = mean_acc
                print(f"[step {state.global_step}] new best: {mean_acc:.4f} → {self.best_dir}")
                import shutil
                if os.path.exists(self.best_dir):
                    shutil.rmtree(self.best_dir)
                model.save_pretrained(self.best_dir)
                tokenizer.save_pretrained(self.best_dir)
                with open(os.path.join(self.best_dir, "best_info.txt"), "w") as f:
                    f.write(f"step={state.global_step}\nmean_accuracy={mean_acc:.4f}\n")
        finally:
            if was_training:
                model.train()
        return control


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = rewrite_paths(load_config(args.config))

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

    model = AutoModelForCausalLM.from_pretrained(gen_name, torch_dtype=torch.bfloat16)

    reward_fn = build_correctness_reward_fn()
    train_ds = build_train_dataset(cfg, tokenizer)

    # Ensure generation_batch_size divisible by num_generations.
    per_device_bs = cfg["training"]["per_device_train_batch_size"]
    requested_accum = cfg["training"]["gradient_accumulation_steps"]
    num_gen = cfg["generation"]["num_generations"]
    gen_batch = per_device_bs * requested_accum
    if gen_batch % num_gen != 0:
        bumped = ((gen_batch + num_gen - 1) // num_gen) * num_gen
        requested_accum = bumped // per_device_bs
        print(
            f"[train] bumping grad_accum {cfg['training']['gradient_accumulation_steps']} "
            f"→ {requested_accum} so gen_batch divisible by num_generations"
        )

    grpo_kwargs = {}
    if cfg["training"].get("steps_per_generation") is not None:
        grpo_kwargs["steps_per_generation"] = cfg["training"]["steps_per_generation"]

    grpo_cfg = GRPOConfig(
        output_dir=cfg["training"]["output_dir"],
        learning_rate=cfg["training"]["learning_rate"],
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=requested_accum,
        num_train_epochs=cfg["training"]["num_train_epochs"],
        max_steps=cfg["training"]["max_steps"],
        warmup_steps=cfg["training"].get("warmup_steps", 0),
        warmup_ratio=cfg["training"].get("warmup_ratio", 0.0),
        lr_scheduler_type=cfg["training"].get("lr_scheduler_type", "linear"),
        logging_steps=cfg["training"]["logging_steps"],
        save_steps=cfg["training"]["save_steps"],
        bf16=cfg["training"]["bf16"],
        seed=cfg["training"]["seed"],
        num_generations=cfg["generation"]["num_generations"],
        temperature=cfg["generation"]["temperature"],
        max_completion_length=cfg["generation"]["max_new_tokens"],
        beta=cfg["training"]["kl_coef"],
        max_grad_norm=cfg["training"].get("max_grad_norm", 1.0),
        gradient_checkpointing=True,
        save_total_limit=1,
        report_to=["wandb"],
        **grpo_kwargs,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=train_ds,
        callbacks=[
            GradientStepPrintCallback(),
            PeriodicEvalCallback(cfg, cfg["training"]["eval_steps"]),
        ],
    )

    if not cfg["training"].get("skip_baseline_eval", False):
        from eval import evaluate_all
        model.eval()
        tokenizer.padding_side = "left"
        device = next(model.parameters()).device
        print("[step 0] baseline benchmarks")
        baseline = evaluate_all(model, tokenizer, cfg, device, max_samples=50, debug=True)
        if wandb.run is not None:
            wandb.log({f"eval/{name}_pass@1": r["pass@1"] for name, r in baseline.items()}, step=0)
        model.train()

    ckpts = sorted(glob.glob(f"{cfg['training']['output_dir']}/checkpoint-*"))
    resume_ckpt = ckpts[-1] if ckpts else None
    if resume_ckpt:
        print(f"Resuming from {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(cfg["training"]["output_dir"])

    final = run_eval(cfg["training"]["output_dir"], cfg)
    if wandb.run is not None:
        wandb.log({f"eval/final_{name}_pass@1": r["pass@1"] for name, r in final.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
