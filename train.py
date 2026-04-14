"""GRPO training with cross-model perplexity (CME) reward."""

from __future__ import annotations

import os

import torch
import yaml
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import GRPOConfig, GRPOTrainer

from reward import CMERewardModel, build_cme_reward_fn
from cme_trainer import CMETokenLevelGRPOTrainer
from eval import run_eval


PROMPT_TEMPLATE = (
    "Solve the following math problem. Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}\n\nSolution:"
)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_train_dataset(cfg: dict):
    ds = load_dataset(cfg["data"]["train_dataset"], split="train")

    def _map(ex):
        return {"prompt": PROMPT_TEMPLATE.format(problem=ex["problem"])}

    return ds.map(_map, remove_columns=[c for c in ds.column_names if c != "problem"])


class PeriodicEvalCallback(TrainerCallback):
    def __init__(self, cfg: dict, eval_steps: int, baseline_samples=None):
        self.cfg = cfg
        self.eval_steps = eval_steps
        self.baseline_samples = baseline_samples or []  # list of (problem, gold, text)

    def _generate_sample(self, model, tokenizer, problem, device, max_new_tokens=1024):
        from eval import format_prompt
        import torch as _torch
        prompt = format_prompt(problem, tokenizer)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with _torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)

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
            from eval import evaluate_all, extract_boxed, is_correct
            tokenizer.padding_side = "left"
            device = next(model.parameters()).device
            print(f"[step {state.global_step}] running benchmarks")
            results = evaluate_all(model, tokenizer, self.cfg, device, max_samples=50)
            if wandb.run is not None:
                wandb.define_metric("eval/*", step_metric="train/global_step", step_sync=False)
                wandb.log(
                    {
                        "train/global_step": state.global_step,
                        **{f"eval/{name}_pass@1": r["pass@1"] for name, r in results.items()},
                    },
                )

            # Side-by-side sample generations.
            if self.baseline_samples:
                print(f"\n{'=' * 70}")
                print(f"[step {state.global_step}] SIDE-BY-SIDE SAMPLES (5 problems)")
                print(f"{'=' * 70}")
                for i, (prob, gold, base_text) in enumerate(self.baseline_samples):
                    ft_text = self._generate_sample(model, tokenizer, prob, device)
                    b_pred = extract_boxed(base_text)
                    f_pred = extract_boxed(ft_text)
                    b_ok = "✓" if is_correct(b_pred, gold) else "✗"
                    f_ok = "✓" if is_correct(f_pred, gold) else "✗"
                    print(f"\n--- Problem {i+1} (gold={gold}) ---")
                    print(f"BASE [{b_ok} pred={b_pred}]: {base_text[:500]}...")
                    print(f"FINETUNED [{f_ok} pred={f_pred}]: {ft_text[:500]}...")
        finally:
            if was_training:
                model.train()
        return control


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
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

    model = AutoModelForCausalLM.from_pretrained(
        gen_name,
        torch_dtype=torch.bfloat16,
    )

    # Verifier on separate GPU if available.
    verifier_device = "cuda:1" if torch.cuda.device_count() > 1 else (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    reward_model = CMERewardModel(
        verifier_name=cfg["model"]["verifier"],
        device=verifier_device,
        max_length=cfg["reward"]["max_verifier_length"],
    )
    token_level = cfg.get("reward", {}).get("token_level", False)
    answer_only = cfg.get("reward", {}).get("answer_only_cme", False)
    no_box_penalty = cfg.get("reward", {}).get("no_box_penalty", 5.0)
    reward_fn = build_cme_reward_fn(
        reward_model,
        token_level=token_level,
        gen_tokenizer=tokenizer if token_level else None,
        answer_only=answer_only,
        no_box_penalty=no_box_penalty,
    )

    train_ds = build_train_dataset(cfg)

    # TRL requires generation_batch_size (per_device * world * grad_accum) to be
    # divisible by num_generations. Auto-bump grad_accum so this always holds.
    per_device_bs = cfg["training"]["per_device_train_batch_size"]
    requested_accum = cfg["training"]["gradient_accumulation_steps"]
    num_gen = cfg["generation"]["num_generations"]
    gen_batch = per_device_bs * requested_accum
    if gen_batch % num_gen != 0:
        bumped = ((gen_batch + num_gen - 1) // num_gen) * num_gen
        requested_accum = bumped // per_device_bs
        print(
            f"[train] bumping gradient_accumulation_steps "
            f"{cfg['training']['gradient_accumulation_steps']} -> {requested_accum} "
            f"so generation_batch_size ({per_device_bs * requested_accum}) is divisible by "
            f"num_generations ({num_gen})"
        )

    grpo_cfg = GRPOConfig(
        output_dir=cfg["training"]["output_dir"],
        learning_rate=cfg["training"]["learning_rate"],
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=requested_accum,
        num_train_epochs=cfg["training"]["num_train_epochs"],
        max_steps=cfg["training"]["max_steps"],
        warmup_steps=cfg["training"]["warmup_steps"],
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
        report_to=["wandb"],
    )

    # Cache baseline generations (5 MATH-500 problems) for side-by-side logging.
    print("\nCaching baseline generations for side-by-side comparison...")
    from eval import format_prompt
    math500_bench = next((b for b in cfg["benchmarks"] if b["name"] == "math500"), cfg["benchmarks"][0])
    _ds = load_dataset(math500_bench["dataset"], split=math500_bench["split"])
    _ds = _ds.shuffle(seed=42).select(range(5))
    baseline_samples = []
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(_device)
    model.eval()
    tokenizer.padding_side = "left"
    with torch.no_grad():
        for ex in _ds:
            prob = ex[math500_bench["problem_key"]]
            gold = ex[math500_bench["answer_key"]]
            prompt = format_prompt(prob, tokenizer)
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(_device)
            out = model.generate(
                **enc, max_new_tokens=1024, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            text = tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            baseline_samples.append((prob, gold, text))
    print(f"  cached {len(baseline_samples)} baseline samples\n")

    TrainerCls = CMETokenLevelGRPOTrainer if token_level else GRPOTrainer
    trainer = TrainerCls(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=train_ds,
        callbacks=[PeriodicEvalCallback(cfg, cfg["training"]["eval_steps"], baseline_samples=baseline_samples)],
    )

    if not cfg["training"].get("skip_baseline_eval", False):
        from eval import evaluate_all
        model.eval()
        tokenizer.padding_side = "left"
        device = next(model.parameters()).device
        print("[step 0] baseline benchmarks")
        baseline = evaluate_all(model, tokenizer, cfg, device, max_samples=50, debug=True)
        if wandb.run is not None:
            wandb.log(
                {f"eval/{name}_pass@1": r["pass@1"] for name, r in baseline.items()},
                step=0,
            )
        model.train()

    import glob
    ckpts = sorted(glob.glob(f"{cfg['training']['output_dir']}/checkpoint-*"))
    resume_ckpt = ckpts[-1] if ckpts else None
    if resume_ckpt:
        print(f"Resuming from {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(cfg["training"]["output_dir"])

    # Final eval.
    final = run_eval(cfg["training"]["output_dir"], cfg)
    if wandb.run is not None:
        wandb.log(
            {f"eval/final_{name}_pass@1": r["pass@1"] for name, r in final.items()}
        )
        wandb.finish()


if __name__ == "__main__":
    main()
