"""GRPO training with CME reward for open-ended instruction following.

Uses the same cross-model perplexity (CME) reward signal as the math training
script, but on UltraFeedback instructions with LLM-judge evaluation instead of
answer correctness checking.
"""

from __future__ import annotations

import os
import random
from typing import List

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

from reward import CMERewardModel


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)


def load_config(path: str = "config_quality1.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def format_prompt(instruction: str, tokenizer) -> str:
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": instruction}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return PROMPT_TEMPLATE.format(instruction=instruction)


def build_train_dataset(cfg: dict, tokenizer):
    ds = load_dataset(cfg["data"]["train_dataset"], split="train")
    max_samples = cfg["data"].get("max_train_samples", 5000)
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=42).select(range(max_samples))

    def _map(ex):
        instruction = ex.get("instruction", ex.get("prompt", ""))
        return {"prompt": format_prompt(instruction, tokenizer)}

    return ds.map(_map, remove_columns=[c for c in ds.column_names])


def build_quality_reward_fn(reward_model: CMERewardModel, reward_metric: str = "entropy"):
    """TRL-compatible reward function using sequence-level CME for open-ended tasks.

    No answer extraction or correctness checking — just how surprised the verifier
    is by the generator's response (lower surprise = higher reward).
    """
    _call_count = [0]

    def reward_fn(prompts, completions, **kwargs) -> List[float]:
        prompt_texts: List[str] = []
        completion_texts: List[str] = []
        for p, c in zip(prompts, completions):
            if isinstance(p, list):
                p = "\n".join(m.get("content", "") for m in p)
            if isinstance(c, list):
                c = "\n".join(m.get("content", "") for m in c)
            prompt_texts.append(p)
            completion_texts.append(c)

        debug_this = _call_count[0] < 3
        if debug_this:
            print(f"\n[DEBUG reward_fn call {_call_count[0]}] CME quality reward")
            print(f"  prompt[0][:150]: {repr(prompt_texts[0][:150])}")
            print(f"  completion[0][:200]: {repr(completion_texts[0][:200])}")
            _call_count[0] += 1

        rewards = reward_model.score(
            prompt_texts, completion_texts,
            token_level=False, answer_only=False,
            reward_metric=reward_metric,
        )

        if _call_count[0] <= 3:
            print(f"  rewards: {[f'{r:.4f}' for r in rewards]}")
            print(f"  mean={sum(rewards)/len(rewards):.4f} std={torch.tensor(rewards).std().item():.4f}")

        return rewards

    reward_fn.token_level = False
    return reward_fn


def _generate_responses(model, tokenizer, prompts: list[str], device: str, max_new_tokens: int = 1024) -> list[str]:
    """Generate greedy responses for a batch of already-formatted prompts."""
    model.eval()
    responses = []
    batch_size = 4
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
    return responses


def _judge_pairwise(
    instructions: list[str],
    responses_a: list[str],
    responses_b: list[str],
    judge_model: str = "gpt-5.2",
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """Use an LLM judge to compare two sets of responses pairwise.

    Returns dict with wins_a, wins_b, ties, winrate_a, winrate_b.
    """
    from openai import OpenAI
    client = OpenAI()

    wins_a, wins_b, ties = 0, 0, 0

    for instruction, resp_a, resp_b in zip(instructions, responses_a, responses_b):
        # Randomize presentation order to mitigate position bias.
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
            print(f"  Judge error: {e}")
            verdict = "TIE"

        if verdict == "A":
            winner = "a" if order == "ab" else "b"
        elif verdict == "B":
            winner = "b" if order == "ab" else "a"
        else:
            winner = "tie"

        winner_label = {
            "a": label_a, "b": label_b, "tie": "TIE",
        }[winner]
        print(f"    [{i+1}/{len(instructions)}] winner={winner_label} | {instruction[:70]}")

        if winner == "a":
            wins_a += 1
        elif winner == "b":
            wins_b += 1
        else:
            ties += 1

    total = wins_a + wins_b + ties
    return {
        "wins_a": wins_a, "wins_b": wins_b, "ties": ties, "total": total,
        "winrate_a": wins_a / total if total else 0,
        "winrate_b": wins_b / total if total else 0,
    }


class LLMJudgeEvalCallback(TrainerCallback):
    """Periodic LLM-judge evaluation during training.

    At startup, caches responses from the base model and instruct model on a
    fixed set of eval prompts. Every eval_steps, generates from the current
    (finetuned) model and runs three pairwise comparisons via LLM judge:
      - finetuned vs base
      - finetuned vs instruct
    Logs win rates to wandb.
    """

    def __init__(
        self,
        cfg: dict,
        eval_steps: int,
        eval_instructions: list[str],
        eval_prompts: list[str],
        base_responses: list[str],
        instruct_responses: list[str],
    ):
        self.cfg = cfg
        self.eval_steps = eval_steps
        self.eval_instructions = eval_instructions
        self.eval_prompts = eval_prompts
        self.base_responses = base_responses
        self.instruct_responses = instruct_responses
        self.judge_model = cfg.get("eval", {}).get("judge_model", "gpt-5.2")
        self.best_winrate = -1.0
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
            device = next(model.parameters()).device
            tokenizer.padding_side = "left"
            max_new_tokens = self.cfg.get("eval", {}).get("max_new_tokens", 2048)

            print(f"\n[step {state.global_step}] generating eval responses...")
            ft_responses = _generate_responses(
                model, tokenizer, self.eval_prompts, str(device), max_new_tokens,
            )

            print(f"[step {state.global_step}] running LLM judge (finetuned vs base)...")
            vs_base = _judge_pairwise(
                self.eval_instructions, ft_responses, self.base_responses,
                judge_model=self.judge_model, label_a="finetuned", label_b="base",
            )

            print(f"[step {state.global_step}] running LLM judge (finetuned vs instruct)...")
            vs_instruct = _judge_pairwise(
                self.eval_instructions, ft_responses, self.instruct_responses,
                judge_model=self.judge_model, label_a="finetuned", label_b="instruct",
            )

            print(f"[step {state.global_step}] JUDGE RESULTS:")
            print(f"  vs base:    finetuned wins {vs_base['wins_a']}, base wins {vs_base['wins_b']}, ties {vs_base['ties']} → winrate {vs_base['winrate_a']:.1%}")
            print(f"  vs instruct: finetuned wins {vs_instruct['wins_a']}, instruct wins {vs_instruct['wins_b']}, ties {vs_instruct['ties']} → winrate {vs_instruct['winrate_a']:.1%}")

            if wandb.run is not None:
                wandb.define_metric("eval/*", step_metric="train/global_step", step_sync=False)
                wandb.log({
                    "train/global_step": state.global_step,
                    "eval/winrate_vs_base": vs_base["winrate_a"],
                    "eval/winrate_vs_instruct": vs_instruct["winrate_a"],
                    "eval/wins_vs_base": vs_base["wins_a"],
                    "eval/wins_vs_instruct": vs_instruct["wins_a"],
                })

            # Save best checkpoint based on win rate vs base.
            if vs_base["winrate_a"] > self.best_winrate:
                self.best_winrate = vs_base["winrate_a"]
                print(f"[step {state.global_step}] new best winrate vs base: {vs_base['winrate_a']:.1%} — saving to {self.best_dir}")
                import shutil
                if os.path.exists(self.best_dir):
                    shutil.rmtree(self.best_dir)
                model.save_pretrained(self.best_dir)
                if tokenizer is not None:
                    tokenizer.save_pretrained(self.best_dir)
                with open(os.path.join(self.best_dir, "best_info.txt"), "w") as f:
                    f.write(
                        f"step={state.global_step}\n"
                        f"winrate_vs_base={vs_base['winrate_a']:.4f}\n"
                        f"winrate_vs_instruct={vs_instruct['winrate_a']:.4f}\n"
                    )

            # Print a few side-by-side samples.
            print(f"\n{'=' * 70}")
            print(f"[step {state.global_step}] SAMPLE COMPARISONS (3 prompts)")
            print(f"{'=' * 70}")
            for i in range(min(3, len(self.eval_instructions))):
                print(f"\n--- Prompt {i+1}: {self.eval_instructions[i][:100]}...")
                print(f"  BASE:      {self.base_responses[i][:300]}...")
                print(f"  FINETUNED: {ft_responses[i][:300]}...")
                print(f"  INSTRUCT:  {self.instruct_responses[i][:300]}...")
        finally:
            if was_training:
                model.train()
        return control


def _cache_model_responses(model_name: str, prompts: list[str], device: str, max_new_tokens: int = 1024) -> list[str]:
    """Load a model, generate on prompts, unload, return responses."""
    print(f"  Caching responses from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    responses = _generate_responses(model, tokenizer, prompts, device, max_new_tokens)

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return responses


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_quality1.yaml")
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
        gen_name, torch_dtype=torch.bfloat16,
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
    reward_metric = cfg.get("reward", {}).get("reward_metric", "entropy")
    reward_fn = build_quality_reward_fn(reward_model, reward_metric=reward_metric)

    train_ds = build_train_dataset(cfg, tokenizer)

    # Auto-bump gradient_accumulation_steps for TRL divisibility requirement.
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
        save_total_limit=1,
        report_to=["wandb"],
    )

    # ── Cache eval prompts and baseline responses for LLM judge ──
    judge_num_samples = cfg.get("eval", {}).get("judge_num_samples", 50)
    eval_max_tokens = cfg.get("eval", {}).get("max_new_tokens", 2048)
    gen_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("\nLoading eval prompts from UltraFeedback...")
    eval_ds = load_dataset(cfg["data"]["train_dataset"], split="train")
    eval_ds = eval_ds.shuffle(seed=99).select(range(judge_num_samples))
    eval_instructions = [
        ex.get("instruction", ex.get("prompt", "")) for ex in eval_ds
    ]
    eval_prompts = [format_prompt(inst, tokenizer) for inst in eval_instructions]

    print("Caching baseline model responses for LLM judge eval...")
    base_name = cfg["model"]["base"]
    instruct_name = cfg["model"]["instruct"]

    # Cache base model responses (the generator before training).
    model.to(gen_device)
    model.eval()
    tokenizer.padding_side = "left"
    base_responses = _generate_responses(model, tokenizer, eval_prompts, gen_device, eval_max_tokens)
    print(f"  cached {len(base_responses)} base responses")

    # Cache instruct model responses (loaded separately, then unloaded).
    # Format prompts with instruct model's tokenizer (may have chat template).
    instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_name)
    if instruct_tokenizer.pad_token is None:
        instruct_tokenizer.pad_token = instruct_tokenizer.eos_token
    instruct_prompts = [format_prompt(inst, instruct_tokenizer) for inst in eval_instructions]
    instruct_responses = _cache_model_responses(
        instruct_name, instruct_prompts, gen_device, eval_max_tokens,
    )
    print(f"  cached {len(instruct_responses)} instruct responses\n")

    eval_callback = LLMJudgeEvalCallback(
        cfg=cfg,
        eval_steps=cfg["training"]["eval_steps"],
        eval_instructions=eval_instructions,
        eval_prompts=eval_prompts,
        base_responses=base_responses,
        instruct_responses=instruct_responses,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=train_ds,
        callbacks=[eval_callback],
    )

    # Run baseline judge eval before training.
    if not cfg["training"].get("skip_baseline_eval", False):
        print("[step 0] baseline LLM judge eval")
        vs_base = _judge_pairwise(
            eval_instructions, base_responses, base_responses,
            judge_model=cfg.get("eval", {}).get("judge_model", "gpt-5.2"),
        )
        print(f"  baseline vs self: winrate {vs_base['winrate_a']:.1%} (sanity check — should be ~50%)")
        if wandb.run is not None:
            wandb.log({"eval/winrate_vs_base": 0.5, "eval/winrate_vs_instruct": 0.0}, step=0)

    import glob
    ckpts = sorted(glob.glob(f"{cfg['training']['output_dir']}/checkpoint-*"))
    resume_ckpt = ckpts[-1] if ckpts else None
    if resume_ckpt:
        print(f"Resuming from {resume_ckpt}")
    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(cfg["training"]["output_dir"])

    # Final judge eval.
    print("\n[FINAL] running LLM judge evaluation...")
    model.eval()
    tokenizer.padding_side = "left"
    device = next(model.parameters()).device
    final_responses = _generate_responses(model, tokenizer, eval_prompts, str(device), eval_max_tokens)
    judge_model = cfg.get("eval", {}).get("judge_model", "gpt-5.2")

    final_vs_base = _judge_pairwise(eval_instructions, final_responses, base_responses, judge_model=judge_model)
    final_vs_instruct = _judge_pairwise(eval_instructions, final_responses, instruct_responses, judge_model=judge_model)

    print(f"\nFINAL RESULTS:")
    print(f"  vs base:     winrate {final_vs_base['winrate_a']:.1%}")
    print(f"  vs instruct: winrate {final_vs_instruct['winrate_a']:.1%}")

    if wandb.run is not None:
        wandb.log({
            "eval/final_winrate_vs_base": final_vs_base["winrate_a"],
            "eval/final_winrate_vs_instruct": final_vs_instruct["winrate_a"],
        })
        wandb.finish()


if __name__ == "__main__":
    main()
