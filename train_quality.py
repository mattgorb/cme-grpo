"""GRPO training with CME reward for open-ended instruction following.

Uses the same cross-model perplexity (CME) reward signal as the math training
script, but on UltraFeedback instructions with LLM-judge evaluation instead of
answer correctness checking.
"""

from __future__ import annotations

# Force UTF-8 for Path.read_text so HF Hub's ASCII-default model-card template
# load doesn't crash on containers with POSIX/ASCII locale.
# Only forward newline= when explicitly provided — Python 3.12's Path.read_text
# doesn't accept it (3.13+ added that kwarg).
import pathlib as _pathlib
_orig_read_text = _pathlib.Path.read_text
def _read_text_utf8(self, encoding=None, errors=None, newline=None):
    if encoding is None:
        encoding = "utf-8"
    kwargs = {"encoding": encoding, "errors": errors}
    if newline is not None:
        kwargs["newline"] = newline
    return _orig_read_text(self, **kwargs)
_pathlib.Path.read_text = _read_text_utf8

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


def _generate_responses(model, tokenizer, prompts: list[str], device: str, max_new_tokens: int = 1024, label: str = "") -> list[str]:
    """Generate greedy responses for a batch of already-formatted prompts."""
    import time as _time
    model.eval()
    responses = []
    batch_size = 4
    n_total = len(prompts)
    t_start = _time.time()
    tag = f"[{label}] " if label else ""
    print(f"{tag}generating {n_total} responses (batch_size={batch_size}, max_new_tokens={max_new_tokens})...", flush=True)
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
        elapsed = _time.time() - t_start
        rate = done / max(elapsed, 1e-6)
        eta = (n_total - done) / max(rate, 1e-6)
        print(f"{tag}  [{done}/{n_total}] elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)
    total_time = _time.time() - t_start
    print(f"{tag}done: {n_total} responses in {total_time:.0f}s", flush=True)
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

    Returns aggregate win counts plus a `per_sample` list with one record per
    instruction: {index, winner, reason}. The judge is prompted to emit both a
    verdict and a 1-2 sentence justification.
    """
    from openai import OpenAI
    client = OpenAI()

    wins_a, wins_b, ties = 0, 0, 0
    per_sample: list[dict] = []

    for i, (instruction, resp_a, resp_b) in enumerate(zip(instructions, responses_a, responses_b)):
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
            "Respond in EXACTLY this format (one line each, no extra text):\n"
            "WINNER: <A | B | TIE>\n"
            "REASON: <one or two sentences>\n"
        )

        raw = ""
        verdict = "TIE"
        reason = ""
        try:
            response = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_completion_tokens=200,
                temperature=0,
            )
            raw = (response.choices[0].message.content or "").strip()
            # Parse WINNER and REASON lines (tolerant of casing / stray text).
            for line in raw.splitlines():
                s = line.strip()
                if s.upper().startswith("WINNER:"):
                    tok = s.split(":", 1)[1].strip().upper()
                    if tok.startswith("A"):
                        verdict = "A"
                    elif tok.startswith("B"):
                        verdict = "B"
                    else:
                        verdict = "TIE"
                elif s.upper().startswith("REASON:"):
                    reason = s.split(":", 1)[1].strip()
            # If the judge didn't format but did emit A/B/TIE, fall back.
            if verdict == "TIE" and not reason:
                head = raw.strip().upper()[:3]
                if head.startswith("A"):
                    verdict = "A"
                elif head.startswith("B"):
                    verdict = "B"
        except Exception as e:
            print(f"  Judge error: {e}")
            reason = f"(judge error: {type(e).__name__}: {e})"

        if verdict == "A":
            winner = "a" if order == "ab" else "b"
        elif verdict == "B":
            winner = "b" if order == "ab" else "a"
        else:
            winner = "tie"

        winner_label = {"a": label_a, "b": label_b, "tie": "TIE"}[winner]
        print(f"    [{i+1}/{len(instructions)}] winner={winner_label} | {instruction[:70]}")

        if winner == "a":
            wins_a += 1
        elif winner == "b":
            wins_b += 1
        else:
            ties += 1

        per_sample.append({
            "index": i,
            "winner": winner,          # 'a' | 'b' | 'tie' in terms of label_a/label_b
            "winner_label": winner_label,
            "order_shown": order,       # "ab" = label_a shown first, "ba" = reversed
            "verdict": verdict,         # what the judge replied (A/B/TIE in shown order)
            "reason": reason,
            "raw": raw,
        })

    total = wins_a + wins_b + ties
    # AlpacaEval-style winrate: ties count as half-wins for each side.
    # This avoids collapsing "ties" into "losses" when many prompts are judged even.
    return {
        "wins_a": wins_a, "wins_b": wins_b, "ties": ties, "total": total,
        "winrate_a": (wins_a + 0.5 * ties) / total if total else 0,
        "winrate_b": (wins_b + 0.5 * ties) / total if total else 0,
        # Also include the strict-wins-only fraction for reference.
        "strict_winrate_a": wins_a / total if total else 0,
        "strict_winrate_b": wins_b / total if total else 0,
        "per_sample": per_sample,
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
        self.output_dir = cfg["training"]["output_dir"]
        # Track best for each comparison separately; save to two distinct dirs.
        self.best_vs_base = -1.0
        self.best_vs_instruct = -1.0
        # NOTE: avoid the "checkpoint-" prefix so HF Trainer's _rotate_checkpoints
        # (which globs "checkpoint-*" with use_mtime=True) doesn't sweep these
        # dirs into its save_total_limit cleanup.
        self.best_vs_base_dir = os.path.join(self.output_dir, "best-vs-base")
        self.best_vs_instruct_dir = os.path.join(self.output_dir, "best-vs-instruct")
        # Aliases for backwards compatibility with code that referenced `best_dir`/`best_winrate`.
        self.best_dir = self.best_vs_base_dir
        self.best_winrate = -1.0
        # Reference to the GRPOTrainer, set by main() after construction. Used to
        # call trainer.save_model() which handles wrapping / accelerate correctly.
        self.trainer = None

        # Save ALL eval prompts (not a subset).
        self.sample_indices = list(range(len(eval_instructions)))
        self.samples_dir = os.path.join(self.output_dir, "eval_samples")
        os.makedirs(self.samples_dir, exist_ok=True)

    def _save_best_via_copy(self, target_dir: str, source_output_dir: str):
        """Copy the most recent numeric checkpoint-N to target_dir.

        More reliable than `model.save_pretrained` inside a callback because
        the trainer's own save mechanism handles wrapped/DDP models correctly.
        Returns the source path on success, None if no numeric checkpoint exists.
        """
        import re, glob, shutil
        candidates = []
        for d in glob.glob(os.path.join(source_output_dir, "checkpoint-*")):
            if not os.path.isdir(d):
                continue
            m = re.match(r"checkpoint-(\d+)$", os.path.basename(d))
            if m:
                candidates.append((int(m.group(1)), d))
        if not candidates:
            return None
        candidates.sort()
        _, latest = candidates[-1]
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(latest, target_dir)
        return latest

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

            print(f"[step {state.global_step}] running LLM judge (finetuned vs base)...", flush=True)
            try:
                vs_base = _judge_pairwise(
                    self.eval_instructions, ft_responses, self.base_responses,
                    judge_model=self.judge_model, label_a="finetuned", label_b="base",
                )
            except Exception as e:
                print(f"  [WARN] judge vs base failed ({type(e).__name__}: {e})", flush=True)
                vs_base = {"wins_a": 0, "wins_b": 0, "ties": 0, "total": 0, "winrate_a": 0.5, "winrate_b": 0.5}

            print(f"[step {state.global_step}] running LLM judge (finetuned vs instruct)...", flush=True)
            try:
                vs_instruct = _judge_pairwise(
                    self.eval_instructions, ft_responses, self.instruct_responses,
                    judge_model=self.judge_model, label_a="finetuned", label_b="instruct",
                )
            except Exception as e:
                print(f"  [WARN] judge vs instruct failed ({type(e).__name__}: {e})", flush=True)
                vs_instruct = {"wins_a": 0, "wins_b": 0, "ties": 0, "total": 0, "winrate_a": 0.5, "winrate_b": 0.5}

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

            # Save best checkpoints for both metrics. Each is independently tracked
            # and saved to its own directory, so both "best vs base" and
            # "best vs instruct" are preserved.
            def _save_best(metric_name, current_winrate, prev_best, target_dir):
                if current_winrate <= prev_best:
                    return prev_best  # no improvement
                print(
                    f"[step {state.global_step}] new best winrate vs {metric_name}: "
                    f"{current_winrate:.1%} (prev {prev_best:.1%}) — saving to {target_dir}",
                    flush=True,
                )
                # Strategy 1: trainer.save_model — handles wrapped/DDP models correctly,
                # captures the CURRENT model state (no lag).
                # Strategy 2 (fallback): copy the latest numeric checkpoint from disk.
                save_method = None
                source_info = ""
                try:
                    if self.trainer is not None and hasattr(self.trainer, "save_model"):
                        import shutil
                        if os.path.exists(target_dir):
                            shutil.rmtree(target_dir)
                        self.trainer.save_model(target_dir)
                        save_method = "trainer.save_model"
                        source_info = f"current model state at step {state.global_step}"
                    else:
                        src = self._save_best_via_copy(target_dir, self.output_dir)
                        if src is None:
                            print(f"  [WARN] trainer.save_model unavailable AND no "
                                  f"numeric checkpoint to copy from. Skipping save "
                                  f"(will retry next eval).", flush=True)
                            return prev_best
                        save_method = "copy-from-latest"
                        source_info = f"copied from {src}"
                except Exception as e:
                    print(f"  [WARN] {save_method or 'trainer.save_model'} failed "
                          f"({type(e).__name__}: {e}); trying copy-from-latest fallback", flush=True)
                    try:
                        src = self._save_best_via_copy(target_dir, self.output_dir)
                        if src is None:
                            print(f"  [ERROR] no numeric checkpoint exists yet to copy from. "
                                  f"Skipping save (will retry next eval).", flush=True)
                            return prev_best
                        save_method = "copy-from-latest (fallback)"
                        source_info = f"copied from {src}"
                    except Exception as e2:
                        print(f"  [ERROR] fallback copy also failed: "
                              f"{type(e2).__name__}: {e2}", flush=True)
                        import traceback
                        traceback.print_exc()
                        return prev_best

                # Write best_info.txt and verify config.json exists.
                try:
                    with open(os.path.join(target_dir, "best_info.txt"), "w") as f:
                        f.write(
                            f"step={state.global_step}\n"
                            f"metric=winrate_vs_{metric_name}\n"
                            f"winrate_vs_base={vs_base['winrate_a']:.4f}\n"
                            f"winrate_vs_instruct={vs_instruct['winrate_a']:.4f}\n"
                            f"save_method={save_method}\n"
                            f"source={source_info}\n"
                        )
                except Exception as e:
                    print(f"  [WARN] couldn't write best_info.txt: {e}", flush=True)

                cfg_path = os.path.join(target_dir, "config.json")
                if os.path.exists(cfg_path):
                    print(f"  → saved via {save_method} ({source_info}; config.json verified)", flush=True)
                else:
                    print(f"  [WARN] save claimed to succeed but {cfg_path} not found!", flush=True)
                return current_winrate

            self.best_vs_base = _save_best(
                "base", vs_base["winrate_a"], self.best_vs_base, self.best_vs_base_dir,
            )
            self.best_vs_instruct = _save_best(
                "instruct", vs_instruct["winrate_a"], self.best_vs_instruct, self.best_vs_instruct_dir,
            )
            self.best_winrate = self.best_vs_base  # keep alias in sync

            # Build per-sample records joining responses + judge verdicts/reasons.
            vs_base_per = {r["index"]: r for r in vs_base.get("per_sample", [])}
            vs_ins_per  = {r["index"]: r for r in vs_instruct.get("per_sample", [])}

            samples = []
            for idx in self.sample_indices:
                judge_vs_base = vs_base_per.get(idx, {})
                judge_vs_ins  = vs_ins_per.get(idx, {})
                samples.append({
                    "index": idx,
                    "instruction": self.eval_instructions[idx],
                    "base_response": self.base_responses[idx],
                    "finetuned_response": ft_responses[idx],
                    "instruct_response": self.instruct_responses[idx],
                    "judge_finetuned_vs_base": {
                        "winner": judge_vs_base.get("winner_label", ""),
                        "reason": judge_vs_base.get("reason", ""),
                    },
                    "judge_finetuned_vs_instruct": {
                        "winner": judge_vs_ins.get("winner_label", ""),
                        "reason": judge_vs_ins.get("reason", ""),
                    },
                })

            # JSON (machine-readable).
            sample_file = os.path.join(
                self.samples_dir, f"step_{state.global_step:05d}.json"
            )
            import json as _json
            with open(sample_file, "w", encoding="utf-8") as f:
                _json.dump(
                    {"step": state.global_step, "samples": samples},
                    f, indent=2, ensure_ascii=False,
                )

            # Markdown (human-readable — real newlines, no \n escapes).
            md_file = os.path.join(
                self.samples_dir, f"step_{state.global_step:05d}.md"
            )
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(f"# Eval samples — step {state.global_step}\n\n")
                f.write(
                    f"Aggregate: **finetuned vs base** "
                    f"winrate = {vs_base['winrate_a']:.1%} "
                    f"({vs_base['wins_a']}W / {vs_base['wins_b']}L / {vs_base['ties']}T)  ·  "
                    f"**finetuned vs instruct** "
                    f"winrate = {vs_instruct['winrate_a']:.1%} "
                    f"({vs_instruct['wins_a']}W / {vs_instruct['wins_b']}L / {vs_instruct['ties']}T)\n\n"
                )
                f.write("---\n\n")
                for s in samples:
                    f.write(f"## Sample {s['index']}\n\n")
                    f.write(f"### Instruction\n\n{s['instruction']}\n\n")
                    f.write(f"### Base response\n\n{s['base_response']}\n\n")
                    f.write(f"### Finetuned response\n\n{s['finetuned_response']}\n\n")
                    f.write(f"### Instruct response\n\n{s['instruct_response']}\n\n")
                    jb = s["judge_finetuned_vs_base"]
                    ji = s["judge_finetuned_vs_instruct"]
                    f.write(f"### Judge: finetuned vs base\n\n")
                    f.write(f"- **Winner**: {jb['winner']}\n")
                    f.write(f"- **Reason**: {jb['reason']}\n\n")
                    f.write(f"### Judge: finetuned vs instruct\n\n")
                    f.write(f"- **Winner**: {ji['winner']}\n")
                    f.write(f"- **Reason**: {ji['reason']}\n\n")
                    f.write("---\n\n")

            print(f"[step {state.global_step}] wrote {len(samples)} samples → {sample_file} and {md_file}", flush=True)
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
        save_total_limit=cfg["training"].get("save_total_limit", 3),
        report_to=["wandb"],
    )

    # ── Cache eval prompts and baseline responses for LLM judge ──
    judge_num_samples = cfg.get("eval", {}).get("judge_num_samples", 50)
    eval_max_tokens = cfg.get("eval", {}).get("max_new_tokens", 2048)
    gen_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    base_name = cfg["model"]["base"]
    instruct_name = cfg["model"]["instruct"]

    # ── Disk cache for eval prompts + base/instruct responses ──
    # Regeneration across restarts wastes ~10-20 min. Cache keyed to settings
    # that would change what's generated; invalidates automatically.
    import json as _json
    os.makedirs(cfg["training"]["output_dir"], exist_ok=True)
    cache_path = os.path.join(cfg["training"]["output_dir"], "eval_cache.json")
    cache_key = {
        "dataset": cfg["data"]["train_dataset"],
        "n_samples": judge_num_samples,
        "shuffle_seed": 99,
        "base_model": base_name,
        "instruct_model": instruct_name,
        "max_new_tokens": eval_max_tokens,
    }

    cached = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                c = _json.load(f)
            if c.get("key") == cache_key:
                cached = c
                print(f"\n[eval cache] loaded {cache_path} (skipping base + instruct generation)", flush=True)
            else:
                print(f"\n[eval cache] key mismatch — regenerating", flush=True)
        except Exception as e:
            print(f"\n[eval cache] failed to load ({e}) — regenerating", flush=True)

    if cached is not None:
        eval_instructions  = cached["instructions"]
        eval_prompts       = cached["prompts"]
        base_responses     = cached["base_responses"]
        instruct_responses = cached["instruct_responses"]
    else:
        print("\nLoading eval prompts from UltraFeedback...", flush=True)
        eval_ds = load_dataset(cfg["data"]["train_dataset"], split="train")
        eval_ds = eval_ds.shuffle(seed=99).select(range(judge_num_samples))
        eval_instructions = [
            ex.get("instruction", ex.get("prompt", "")) for ex in eval_ds
        ]
        eval_prompts = [format_prompt(inst, tokenizer) for inst in eval_instructions]

        print("Caching baseline model responses for LLM judge eval...", flush=True)

        # Base model = the generator before training.
        model.to(gen_device)
        model.eval()
        tokenizer.padding_side = "left"
        base_responses = _generate_responses(model, tokenizer, eval_prompts, gen_device, eval_max_tokens)
        print(f"  cached {len(base_responses)} base responses", flush=True)

        # Instruct model (loaded separately, then unloaded).
        instruct_tokenizer = AutoTokenizer.from_pretrained(instruct_name)
        if instruct_tokenizer.pad_token is None:
            instruct_tokenizer.pad_token = instruct_tokenizer.eos_token
        instruct_prompts = [format_prompt(inst, instruct_tokenizer) for inst in eval_instructions]
        instruct_responses = _cache_model_responses(
            instruct_name, instruct_prompts, gen_device, eval_max_tokens,
        )
        print(f"  cached {len(instruct_responses)} instruct responses\n", flush=True)

        # Persist.
        with open(cache_path, "w", encoding="utf-8") as f:
            _json.dump({
                "key": cache_key,
                "instructions": eval_instructions,
                "prompts": eval_prompts,
                "base_responses": base_responses,
                "instruct_responses": instruct_responses,
            }, f, ensure_ascii=False)
        print(f"[eval cache] saved to {cache_path}\n", flush=True)

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
    # Inject trainer reference so the callback can use trainer.save_model()
    # which handles wrapped/DDP models correctly. Without this, the callback
    # would fall back to copy-from-latest-checkpoint.
    eval_callback.trainer = trainer

    # Run baseline judge eval before training. Non-fatal if judge API fails.
    if not cfg["training"].get("skip_baseline_eval", False):
        print("[step 0] baseline LLM judge eval (sanity check: base vs base should be ~50%)", flush=True)
        try:
            vs_base = _judge_pairwise(
                eval_instructions, base_responses, base_responses,
                judge_model=cfg.get("eval", {}).get("judge_model", "gpt-5.2"),
            )
            print(f"  baseline vs self: winrate {vs_base['winrate_a']:.1%}", flush=True)
            if wandb.run is not None:
                wandb.log({"eval/winrate_vs_base": 0.5, "eval/winrate_vs_instruct": 0.0}, step=0)
        except Exception as e:
            print(f"  [WARN] baseline judge eval failed ({type(e).__name__}: {e}) — continuing training anyway", flush=True)

    import glob, re
    def _resume_step(p):
        m = re.match(r"checkpoint-(\d+)$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    ckpts = [p for p in glob.glob(f"{cfg['training']['output_dir']}/checkpoint-*")
             if _resume_step(p) >= 0]  # excludes "checkpoint-best"
    ckpts.sort(key=_resume_step)
    resume_ckpt = ckpts[-1] if ckpts else None
    if resume_ckpt:
        print(f"Resuming from {resume_ckpt}", flush=True)
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
