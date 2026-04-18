"""Toy CME-GRPO training — runs locally on MacBook.

Task: steer distilgpt2 toward producing Python code using a code-finetuned
verifier (tiny_starcoder_py) as the CME reward signal.
"""

from __future__ import annotations

import os
import sys

# Add parent dir so we can import shared modules (reward, cme_trainer).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import yaml
import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from reward import CMERewardModel, build_cme_reward_fn
from cme_trainer import CMETokenLevelGRPOTrainer
from eval_models import eval_model, judge_winrate


# Simple code-related prompts — enough to test the mechanics.
TOY_PROMPTS = [
    "# Python function to add two numbers\ndef add(",
    "# Python function to check if a number is even\ndef is_even(",
    "# Python function to reverse a string\ndef reverse(",
    "# Python function to find the maximum of a list\ndef find_max(",
    "# Python function to compute factorial\ndef factorial(",
    "# Python function to count words in a string\ndef count_words(",
    "# Python function to check if a string is a palindrome\ndef is_palindrome(",
    "# Python function to sort a list of integers\ndef sort_list(",
    "# Python function to compute the sum of a list\ndef sum_list(",
    "# Python function to merge two dictionaries\ndef merge_dicts(",
    "# Python function to flatten a nested list\ndef flatten(",
    "# Python function to remove duplicates from a list\ndef remove_duplicates(",
    "# Python function to compute fibonacci numbers\ndef fibonacci(",
    "# Python function to convert celsius to fahrenheit\ndef to_fahrenheit(",
    "# Python function to find common elements in two lists\ndef intersection(",
    "# Python function to capitalize all words in a string\ndef capitalize_words(",
    "# Python function to check if a list is sorted\ndef is_sorted(",
    "# Python function to compute the GCD of two numbers\ndef gcd(",
    "# Python function to generate a random password\ndef gen_password(",
    "# Python function to read lines from a file\ndef read_lines(",
    "# Python function to count vowels in a string\ndef count_vowels(",
    "# Python function to zip two lists together\ndef zip_lists(",
    "# Python function to compute the power of a number\ndef power(",
    "# Python function to transpose a matrix\ndef transpose(",
    "# Python function to find the median of a list\ndef median(",
    "# Python function to convert a list to a dictionary\ndef list_to_dict(",
    "# Python function to check if two strings are anagrams\ndef is_anagram(",
    "# Python function to binary search a sorted list\ndef binary_search(",
    "# Python function to compute the mean of a list\ndef mean(",
    "# Python function to rotate a list by k positions\ndef rotate(",
    "# Python function to compute the absolute value\ndef absolute(",
    "# Python function to check if a number is prime\ndef is_prime(",
    "# Python function to convert a string to an integer\ndef to_int(",
    "# Python function to split a string by delimiter\ndef split_str(",
    "# Python function to join a list of strings\ndef join_strings(",
    "# Python function to find the index of an element\ndef find_index(",
    "# Python function to swap two variables\ndef swap(",
    "# Python function to clamp a value between min and max\ndef clamp(",
    "# Python function to compute the dot product of two vectors\ndef dot_product(",
    "# Python function to compute the length of a string without len\ndef str_length(",
    "# Python function to filter even numbers from a list\ndef filter_evens(",
    "# Python function to map a function over a list\ndef map_list(",
    "# Python function to reduce a list with a function\ndef reduce_list(",
    "# Python function to create a range of numbers\ndef make_range(",
    "# Python function to check if a key exists in a dictionary\ndef has_key(",
    "# Python function to get the unique elements of a list\ndef unique(",
    "# Python function to compute the variance of a list\ndef variance(",
    "# Python function to compute the standard deviation\ndef std_dev(",
    "# Python function to deep copy a nested structure\ndef deep_copy(",
    "# Python function to chunk a list into groups of n\ndef chunk(",
    "# Python function to interleave two lists\ndef interleave(",
    "# Python function to compute the nth triangle number\ndef triangle_number(",
    "# Python function to check if a year is a leap year\ndef is_leap_year(",
    "# Python function to convert snake_case to camelCase\ndef to_camel(",
    "# Python function to convert camelCase to snake_case\ndef to_snake(",
    "# Python function to compute the hamming distance\ndef hamming_distance(",
    "# Python function to encode a string to base64\ndef encode_b64(",
    "# Python function to decode a base64 string\ndef decode_b64(",
    "# Python function to compute the Levenshtein distance\ndef levenshtein(",
    "# Python function to generate permutations of a list\ndef permutations(",
    "# Python function to generate combinations of k items\ndef combinations(",
    "# Python function to check if a graph has a cycle\ndef has_cycle(",
    "# Python function to perform a depth-first search\ndef dfs(",
    "# Python function to perform a breadth-first search\ndef bfs(",
    "# Python function to invert a dictionary\ndef invert_dict(",
    "# Python function to find the second largest element\ndef second_largest(",
    "# Python function to compute the running average\ndef running_avg(",
    "# Python function to pad a string to a given length\ndef pad_string(",
    "# Python function to truncate a string with ellipsis\ndef truncate(",
    "# Python function to compute the product of a list\ndef product(",
    "# Python function to check if all elements are truthy\ndef all_truthy(",
    "# Python function to check if any element is truthy\ndef any_truthy(",
    "# Python function to zip three lists together\ndef zip3(",
    "# Python function to unzip a list of tuples\ndef unzip(",
    "# Python function to group elements by a key function\ndef group_by(",
    "# Python function to compute the cumulative sum\ndef cumsum(",
    "# Python function to find the mode of a list\ndef mode(",
    "# Python function to implement a simple stack\ndef make_stack(",
    "# Python function to implement a simple queue\ndef make_queue(",
    "# Python function to compute the LCM of two numbers\ndef lcm(",
    "# Python function to convert decimal to binary string\ndef to_binary(",
    "# Python function to convert binary string to decimal\ndef from_binary(",
    "# Python function to find the longest common prefix\ndef longest_prefix(",
    "# Python function to count occurrences of each character\ndef char_count(",
    "# Python function to remove all whitespace from a string\ndef strip_all(",
    "# Python function to validate an email address\ndef is_valid_email(",
    "# Python function to generate a list of squares\ndef squares(",
    "# Python function to matrix multiply two 2D lists\ndef matmul(",
    "# Python function to compute the trace of a matrix\ndef trace(",
    "# Python function to create an identity matrix\ndef identity(",
    "# Python function to memoize another function\ndef memoize(",
    "# Python function to retry a function n times\ndef retry(",
    "# Python function to time the execution of a function\ndef timeit(",
    "# Python function to compose two functions\ndef compose(",
    "# Python function to partially apply arguments\ndef partial_apply(",
    "# Python function to debounce a function call\ndef debounce(",
]


def diversity_reward_fn(prompts, completions, **kwargs):
    """Penalize repetitive completions. Catches both token-level and character-level repetition."""
    rewards = []
    for c in completions:
        text = c if isinstance(c, str) else "\n".join(m.get("content", "") for m in c)
        if len(text) == 0:
            rewards.append(-5.0)
            continue

        # Token-level: unique ratio
        tokens = text.split()
        token_ratio = len(set(tokens)) / max(len(tokens), 1)

        # Character-level: catches digit-string exploit (e.g., "111111...")
        chars = list(text)
        char_ratio = len(set(chars)) / max(len(chars), 1)

        # Worst of both — scale to match CME reward range (-1 to -4)
        worst_ratio = min(token_ratio, char_ratio)
        rewards.append(-5.0 * (1.0 - worst_ratio))
    return rewards
    return rewards


class KLTrackingCallback(TrainerCallback):
    """Periodically measure KL(policy || base) on a few prompts and log to wandb."""

    def __init__(self, base_model, tokenizer, prompts, device, every_steps=25):
        self.base_model = base_model
        self.base_model.eval()
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.device = device
        self.every_steps = every_steps

    @torch.no_grad()
    def _estimate_kl(self, policy_model):
        kls = []
        for prompt in self.prompts:
            enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            policy_logits = policy_model(**enc).logits[:, :-1, :]
            base_logits = self.base_model(**enc).logits[:, :-1, :]
            p = torch.softmax(policy_logits, dim=-1)
            log_p = torch.log_softmax(policy_logits, dim=-1)
            log_q = torch.log_softmax(base_logits, dim=-1)
            kl = (p * (log_p - log_q)).sum(dim=-1).mean()
            kls.append(kl.item())
        return sum(kls) / len(kls)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_steps != 0:
            return
        model = kwargs.get("model")
        if model is None:
            return
        was_training = model.training
        model.eval()
        kl = self._estimate_kl(model)
        print(f"\n[step {state.global_step}] KL(policy || base) = {kl:.4f}")
        if wandb.run is not None:
            wandb.log({"kl/policy_vs_base": kl, "train/global_step": state.global_step})
        if was_training:
            model.train()


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def build_train_dataset() -> Dataset:
    """Load MBPP train split as training prompts."""
    from datasets import load_dataset as _load_ds
    ds = _load_ds("google-research-datasets/mbpp", "sanitized", split="train")
    # Keep only the prompt column — TRL expects a "prompt" column.
    ds = ds.map(lambda ex: {"prompt": ex["prompt"] + "\n"}, remove_columns=[
        c for c in ds.column_names if c != "prompt"
    ])
    return ds


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = pick_device()
    print(f"Using device: {device}")

    # W&B setup.
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ.setdefault("WANDB_PROJECT", cfg["wandb"]["project"])
        wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["wandb"]["run_name"],
            config=cfg,
        )

    # Pre-training eval: verify model capabilities on MBPP.
    print("\n" + "=" * 60)
    print("PRE-TRAINING EVAL — MBPP (verifier should beat generator)")
    print("=" * 60)
    from datasets import load_dataset as _load_ds
    mbpp_ds = _load_ds("google-research-datasets/mbpp", "sanitized", split="test")
    mbpp_ds = mbpp_ds.select(range(min(50, len(mbpp_ds))))

    gen_name = cfg["model"]["generator"]
    ver_name = cfg["model"]["verifier"]
    gen_summary = eval_model(gen_name, mbpp_ds, device)
    ver_summary = eval_model(ver_name, mbpp_ds, device)

    print(f"\n{'Model':<40} {'Syntax':>8} {'Tests':>8} {'Ref CE':>8} {'Ref PPL':>9} {'Entropy':>9}")
    print(f"{'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")
    print(f"{gen_name:<40} {gen_summary['syntax_rate']:>7.1%} {gen_summary['test_pass_rate']:>7.1%} {gen_summary['mean_ref_ce']:>8.3f} {gen_summary['mean_ref_ppl']:>9.1f} {gen_summary['mean_ref_entropy']:>9.2f}")
    print(f"{ver_name:<40} {ver_summary['syntax_rate']:>7.1%} {ver_summary['test_pass_rate']:>7.1%} {ver_summary['mean_ref_ce']:>8.3f} {ver_summary['mean_ref_ppl']:>9.1f} {ver_summary['mean_ref_entropy']:>9.2f}")

    ce_gap = gen_summary["mean_ref_ce"] - ver_summary["mean_ref_ce"]
    if ce_gap < 0.1:
        print(f"\n⚠ Warning: verifier CE is not clearly lower than generator (gap={ce_gap:.3f}).")
        print("  CME reward signal may be too weak. Consider a stronger verifier.")
    else:
        print(f"\n✓ Verifier has {ce_gap:.3f} lower CE — CME signal should work.")

    if wandb.run is not None:
        wandb.log({
            "pre_eval/generator_ce": gen_summary["mean_ref_ce"],
            "pre_eval/generator_ppl": gen_summary["mean_ref_ppl"],
            "pre_eval/verifier_ce": ver_summary["mean_ref_ce"],
            "pre_eval/verifier_ppl": ver_summary["mean_ref_ppl"],
            "pre_eval/ce_gap": ce_gap,
        })

    # Free eval memory before training.
    if device == "mps":
        torch.mps.empty_cache()
    import gc; gc.collect()

    # Generator.
    tokenizer = AutoTokenizer.from_pretrained(gen_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(gen_name, torch_dtype=torch.float32)

    # Verifier (CME reward).
    reward_model = CMERewardModel(
        verifier_name=cfg["model"]["verifier"],
        device=device,
        max_length=cfg["reward"]["max_verifier_length"],
        dtype=torch.float32,
    )
    token_level = cfg.get("reward", {}).get("token_level", False)
    answer_only = cfg.get("reward", {}).get("answer_only_cme", False)
    reward_fn = build_cme_reward_fn(
        reward_model,
        token_level=token_level,
        gen_tokenizer=tokenizer if token_level else None,
        answer_only=answer_only,
    )

    # Quick sanity check: verify the reward is directional.
    print("\n--- Reward sanity check ---")
    test_prompt = "# Python function to add two numbers\ndef add("
    good_response = "a, b):\n    return a + b"
    bad_response = "the quick brown fox jumps over the lazy dog"
    scores = reward_model.score([test_prompt, test_prompt], [good_response, bad_response])
    print(f"  Code completion reward: {scores[0]:.4f}")
    print(f"  Nonsense completion reward: {scores[1]:.4f}")
    if scores[0] > scores[1]:
        print("  ✓ Verifier prefers code — reward signal is directional!")
    else:
        print("  ✗ Unexpected — verifier doesn't prefer code. Check models.")
    print()

    # Dataset — MBPP train split.
    train_ds = build_train_dataset()
    print(f"Training dataset: {len(train_ds)} examples (MBPP train)")

    # GRPO config.
    tcfg = cfg["training"]
    gcfg = cfg["generation"]

    per_device_bs = tcfg["per_device_train_batch_size"]
    grad_accum = tcfg["gradient_accumulation_steps"]
    num_gen = gcfg["num_generations"]
    gen_batch = per_device_bs * grad_accum
    if gen_batch % num_gen != 0:
        grad_accum = ((gen_batch + num_gen - 1) // num_gen) * num_gen // per_device_bs
        print(f"Bumped gradient_accumulation_steps to {grad_accum}")

    use_bf16 = tcfg["bf16"] and device != "cpu"

    grpo_cfg = GRPOConfig(
        output_dir=tcfg["output_dir"],
        learning_rate=tcfg["learning_rate"],
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=tcfg["num_train_epochs"],
        max_steps=tcfg["max_steps"],
        warmup_steps=tcfg["warmup_steps"],
        logging_steps=tcfg["logging_steps"],
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg.get("save_total_limit", 2),
        bf16=use_bf16,
        seed=tcfg["seed"],
        num_generations=gcfg["num_generations"],
        temperature=gcfg["temperature"],
        max_completion_length=gcfg["max_new_tokens"],
        repetition_penalty=gcfg.get("repetition_penalty", 1.0),
        beta=0.0,  # skip TRL ref model; we track KL separately in callback
        max_grad_norm=tcfg.get("max_grad_norm", 1.0),
        gradient_checkpointing=False,  # not needed for tiny models
        report_to=["wandb"] if not args.no_wandb else [],
        log_completions=True,  # log prompt + generations to wandb + terminal
        num_completions_to_print=4,  # print 4 per generation step
    )

    TrainerCls = CMETokenLevelGRPOTrainer if token_level else GRPOTrainer
    trainer = TrainerCls(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_fn, diversity_reward_fn],
        args=grpo_cfg,
        train_dataset=train_ds,
    )

    print("\nStarting training...")
    trainer.train()
    trainer.save_model(tcfg["output_dir"])

    # Post-training eval on MBPP.
    print("\n" + "=" * 60)
    print("POST-TRAINING EVAL — MBPP (finetuned generator)")
    print("=" * 60)
    model.eval()
    ft_summary = eval_model(tcfg["output_dir"], mbpp_ds, device)

    print(f"\n{'Model':<40} {'Syntax':>8} {'Tests':>8}")
    print(f"{'-'*40} {'-'*8} {'-'*8}")
    print(f"{'generator (base)':<40} {gen_summary['syntax_rate']:>7.1%} {gen_summary['test_pass_rate']:>7.1%}")
    print(f"{'generator (finetuned)':<40} {ft_summary['syntax_rate']:>7.1%} {ft_summary['test_pass_rate']:>7.1%}")
    print(f"{'verifier':<40} {ver_summary['syntax_rate']:>7.1%} {ver_summary['test_pass_rate']:>7.1%}")

    delta = ft_summary["test_pass_rate"] - gen_summary["test_pass_rate"]
    if delta > 0:
        print(f"\n✓ Finetuned generator improved by {delta:.1%} on MBPP tests!")
    elif delta == 0:
        print(f"\n— No change in test pass rate.")
    else:
        print(f"\n✗ Finetuned generator regressed by {abs(delta):.1%} on MBPP tests.")

    if wandb.run is not None:
        wandb.log({
            "post_eval/finetuned_syntax": ft_summary["syntax_rate"],
            "post_eval/finetuned_test_pass": ft_summary["test_pass_rate"],
            "post_eval/delta_test_pass": delta,
        })

    # LLM judge: base vs finetuned.
    print("\n" + "=" * 60)
    print("LLM JUDGE — base vs finetuned (GPT-5.2)")
    print("=" * 60)
    judge_result = judge_winrate(
        gen_name, tcfg["output_dir"], mbpp_ds, device, num_samples=50,
    )

    if wandb.run is not None:
        wandb.log({
            "judge/base_wins": judge_result["wins_a"],
            "judge/finetuned_wins": judge_result["wins_b"],
            "judge/ties": judge_result["ties"],
            "judge/finetuned_winrate": judge_result["winrate_b"],
        })
        wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()
