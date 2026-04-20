"""Quick smoke test for CME quality training.

Run BEFORE committing to a full training run. Checks:
  1. Generator and verifier load correctly in bf16
  2. CME reward has nonzero variance across responses
  3. Advantage estimates are reasonable
  4. GPU memory fits within budget

Usage:
    python sanity_check.py --config config_quality1.yaml
    python sanity_check.py --config config_quality2.yaml

Should complete in under 2 minutes on a single GPU.
"""

from __future__ import annotations

import argparse
import time

import torch
import yaml

from reward import CMERewardModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_quality1.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    gen_name = cfg["model"]["generator"]
    ver_name = cfg["model"]["verifier"]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    is_cuda = device.startswith("cuda")

    print(f"{'=' * 60}")
    print(f"CME Quality Training — Sanity Check")
    print(f"{'=' * 60}")
    print(f"  Generator: {gen_name}")
    print(f"  Verifier:  {ver_name}")
    print(f"  Device:    {device}")
    print()

    # ── Step 1: Load generator ──
    print("[1/5] Loading generator in bf16...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(gen_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_name, torch_dtype=torch.bfloat16, device_map=device,
    )
    gen_model.eval()
    gen_params = sum(p.numel() for p in gen_model.parameters()) / 1e6
    print(f"  OK — {gen_params:.0f}M params, loaded in {time.time()-t0:.1f}s")

    if is_cuda:
        mem_gen = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory after generator: {mem_gen:.2f} GB")

    # ── Step 2: Load verifier ──
    print("\n[2/5] Loading verifier in bf16...")
    t0 = time.time()
    reward_model = CMERewardModel(
        verifier_name=ver_name, device=device,
        max_length=cfg["reward"]["max_verifier_length"],
    )
    ver_params = sum(p.numel() for p in reward_model.model.parameters()) / 1e6
    print(f"  OK — {ver_params:.0f}M params, loaded in {time.time()-t0:.1f}s")

    if is_cuda:
        mem_both = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory (gen + ver): {mem_both:.2f} GB")

    # ── Step 3: Sample prompts from UltraFeedback ──
    print("\n[3/5] Loading 5 prompts from UltraFeedback...")
    from datasets import load_dataset
    ds = load_dataset(cfg["data"]["train_dataset"], split="train")
    ds = ds.shuffle(seed=42).select(range(5))
    instructions = [ex.get("instruction", ex.get("prompt", "")) for ex in ds]

    from train_quality import format_prompt
    prompts = [format_prompt(inst, tokenizer) for inst in instructions]
    for i, inst in enumerate(instructions):
        print(f"  [{i}] {inst[:80]}...")

    # ── Step 4: Generate G=4 responses per prompt ──
    G = 4
    print(f"\n[4/5] Generating {G} responses per prompt...")
    tokenizer.padding_side = "left"
    all_prompts = []
    all_responses = []

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = gen_model.generate(
                **enc, max_new_tokens=512, do_sample=True,
                temperature=1.0, top_p=1.0, num_return_sequences=G,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j in range(G):
            resp = tokenizer.decode(out[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
            all_prompts.append(prompt)
            all_responses.append(resp)

    print(f"  Generated {len(all_responses)} total responses")
    for i in range(min(3, len(all_responses))):
        print(f"  response[{i}] ({len(all_responses[i])} chars): {all_responses[i][:150]}...")

    # ── Step 5: Compute CME rewards ──
    print(f"\n[5/5] Computing CME rewards...")
    t0 = time.time()
    reward_metric = cfg.get("reward", {}).get("reward_metric", "entropy")
    rewards = reward_model.score(
        all_prompts, all_responses,
        token_level=False, answer_only=False,
        reward_metric=reward_metric,
    )
    score_time = time.time() - t0
    print(f"  Scored {len(rewards)} responses in {score_time:.1f}s")

    if is_cuda:
        mem_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory: {mem_peak:.2f} GB")

    # ── Analysis ──
    print(f"\n{'=' * 60}")
    print("REWARD ANALYSIS")
    print(f"{'=' * 60}")

    rewards_t = torch.tensor(rewards)
    print(f"  All rewards:  mean={rewards_t.mean():.4f}  std={rewards_t.std():.4f}  "
          f"min={rewards_t.min():.4f}  max={rewards_t.max():.4f}")

    # Per-prompt analysis.
    issues = []
    for i in range(5):
        group = rewards_t[i * G : (i + 1) * G]
        g_mean = group.mean().item()
        g_std = group.std().item()
        print(f"  Prompt {i}: rewards={[f'{r:.4f}' for r in group.tolist()]}  mean={g_mean:.4f}  std={g_std:.4f}")
        if g_std < 0.01:
            issues.append(f"Prompt {i} has near-zero variance (std={g_std:.4f})")

    # Advantage estimates (what GRPO would compute).
    print(f"\n  ADVANTAGE ESTIMATES (per-prompt normalized):")
    for i in range(5):
        group = rewards_t[i * G : (i + 1) * G]
        g_mean = group.mean()
        g_std = group.std()
        advantages = (group - g_mean) / (g_std + 1e-4)
        print(f"  Prompt {i}: advantages={[f'{a:.3f}' for a in advantages.tolist()]}")

    # Overall variance check.
    overall_std = rewards_t.std().item()

    # ── Verdict ──
    print(f"\n{'=' * 60}")
    print("DIAGNOSTICS")
    print(f"{'=' * 60}")

    ok = True

    # Check 1: Nonzero variance.
    if overall_std < 0.01:
        print(f"  [WARN] Overall reward std is very low ({overall_std:.4f})")
        print(f"         CME signal may be too weak for GRPO to learn.")
        print(f"         Consider using a larger/different verifier.")
        ok = False
    else:
        print(f"  [OK] Reward variance: std={overall_std:.4f}")

    # Check 2: Per-prompt variance.
    if issues:
        for issue in issues:
            print(f"  [WARN] {issue}")
        if len(issues) >= 4:
            print(f"         Most prompts have no within-group variance. GRPO advantages will be ~0.")
            ok = False
    else:
        print(f"  [OK] All prompts have meaningful within-group variance")

    # Check 3: Rewards not degenerate.
    if rewards_t.isnan().any() or rewards_t.isinf().any():
        print(f"  [FAIL] Found NaN/Inf in rewards!")
        ok = False
    else:
        print(f"  [OK] No NaN/Inf in rewards")

    # Check 4: Memory estimate.
    if is_cuda:
        total_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        # Rough estimate: training needs ~3x inference memory (optimizer states, gradients).
        est_train_mem = mem_both * 3
        print(f"  [INFO] GPU total: {total_mem:.1f} GB, inference: {mem_both:.1f} GB, est. training: {est_train_mem:.1f} GB")
        if est_train_mem > total_mem * 0.95:
            print(f"  [WARN] Training may OOM. Consider gradient checkpointing or smaller batch size.")
            ok = False
        else:
            print(f"  [OK] Memory estimate fits within GPU budget")
    else:
        print(f"  [INFO] Running on CPU — skip memory estimate")

    # Check 5: Response quality.
    empty_responses = sum(1 for r in all_responses if len(r.strip()) < 10)
    if empty_responses > len(all_responses) * 0.5:
        print(f"  [WARN] {empty_responses}/{len(all_responses)} responses are very short (<10 chars)")
        print(f"         Base model may need a different prompt format or higher temperature.")
        ok = False
    else:
        print(f"  [OK] {len(all_responses) - empty_responses}/{len(all_responses)} responses have meaningful length")

    print(f"\n{'=' * 60}")
    if ok:
        print("GO — sanity check passed. Proceed with training.")
    else:
        print("NO-GO — issues detected above. Review before launching full training.")
    print(f"{'=' * 60}")

    # Cleanup.
    del gen_model, reward_model
    if is_cuda:
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
