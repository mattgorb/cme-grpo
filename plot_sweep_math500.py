"""Plot MATH-500 pass@1 eval curves for all unique verifiers in the sweep."""
import matplotlib.pyplot as plt
import wandb

EVAL_KEY = "eval/math500_pass@1"
EVAL_STEPS = [0, 200, 400, 600]  # training steps, evenly spaced

# Verifier labels + capability numbers + cross-family flag.
# Ordered by approximate verifier MATH-500 pass@1 (strong → weak within Qwen, then cross-family).
RUNS = [
    ("v1", "92vgwhnc", "Qwen2.5-Math-7B-Instruct", 83, False),
    ("v2", "o97rcyuq", "Qwen2.5-Math-1.5B-Instruct", 75, False),
    ("v3", "vlf94bqt", "Qwen2.5-Math-1.5B", 65, False),
    ("v4", "6u6geg2b", "Qwen2.5-0.5B-Instruct", 35, False),
    ("v5", "007sy3kl", "Qwen2.5-0.5B", 20, False),
    ("v6", "mwgt14ic", "Llama-3.1-8B-Instruct", None, True),
    ("v7", "h8gsar2y", "gemma-2-2b-it", 25, True),
    ("v8", "9hbdvdu1", "OLMo-2-7B-Instruct", None, True),
]

api = wandb.Api()

def pull(run_id):
    r = api.run(f"matthewgorbett/cme-grpo/{run_id}")
    rows = list(r.scan_history(keys=["_step", EVAL_KEY]))
    rows.sort(key=lambda x: x["_step"])
    return [row[EVAL_KEY] for row in rows]

fig, ax = plt.subplots(figsize=(8, 5.5))
qwen_cmap = plt.cm.Blues
cross_cmap = plt.cm.Oranges

qwen_runs = [r for r in RUNS if not r[4]]
cross_runs = [r for r in RUNS if r[4]]

for i, (label, rid, name, cap, cross) in enumerate(qwen_runs):
    ys = pull(rid)
    color = qwen_cmap(0.95 - 0.15 * i)
    cap_str = f" ({cap}%)" if cap is not None else ""
    ax.plot(EVAL_STEPS, ys, "-o", color=color, lw=2, ms=6,
            label=f"{label}: {name}{cap_str}")

for i, (label, rid, name, cap, cross) in enumerate(cross_runs):
    ys = pull(rid)
    color = cross_cmap(0.85 - 0.2 * i)
    cap_str = f" ({cap}%)" if cap is not None else ""
    ax.plot(EVAL_STEPS, ys, "--s", color=color, lw=2, ms=6,
            label=f"{label}: {name}{cap_str}")

ax.set_xlabel("Training step")
ax.set_ylabel("MATH-500 pass@1")
ax.set_title("Generator MATH-500 across verifier sweep\n"
             "Qwen2.5-Math-1.5B base | 8 verifiers spanning ~4× capability + 4 families")
ax.set_xticks(EVAL_STEPS)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.40, 0.70)
ax.axhline(0.44, color="grey", lw=0.8, ls=":", alpha=0.7)
ax.text(5, 0.445, "base model (step 0)", fontsize=8, color="grey")

ax.legend(loc="lower right", fontsize=8, framealpha=0.95,
          title="verifier (own MATH-500 pass@1)", title_fontsize=8)

plt.tight_layout()
plt.savefig("sweep_math500_curves.png", dpi=200, bbox_inches="tight")
plt.savefig("sweep_math500_curves.pdf", bbox_inches="tight")
print("wrote sweep_math500_curves.png / .pdf")
