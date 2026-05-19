"""Grouped bar chart for the quality verifier-capability sweep.

Generator: Qwen/Qwen2.5-0.5B. Six verifier conditions across four families
plus a random-weighted control. Two bars per condition: finetuned win rate
vs the base comparator and vs the instruct comparator. Judge: gpt-5.2.
"""
import matplotlib.pyplot as plt
import numpy as np

# label : display name : family : WR vs base (%) : WR vs instruct (%)
SWEEP = [
    ("q1", "random\n(~tiny)",                   "Random", 55.8, 22.8),
    ("q2", "gemma-3-270m-it\n(270M)",           "Gemma",  59.2, 27.0),
    ("q5", "OLMo-2-1B-DPO\n(1B)",               "OLMo",   65.0, 32.2),
    ("q4", "Llama-3.2-1B-Instruct\n(1B)",       "Llama",  70.0, 32.0),
    ("q3", "Qwen2.5-1.5B-Instruct\n(1.5B)",     "Qwen",   64.2, 25.8),
    ("q6", "gemma-4-E4B-it\n(~4B)",             "Gemma",  70.0, 34.0),
]

FAMILY_COLOR = {
    "Random": "#7f7f7f",
    "Gemma":  "#ff7f0e",
    "OLMo":   "#9467bd",
    "Llama":  "#d62728",
    "Qwen":   "#1f77b4",
}

labels = [r[1] for r in SWEEP]
wr_base = [r[3] for r in SWEEP]
wr_instr = [r[4] for r in SWEEP]
colors = [FAMILY_COLOR[r[2]] for r in SWEEP]

x = np.arange(len(SWEEP))
bw = 0.36

fig, ax = plt.subplots(figsize=(10.5, 5.5))

# vs base: solid color
bars_base = ax.bar(x - bw / 2, wr_base, bw,
                   color=colors, edgecolor="black", lw=0.6,
                   label="vs base")
# vs instruct: same color, hatched + lighter alpha
bars_instr = ax.bar(x + bw / 2, wr_instr, bw,
                    color=colors, edgecolor="black", lw=0.6, alpha=0.55,
                    hatch="//", label="vs instruct")

# 50% tie line
ax.axhline(50, color="grey", lw=1.6, ls="--", alpha=0.7, zorder=1)
ax.text(x[-1] + 0.4, 51.5, "tie (50%)",
        fontsize=12, color="grey", ha="right")


# Annotate bar values
for bars, vals in [(bars_base, wr_base), (bars_instr, wr_instr)]:
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.0,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10, rotation=25, ha="right")
ax.set_ylabel("Finetuned-generator win rate (%)", fontsize=14)
ax.tick_params(axis="y", labelsize=11)
ax.set_ylim(0, 82)
ax.grid(True, axis="y", alpha=0.3)

# Custom legend explaining bar styles (family colors are implicit in x-axis labels)
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="grey", edgecolor="black", label="vs base"),
    Patch(facecolor="grey", edgecolor="black", alpha=0.55, hatch="//", label="vs instruct"),
]
ax.legend(handles=legend_handles, loc="upper left", fontsize=12, framealpha=0.95)

plt.tight_layout()
plt.savefig("sweep_quality_winrates.png", dpi=200, bbox_inches="tight")
plt.savefig("sweep_quality_winrates.pdf", bbox_inches="tight")
print("wrote sweep_quality_winrates.png / .pdf")
