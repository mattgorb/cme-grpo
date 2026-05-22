"""Scatter: verifier MATH-500 pass@1 vs trained-generator MATH-500 pass@1.

Headline figure for the verifier-capability sweep section. The flatness of the
relationship is the claim: trained-generator outcome is decoupled from
verifier capability across a ~75-point range and four model families.

V8 (OLMo-2) is plotted at its measured value of 0% but marked as a parsing
caveat — its raw eval failed to extract \\boxed answers, not because the
model can't do math.
"""
from pathlib import Path
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent.parent / "out"

# label : verifier name : verifier MATH-500 pass@1 : trained gen MATH-500 pass@1 : family
SWEEP = [
    ("V1", "Qwen2.5-Math-7B-Instruct",   76, 0.60, "Qwen"),
    ("V2", "Qwen2.5-Math-1.5B-Instruct", 69, 0.60, "Qwen"),
    ("V3", "Qwen2.5-Math-1.5B",          45, 0.59, "Qwen"),
    ("V4", "Qwen2.5-0.5B-Instruct",      30, 0.61, "Qwen"),
    ("V5", "Qwen2.5-0.5B",               27, 0.60, "Qwen"),
    ("V6", "Llama-3.1-8B-Instruct",      42, 0.59, "Llama"),
    ("V7", "gemma-2-2b-it",              10, 0.59, "Gemma"),
    ("V8", "OLMo-2-1124-7B-Instruct",     0, 0.58, "OLMo"),  # parsing failure
]

GENERATOR_BASELINE = 0.44  # Qwen2.5-Math-1.5B step-0 pass@1
GENERATOR_VERIFIER_CAP = 45  # same model on its own benchmark (V3)
V8_IS_PARSING_FAILURE = True

FAMILY_COLOR = {
    "Qwen":  "#1f77b4",
    "Llama": "#d62728",
    "Gemma": "#ff7f0e",
    "OLMo":  "#9467bd",
}

fig, ax = plt.subplots(figsize=(9.0, 5.5))

# horizontal band marking the trained-generator convergence range
ymin = min(r[3] for r in SWEEP)
ymax = max(r[3] for r in SWEEP)
ax.axhspan(ymin, ymax, color="grey", alpha=0.10, zorder=0)
ax.axhline(GENERATOR_BASELINE, color="grey", lw=1.6, ls=":", zorder=1)
ax.text(78, GENERATOR_BASELINE + 0.006, "generator pre-training (0.44)",
        fontsize=13, color="grey", ha="right")
ax.axvline(GENERATOR_VERIFIER_CAP, color="grey", lw=1.6, ls="--", alpha=0.6, zorder=1)
ax.text(GENERATOR_VERIFIER_CAP + 1, 0.435,
        "verifier = generator capability",
        fontsize=13, color="grey", rotation=90, va="bottom", ha="left")

# OLS slope (excluding V8 parsing failure)
clean = [r for r in SWEEP if not (V8_IS_PARSING_FAILURE and r[0] == "V8")]
xs = [r[2] for r in clean]
ys = [r[3] for r in clean]
n = len(xs)
xbar = sum(xs) / n
ybar = sum(ys) / n
num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
den = sum((x - xbar) ** 2 for x in xs)
slope = num / den
intercept = ybar - slope * xbar
xline = [0, 80]
yline = [slope * x + intercept for x in xline]
ax.plot(xline, yline, color="black", lw=1, alpha=0.5, ls="-",
        label=f"OLS fit (excl. V8): slope = {slope:+.4f}")

# Per-point label offsets (x, y in points) and horizontal alignment, tuned
# to avoid collisions given the actual coordinate layout.
LABEL_POS = {
    "V1": (5, 9, "right"),       # 76, 0.60 — above dot, nudged right
    "V2": (0, -16, "center"),    # 69, 0.60 — centered under dot
    "V3": (10, 9, "left"),       # 45, 0.59 — above dot, starts right of dot
    "V4": (10, 6, "left"),       # 30, 0.61
    "V5": (-10, -16, "right"),   # 27, 0.60 — below dot, ends left of dot
    "V6": (10, -16, "left"),     # 42, 0.59 — below dot, starts right of dot
    "V7": (10, 9, "left"),       # 10, 0.59 — above dot
    "V8": (14, -2, "left"),      # 0, 0.58 — right of X marker
}

for label, name, vcap, gcap, fam in SWEEP:
    color = FAMILY_COLOR[fam]
    dx, dy, ha = LABEL_POS[label]
    if V8_IS_PARSING_FAILURE and label == "V8":
        ax.scatter(vcap, gcap, s=140, marker="X", facecolor="white",
                   edgecolor=color, linewidth=2, zorder=4)
        text = f"{name} (parse fail)"
    else:
        ax.scatter(vcap, gcap, s=110, color=color, edgecolor="white",
                   linewidth=1, zorder=5)
        text = name
    ax.annotate(text, (vcap, gcap), xytext=(dx, dy),
                textcoords="offset points", fontsize=10, color=color,
                fontweight="bold", ha=ha)

ax.set_xlabel("Verifier MATH-500 pass@1 (%)", fontsize=17)
ax.set_ylabel("Generator MATH-500 pass@1", fontsize=17)
ax.tick_params(axis="both", labelsize=14)
ax.set_xlim(-3, 82)
ax.set_ylim(0.42, 0.66)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "sweep_math500_scatter.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT_DIR / "sweep_math500_scatter.pdf", bbox_inches="tight")
print(f"wrote {OUT_DIR}/sweep_math500_scatter.png / .pdf  (slope = {slope:+.4f})")
