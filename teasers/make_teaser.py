"""Teaser figures comparing verifier CE vs generator self-CE on boxed-answer spans.

Shows that `ce_ver_answer` separates correct from incorrect generations much
better than `ce_gen_answer` — supporting the CME-as-reward claim.

Inputs: glob-matched CSVs from analyze_cme_correctness.py
(rows with: benchmark, generator, correct, ce_gen_answer, ce_ver_answer, ...)

Outputs:
  teaser_bar_auroc.png   — AUROC(wrong > correct) for gen vs ver CE, per (model, dataset)
  teaser_bar_gap.png     — Mean CE gap (wrong - correct) for gen vs ver CE
  teaser_scatter.png     — ce_gen_answer (x) vs ce_ver_answer (y) colored by correctness

Usage (run from project root):
    python teasers/make_teaser.py
    python teasers/make_teaser.py --glob "cme_qwen*.csv"
    python teasers/make_teaser.py --output-dir teasers/figures/custom
"""

from __future__ import annotations

import argparse
import glob
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def auroc(scores: list[float], labels: list[int]) -> float | None:
    """AUROC where higher score should correspond to label=1."""
    paired = [(s, l) for s, l in zip(scores, labels) if not math.isnan(s)]
    pos = [s for s, l in paired if l == 1]
    neg = [s for s, l in paired if l == 0]
    if not pos or not neg:
        return None
    wins = sum(1 for p in pos for n in neg if p > n) + 0.5 * sum(
        1 for p in pos for n in neg if p == n
    )
    return wins / (len(pos) * len(neg))


def short_name(name: str) -> str:
    """Shorten HF model name for labels."""
    return name.split("/")[-1]


def load_data(pattern: str) -> tuple[pd.DataFrame, str]:
    """Load matching CSVs. Returns (df, mode) where mode is 'perrow' or 'summary'.
    Prefers per-row CSVs (richer data). Falls back to summary CSVs if only those exist."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files match {pattern!r}.\n"
            f"Expected cme_*.csv (per-row) or cme_*.summary.csv (summary).\n"
            f"Run `bash run_analyze_cme.sh` first, or pass --glob with a correct path."
        )
    perrow_files = [f for f in files if "summary" not in f]
    summary_files = [f for f in files if "summary" in f]

    if perrow_files:
        dfs = []
        for f in perrow_files:
            df = pd.read_csv(f)
            if "generator" not in df.columns:
                stem = os.path.basename(f).replace("cme_", "").replace(".csv", "")
                parts = stem.rsplit("_", 1)
                df["generator"] = parts[0] if len(parts) == 2 else stem
                df["benchmark"] = parts[1] if len(parts) == 2 else "?"
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        df = df.dropna(subset=["ce_gen_answer", "ce_ver_answer"])
        return df, "perrow"

    # Fall back to summary files.
    dfs = [pd.read_csv(f) for f in summary_files]
    df = pd.concat(dfs, ignore_index=True)
    return df, "summary"


def compute_stats_from_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format summary CSV into one row per (generator, benchmark)."""
    df = df[df["metric"].isin(["ce_gen_answer", "ce_ver_answer"])]
    rows = []
    for (gen, bench), grp in df.groupby(["generator", "benchmark"]):
        n = int(grp["n_total"].iloc[0])
        acc = float(grp["accuracy"].iloc[0])
        row = {
            "generator": short_name(gen),
            "benchmark": bench,
            "n": n,
            "n_correct": int(round(n * acc)),
            "n_wrong": int(round(n * (1 - acc))),
        }
        for _, mrow in grp.iterrows():
            m = mrow["metric"]
            row[f"{m}_auroc"] = mrow["auroc_wrong_gt_correct"]
            row[f"{m}_gap"] = mrow["gap_wrong_minus_correct"]
            row[f"{m}_cor_mean"] = mrow["correct_mean"]
            row[f"{m}_wro_mean"] = mrow["incorrect_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_per_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """For each (generator, benchmark), compute AUROC and gap for gen and ver."""
    rows = []
    for (gen, bench), grp in df.groupby(["generator", "benchmark"]):
        # Filter to rows where the boxed span was found for BOTH metrics.
        grp = grp.dropna(subset=["ce_gen_answer", "ce_ver_answer"])
        if len(grp) < 5:
            continue
        labels_wrong = [1 - int(c) for c in grp["correct"]]
        stats = {
            "generator": short_name(gen),
            "benchmark": bench,
            "n": len(grp),
            "n_correct": int(grp["correct"].sum()),
            "n_wrong": int((1 - grp["correct"]).sum()),
        }
        for metric in ("ce_gen_answer", "ce_ver_answer"):
            vals = grp[metric].tolist()
            au = auroc(vals, labels_wrong)
            cor = grp.loc[grp["correct"] == 1, metric].dropna()
            wro = grp.loc[grp["correct"] == 0, metric].dropna()
            stats[f"{metric}_auroc"] = au if au is not None else np.nan
            stats[f"{metric}_gap"] = (
                wro.mean() - cor.mean()
                if len(cor) and len(wro)
                else np.nan
            )
            stats[f"{metric}_cor_mean"] = cor.mean() if len(cor) else np.nan
            stats[f"{metric}_wro_mean"] = wro.mean() if len(wro) else np.nan
        rows.append(stats)
    return pd.DataFrame(rows)


def plot_bar_auroc(stats: pd.DataFrame, output_path: str) -> None:
    """Grouped bar chart: AUROC(wrong>correct) for gen vs ver per (model, benchmark)."""
    stats = stats.copy()
    stats["label"] = stats["generator"] + "\n" + stats["benchmark"]
    stats = stats.sort_values(["generator", "benchmark"]).reset_index(drop=True)

    x = np.arange(len(stats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(stats) * 0.9), 5))
    b1 = ax.bar(
        x - w / 2, stats["ce_gen_answer_auroc"], w,
        label="generator self-CE on answer span",
        color="#a0a0a0", edgecolor="black",
    )
    b2 = ax.bar(
        x + w / 2, stats["ce_ver_answer_auroc"], w,
        label="verifier CE on answer span (CME)",
        color="#4c78a8", edgecolor="black",
    )
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6, label="chance (AUROC=0.5)")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC (wrong > correct)")
    ax.set_title("Verifier CE separates correct from incorrect better than generator self-CE")
    ax.set_xticks(x)
    ax.set_xticklabels(stats["label"], rotation=30, ha="right", fontsize=9)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for b in list(b1) + list(b2):
        h = b.get_height()
        if not np.isnan(h):
            ax.text(
                b.get_x() + b.get_width() / 2, h + 0.005,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"wrote {output_path}")


def plot_bar_gap(stats: pd.DataFrame, output_path: str) -> None:
    """Grouped bar chart: mean CE gap (wrong − correct) for gen vs ver."""
    stats = stats.copy()
    stats["label"] = stats["generator"] + "\n" + stats["benchmark"]
    stats = stats.sort_values(["generator", "benchmark"]).reset_index(drop=True)

    x = np.arange(len(stats))
    w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(stats) * 0.9), 5))
    ax.bar(
        x - w / 2, stats["ce_gen_answer_gap"], w,
        label="generator self-CE gap",
        color="#a0a0a0", edgecolor="black",
    )
    ax.bar(
        x + w / 2, stats["ce_ver_answer_gap"], w,
        label="verifier CE gap (CME)",
        color="#e45756", edgecolor="black",
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Mean CE gap (wrong − correct)")
    ax.set_title("Verifier CE opens a wider margin between wrong and correct answers")
    ax.set_xticks(x)
    ax.set_xticklabels(stats["label"], rotation=30, ha="right", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"wrote {output_path}")


def filter_valid_pairs(df: pd.DataFrame, min_per_bucket: int = 3) -> pd.DataFrame:
    """Drop (generator, benchmark) pairs with too few correct or wrong examples."""
    valid = []
    for (gen, bench), grp in df.groupby(["generator", "benchmark"]):
        n_cor = int((grp["correct"] == 1).sum())
        n_wro = int((grp["correct"] == 0).sum())
        if n_cor >= min_per_bucket and n_wro >= min_per_bucket:
            valid.append((gen, bench))
    if not valid:
        return df.iloc[0:0]
    keep = df.set_index(["generator", "benchmark"]).index.isin(valid)
    return df[keep].reset_index(drop=True)


def plot_qualitative_example(ax) -> None:
    """Panel (a): flow diagram — same answer piped through generator AND verifier,
    showing their respective "surprise" (CE). Verifier's surprise clearly separates
    correct from wrong; generator's doesn't.

    PLACEHOLDER DATA — edit the 5 values below after measuring on real models:
        Generator: meta-llama/Llama-3.2-1B (base)
        Verifier:  Qwen/Qwen2.5-Math-7B-Instruct
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── PLACEHOLDER CONTENT ──
    problem_text = "Problem: Area of right triangle with legs 3, 4"
    expr_a = r"$A = \frac{1}{2}\cdot 3\cdot 4 = 6$"
    expr_b = r"$A = 3\cdot 4 = 12$"
    ce_gen_a, ce_ver_a = 2.1, 0.4
    ce_gen_b, ce_ver_b = 2.3, 5.2
    # ─────────────────────────

    c_ok = "#2ca02c"
    c_bad = "#d62728"
    c_gen = "#7a7a7a"
    c_ver = "#3a6a9c"

    # Problem header.
    ax.text(0.5, 0.96, problem_text, ha="center", va="top",
            fontsize=11, fontweight="bold")

    def _draw_row(y, mark, mark_color, expr, ce_g, ce_v):
        # Response text on the left: marker hangs further left of the expression.
        ax.text(-0.08, y, mark, fontsize=13, fontweight="bold",
                color=mark_color, va="center", clip_on=False)
        ax.text(0.05, y, expr, fontsize=10, va="center")

        # Fork: single trunk splits into two branches (gen above, ver below).
        fork_x = 0.56
        y_u = y + 0.055
        y_d = y - 0.055

        ax.annotate("", xy=(fork_x, y), xytext=(0.48, y),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-", lw=1.0, color="gray"))
        ax.plot([fork_x, fork_x], [y_u, y_d],
                transform=ax.transAxes, color="gray", linewidth=1.0,
                clip_on=False)

        # Gen branch (top).
        ax.annotate("", xy=(0.74, y_u), xytext=(fork_x, y_u),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=1.0, color=c_gen))
        ax.text(0.65, y_u + 0.025, "generator", ha="center", va="bottom",
                fontsize=9, color=c_gen, fontweight="bold")
        ax.text(0.76, y_u, f"CE = {ce_g:.1f}", ha="left", va="center",
                fontsize=11, fontweight="bold", color=c_gen)

        # Ver branch (bottom).
        ax.annotate("", xy=(0.74, y_d), xytext=(fork_x, y_d),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=1.0, color=c_ver))
        ax.text(0.65, y_d - 0.04, "verifier", ha="center", va="top",
                fontsize=9, color=c_ver, fontweight="bold")
        ax.text(0.76, y_d, f"CE = {ce_v:.1f}", ha="left", va="center",
                fontsize=11, fontweight="bold", color=c_ver)

    _draw_row(0.67, "A)", c_ok, expr_a, ce_gen_a, ce_ver_a)
    _draw_row(0.30, "B)", c_bad, expr_b, ce_gen_b, ce_ver_b)

    # Bottom summary of the two gaps.
    gap_gen = abs(ce_gen_b - ce_gen_a)
    gap_ver = abs(ce_ver_b - ce_ver_a)
    ax.plot([0.05, 0.95], [0.10, 0.10], transform=ax.transAxes,
            color="gray", linewidth=0.4, clip_on=False)
    ax.text(
        0.5, 0.04,
        f"$\\Delta$CE$_{{\\,\\rm gen}}$ = {gap_gen:.1f}  (ambiguous)"
        " | "
        f"$\\Delta$CE$_{{\\,\\rm ver}}$ = {gap_ver:.1f}  (separates)",
        ha="center", fontsize=10, style="italic",
    )


def compute_all_metric_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute AUROC for all 4 metric families (CE-full, CE-ans, H-full, H-ans),
    gen vs ver, for each (generator, benchmark) pair. Returns long format:
    one row per (gen, bench, metric_family)."""
    pairs = [
        ("ce_full", "ce_gen_full", "ce_ver_full"),
        ("ce_answer", "ce_gen_answer", "ce_ver_answer"),
        ("entropy_full", "entropy_gen_full", "entropy_ver_full"),
        ("entropy_answer", "entropy_gen_answer", "entropy_ver_answer"),
    ]
    rows = []
    for (gen, bench), grp in df.groupby(["generator", "benchmark"]):
        n_cor = int((grp["correct"] == 1).sum())
        n_wro = int((grp["correct"] == 0).sum())
        if n_cor < 3 or n_wro < 3:
            continue
        for family, gen_col, ver_col in pairs:
            if gen_col not in grp.columns or ver_col not in grp.columns:
                continue
            sub = grp.dropna(subset=[gen_col, ver_col])
            if len(sub) < 5:
                continue
            lw = [1 - int(c) for c in sub["correct"]]
            au_gen = auroc(sub[gen_col].tolist(), lw)
            au_ver = auroc(sub[ver_col].tolist(), lw)
            rows.append({
                "generator": short_name(gen),
                "benchmark": bench,
                "metric": family,
                "auroc_gen": au_gen if au_gen is not None else np.nan,
                "auroc_ver": au_ver if au_ver is not None else np.nan,
                "n": len(sub),
            })
    return pd.DataFrame(rows)


def paper_figure(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    output_path: str,
    bar_datasets: tuple[str, str] = ("math500", "amc23"),
) -> None:
    """Three-panel paper-quality PDF.

    (a) Bar chart of CE-on-answer-span AUROC for dataset 1 (e.g. math500).
    (b) Same bar chart for dataset 2 (e.g. amc23).
    (c) Win-loss scatter: generator-AUROC vs verifier-AUROC across all
        (model × benchmark × metric-family) combinations.
    """
    stats = stats[(stats["n_correct"] >= 3) & (stats["n_wrong"] >= 3)].reset_index(drop=True)
    if stats.empty:
        print(f"no valid (model, benchmark) pairs for {output_path}")
        return

    all_metric_stats = compute_all_metric_stats(df)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    c_gen = "#a0a0a0"
    c_ver = "#4c78a8"
    c_chance = "#d62728"

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(13, 4.0))
    gs = GridSpec(2, 3, figure=fig,
                  width_ratios=[1, 1, 1], hspace=0.5, wspace=0.32)
    ax_a = fig.add_subplot(gs[:, 0])       # qualitative example, spans rows
    ax_b = fig.add_subplot(gs[:, 1])       # AUROC bar chart, spans rows
    ax_c_top = fig.add_subplot(gs[0, 2])   # gen density
    ax_c_bot = fig.add_subplot(gs[1, 2])   # ver density

    # ───────── Panel (a): qualitative example ─────────
    plot_qualitative_example(ax_a)

    # ───────── Panel (b): AUROC bar chart for one dataset ─────────
    bar_exclusions = {
        "math500": {"gemma-2-2b"},
    }
    bench = bar_datasets[1] if len(bar_datasets) >= 2 else bar_datasets[0]
    sub_b = stats[stats["benchmark"] == bench]
    excluded = bar_exclusions.get(bench, set())
    if excluded:
        sub_b = sub_b[~sub_b["generator"].isin(excluded)]
    sub_b = sub_b.sort_values("generator").reset_index(drop=True)
    if sub_b.empty:
        ax_b.text(0.5, 0.5, f"(no data for {bench})", transform=ax_b.transAxes,
                  ha="center", va="center", fontsize=10, color="gray")
        ax_b.set_xticks([])
        ax_b.set_yticks([])
        ax_b.set_title(bench, fontsize=10)
    else:
        x = np.arange(len(sub_b))
        w = 0.38
        ax_b.bar(x - w / 2, sub_b["ce_gen_answer_auroc"], w,
                 color=c_gen, edgecolor="black", linewidth=0.6,
                 label="generator self-CE")
        ax_b.bar(x + w / 2, sub_b["ce_ver_answer_auroc"], w,
                 color=c_ver, edgecolor="black", linewidth=0.6,
                 label="verifier CE (CME)")
        ax_b.axhline(0.5, color=c_chance, linestyle="--", linewidth=0.8, alpha=0.7)
        ax_b.set_ylim(0.4, 1.0)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(sub_b["generator"], rotation=30, ha="right", fontsize=8)
        ax_b.set_ylabel("AUROC (wrong $>$ correct)")
        ax_b.set_title(bench, fontsize=11)
        ax_b.grid(axis="y", alpha=0.25, linewidth=0.5)
        for xi, v in zip(x - w / 2, sub_b["ce_gen_answer_auroc"]):
            if not np.isnan(v):
                ax_b.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        for xi, v in zip(x + w / 2, sub_b["ce_ver_answer_auroc"]):
            if not np.isnan(v):
                ax_b.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax_b.legend(loc="lower right", fontsize=8, frameon=True)

    # ───────── Panel (c): density plot ─────────
    # Use a single representative (generator, benchmark) pair so we don't blend
    # CE scales across models. Default to qwen-math-1.5b on math500 if present,
    # else fall back to the first pair with enough data.
    c_correct = "#2ca02c"
    c_wrong = "#d62728"

    preferred = [("qwen-math-1.5b", "math500"), ("qwen-1.5b", "math500")]
    chosen = None
    for gen_key, bench_key in preferred:
        sub = df[(df["generator"] == gen_key) & (df["benchmark"] == bench_key)]
        if (sub["correct"] == 1).sum() >= 5 and (sub["correct"] == 0).sum() >= 5:
            chosen = (gen_key, bench_key, sub)
            break
    if chosen is None:
        for (gen_key, bench_key), sub in df.groupby(["generator", "benchmark"]):
            if (sub["correct"] == 1).sum() >= 5 and (sub["correct"] == 0).sum() >= 5:
                chosen = (gen_key, bench_key, sub)
                break
    if chosen is None:
        ax_c_top.text(0.5, 0.5, "(no data)", transform=ax_c_top.transAxes,
                      ha="center", va="center", color="gray")
        ax_c_bot.text(0.5, 0.5, "(no data)", transform=ax_c_bot.transAxes,
                      ha="center", va="center", color="gray")
    else:
        gen_key, bench_key, sub = chosen
        gen_cor = sub.loc[sub["correct"] == 1, "ce_gen_answer"].dropna().values
        gen_wro = sub.loc[sub["correct"] == 0, "ce_gen_answer"].dropna().values
        ver_cor = sub.loc[sub["correct"] == 1, "ce_ver_answer"].dropna().values
        ver_wro = sub.loc[sub["correct"] == 0, "ce_ver_answer"].dropna().values

        try:
            from scipy.stats import gaussian_kde
            has_kde = True
        except ImportError:
            has_kde = False

        def _plot_density(ax, cor, wro, title, title_color):
            all_vals = np.concatenate([cor, wro]) if len(cor) and len(wro) else np.array([0.0, 1.0])
            x_lo = max(0.0, float(np.quantile(all_vals, 0.02)))
            x_hi = float(np.quantile(all_vals, 0.95))
            if x_hi - x_lo < 0.2:
                x_hi = x_lo + 1.0
            x_grid = np.linspace(x_lo, x_hi, 250)

            if has_kde:
                if len(wro) >= 2:
                    y = gaussian_kde(wro)(x_grid)
                    ax.fill_between(x_grid, y, color=c_wrong, alpha=0.35)
                    ax.plot(x_grid, y, color=c_wrong, linewidth=1.8,
                            label="wrong")
                if len(cor) >= 2:
                    y = gaussian_kde(cor)(x_grid)
                    ax.fill_between(x_grid, y, color=c_correct, alpha=0.35)
                    ax.plot(x_grid, y, color=c_correct, linewidth=1.8,
                            label="correct")
            else:
                bins = np.linspace(x_lo, x_hi, 35)
                ax.hist(wro, bins=bins, density=True, alpha=0.5,
                        color=c_wrong, label="wrong",
                        edgecolor="white", linewidth=0.3)
                ax.hist(cor, bins=bins, density=True, alpha=0.5,
                        color=c_correct, label="correct",
                        edgecolor="white", linewidth=0.3)

            ax.set_xlim(x_lo, x_hi)
            ax.set_title(title, fontsize=10, color=title_color, fontweight="bold")
            ax.set_ylabel("density", fontsize=9)
            ax.grid(alpha=0.25, linewidth=0.5)

        _plot_density(ax_c_top, gen_cor, gen_wro, "Generator Self-CE", c_gen)
        _plot_density(ax_c_bot, ver_cor, ver_wro, "Verifier CE", c_ver)
        ax_c_bot.set_xlabel("CE on \\boxed{} span", fontsize=9)
        ax_c_top.legend(loc="upper right", frameon=False, fontsize=8)

    # Panel labels (a) (b) (c).
    for ax_i, lbl in zip([ax_a, ax_b, ax_c_top], "abc"):
        ax_i.text(-0.14, 1.05, f"({lbl})",
                  transform=ax_i.transAxes, fontweight="bold", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"wrote {output_path}")


def plot_scatter(df: pd.DataFrame, output_path: str) -> None:
    """Scatter: ce_gen_answer (x) vs ce_ver_answer (y), colored by correctness,
    faceted by (generator, benchmark)."""
    groups = list(df.groupby(["generator", "benchmark"]))
    groups = [(g, sub) for g, sub in groups if len(sub) >= 5]
    n = len(groups)
    if n == 0:
        print(f"no groups with enough data for scatter")
        return

    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False,
    )

    # Determine shared axis range for comparability.
    x_all = df["ce_gen_answer"].dropna()
    y_all = df["ce_ver_answer"].dropna()
    x_lo, x_hi = float(x_all.quantile(0.01)), float(x_all.quantile(0.99))
    y_lo, y_hi = float(y_all.quantile(0.01)), float(y_all.quantile(0.99))

    for idx, ((gen, bench), sub) in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        cor = sub[sub["correct"] == 1]
        wro = sub[sub["correct"] == 0]
        ax.scatter(
            wro["ce_gen_answer"], wro["ce_ver_answer"],
            c="#e45756", s=18, alpha=0.55, label=f"wrong (n={len(wro)})",
            edgecolors="none",
        )
        ax.scatter(
            cor["ce_gen_answer"], cor["ce_ver_answer"],
            c="#4c78a8", s=18, alpha=0.65, label=f"correct (n={len(cor)})",
            edgecolors="none",
        )
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("ce_gen_answer", fontsize=9)
        ax.set_ylabel("ce_ver_answer", fontsize=9)
        ax.set_title(f"{short_name(gen)} — {bench}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    # Hide unused panels.
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "Verifier CE (y-axis) separates wrong from correct; generator self-CE (x-axis) does not",
        fontsize=12,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"wrote {output_path}")


def main():
    ap = argparse.ArgumentParser()
    # CSVs live alongside this script in teasers/; figures go in teasers/figures/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_glob = os.path.join(script_dir, "cme_*.csv")
    default_output = os.path.join(script_dir, "figures")
    ap.add_argument("--glob", default=default_glob, help="Glob for per-row CSVs")
    ap.add_argument("--output-dir", default=default_output, help="Where to write PNGs")
    ap.add_argument("--bar-datasets", default="math500,amc23",
                    help="Comma-separated names of the two datasets for left-side bar charts")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df, mode = load_data(args.glob)
    print(f"loaded {len(df)} rows ({mode} mode)")

    if mode == "perrow":
        stats = compute_per_group_stats(df)
        perrow_df = df
    else:
        stats = compute_stats_from_summary(df)
        perrow_df = None

    if stats.empty:
        print("no groups had enough data. Exiting.")
        return

    # Print a quick textual summary.
    print("\nPer-group stats:")
    cols = [
        "generator", "benchmark", "n", "n_correct",
        "ce_gen_answer_auroc", "ce_ver_answer_auroc",
        "ce_gen_answer_gap", "ce_ver_answer_gap",
    ]
    print(stats[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    stats_path = os.path.join(args.output_dir, "teaser_stats.csv")
    stats.to_csv(stats_path, index=False)
    print(f"\nwrote {stats_path}")

    plot_bar_auroc(stats, os.path.join(args.output_dir, "teaser_bar_auroc.png"))
    plot_bar_gap(stats, os.path.join(args.output_dir, "teaser_bar_gap.png"))
    if perrow_df is not None:
        perrow_df_valid = filter_valid_pairs(perrow_df)
        plot_scatter(perrow_df_valid, os.path.join(args.output_dir, "teaser_scatter.png"))
        bar_ds = tuple(args.bar_datasets.split(","))
        paper_figure(
            perrow_df_valid, stats,
            os.path.join(args.output_dir, "teaser_paper.pdf"),
            bar_datasets=bar_ds[:2] if len(bar_ds) >= 2 else ("math500", "amc23"),
        )
    else:
        print("\nSkipping scatter plot and paper PDF: only summary CSVs found (need per-row data).")


if __name__ == "__main__":
    main()
