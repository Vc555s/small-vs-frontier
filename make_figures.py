#!/usr/bin/env python3
"""Figures for the T1 study, computed from the prediction files in results/.

  python make_figures.py      writes results/figures/{label_efficiency,stress_tests,out_of_scope}.png

PNG, 200 dpi, white background. Colour is fixed per system across all figures (validated
colour-blind-safe pair): fine-tuned small model = blue, Claude = orange. Small-model bars and
points show the mean over seeds with min-max whiskers; Claude was run once.
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from stress_common import NONE_LABEL, read_jsonl  # noqa: E402

TAG = "Qwen3-4B-Instruct-2507"
SEEDS = [0, 1, 2]
BUDGETS = [0, 100, 500, 2000, 9993]
SMALL, CLAUDE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
OUT = "results/figures"
SMALL_NAME = "Qwen3-4B + LoRA (fine-tuned)"
CLAUDE_NAME = "Claude via Claude Code CLI"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 11,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def style(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def acc(rows, key=lambda r: r["pred"] == r["gold"]):
    return sum(key(r) for r in rows) / len(rows)


def summary(vals):
    return sum(vals) / len(vals), min(vals), max(vals)


def bar_with_range(ax, x, vals, color, width, label=None):
    m, lo, hi = summary(vals)
    ax.bar(x, m, width, color=color, label=label, zorder=2)
    if len(vals) > 1:
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=1.2, capsize=4, zorder=3)
    ax.text(x, (hi if len(vals) > 1 else m) + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=10, color=INK)
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    idx = json.load(open("results/sample500.json"))["indices"]
    claude = {r["idx"]: r for r in read_jsonl("results/preds_claude_code_sample500.jsonl")}
    c_k = sum(claude[i]["pred"] == claude[i]["gold"] for i in idx)
    c_acc, (c_lo, c_hi) = c_k / len(idx), wilson(c_k, len(idx))

    def small_sample_acc(n, s):
        p = f"results/preds_{TAG}_n{n}_s{s}.jsonl"
        if not os.path.exists(p):
            return None
        rows = read_jsonl(p)
        return sum(rows[i]["pred"] == rows[i]["gold"] for i in idx) / len(idx)

    # ---------------------------------------------------------------- 1. label efficiency
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = list(range(len(BUDGETS)))
    ax.axhspan(c_lo, c_hi, color=CLAUDE, alpha=0.13, lw=0, zorder=1)
    ax.axhline(c_acc, color=CLAUDE, lw=2, zorder=2, label=f"{CLAUDE_NAME} (95% interval shaded)")
    pts = []
    for x, n in zip(xs, BUDGETS):
        vals = [v for v in (small_sample_acc(n, s) for s in SEEDS) if v is not None]
        if vals:
            pts.append((x, n, vals))
    multi = [(x, summary(v)) for x, n, v in pts if len(v) > 1 or n == 0]
    ax.plot([x for x, _ in multi], [s[0] for _, s in multi], color=SMALL, lw=2, zorder=3)
    for x, n, vals in pts:
        m, lo, hi = summary(vals)
        single = len(vals) == 1 and n != 0
        if len(vals) > 1:
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=SMALL, elinewidth=1.5, capsize=5, zorder=3)
        ax.plot(x, m, "o", ms=9, zorder=4, color=SMALL if not single else "white",
                markeredgecolor=SMALL, markeredgewidth=2)
        lab = f"{m:.3f}" + ("\n(1 seed)" if single else "")
        ax.annotate(lab, (x, m), textcoords="offset points", xytext=(0, -26 if single else 12),
                    ha="center", fontsize=10, color=INK)
    if pts and pts[-1][1] == 9993 and len(pts[-1][2]) == 1:
        ax.plot([pts[-2][0], pts[-1][0]], [summary(pts[-2][2])[0], pts[-1][2][0]], color=SMALL, lw=2,
                ls=(0, (3, 2)), zorder=3)
    ax.plot([], [], "o-", color=SMALL, lw=2, ms=8, label=f"{SMALL_NAME}: mean, min–max over 3 seeds")
    ax.plot([], [], "o", color="white", markeredgecolor=SMALL, markeredgewidth=2, ms=8, label="single seed")
    ax.set_xticks(xs, [f"{n:,}" for n in BUDGETS])
    ax.set_xlabel("Labelled training examples (Banking77)")
    style(ax, "Accuracy on the 500-message sample")
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/label_efficiency.png", dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------- 2. stress tests
    cats = ["Clean", "Typos", "Two issues\n(either label)"]
    small_vals = [
        [small_sample_acc(2000, s) for s in SEEDS],
        [acc(read_jsonl(f"results/stress/preds_small_A_s{s}.jsonl")) for s in SEEDS],
        [acc(read_jsonl(f"results/stress/preds_small_B_s{s}.jsonl"), lambda r: r["pred"] in (r["gold_a"], r["gold_b"])) for s in SEEDS],
    ]
    claude_vals = [
        [c_acc],
        [acc(read_jsonl("results/stress/preds_claude_A.jsonl"))],
        [acc(read_jsonl("results/stress/preds_claude_B.jsonl"), lambda r: r["pred"] in (r["gold_a"], r["gold_b"]))],
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    w = 0.3
    for i in range(3):
        bar_with_range(ax, i - w / 2 - 0.01, small_vals[i], SMALL, w, SMALL_NAME + " (2,000 labels, 3 seeds)" if i == 0 else None)
        bar_with_range(ax, i + w / 2 + 0.01, claude_vals[i], CLAUDE, w, CLAUDE_NAME if i == 0 else None)
    ax.set_xticks(range(3), cats)
    style(ax, "Accuracy")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=1)
    fig.tight_layout()
    fig.savefig(f"{OUT}/stress_tests.png", dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------- 3. out of scope
    def oos_stats(rows):
        oos = [r for r in rows if r["gold"] == NONE_LABEL]
        return sum(r["pred"] == NONE_LABEL for r in oos) / len(oos), acc(rows)

    variants = [("C1\nClaude,\nextended prompt", [read_jsonl("results/stress/preds_claude_C1.jsonl")], CLAUDE),
                ("S1\nextended\nprompt", [read_jsonl(f"results/stress/preds_small_S1_s{s}.jsonl") for s in SEEDS], SMALL),
                ("S2\nconfidence\nthreshold", [read_jsonl(f"results/stress/preds_small_S2_s{s}.jsonl") for s in SEEDS], SMALL),
                ("S3\nretrained with\nout-of-scope", [read_jsonl(f"results/stress/preds_small_S3_s{s}.jsonl") for s in SEEDS], SMALL)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), sharey=True)
    for k, (ax, title) in enumerate(zip(axes, ["Out-of-scope recall", "Overall accuracy (700 messages)"])):
        for x, (name, runs, color) in enumerate(variants):
            bar_with_range(ax, x, [oos_stats(r)[k] for r in runs], color, 0.5)
        ax.set_xticks(range(len(variants)), [v[0] for v in variants], fontsize=10)
        ax.set_title(title, color=INK, loc="left")
        style(ax, "Share" if k == 0 else "")
    axes[0].set_ylim(0, 1.08)
    fig.legend(handles=[Patch(color=SMALL, label=SMALL_NAME + " (3 seeds)"), Patch(color=CLAUDE, label=CLAUDE_NAME)],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{OUT}/out_of_scope.png", dpi=200)
    plt.close(fig)
    print("wrote", ", ".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
