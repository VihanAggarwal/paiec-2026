"""Figures for the report, from eval/results/*.json and *_pairs.csv (no numbers typed by hand).

    python report/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "eval" / "results"
OUT = ROOT / "report"
BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
# validated categorical slots 1-4 (dataviz reference palette, light mode)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})


def load(stem):
    return json.loads((RES / f"{stem}.json").read_text())


def fig_budget():
    series = [("v4 (GP + item kernels)", "v4_pair_k7_s0", SLOTS[0], "o"),
              ("v3 (+ pair offset)", "v3_pair_k7_s0", SLOTS[1], "s"),
              ("v2 (benchmark offset)", "v2_pair_k7_s0", SLOTS[2], "^"),
              ("v0 (empirical mean)", "v0_pair_k7_s0", SLOTS[3], "D")]
    x = np.log2(np.array(BUDGETS) + 1)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.axhline(0.25, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(x[-1], 0.2515, "constant 0.5", color=MUTED, ha="right", va="bottom", fontsize=7)
    for label, stem, color, marker in series:
        r = load(stem)
        y = [r[f"B{b}"] for b in BUDGETS]
        ax.plot(x, y, color=color, lw=2, marker=marker, ms=4, label=f"{label}: ALC {r['ALC']:.4f}", zorder=3)
    ax.set_xticks(x, [str(b) for b in BUDGETS])
    ax.set_xlabel("label budget per pair (log2(1+n) spacing)")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_ylim(0.17, 0.37)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.66))
    fig.tight_layout()
    fig.savefig(OUT / "fig_budget.pdf")
    plt.close(fig)


def fig_noise():
    d = pd.read_csv(RES / "v4_pair_k7_s0_pairs.csv")
    d["ALC"] = sum(w * d[f"B{b}"] for w, b in zip(WEIGHTS, BUDGETS))
    s = d.groupby(["fold", "snap"]).ALC.mean()
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.hist(s, bins=24, color=SLOTS[0], edgecolor="white", linewidth=1.0)
    ax.axvline(s.mean(), color=INK, lw=1)
    ax.text(s.mean() - 0.003, ax.get_ylim()[1] * 0.92, f"mean {s.mean():.3f}", color=INK, fontsize=7, ha="right")
    ax.set_xlabel("ALC of one simulated formative evaluation (snapshot)")
    ax.set_ylabel("snapshots")
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig_noise.pdf")
    plt.close(fig)
    rng = np.random.default_rng(0)
    a = s.to_numpy()
    best = {n: float(np.mean([a[rng.integers(0, len(a), n)].min() for _ in range(5000)])) for n in (1, 5, 10, 20)}
    stats = {"snapshots": int(len(a)), "pairs_per_snapshot": float(d.groupby(["fold", "snap"]).size().mean()),
             "mean": float(a.mean()), "sd": float(a.std(ddof=1)), "q05": float(np.quantile(a, 0.05)),
             "q95": float(np.quantile(a, 0.95)), "expected_best_of": best}
    (OUT / "noise_stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == "__main__":
    fig_budget()
    fig_noise()
