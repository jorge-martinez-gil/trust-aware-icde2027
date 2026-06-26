"""Publication-quality figures for the federated trust-aware retrieval study.

Generates vector PDF (for the paper) and PNG (for preview) from the saved
result JSONs. Run after the experiment::

    python -m trust_aware.federated.plots
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_REPO, ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

RES = os.path.join(_REPO, "results", "federated")
FIGDIR = os.path.join(_REPO, "figures", "federated")
PAPER_FIGDIR = os.path.join(_REPO, "paper", "figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": False,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "axes.linewidth": 0.8,
    "lines.solid_capstyle": "round",
    "figure.dpi": 150,
    "savefig.transparent": False,
})

# Consistent method styling.
STYLE = {
    "Oracle":      dict(color="#333333", marker="*", ls=":",  lw=1.4, ms=7),
    "Trust-Aware": dict(color="#0072B2", marker="o", ls="-",  lw=2.0, ms=4.5),
    "CORI+Mean":   dict(color="#009E73", marker="D", ls="-.", lw=1.5, ms=3.8),
    "CORI+UCB":    dict(color="#CC79A7", marker="v", ls="-.", lw=1.5, ms=4),
    "CORI":        dict(color="#D55E00", marker="s", ls="--", lw=1.6, ms=4),
    "ReDDE":       dict(color="#E69F00", marker="^", ls="--", lw=1.6, ms=4),
    "Random":      dict(color="#888888", marker="x", ls="-.", lw=1.2, ms=4),
}
ORDER = ["Oracle", "Trust-Aware", "CORI", "ReDDE", "Random"]
ONLINE_ORDER = ["CORI", "CORI+Mean", "CORI+UCB", "Trust-Aware"]


def _save(fig, name):
    for folder in [FIGDIR, PAPER_FIGDIR]:
        os.makedirs(folder, exist_ok=True)
        fig.savefig(os.path.join(folder, name + ".pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(folder, name + ".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


def _polish(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y")


def _load_sweep_multiseed():
    seeds = [7, 11, 13]
    data = {}
    for seed in seeds:
        with open(os.path.join(RES, f"sweep_agg_{seed}.json"), encoding="utf-8") as fh:
            data[seed] = json.load(fh)
    counts = sorted((int(c) for c in data[7].keys()))
    return data, counts


def _multiseed_has_methods(methods: List[str]) -> bool:
    try:
        with open(os.path.join(RES, "multiseed.json"), encoding="utf-8") as fh:
            multiseed = json.load(fh)
    except FileNotFoundError:
        return False
    seed = next(iter(multiseed["by_seed"]))
    n = next(iter(multiseed["by_seed"][seed]))
    available = multiseed["by_seed"][seed][n]["aggregate"]
    return all(method in available for method in methods)


def fig_robustness(metric_key: str, ylabel: str, name: str):
    data, counts = _load_sweep_multiseed()
    fig, ax = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)
    for m in ORDER:
        means, stds = [], []
        for c in counts:
            vals = [data[s][str(c)][m][metric_key] for s in data]
            means.append(np.mean(vals)); stds.append(np.std(vals))
        means, stds = np.array(means), np.array(stds)
        st = STYLE[m]
        ax.plot(counts, means, label=m, **st)
        ax.fill_between(
            counts, means - stds, means + stds,
            color=st["color"], alpha=0.12, linewidth=0,
        )
    ax.set_xlabel("Untrusted duplicate sources")
    ax.set_ylabel(ylabel)
    ax.set_xticks(counts)
    ax.set_ylim(0, 1.02)
    ax.legend(ncol=2, frameon=False, loc="lower left", handlelength=2.4)
    _polish(ax)
    _save(fig, name)


def fig_learning_curve(name="fig_learning_curve"):
    """Rolling routing accuracy over the query stream at n=6 (online calibration)."""
    with open(os.path.join(RES, "multiseed.json"), encoding="utf-8") as fh:
        multiseed = json.load(fh)
    fig, ax = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)
    W = 60
    for method in ["Trust-Aware", "CORI", "ReDDE"]:
        curves = []
        for seed in multiseed["by_seed"]:
            records = multiseed["by_seed"][seed]["6"]["records"][method]
            hits = np.array([r["home_rank"] == 1 for r in records], dtype=float)
            curves.append(np.convolve(hits, np.ones(W) / W, mode="valid"))
        curves = np.asarray(curves)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        x = np.arange(W, W + len(mean))
        style = STYLE[method]
        ax.plot(x, mean, color=style["color"], ls=style["ls"],
                lw=style["lw"], label=method)
        ax.fill_between(x, mean - std, mean + std, color=style["color"],
                        alpha=0.12, linewidth=0)
    ax.set_xlabel("Queries processed")
    ax.set_ylabel(f"Rolling R@1 (window={W})")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower right")
    _polish(ax)
    _save(fig, name)


def fig_trust_bars(name="fig_learned_trust"):
    """Learned trust lower bound per source after the n=6 run (real vs twin)."""
    with open(os.path.join(RES, "sweep_seed7.json"), encoding="utf-8") as fh:
        study = json.load(fh)
    tr = study["raw_for_stats"]["6"]["trust_report"]
    names = list(tr.keys())
    lbs = [tr[n]["lower_bound"] for n in names]
    is_twin = [n.startswith("TWIN") for n in names]
    order = sorted(range(len(names)), key=lambda i: (is_twin[i], -lbs[i]))
    names = [names[i] for i in order]; lbs = [lbs[i] for i in order]
    is_twin = [is_twin[i] for i in order]
    colors = [STYLE["CORI"]["color"] if twin else STYLE["Trust-Aware"]["color"]
              for twin in is_twin]
    fig, ax = plt.subplots(figsize=(3.55, 2.65), constrained_layout=True)
    ax.bar(range(len(names)), lbs, color=colors, width=0.78,
           edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(names)))
    short = [n.replace("TWIN-", "tw:").replace("-1", "").replace("-2", "")
             for n in names]
    ax.set_xticklabels(short, rotation=52, ha="right", fontsize=6.8)
    ax.set_ylabel("Trust lower bound")
    prior_mean = 8.0 / 10.0
    prior_sd = np.sqrt((8.0 * 2.0) / ((10.0 ** 2) * 11.0))
    prior_lower_bound = prior_mean - prior_sd
    ax.axhline(prior_lower_bound, color="#555555", lw=1.0, ls=":")
    ax.set_ylim(0, 0.84)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Patch(color=STYLE["Trust-Aware"]["color"], label="genuine"),
        Patch(color=STYLE["CORI"]["color"], label="untrusted twin"),
        Line2D([0], [0], color="#555555", ls=":", lw=1.0,
               label="prior lower bound"),
    ], frameon=False, loc="upper right")
    _polish(ax)
    _save(fig, name)


def fig_clean_vs_untrusted(name="fig_clean_vs_untrusted"):
    with open(os.path.join(RES, "multiseed.json"), encoding="utf-8") as fh:
        ms = json.load(fh)
    methods = ["Trust-Aware", "CORI", "ReDDE", "Random"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)
    for ax, (mk, lab) in zip(axes, [("R@1", "Routing accuracy R@1"),
                                    ("nDCG@10(b=1)", "nDCG@10 (budget=1)")]):
        x = np.arange(len(methods)); w = 0.38
        for j, (n, hatch, alpha) in enumerate([("0", "", 1.0), ("6", "//", 0.65)]):
            means = [np.mean([ms["by_seed"][s][n]["aggregate"][m][mk]
                              for s in ms["by_seed"]]) for m in methods]
            stds = [np.std([ms["by_seed"][s][n]["aggregate"][m][mk]
                            for s in ms["by_seed"]]) for m in methods]
            cols = [STYLE[m]["color"] for m in methods]
            ax.bar(x + (j - 0.5) * w, means, w, yerr=stds, capsize=3,
                   color=cols, alpha=alpha, hatch=hatch,
                   label=("clean (0 twins)" if j == 0 else "untrusted (6 twins)"))
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_title(lab); ax.set_ylim(0, 1.0)
        _polish(ax)
    axes[0].legend(frameon=False, loc="upper right", fontsize=7)
    _save(fig, name)


def fig_online_baselines(name="fig_online_baselines"):
    """Online reputation baselines at the six-twin operating point."""
    if not _multiseed_has_methods(ONLINE_ORDER):
        return
    with open(os.path.join(RES, "multiseed.json"), encoding="utf-8") as fh:
        ms = json.load(fh)
    metrics = [("R@1", "Routing accuracy R@1"), ("nDCG@10(b=1)", "nDCG@10")]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), constrained_layout=True)
    x = np.arange(len(ONLINE_ORDER))
    for ax, (metric, title) in zip(axes, metrics):
        means = []
        stds = []
        for method in ONLINE_ORDER:
            vals = [
                ms["by_seed"][seed]["6"]["aggregate"][method][metric]
                for seed in ms["by_seed"]
            ]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        colors = [STYLE[m]["color"] for m in ONLINE_ORDER]
        ax.bar(x, means, yerr=stds, capsize=3, color=colors,
               edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(ONLINE_ORDER, rotation=24, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        _polish(ax)
    _save(fig, name)


def fig_adversary_variants(name="fig_adversary_variants"):
    """Sensitivity to exact, corrupted, and partial untrusted mirrors."""
    path = os.path.join(RES, "variant_sensitivity.json")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    variants = data["variants"]
    methods = ["CORI", "Trust-Aware"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), constrained_layout=True)
    x = np.arange(len(variants))
    width = 0.34
    for ax, metric, title in [
        (axes[0], "R@1", "Routing accuracy R@1"),
        (axes[1], "nDCG@10(b=1)", "nDCG@10"),
    ]:
        for j, method in enumerate(methods):
            means, stds = [], []
            for variant in variants:
                vals = [
                    data["by_seed"][seed][variant]["aggregate"][method][metric]
                    for seed in data["by_seed"]
                ]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            ax.bar(x + (j - 0.5) * width, means, width, yerr=stds, capsize=3,
                   color=STYLE[method]["color"], label=method,
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=18, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        _polish(ax)
    axes[0].legend(frameon=False, loc="upper right")
    _save(fig, name)


def fig_testbed(name="fig_testbed"):
    with open(os.path.join(RES, "sweep_seed7.json"), encoding="utf-8") as fh:
        study = json.load(fh)
    rows = study["testbed_summary"]
    names = [r["collection"] for r in rows]
    docs = [r["docs"] for r in rows]; q = [r["queries"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(names))
    ax1.bar(x - 0.2, docs, 0.4, color="#1f77b4", label="documents")
    ax1.set_ylabel("documents", color="#1f77b4")
    ax1.set_yscale("log")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, q, 0.4, color="#ff7f0e", label="queries")
    ax2.set_ylabel("queries (with qrels)", color="#ff7f0e")
    ax1.set_xticks(x); ax1.set_xticklabels(
        [f"{r['collection']}\n({r['domain'].split()[0]})" for r in rows], fontsize=8)
    ax1.set_title("Federated testbed: five heterogeneous real collections")
    _save(fig, name)


def generate_all():
    fig_robustness("R@1", "Routing accuracy (R@1)", "fig_robustness_routing")
    fig_robustness(
        "nDCG@10(b=1)", "nDCG@10 (budget=1)", "fig_robustness_ndcg"
    )
    fig_learning_curve()
    fig_trust_bars()
    fig_clean_vs_untrusted()
    fig_online_baselines()
    fig_adversary_variants()
    fig_testbed()
    print("Figures written to", FIGDIR, "and", PAPER_FIGDIR)


if __name__ == "__main__":
    generate_all()
