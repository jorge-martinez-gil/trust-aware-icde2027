"""Generate publication tables and reports from saved federated results."""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from .statistics import compare_methods_clustered, holm_bonferroni

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
RES = os.path.join(_REPO, "results", "federated")
TABDIR = os.path.join(RES, "tables")
PAPER_TABDIR = os.path.join(_REPO, "paper", "tables")

METHODS = ["Random", "ReDDE", "CORI", "Trust-Aware", "Oracle"]
ONLINE_METHODS = ["CORI", "CORI+Mean", "CORI+UCB", "Trust-Aware"]


def _load_json(filename: str) -> Dict:
    with open(os.path.join(RES, filename), encoding="utf-8") as fh:
        return json.load(fh)


def _latex_escape(text: str) -> str:
    return str(text).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")


def _seeds_agg(n: str, metric: str, methods=None):
    methods = methods or METHODS
    multiseed = _load_json("multiseed.json")
    return {
        method: [
            multiseed["by_seed"][seed][n]["aggregate"][method][metric]
            for seed in multiseed["by_seed"]
        ]
        for method in methods
    }


def _has_multiseed_methods(methods) -> bool:
    multiseed = _load_json("multiseed.json")
    seed = next(iter(multiseed["by_seed"]))
    n = next(iter(multiseed["by_seed"][seed]))
    available = multiseed["by_seed"][seed][n]["aggregate"]
    return all(method in available for method in methods)


def _clustered_records(n: str):
    multiseed = _load_json("multiseed.json")
    clustered = {m: [] for m in ["Trust-Aware", "CORI", "ReDDE"]}
    for seed in multiseed["by_seed"]:
        records = multiseed["by_seed"][seed][n]["records"]
        for method in clustered:
            for record in records[method]:
                replicated = dict(record)
                replicated["seed"] = int(seed)
                clustered[method].append(replicated)
    return clustered


def table_testbed() -> str:
    rows = _load_json("sweep_seed7.json")["testbed_summary"]
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Collection & Domain & Docs & Queries & Avg.\ len & Rel./q \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(row['collection'])} & {_latex_escape(row['domain'])} & "
            f"{row['docs']} & {row['queries']} & {row['avg_doc_len']:.0f} & "
            f"{row['avg_rel_per_query']:.1f} \\\\"
        )
    total_docs = sum(row["docs"] for row in rows)
    total_queries = sum(row["queries"] for row in rows)
    lines += [
        r"\midrule",
        f"\\textbf{{Total}} & 5 domains & {total_docs} & {total_queries} & -- & -- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines)


def table_main() -> str:
    metrics = [
        ("R@1", "R@1"),
        ("nDCG@10(b=1)", "nDCG@10"),
        ("MAP(b=1)", "MAP"),
        ("MRR", "MRR"),
    ]

    def block(n: str):
        aggregate = {metric: _seeds_agg(n, metric) for metric, _ in metrics}
        rows = []
        for method in METHODS:
            cells = []
            for metric, _ in metrics:
                values = aggregate[metric][method]
                cells.append(f"{np.mean(values):.3f}\\,$\\pm$\\,{np.std(values):.3f}")
            name = r"\textbf{Trust-Aware}" if method == "Trust-Aware" else method
            rows.append(f"{name} & " + " & ".join(cells) + r" \\")
        return rows

    lines = [
        r"\begin{tabular}{l cccc}",
        r"\toprule",
        r"Method & R@1 & nDCG@10 & MAP & MRR \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{(a) Clean federation (0 untrusted sources)}}\\",
        r"\midrule",
    ]
    lines += block("0")
    lines += [
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{(b) Untrusted federation (6 evil-twin sources)}}\\",
        r"\midrule",
    ]
    lines += block("6")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def significance_analysis() -> Dict:
    """Build the confirmatory family shared by JSON, Markdown, and LaTeX."""

    records = _clustered_records("6")
    family = {}
    for metric, label in [("ndcg@1", "nDCG@10"), ("map@1", "MAP")]:
        result = compare_methods_clustered(
            records,
            metric,
            "Trust-Aware",
            ["CORI", "ReDDE"],
            apply_holm=False,
        )
        for other in ["CORI", "ReDDE"]:
            family[f"{metric}:{other}"] = {
                "setting": "6 twins",
                "metric": metric,
                "metric_label": label,
                "reference": "Trust-Aware",
                "comparison": other,
                **result["comparisons"][other],
            }

    adjusted = holm_bonferroni(
        {key: row["p_raw"] for key, row in family.items()}
    )
    for key, row in family.items():
        row.update(adjusted[key])
    return {
        "schema_version": 1,
        "independent_unit": "query",
        "replicate_aggregation": "arithmetic mean across seeds within query",
        "n_seed_replicates": 3,
        "test": "two-sided paired Wilcoxon signed-rank",
        "effect_size": "matched-pairs rank-biserial correlation",
        "confidence_interval": (
            "10,000-resample query-cluster bootstrap of mean paired difference"
        ),
        "multiplicity_correction": (
            "Holm-Bonferroni across all four confirmatory comparisons"
        ),
        "comparisons": family,
    }


def table_significance(analysis: Dict | None = None) -> str:
    analysis = analysis or significance_analysis()
    lines = [
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"Metric & vs.\ & $N_q$ & $\Delta$ & 95\% CI & $p_{\text{Holm}}$ & $r_{\mathrm{rb}}$ (effect) \\",
        r"\midrule",
    ]
    for row in analysis["comparisons"].values():
        lines.append(
            f"{row['metric_label']} & {row['comparison']} & {row['n_clusters']} & "
            f"{row['mean_diff']:+.3f} & "
            f"[{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}] & "
            f"{row['p_holm']:.1e} & "
            f"{row['rank_biserial']:+.3f} ({row['effect']}) \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_online_baselines() -> str | None:
    if not _has_multiseed_methods(ONLINE_METHODS):
        return None
    metrics = [
        ("R@1", "R@1"),
        ("nDCG@10(b=1)", "nDCG@10"),
        ("MAP(b=1)", "MAP"),
        ("cost(b=1)", "cost"),
    ]
    lines = [
        r"\begin{tabular}{l cccc}",
        r"\toprule",
        r"Method & R@1 & nDCG@10 & MAP & cost \\",
        r"\midrule",
    ]
    aggregates = {
        metric: _seeds_agg("6", metric, methods=ONLINE_METHODS)
        for metric, _ in metrics
    }
    means = {
        metric: {
            method: float(np.mean(aggregates[metric][method]))
            for method in ONLINE_METHODS
        }
        for metric, _label in metrics
    }
    best = {}
    for metric, _label in metrics:
        if metric.startswith("cost"):
            best[metric] = min(means[metric].values())
        else:
            best[metric] = max(means[metric].values())
    for method in ONLINE_METHODS:
        cells = []
        for metric, _label in metrics:
            values = aggregates[metric][method]
            if metric.startswith("cost"):
                cell = f"{np.mean(values):.3f}"
            else:
                cell = f"{np.mean(values):.3f}\\,$\\pm$\\,{np.std(values):.3f}"
            if abs(means[metric][method] - best[metric]) < 1e-12:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        name = method
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_variant_sensitivity() -> str | None:
    try:
        data = _load_json("variant_sensitivity.json")
    except FileNotFoundError:
        return None
    methods = ["CORI", "Trust-Aware"]
    lines = [
        r"\begin{tabular}{l ccc ccc}",
        r"\toprule",
        r"Variant & CORI R@1 & Trust R@1 & $\Delta$ & CORI nDCG & Trust nDCG & $\Delta$ \\",
        r"\midrule",
    ]
    for variant in data["variants"]:
        rows = {method: {"R@1": [], "nDCG@10(b=1)": []} for method in methods}
        for seed in data["by_seed"]:
            aggregate = data["by_seed"][seed][variant]["aggregate"]
            for method in methods:
                rows[method]["R@1"].append(aggregate[method]["R@1"])
                rows[method]["nDCG@10(b=1)"].append(
                    aggregate[method]["nDCG@10(b=1)"]
                )
        cori_r = float(np.mean(rows["CORI"]["R@1"]))
        trust_r = float(np.mean(rows["Trust-Aware"]["R@1"]))
        cori_n = float(np.mean(rows["CORI"]["nDCG@10(b=1)"]))
        trust_n = float(np.mean(rows["Trust-Aware"]["nDCG@10(b=1)"]))
        lines.append(
            f"{_latex_escape(variant)} & {cori_r:.3f} & {trust_r:.3f} & "
            f"{trust_r - cori_r:+.3f} & {cori_n:.3f} & {trust_n:.3f} & "
            f"{trust_n - cori_n:+.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def generate_all():
    os.makedirs(TABDIR, exist_ok=True)
    os.makedirs(PAPER_TABDIR, exist_ok=True)
    analysis = significance_analysis()
    tables = {
        "table_testbed.tex": table_testbed(),
        "table_main.tex": table_main(),
        "table_significance.tex": table_significance(analysis),
    }
    optional_tables = {
        "table_online_baselines.tex": table_online_baselines(),
        "table_variant_sensitivity.tex": table_variant_sensitivity(),
    }
    tables.update({
        filename: content
        for filename, content in optional_tables.items()
        if content is not None
    })
    for filename, content in tables.items():
        for folder in [TABDIR, PAPER_TABDIR]:
            with open(os.path.join(folder, filename), "w", encoding="utf-8") as fh:
                fh.write(content + "\n")
    with open(
        os.path.join(RES, "statistical_analysis.json"), "w", encoding="utf-8"
    ) as fh:
        json.dump(analysis, fh, indent=2, sort_keys=True)
        fh.write("\n")

    markdown = [
        "# Federated Trust-Aware Retrieval - Results\n",
        "Generated from `results/federated/*.json` (seeds 7, 11, 13).\n",
        "## Testbed\n",
        "```\n" + table_testbed() + "\n```\n",
        "## Main results (mean +/- std over 3 seeds)\n",
        "```\n" + table_main() + "\n```\n",
        (
            "## Confirmatory inference (query-clustered paired Wilcoxon, "
            "family-wise Holm correction)\n"
        ),
        "```\n" + table_significance(analysis) + "\n```\n",
    ]
    online = table_online_baselines()
    if online is not None:
        markdown += [
            "## Online reputation baselines (6 evil twins)\n",
            "```\n" + online + "\n```\n",
        ]
    variants = table_variant_sensitivity()
    if variants is not None:
        markdown += [
            "## Adversary variant sensitivity (6 untrusted sources)\n",
            "```\n" + variants + "\n```\n",
        ]
    markdown += [
        "Machine-readable details: `statistical_analysis.json`.\n",
    ]
    with open(os.path.join(RES, "RESULTS.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(markdown))
    print("Wrote publication tables, RESULTS.md, and statistical_analysis.json")


if __name__ == "__main__":
    generate_all()
