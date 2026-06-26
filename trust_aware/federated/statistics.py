"""Statistical comparison utilities for the federated study.

Provides paired significance tests, cluster-aware bootstrap confidence
intervals, matched-pairs effect sizes, and Holm-Bonferroni correction for
multiple comparisons. These back the rigor claims in the paper.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats as sps


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Wilcoxon signed-rank test on paired samples (a - b).

    Returns (statistic, p_value). Falls back gracefully when all diffs are zero.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    if np.allclose(diff, 0):
        return 0.0, 1.0
    try:
        stat, p = sps.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(stat), float(p)
    except ValueError:
        return 0.0, 1.0


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta effect size in [-1, 1] (P(a>b) - P(a<b))."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # O(n log n) via sorting.
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return 0.0
    b_sorted = np.sort(b)
    greater = np.searchsorted(b_sorted, a, side="left").sum()
    less = (n_b - np.searchsorted(b_sorted, a, side="right")).sum()
    return float((greater - less) / (n_a * n_b))


def cliffs_magnitude(d: float) -> str:
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def paired_rank_biserial(a: Sequence[float], b: Sequence[float]) -> float:
    """Matched-pairs rank-biserial correlation in [-1, 1].

    Unlike Cliff's delta, this effect size preserves the query-level pairing
    used by the Wilcoxon signed-rank test.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    diff = diff[~np.isclose(diff, 0.0)]
    if len(diff) == 0:
        return 0.0
    ranks = sps.rankdata(np.abs(diff), method="average")
    positive = float(ranks[diff > 0].sum())
    negative = float(ranks[diff < 0].sum())
    denominator = positive + negative
    return 0.0 if denominator == 0 else (positive - negative) / denominator


def paired_effect_magnitude(value: float) -> str:
    absolute = abs(value)
    if absolute < 0.10:
        return "negligible"
    if absolute < 0.30:
        return "small"
    if absolute < 0.50:
        return "medium"
    return "large"


def bootstrap_diff_ci(a: Sequence[float], b: Sequence[float], n_boot: int = 10000,
                      alpha: float = 0.05, seed: int = 7) -> Tuple[float, float, float]:
    """Bootstrap CI for the mean paired difference (a - b).

    Returns (mean_diff, lo, hi).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return float(diff.mean()), lo, hi


def holm_bonferroni(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, dict]:
    """Holm-Bonferroni step-down correction. Returns per-key adjusted decision."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, dict] = {}
    prev_adj = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        adj = max(adj, prev_adj)  # enforce monotonicity
        prev_adj = adj
        out[k] = {"p_raw": p, "p_holm": adj, "reject_H0": adj < alpha}
    return out


def compare_methods(records_by_method: Dict[str, List[dict]], metric_key: str,
                    reference: str, others: List[str], seed: int = 7) -> Dict:
    """Compare `reference` against each `other` on a per-query metric.

    Aligns records by query id and runs Wilcoxon + bootstrap CI + Cliff's delta,
    with Holm-Bonferroni across the family of comparisons.
    """
    ref = {r["qid"]: r[metric_key] for r in records_by_method[reference]}
    results: Dict[str, dict] = {}
    pvals: Dict[str, float] = {}
    for o in others:
        om = {r["qid"]: r[metric_key] for r in records_by_method[o]}
        common = [q for q in ref if q in om]
        a = [ref[q] for q in common]
        b = [om[q] for q in common]
        stat, p = paired_wilcoxon(a, b)
        md, lo, hi = bootstrap_diff_ci(a, b, seed=seed)
        d = cliffs_delta(a, b)
        results[o] = {
            "n": len(common),
            "mean_ref": float(np.mean(a)),
            "mean_other": float(np.mean(b)),
            "mean_diff": md, "ci95": [lo, hi],
            "wilcoxon_stat": stat, "p_raw": p,
            "cliffs_delta": d, "effect": cliffs_magnitude(d),
        }
        pvals[o] = p
    holm = holm_bonferroni(pvals)
    for o in others:
        results[o].update({"p_holm": holm[o]["p_holm"],
                           "reject_H0": holm[o]["reject_H0"]})
    return {"reference": reference, "metric": metric_key, "comparisons": results}


def compare_methods_clustered(
    records_by_method: Dict[str, List[dict]],
    metric_key: str,
    reference: str,
    others: List[str],
    *,
    cluster_key: str = "qid",
    seed: int = 7,
    apply_holm: bool = True,
) -> Dict:
    """Compare methods after aggregating repeated observations by query.

    Multi-seed experiments observe the same query several times. Treating those
    seed replicates as independent inflates the effective sample size. This
    function averages replicates within each query cluster, then performs paired
    inference across the independent query clusters.
    """

    def clustered(records: List[dict]) -> Dict[str, List[float]]:
        values: Dict[str, List[float]] = defaultdict(list)
        for record in records:
            values[str(record[cluster_key])].append(float(record[metric_key]))
        return dict(values)

    ref = clustered(records_by_method[reference])
    results: Dict[str, dict] = {}
    pvals: Dict[str, float] = {}
    for other in others:
        comparison = clustered(records_by_method[other])
        common = sorted(set(ref).intersection(comparison))
        if not common:
            raise ValueError(
                f"No shared {cluster_key!r} values for {reference!r} and {other!r}"
            )
        a = [float(np.mean(ref[qid])) for qid in common]
        b = [float(np.mean(comparison[qid])) for qid in common]
        differences = np.asarray(a) - np.asarray(b)
        statistic, p_value = paired_wilcoxon(a, b)
        mean_diff, lower, upper = bootstrap_diff_ci(a, b, seed=seed)
        effect = paired_rank_biserial(a, b)
        wins = int(np.sum(differences > 0))
        ties = int(np.sum(np.isclose(differences, 0.0)))
        losses = int(np.sum(differences < 0))
        results[other] = {
            "n_clusters": len(common),
            "n_observations": sum(
                min(len(ref[qid]), len(comparison[qid])) for qid in common
            ),
            "mean_ref": float(np.mean(a)),
            "mean_other": float(np.mean(b)),
            "mean_diff": mean_diff,
            "median_diff": float(np.median(differences)),
            "ci95": [lower, upper],
            "wilcoxon_stat": statistic,
            "p_raw": p_value,
            "rank_biserial": effect,
            "effect": paired_effect_magnitude(effect),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "common_language_effect": float((wins + 0.5 * ties) / len(common)),
        }
        pvals[other] = p_value

    if apply_holm:
        holm = holm_bonferroni(pvals)
        for other in others:
            results[other].update(
                {
                    "p_holm": holm[other]["p_holm"],
                    "reject_H0": holm[other]["reject_H0"],
                }
            )
    return {
        "reference": reference,
        "metric": metric_key,
        "cluster_key": cluster_key,
        "comparisons": results,
    }
