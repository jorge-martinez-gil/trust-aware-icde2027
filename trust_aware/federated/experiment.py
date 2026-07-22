"""End-to-end federated trust-aware retrieval study.

Two settings are evaluated on the same real collections:

A. **Clean federation** -- the six real collections as sources.
B. **Untrusted federation** -- additional lexically-attractive but unreliable
   sources are injected (exact-content mirrors that never contain judged
   relevant documents). This models the AI-native reality of low-quality or
   hallucinated sources and is where first-class calibrated trust matters.

Protocol: queries from all collections are pooled and streamed in a fixed,
seeded order. Each selector ranks sources per query; we evaluate routing and
end-to-end merged retrieval at several cost budgets. The Trust-Aware router is
evaluated *prequentially* (scored on a query, then updated from the observed
outcome), so its trust at query t depends only on queries < t.
"""

from __future__ import annotations

import json
import os
import platform
import random
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

from .testbed import Testbed, load_testbed, Collection
from .retrieval import Federation, build_federation, BM25Index, tokenize
from .selection import build_selectors
from .metrics import ndcg_at_k, average_precision, reciprocal_rank_of

BUDGETS = (1, 2, 3)
FEEDBACK_DEPTH = 100


ADVERSARY_VARIANTS = {
    "exact": {"corruption": 0.0, "doc_fraction": 1.0},
    "corrupted-20": {"corruption": 0.20, "doc_fraction": 1.0},
    "partial-50": {"corruption": 0.0, "doc_fraction": 0.50},
}


def experiment_metadata(seed: int, docs_each: int) -> Dict:
    """Machine-readable protocol metadata stored with every result bundle."""

    return {
        "schema_version": 2,
        "seed": seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "budgets": list(BUDGETS),
        "feedback_depth": FEEDBACK_DEPTH,
        "docs_each": docs_each,
        "trust_prior": [8.0, 2.0],
        "trust_z": 1.0,
        "cost_tiebreak": 0.02,
        "evaluation": "prequential",
        "adversary_variants": ADVERSARY_VARIANTS,
    }


# ---------------------------------------------------------------------------
# Untrusted-source construction
# ---------------------------------------------------------------------------


def make_mirror_collection(
    name: str,
    source: Collection,
    seed: int,
    corruption: float = 0.0,
    doc_fraction: float = 1.0,
) -> Collection:
    """An *evil-twin* mirror: an exact-content duplicate with invalid provenance.

    The mirror copies a genuine collection's documents under fresh ids that
    appear in no relevance judgement. Content-based resource selectors (CORI,
    ReDDE, BM25) cannot distinguish an exact twin from the original, so they are
    forced to a coin flip; only a method that learns from observed outcomes can
    route around it. ``corruption`` optionally perturbs a fraction of tokens, and
    ``doc_fraction`` optionally keeps a source-sized subset, to model less
    stylized degraded mirrors in sensitivity studies.
    """
    rng = random.Random(seed)
    docs: Dict[str, str] = {}
    items = list(source.docs.items())
    if doc_fraction < 1.0:
        keep = max(1, int(round(len(items) * doc_fraction)))
        rng.shuffle(items)
        items = items[:keep]
    for did, text in items:
        if corruption > 0:
            toks = text.split()
            for i in range(len(toks)):
                if rng.random() < corruption:
                    toks[i] = f"corrupt_{rng.randrange(4096):04x}"
            text = " ".join(toks)
        docs[f"tw_{did}"] = text
    return Collection(name=name, domain="untrusted", docs=docs, queries={}, qrels={})


def build_federation_with_untrusted(
    testbed: Testbed,
    n_untrusted: int,
    docs_each: int,
    seed: int,
    adversary: str = "exact",
):
    """Add ``n_untrusted`` evil-twin mirrors, cycling over the real collections."""
    if adversary not in ADVERSARY_VARIANTS:
        raise ValueError(
            f"unknown adversary variant {adversary!r}; "
            f"expected one of {sorted(ADVERSARY_VARIANTS)}"
        )
    variant = ADVERSARY_VARIANTS[adversary]
    tb = Testbed(collections=dict(testbed.collections))
    real = [n for n, c in testbed.collections.items() if c.domain != "untrusted"]
    for u in range(n_untrusted):
        target = real[u % len(real)]
        prefix = "TWIN" if adversary == "exact" else f"TWIN-{adversary}"
        nm = f"{prefix}-{target}-{u+1}"
        tb.collections[nm] = make_mirror_collection(
            nm,
            testbed.collections[target],
            seed + u,
            corruption=variant["corruption"],
            doc_fraction=variant["doc_fraction"],
        )
    fed = build_federation(tb, seed=seed)
    return tb, fed


def merged_search(fed: Federation, sources: List[str], query: str,
                  per_source_k: int = 100) -> List[str]:
    """Search each selected source, min-max normalise within source, merge.

    Returns a single ranked list of namespaced doc ids ("SRC::docid").
    The same merge is used for every selector, so comparisons are fair.
    """
    merged: List[Tuple[str, float]] = []
    for s in sources:
        res = fed.indexes[s].search(query, top_k=per_source_k)
        if not res:
            continue
        scores = [sc for _, sc in res]
        smin, smax = min(scores), max(scores)
        rng = (smax - smin) or 1.0
        for did, sc in res:
            merged.append((f"{s}::{did}", (sc - smin) / rng))
    merged.sort(key=lambda x: -x[1])
    return [d for d, _ in merged]


# ---------------------------------------------------------------------------
# Query stream
# ---------------------------------------------------------------------------


@dataclass
class QueryItem:
    qid: str            # globally unique
    home: str           # home collection name
    text: str
    rel: Set[str]       # namespaced relevant doc ids


def build_query_stream(testbed: Testbed, seed: int) -> List[QueryItem]:
    items: List[QueryItem] = []
    for name, coll in testbed.collections.items():
        if coll.domain == "untrusted":
            continue
        for qid in coll.usable_query_ids():
            rel = {f"{name}::{d}" for d in coll.qrels[qid]}
            items.append(QueryItem(f"{name}:{qid}", name, coll.queries[qid], rel))
    rng = random.Random(seed)
    rng.shuffle(items)
    return items


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_setting(testbed: Testbed, fed: Federation, seed: int,
                     budgets=BUDGETS) -> Dict:
    stream = build_query_stream(testbed, seed)
    home_of = {it.text: it.home for it in stream}
    selectors = build_selectors(fed, home_of, seed=seed)

    # Precompute per-query, per-source normalised BM25 results once (big speedup).
    src_names = fed.source_names()
    qcache: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
    feedback_hit: Dict[str, Dict[str, bool]] = {}
    for it in stream:
        per_src = {}
        fb = {}
        for s in src_names:
            res = fed.indexes[s].search(it.text, top_k=100)
            if res:
                scores = [sc for _, sc in res]
                smin, smax = min(scores), max(scores)
                rng = (smax - smin) or 1.0
                per_src[s] = [(f"{s}::{d}", (sc - smin) / rng) for d, sc in res]
                fb[s] = any(f"{s}::{d}" in it.rel for d, _ in res[:FEEDBACK_DEPTH])
            else:
                per_src[s] = []
                fb[s] = False
        qcache[it.qid] = per_src
        feedback_hit[it.qid] = fb

    def merged_from_cache(qid: str, sources: List[str]) -> List[str]:
        merged: List[Tuple[str, float]] = []
        cache = qcache[qid]
        for s in sources:
            merged.extend(cache.get(s, ()))
        merged.sort(key=lambda x: -x[1])
        return [d for d, _ in merged]

    # Per-(method) per-query records.
    records: Dict[str, List[dict]] = {m: [] for m in selectors}

    for it in stream:
        for mname, sel in selectors.items():
            order = sel.rank_sources(it.text)
            rr = reciprocal_rank_of(order, it.home)
            rec = {"qid": it.qid, "home": it.home, "rr": rr,
                   "home_rank": order.index(it.home) + 1 if it.home in order else 99}
            for b in budgets:
                if mname == "Search-All":
                    chosen = order  # exhaustive regardless of budget
                else:
                    chosen = order[:b]
                ranked = merged_from_cache(it.qid, chosen)
                rec[f"ndcg@{b}"] = ndcg_at_k(ranked, it.rel, 10)
                rec[f"map@{b}"] = average_precision(ranked, it.rel)
                rec[f"cost@{b}"] = round(
                    sum(fed.cost_per_query[s] for s in chosen), 4)
                rec[f"lat@{b}"] = round(
                    max((fed.latency_ms[s] for s in chosen), default=0.0), 2)
            records[mname].append(rec)

            # Online feedback is prequential: update AFTER scoring this query.
            top = order[0]
            sel.observe(top, feedback_hit[it.qid][top])

    out = {
        "n_queries": len(stream),
        "sources": fed.source_names(),
        "records": records,
    }
    # Attach learned trust if available.
    trust_reports = {}
    for mname, sel in selectors.items():
        if hasattr(sel, "trust_report"):
            trust_reports[mname] = sel.trust_report()
    if trust_reports:
        out["trust_reports"] = trust_reports
        out["trust_report"] = trust_reports.get("Trust-Aware", {})
    return out


def aggregate(records: Dict[str, List[dict]], budgets=BUDGETS) -> Dict[str, dict]:
    agg: Dict[str, dict] = {}
    for m, recs in records.items():
        n = len(recs)
        row = {
            "MRR": float(np.mean([r["rr"] for r in recs])),
            "R@1": float(np.mean([1.0 if r["home_rank"] == 1 else 0.0 for r in recs])),
            "R@2": float(np.mean([1.0 if r["home_rank"] <= 2 else 0.0 for r in recs])),
        }
        for b in budgets:
            row[f"nDCG@10(b={b})"] = float(np.mean([r[f"ndcg@{b}"] for r in recs]))
            row[f"MAP(b={b})"] = float(np.mean([r[f"map@{b}"] for r in recs]))
            row[f"cost(b={b})"] = float(np.mean([r[f"cost@{b}"] for r in recs]))
        agg[m] = row
    return agg


def run_one(
    testbed: Testbed,
    seed: int,
    n_untrusted: int,
    docs_each: int,
    adversary: str = "exact",
):
    if n_untrusted == 0:
        fed = build_federation(testbed, seed=seed)
        tb = testbed
    else:
        tb, fed = build_federation_with_untrusted(
            testbed, n_untrusted, docs_each, seed, adversary=adversary)
    res = evaluate_setting(tb, fed, seed=seed)
    return res


def run_study(seed: int = 7, sweep=(0, 2, 4, 6, 8), docs_each: int = 1500,
              stat_counts=(0, 6), out_dir: str = None) -> Dict:
    """Robustness sweep over the number of untrusted sources (single seed)."""
    tb = load_testbed()
    out_dir = out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results", "federated")
    os.makedirs(out_dir, exist_ok=True)

    sweep_agg = {}
    raw_keep = {}
    for n in sweep:
        res = run_one(tb, seed, n, docs_each)
        sweep_agg[n] = aggregate(res["records"])
        if n in stat_counts:
            raw_keep[n] = {"records": res["records"],
                           "trust_report": res.get("trust_report")}
        print(f"[seed {seed}] n_untrusted={n} done", flush=True)

    study = {"seed": seed, "docs_each": docs_each,
             "metadata": experiment_metadata(seed, docs_each),
             "sweep_aggregate": sweep_agg,
             "raw_for_stats": raw_keep,
             "testbed_summary": tb.summary_rows()}
    with open(os.path.join(out_dir, f"sweep_seed{seed}.json"), "w") as fh:
        json.dump(study, fh)
    with open(os.path.join(out_dir, f"sweep_agg_{seed}.json"), "w") as fh:
        json.dump(sweep_agg, fh)
    return study


def run_multiseed(seeds=(7, 11, 13), counts=(0, 6), docs_each: int = 1500,
                  out_dir: str = None) -> Dict:
    """Headline conditions across seeds for variance and clustered inference."""
    tb = load_testbed()
    out_dir = out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results", "federated")
    os.makedirs(out_dir, exist_ok=True)
    out = {"seeds": list(seeds), "counts": list(counts), "by_seed": {},
           "metadata": {
               "schema_version": 2,
               "protocol": "prequential",
               "independent_unit": "query",
               "replicates": "seed",
               "docs_each": docs_each,
               "budgets": list(BUDGETS),
               "feedback_depth": FEEDBACK_DEPTH,
           }}
    for seed in seeds:
        out["by_seed"][seed] = {}
        for n in counts:
            res = run_one(tb, seed, n, docs_each)
            out["by_seed"][seed][n] = {
                "aggregate": aggregate(res["records"]),
                "records": res["records"],
            }
            print(f"[multiseed] seed={seed} n={n} done", flush=True)
    with open(os.path.join(out_dir, "multiseed.json"), "w") as fh:
        json.dump(out, fh)
    return out


def run_variant_study(
    seeds=(7, 11, 13),
    variants=("exact", "corrupted-20", "partial-50"),
    n_untrusted: int = 6,
    docs_each: int = 1500,
    out_dir: str = None,
) -> Dict:
    """Sensitivity study over less stylized untrusted-source variants."""

    tb = load_testbed()
    out_dir = out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "results", "federated")
    os.makedirs(out_dir, exist_ok=True)
    out = {
        "seeds": list(seeds),
        "variants": list(variants),
        "n_untrusted": n_untrusted,
        "metadata": {
            "schema_version": 1,
            "adversary_variants": {
                name: ADVERSARY_VARIANTS[name] for name in variants
            },
            "docs_each": docs_each,
            "budgets": list(BUDGETS),
        },
        "by_seed": {},
    }
    for seed in seeds:
        out["by_seed"][seed] = {}
        for variant in variants:
            res = run_one(tb, seed, n_untrusted, docs_each, adversary=variant)
            out["by_seed"][seed][variant] = {
                "aggregate": aggregate(res["records"]),
            }
            print(
                f"[variants] seed={seed} variant={variant} n={n_untrusted} done",
                flush=True,
            )
    with open(os.path.join(out_dir, "variant_sensitivity.json"), "w") as fh:
        json.dump(out, fh)
    return out


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "sweep":
        run_study(seed=int(sys.argv[2]) if len(sys.argv) > 2 else 7)
    elif mode == "multiseed":
        run_multiseed()
    elif mode == "variants":
        run_variant_study()
    print("DONE", flush=True)
