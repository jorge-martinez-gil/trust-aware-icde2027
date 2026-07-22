"""Extended federated trust experiments (journal revision).

Generalises the stylised evil-twin threat model along three axes that a
top-tier evaluation demands, reusing the real six-collection testbed, the CORI
content signal, and the prequential protocol of the core study:

* EXT-1  Graded / stochastic reliability -> calibration recovery.
         Mirrors that are content-identical to a target but reliable only with
         probability ``p`` (partial corpus poisoning / stale replicas, in the
         spirit of PoisonedRAG).  We show the beta-binomial posterior recovers
         ``p`` from observed outcomes.
* EXT-2  Non-stationary reliability (concept drift).
         A source's reliability changes mid-stream; the online estimator
         re-calibrates, which a static content signal cannot.
* EXT-3  Hyper-parameter sensitivity of the optimistic prior and the lower
         confidence width ``z`` on the six-twin federation.

Run with ``python -m trust_aware.federated.experiment_ext`` or via
``examples/run_extended_study.py``.  Results cache as ``results/federated/ext_*.json``.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .testbed import load_testbed, Testbed
from .retrieval import Federation
from .selection import CORISelector, _BetaTrust
from .experiment import (
    build_query_stream,
    build_federation_with_untrusted,
    FEEDBACK_DEPTH,
)
from .metrics import ndcg_at_k

RES = os.path.join(os.path.dirname(__file__), "..", "..", "results", "federated")
SEEDS = (7, 11, 13)


def _real_home_hits(fed: Federation, stream) -> Dict[str, bool]:
    hit = {}
    for it in stream:
        res = fed.indexes[it.home].search(it.text, top_k=100)
        hit[it.qid] = any(f"{it.home}::{d}" in it.rel for d, _ in res[:FEEDBACK_DEPTH])
    return hit


def _tied_class(fed: Federation) -> Dict[str, List[str]]:
    cls: Dict[str, List[str]] = {}
    for s in fed.source_names():
        if s.startswith("TWIN"):
            cls.setdefault(s.split("-")[1], []).append(s)
    return cls


@dataclass
class Router:
    """Configurable prequential trust router (CORI relevance x calibrated trust)."""

    cori: CORISelector
    sources: List[str]
    prior: Tuple[float, float] = (8.0, 2.0)
    z: float = 1.0
    cost_tiebreak: float = 0.02
    seed: int = 7

    def __post_init__(self):
        self.trust = {s: _BetaTrust(self.prior[0], self.prior[1]) for s in self.sources}
        self.rng = np.random.default_rng(self.seed)
        costs = np.array([self.cori.fed.cost_per_query[s] for s in self.sources])
        cmin, cmax = float(costs.min()), float(costs.max())
        self.cn = {s: (self.cori.fed.cost_per_query[s] - cmin) / (cmax - cmin + 1e-9)
                   for s in self.sources}

    def rank(self, query: str) -> List[str]:
        rel = self.cori.score(query)
        rmax = max(rel.values()) or 1.0
        sc = {s: (rel[s] / rmax) * self.trust[s].lower_bound(self.z)
              - self.cost_tiebreak * self.cn[s] for s in self.sources}
        order = list(self.sources)
        self.rng.shuffle(order)
        order.sort(key=lambda s: -sc[s])
        return order

    def observe(self, source: str, success: bool) -> None:
        self.trust[source].update(success)


# --------------------------- EXT-1 graded calibration ---------------------------
def run_graded_calibration(reliabilities=(0.0, 0.2, 0.4, 0.6, 0.8), seeds=SEEDS) -> Dict:
    tb0 = load_testbed()
    per_seed: List[Dict] = []
    for seed in seeds:
        n = len(reliabilities)
        tb, fed = build_federation_with_untrusted(tb0, n, 1500, seed, adversary="exact")
        real = [c for c, col in tb0.collections.items() if col.domain != "untrusted"]
        twins = [s for s in fed.source_names() if s.startswith("TWIN")]
        cfg = {tw: (tw.split("-")[1], reliabilities[i % len(reliabilities)])
               for i, tw in enumerate(twins)}
        stream = build_query_stream(tb0, seed)
        rh = _real_home_hits(fed, stream)
        tied = _tied_class(fed)
        cori = CORISelector(fed, seed=seed)
        R = Router(cori, fed.source_names(), seed=seed)
        bern = random.Random(seed * 101 + 7)
        obs = {s: [] for s in fed.source_names()}
        # Full within-class feedback: a budgeted router that queries the home's
        # content-tied class observes an outcome for every member.
        for it in stream:
            R.rank(it.text)
            R.observe(it.home, rh[it.qid]); obs[it.home].append(1 if rh[it.qid] else 0)
            for tw in tied.get(it.home, []):
                p = cfg[tw][1]; s = bern.random() < p
                R.observe(tw, s); obs[tw].append(1 if s else 0)
        rec = {}
        for tw, (tg, p) in cfg.items():
            t = R.trust[tw]; o = obs[tw]
            rec[tw] = {"kind": "mirror", "target": tg, "true_reliability": p,
                       "learned_mean": round(t.mean(), 4),
                       "learned_lower": round(t.lower_bound(R.z), 4),
                       "n_obs": len(o),
                       "realised": round(float(np.mean(o)), 4) if o else None}
        for s in real:
            hq = [it for it in stream if it.home == s]
            truth = float(np.mean([1.0 if rh[it.qid] else 0.0 for it in hq]))
            t = R.trust[s]; o = obs[s]
            rec[s] = {"kind": "genuine", "target": s, "true_reliability": round(truth, 4),
                      "learned_mean": round(t.mean(), 4),
                      "learned_lower": round(t.lower_bound(R.z), 4),
                      "n_obs": len(o),
                      "realised": round(float(np.mean(o)), 4) if o else None}
        per_seed.append(rec)
    names = sorted(per_seed[0])
    agg = {}
    for nm in names:
        rows = [ps[nm] for ps in per_seed]
        agg[nm] = {"kind": rows[0]["kind"], "target": rows[0]["target"],
                   "true_reliability": round(float(np.mean([r["true_reliability"] for r in rows])), 4),
                   "learned_mean": round(float(np.mean([r["learned_mean"] for r in rows])), 4),
                   "learned_mean_std": round(float(np.std([r["learned_mean"] for r in rows])), 4),
                   "learned_lower": round(float(np.mean([r["learned_lower"] for r in rows])), 4),
                   "realised": round(float(np.mean([r["realised"] for r in rows])), 4),
                   "n_obs": round(float(np.mean([r["n_obs"] for r in rows])), 1)}
    xs = np.array([r["true_reliability"] for r in agg.values()])
    ys = np.array([r["learned_mean"] for r in agg.values()])
    mxs = np.array([r["true_reliability"] for r in agg.values() if r["kind"] == "mirror"])
    mys = np.array([r["learned_mean"] for r in agg.values() if r["kind"] == "mirror"])
    cal = {"mae_all": round(float(np.mean(np.abs(xs - ys))), 4),
           "pearson_all": round(float(np.corrcoef(xs, ys)[0, 1]), 4),
           "mae_mirror": round(float(np.mean(np.abs(mxs - mys))), 4),
           "pearson_mirror": round(float(np.corrcoef(mxs, mys)[0, 1]), 4)}
    out = {"reliabilities": list(reliabilities), "seeds": list(seeds),
           "per_seed": per_seed, "aggregate": agg, "calibration": cal}
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "ext_graded_calibration.json"), "w"), indent=2)
    return out


# ------------------------------- EXT-2 drift ----------------------------------
def run_drift(seed: int = 7) -> Dict:
    tb0 = load_testbed()
    target = max((c for c in tb0.collections if tb0.collections[c].domain != "untrusted"),
                 key=lambda c: tb0.collections[c].num_queries())
    tb, fed = build_federation_with_untrusted(tb0, 2, 1500, seed, adversary="exact")
    twins = [s for s in fed.source_names() if s.startswith("TWIN")]
    drift = {twins[0]: {"label": "stale->refreshed", "before": 0.0, "after": 0.9},
             twins[1]: {"label": "fresh->poisoned", "before": 0.9, "after": 0.0}}
    stream = build_query_stream(tb0, seed)
    rh = _real_home_hits(fed, stream)
    tpos = [i for i, it in enumerate(stream) if it.home == target]
    mid = tpos[len(tpos) // 2]
    cori = CORISelector(fed, seed=seed)
    R = Router(cori, fed.source_names(), seed=seed)
    bern = random.Random(seed * 977 + 3)
    traj = {tw: [] for tw in twins}; seen = 0
    for i, it in enumerate(stream):
        R.rank(it.text); R.observe(it.home, rh[it.qid])
        if it.home == target:
            seen += 1
            for tw in twins:
                c = drift[tw]; p = c["before"] if i < mid else c["after"]
                R.observe(tw, bern.random() < p)
                traj[tw].append({"t": seen, "lower": round(R.trust[tw].lower_bound(R.z), 4),
                                 "mean": round(R.trust[tw].mean(), 4)})
    out = {"seed": seed, "target": target, "n_target_queries": len(tpos),
           "regime_change_at_target_query": len(tpos) // 2,
           "drift": {tw: drift[tw]["label"] for tw in twins}, "trajectory": traj}
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "ext_drift.json"), "w"), indent=2)
    return out


# ------------------------- EXT-3 hyper-parameter sweep ------------------------
def _precompute(tb0, n_twins, seed):
    tb, fed = build_federation_with_untrusted(tb0, n_twins, 1500, seed, adversary="exact")
    stream = build_query_stream(tb0, seed)
    rh, results = {}, {}
    for it in stream:
        per = {}
        for s in fed.source_names():
            res = fed.indexes[s].search(it.text, top_k=100)
            if res:
                sc = [x for _, x in res]; lo, hi = min(sc), max(sc); rng = (hi - lo) or 1.0
                per[s] = [(f"{s}::{d}", (x - lo) / rng) for d, x in res]
            else:
                per[s] = []
            if s == it.home:
                rh[it.qid] = any(f"{it.home}::{d}" in it.rel for d, _ in res[:FEEDBACK_DEPTH])
        results[it.qid] = per
    return fed, stream, rh, results


def _quality(R, stream, rh, results):
    nh = n = 0; nd = 0.0
    for it in stream:
        order = R.rank(it.text); top = order[0]; n += 1
        if top == it.home:
            nh += 1
        ranked = [d for d, _ in results[it.qid][top]]
        nd += ndcg_at_k(ranked, it.rel, 10)
        R.observe(top, rh[it.qid] if top == it.home else False)
    return nh / n, nd / n


def run_hyperparam_sensitivity(n_twins=6, priors=((1.0, 1.0), (4.0, 2.0), (8.0, 2.0), (20.0, 5.0)),
                               zs=(0.0, 1.0, 2.0), seeds=SEEDS) -> Dict:
    tb0 = load_testbed()
    caches = {s: _precompute(tb0, n_twins, s) for s in seeds}
    grid = {}
    for prior in priors:
        for z in zs:
            r1s, nds = [], []
            for seed in seeds:
                fed, stream, rh, results = caches[seed]
                cori = CORISelector(fed, seed=seed)
                R = Router(cori, fed.source_names(), prior=prior, z=z, seed=seed)
                r1, nd = _quality(R, stream, rh, results); r1s.append(r1); nds.append(nd)
            grid[f"prior={prior[0]:.0f}/{prior[1]:.0f},z={z:g}"] = {
                "prior": list(prior), "z": z,
                "R@1": round(float(np.mean(r1s)), 4), "R@1_std": round(float(np.std(r1s)), 4),
                "nDCG@10": round(float(np.mean(nds)), 4), "nDCG@10_std": round(float(np.std(nds)), 4)}
    out = {"n_twins": n_twins, "seeds": list(seeds), "grid": grid}
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(os.path.join(RES, "ext_hyperparam.json"), "w"), indent=2)
    return out


def run_all_ext():
    c = run_graded_calibration()
    print(f"EXT-1 calibration: mirror MAE={c['calibration']['mae_mirror']} "
          f"pearson={c['calibration']['pearson_mirror']}")
    run_drift(seed=7)
    print("EXT-2 drift: results/federated/ext_drift.json")
    h = run_hyperparam_sensitivity()
    nd = [v["nDCG@10"] for v in h["grid"].values()]
    print(f"EXT-3 sensitivity: nDCG@10 range {min(nd)}..{max(nd)}")
    return c, h


if __name__ == "__main__":
    run_all_ext()
