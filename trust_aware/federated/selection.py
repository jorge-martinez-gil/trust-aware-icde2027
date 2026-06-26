"""Resource-selection / query-routing methods over the federation.

All methods expose the same interface::

    rank = method.rank_sources(query_text)      # -> List[source_name] best-first

so they are directly comparable. We implement well-known federated-search
baselines plus the trust-aware router.

Baselines
---------
- ``RandomSelector``      : seeded random ordering (lower bound).
- ``SearchAllSelector``   : every source (exhaustive upper bound on recall).
- ``OracleSelector``      : the true home collection first (routing upper bound).
- ``CORISelector``        : Callan et al. (1995) CORI collection ranking.
- ``ReDDESelector``       : Si & Callan (2003) relevant-document estimation via a
                            centralized sample index.
- ``CORIMeanSelector``    : CORI multiplied by an online empirical success mean.
- ``CORIUCBSelector``     : CORI multiplied by an optimistic online UCB estimate.

Proposed
--------
- ``TrustAwareSelector``  : combines a CORI relevance signal with online,
                            beta-binomial *calibrated trust* per source (a
                            lower-confidence-bound, as in the core optimizer) and
                            a cost-aware budget. Trust evidence is updated from
                            observed routing outcomes, so the router learns which
                            sources are reliable rather than trusting raw scores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .retrieval import Federation, tokenize


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class Selector:
    name = "base"

    def rank_sources(self, query: str) -> List[str]:
        raise NotImplementedError

    def observe(self, source: str, relevant_found: bool) -> None:
        """Optional online feedback hook (default: no-op)."""
        return None


# ---------------------------------------------------------------------------
# Trivial baselines
# ---------------------------------------------------------------------------


class RandomSelector(Selector):
    name = "Random"

    def __init__(self, federation: Federation, seed: int = 7):
        self.sources = federation.source_names()
        self.rng = np.random.default_rng(seed)

    def rank_sources(self, query: str) -> List[str]:
        order = list(self.sources)
        self.rng.shuffle(order)
        return order


class SearchAllSelector(Selector):
    name = "Search-All"

    def __init__(self, federation: Federation):
        # Deterministic order by descending size (cost-agnostic upper bound).
        self.sources = sorted(
            federation.source_names(),
            key=lambda s: -federation.stats[s].num_docs,
        )

    def rank_sources(self, query: str) -> List[str]:
        return list(self.sources)


class OracleSelector(Selector):
    """Upper bound on routing: needs the query's home collection.

    Constructed with a mapping query_text -> home source. Falls back to CORI-like
    behaviour is unnecessary because the experiment supplies the home label.
    """

    name = "Oracle"

    def __init__(self, federation: Federation, home_of: Dict[str, str]):
        self.federation = federation
        self.home_of = home_of
        self.others = federation.source_names()

    def rank_sources(self, query: str) -> List[str]:
        home = self.home_of.get(query)
        rest = [s for s in self.others if s != home]
        return ([home] + rest) if home else list(self.others)


# ---------------------------------------------------------------------------
# CORI (Callan, 1995)
# ---------------------------------------------------------------------------


class CORISelector(Selector):
    name = "CORI"

    def __init__(self, federation: Federation, b: float = 0.4, seed: int = 7):
        self.fed = federation
        self.b = b
        self.sources = federation.source_names()
        self.rng = np.random.default_rng(seed)
        self.num_collections = len(self.sources)
        # cw_i = number of (running) terms in collection i; avg_cw across colls.
        self.cw = {s: federation.stats[s].total_terms for s in self.sources}
        self.avg_cw = sum(self.cw.values()) / max(1, len(self.cw))
        # cf_k = number of collections containing term k.
        cf: Dict[str, int] = {}
        for s in self.sources:
            for term in federation.stats[s].df:
                cf[term] = cf.get(term, 0) + 1
        self.cf = cf

    def score(self, query: str) -> Dict[str, float]:
        q_terms = [t for t in tokenize(query)]
        scores: Dict[str, float] = {}
        for s in self.sources:
            df_s = self.fed.stats[s].df
            cw_ratio = self.cw[s] / (self.avg_cw or 1.0)
            total = 0.0
            for k in q_terms:
                df = df_s.get(k, 0)
                if df == 0:
                    continue
                T = df / (df + 50.0 + 150.0 * cw_ratio)
                cf = self.cf.get(k, 1)
                I = math.log((self.num_collections + 0.5) / cf) / math.log(
                    self.num_collections + 1.0
                )
                p = self.b + (1.0 - self.b) * T * I
                total += p
            scores[s] = total / max(1, len(q_terms))
        return scores

    def rank_sources(self, query: str) -> List[str]:
        scores = self.score(query)
        order = list(self.sources)
        self.rng.shuffle(order)            # fair random tie-breaking
        order.sort(key=lambda s: -scores[s])  # stable sort keeps shuffle on ties
        return order


# ---------------------------------------------------------------------------
# ReDDE (Si & Callan, 2003) via centralized sample index
# ---------------------------------------------------------------------------


class ReDDESelector(Selector):
    name = "ReDDE"

    def __init__(self, federation: Federation, csi_depth: int = 100, seed: int = 7):
        self.fed = federation
        self.sources = federation.source_names()
        self.csi_depth = csi_depth
        self.rng = np.random.default_rng(seed)
        # scale factor per source = full_size / sample_size.
        self.scale: Dict[str, float] = {}
        sample_counts: Dict[str, int] = {}
        for key, src in federation.sample_doc_source.items():
            sample_counts[src] = sample_counts.get(src, 0) + 1
        for s in self.sources:
            full = federation.stats[s].num_docs
            samp = max(1, sample_counts.get(s, 1))
            self.scale[s] = full / samp

    def score(self, query: str) -> Dict[str, float]:
        csi = self.fed.sample_index
        ranked = csi.search(query, top_k=self.csi_depth)
        est: Dict[str, float] = {s: 0.0 for s in self.sources}
        for key, _score in ranked:
            src = self.fed.sample_doc_source.get(key)
            if src is not None:
                est[src] += self.scale[src]
        return est

    def rank_sources(self, query: str) -> List[str]:
        est = self.score(query)
        order = list(self.sources)
        self.rng.shuffle(order)
        order.sort(key=lambda s: -est[s])
        return order


# ---------------------------------------------------------------------------
# Trust-aware router (proposed)
# ---------------------------------------------------------------------------


@dataclass
class _BetaTrust:
    """Beta-binomial trust evidence with an optimistic prior + lower bound.

    The prior is optimistic (``alpha >> beta``) so an *unobserved* source is
    trusted by default; trust is only *demoted* as failures accumulate. This is
    the optimism-under-uncertainty principle: it prevents rarely-queried but
    reliable sources from being penalised, while still suppressing sources that
    repeatedly fail to return relevant results.
    """

    alpha: float = 8.0
    beta: float = 2.0

    def update(self, success: bool) -> None:
        if success:
            self.alpha += 1.0
        else:
            self.beta += 1.0

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def lower_bound(self, z: float = 1.0) -> float:
        # Posterior-mean minus z posterior std (calibrated, penalises uncertainty).
        n = self.alpha + self.beta
        mean = self.alpha / n
        var = (self.alpha * self.beta) / (n * n * (n + 1.0))
        return max(0.0, mean - z * math.sqrt(var))

    def upper_bound(self, z: float = 1.0) -> float:
        n = self.alpha + self.beta
        mean = self.alpha / n
        var = (self.alpha * self.beta) / (n * n * (n + 1.0))
        return min(1.0, mean + z * math.sqrt(var))


class _OnlineCORITrustSelector(Selector):
    """Shared machinery for online reputation baselines over CORI relevance."""

    name = "online-cori-trust"

    def __init__(
        self,
        federation: Federation,
        *,
        z: float = 1.0,
        cost_tiebreak: float = 0.0,
        seed: int = 7,
    ):
        self.fed = federation
        self.sources = federation.source_names()
        self.cori = CORISelector(federation, seed=seed)
        self.z = z
        self.cost_tiebreak = cost_tiebreak
        self.prior = (8.0, 2.0)
        self.trust: Dict[str, _BetaTrust] = {
            s: _BetaTrust(self.prior[0], self.prior[1]) for s in self.sources
        }
        self.rng = np.random.default_rng(seed)
        costs = np.array([federation.cost_per_query[s] for s in self.sources])
        cmin, cmax = float(costs.min()), float(costs.max())
        self.cost_norm = {
            s: (federation.cost_per_query[s] - cmin) / (cmax - cmin + 1e-9)
            for s in self.sources
        }

    def reliability(self, source: str) -> float:
        raise NotImplementedError

    def score(self, query: str) -> Dict[str, float]:
        rel = self.cori.score(query)
        rmax = max(rel.values()) if rel else 1.0
        rmax = rmax or 1.0
        return {
            s: (rel[s] / rmax) * self.reliability(s)
            - self.cost_tiebreak * self.cost_norm[s]
            for s in self.sources
        }

    def rank_sources(self, query: str) -> List[str]:
        scores = self.score(query)
        order = list(self.sources)
        self.rng.shuffle(order)
        order.sort(key=lambda s: -scores[s])
        return order

    def observe(self, source: str, relevant_found: bool) -> None:
        if source in self.trust:
            self.trust[source].update(relevant_found)

    def trust_report(self) -> Dict[str, dict]:
        return {
            s: {
                "mean": round(self.trust[s].mean(), 4),
                "lower_bound": round(self.trust[s].lower_bound(self.z), 4),
                "upper_bound": round(self.trust[s].upper_bound(self.z), 4),
                "alpha": self.trust[s].alpha,
                "beta": self.trust[s].beta,
            }
            for s in self.sources
        }


class CORIMeanSelector(_OnlineCORITrustSelector):
    """Online reputation baseline: relevance times posterior mean.

    This baseline gets the same feedback as Trust-Aware but does not penalise
    uncertainty. It separates the value of calibrated lower bounds from the
    simpler idea of multiplying CORI by an empirical source success rate.
    """

    name = "CORI+Mean"

    def reliability(self, source: str) -> float:
        return self.trust[source].mean()


class CORIUCBSelector(_OnlineCORITrustSelector):
    """Online reputation baseline: relevance times optimistic UCB trust."""

    name = "CORI+UCB"

    def reliability(self, source: str) -> float:
        return self.trust[source].upper_bound(self.z)


class TrustAwareSelector(Selector):
    """Relevance (CORI) + online calibrated trust + cost-aware ranking.

    score(s) = relevance_norm(s) * trust_lower_bound(s)
             - cost_tiebreak * cost_norm(s)

    Trust is a beta-binomial lower confidence bound updated from observed routing
    outcomes via :meth:`observe`, so unreliable sources are demoted as evidence
    accumulates -- the mechanism that lets trust pay off when a source is noisy.
    """

    name = "Trust-Aware"

    def __init__(
        self,
        federation: Federation,
        z: float = 1.0,
        cost_tiebreak: float = 0.02,
        seed: int = 7,
    ):
        self.fed = federation
        self.sources = federation.source_names()
        self.cori = CORISelector(federation, seed=seed)
        self.z = z
        self.cost_tiebreak = cost_tiebreak
        self.prior = (8.0, 2.0)  # optimistic: unseen sources trusted by default
        self.trust: Dict[str, _BetaTrust] = {
            s: _BetaTrust(self.prior[0], self.prior[1]) for s in self.sources}
        self.rng = np.random.default_rng(seed)
        costs = np.array([federation.cost_per_query[s] for s in self.sources])
        cmin, cmax = float(costs.min()), float(costs.max())
        self.cost_norm = {
            s: (federation.cost_per_query[s] - cmin) / (cmax - cmin + 1e-9)
            for s in self.sources
        }

    def score(self, query: str) -> Dict[str, float]:
        # Expected utility = relevance x calibrated-trust reliability factor.
        # At cold start trust is uniform, so ranking == CORI; as evidence
        # accrues, unreliable sources are multiplicatively suppressed.
        rel = self.cori.score(query)
        rmax = max(rel.values()) if rel else 1.0
        rmax = rmax or 1.0
        out: Dict[str, float] = {}
        for s in self.sources:
            rel_n = rel[s] / rmax
            reliability = self.trust[s].lower_bound(self.z)
            out[s] = rel_n * reliability - self.cost_tiebreak * self.cost_norm[s]
        return out

    def rank_sources(self, query: str) -> List[str]:
        scores = self.score(query)
        order = list(self.sources)
        self.rng.shuffle(order)
        order.sort(key=lambda s: -scores[s])
        return order

    def observe(self, source: str, relevant_found: bool) -> None:
        if source in self.trust:
            self.trust[source].update(relevant_found)

    def trust_report(self) -> Dict[str, dict]:
        return {
            s: {
                "mean": round(self.trust[s].mean(), 4),
                "lower_bound": round(self.trust[s].lower_bound(self.z), 4),
                "upper_bound": round(self.trust[s].upper_bound(self.z), 4),
                "alpha": self.trust[s].alpha,
                "beta": self.trust[s].beta,
            }
            for s in self.sources
        }


def build_selectors(
    federation: Federation, home_of: Dict[str, str], seed: int = 7
) -> Dict[str, Selector]:
    return {
        "Random": RandomSelector(federation, seed=seed),
        "Search-All": SearchAllSelector(federation),
        "CORI": CORISelector(federation, seed=seed),
        "ReDDE": ReDDESelector(federation, seed=seed),
        "CORI+Mean": CORIMeanSelector(federation, seed=seed),
        "CORI+UCB": CORIUCBSelector(federation, seed=seed),
        "Trust-Aware": TrustAwareSelector(federation, seed=seed),
        "Oracle": OracleSelector(federation, home_of),
    }
