"""Multi-criteria scoring strategies for trust-aware query optimization.

Scientific contributions:
- Abstract ScoringStrategy interface enabling plug-in scorer design
- LinearWeightedScorer: weighted additive utility model (baseline)
- TOPSISScorer: Technique for Order of Preference by Similarity to Ideal
  Solution (Hwang & Yoon, 1981) — vector-normalized MADM ranking
- BayesianUCBScorer: Upper Confidence Bound scoring for exploration-
  exploitation trade-off in source selection under uncertainty

References:
    Hwang, C.-L., & Yoon, K. (1981). Multiple attribute decision making:
        methods and applications. Springer-Verlag.
    Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis
        of the multiarmed bandit problem. Machine Learning, 47(2), 235-256.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from .models import DataSource, QueryRequest, ScoreBreakdown, _clamp_0_1


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------

def _latency_component(source: DataSource, request: QueryRequest) -> float:
    """Normalize latency to [0, 1] where 1 means minimum latency."""
    if request.max_latency_ms is None or request.max_latency_ms <= 0:
        return 1.0
    return _clamp_0_1(1.0 - source.latency_ms / request.max_latency_ms)


def _cost_component(source: DataSource) -> float:
    """Normalize cost to (0, 1] where 1 means zero cost."""
    return 1.0 / (1.0 + max(0.0, source.cost_per_query))


def _trust_component(source: DataSource, request: QueryRequest) -> float:
    """Return trust component, optionally using conservative LCB."""
    raw = (
        source.effective_trust_lcb()
        if request.use_conservative_trust
        else source.effective_trust()
    )
    return _clamp_0_1(raw)


def _raw_features(
    source: DataSource, request: QueryRequest
) -> Tuple[float, float, float, float]:
    """Return (trust, latency, freshness, cost) feature vector."""
    return (
        _trust_component(source, request),
        _latency_component(source, request),
        _clamp_0_1(source.freshness_score),
        _cost_component(source),
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ScoringStrategy(ABC):
    """Abstract interface for source scoring strategies.

    Implementations must provide :meth:`score`, which returns a composite
    scalar score in [0, 1] together with a :class:`ScoreBreakdown`.
    """

    @abstractmethod
    def score(
        self,
        source: DataSource,
        request: QueryRequest,
        all_sources: Optional[List[DataSource]] = None,
    ) -> Tuple[float, ScoreBreakdown]:
        """Score a single source.

        Args:
            source:      The source to score.
            request:     The query request with weights and constraints.
            all_sources: All candidate sources (required for context-aware
                         strategies such as TOPSIS).

        Returns:
            ``(composite_score, breakdown)`` where composite_score ∈ [0, 1].
        """


# ---------------------------------------------------------------------------
# Linear weighted sum scorer
# ---------------------------------------------------------------------------

class LinearWeightedScorer(ScoringStrategy):
    """Weighted additive utility model (baseline).

    Computes::

        score = w_t * trust + w_l * latency + w_f * freshness + w_c * cost

    where each component is normalized to [0, 1] and weights are taken
    from the QueryRequest.
    """

    def score(
        self,
        source: DataSource,
        request: QueryRequest,
        all_sources: Optional[List[DataSource]] = None,
    ) -> Tuple[float, ScoreBreakdown]:
        trust, latency, freshness, cost = _raw_features(source, request)
        composite = (
            request.trust_weight * trust
            + request.latency_weight * latency
            + request.freshness_weight * freshness
            + request.cost_weight * cost
        )
        return composite, ScoreBreakdown(trust, latency, freshness, cost, composite)


# ---------------------------------------------------------------------------
# TOPSIS scorer
# ---------------------------------------------------------------------------

class TOPSISScorer(ScoringStrategy):
    """TOPSIS multi-criteria ranking.

    Ranks sources by their *relative closeness* to a positive-ideal solution
    (PIS) and away from a negative-ideal solution (NIS).  Steps:

    1. Build the decision matrix of normalized feature vectors.
    2. Apply criteria weights to produce a weighted normalized matrix.
    3. Identify PIS (column-wise maxima) and NIS (column-wise minima).
    4. Compute Euclidean distances D⁺ (to PIS) and D⁻ (to NIS).
    5. Score = D⁻ / (D⁺ + D⁻) ∈ [0, 1]; higher is better.

    Reference:
        Hwang, C.-L., & Yoon, K. (1981). *Multiple attribute decision making*.
        Springer-Verlag.
    """

    def score(
        self,
        source: DataSource,
        request: QueryRequest,
        all_sources: Optional[List[DataSource]] = None,
    ) -> Tuple[float, ScoreBreakdown]:
        candidates = all_sources if all_sources else [source]
        weights = [
            request.trust_weight,
            request.latency_weight,
            request.freshness_weight,
            request.cost_weight,
        ]
        n_criteria = len(weights)

        matrix = [list(_raw_features(s, request)) for s in candidates]

        # Step 1: Vector normalization — divide each entry by column L2-norm
        col_norms = [
            math.sqrt(sum(matrix[i][j] ** 2 for i in range(len(matrix))))
            or 1.0
            for j in range(n_criteria)
        ]
        norm_matrix = [
            [matrix[i][j] / col_norms[j] for j in range(n_criteria)]
            for i in range(len(matrix))
        ]

        # Step 2: Weighted normalized matrix
        weighted = [
            [norm_matrix[i][j] * weights[j] for j in range(n_criteria)]
            for i in range(len(matrix))
        ]

        # Step 3: PIS and NIS
        pis = [
            max(weighted[i][j] for i in range(len(matrix)))
            for j in range(n_criteria)
        ]
        nis = [
            min(weighted[i][j] for i in range(len(matrix)))
            for j in range(n_criteria)
        ]

        # Steps 4-5: Distances and relative closeness for our source
        idx = next(
            (i for i, s in enumerate(candidates) if s.name == source.name), 0
        )
        d_pos = math.sqrt(
            sum((weighted[idx][j] - pis[j]) ** 2 for j in range(n_criteria))
        )
        d_neg = math.sqrt(
            sum((weighted[idx][j] - nis[j]) ** 2 for j in range(n_criteria))
        )
        composite = d_neg / (d_pos + d_neg) if (d_pos + d_neg) > 0 else 0.5

        raw = matrix[idx]
        return composite, ScoreBreakdown(raw[0], raw[1], raw[2], raw[3], composite)


# ---------------------------------------------------------------------------
# Bayesian UCB scorer
# ---------------------------------------------------------------------------

class BayesianUCBScorer(ScoringStrategy):
    """Bayesian Upper Confidence Bound scorer for exploration-exploitation.

    Augments the linear weighted score with an *exploration bonus* that
    decays exponentially as more evidence is accumulated::

        bonus(n) = exploration_bonus * exp(-n / 20)

    Sources with few observations receive a trust premium, encouraging the
    optimizer to gather evidence about under-explored sources.

    Reference:
        Kaufmann, E., Cappé, O., & Garivier, A. (2012). On Bayesian upper
        confidence bounds for bandit problems. AISTATS 2012.
    """

    def __init__(self, exploration_bonus: float = 0.1) -> None:
        self.exploration_bonus = exploration_bonus

    def score(
        self,
        source: DataSource,
        request: QueryRequest,
        all_sources: Optional[List[DataSource]] = None,
    ) -> Tuple[float, ScoreBreakdown]:
        n = source.evidence.n
        # UCB-augmented trust with Bayesian shrinkage
        weight = min(1.0, n / max(1, n + 10))
        trust_ucb = _clamp_0_1(source.evidence.ucb)
        trust = _clamp_0_1(
            (1.0 - weight) * source.trust_score + weight * trust_ucb
        )
        # Exploration bonus inversely proportional to evidence count
        bonus = self.exploration_bonus * math.exp(-n / 20.0)
        trust = _clamp_0_1(trust + bonus)

        latency = _latency_component(source, request)
        freshness = _clamp_0_1(source.freshness_score)
        cost = _cost_component(source)

        composite = (
            request.trust_weight * trust
            + request.latency_weight * latency
            + request.freshness_weight * freshness
            + request.cost_weight * cost
        )
        return composite, ScoreBreakdown(trust, latency, freshness, cost, composite)
