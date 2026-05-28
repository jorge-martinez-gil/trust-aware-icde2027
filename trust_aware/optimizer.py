"""Trust-aware query optimizer for AI-native data systems.

Scientific contributions:
- Multi-criteria source ranking with pluggable scoring strategies
- Pareto frontier analysis across all four objectives (trust, latency,
  freshness, cost) to surface non-dominated candidate sets
- Confidence-bounded plan quality estimation based on score margin
- Adaptive execution-strategy selection (SINGLE / FALLBACK / ENSEMBLE)

Re-exports all public model types so callers that import from
``trust_aware.optimizer`` continue to work without changes.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from .models import (
    DataSource,
    ExecutionStrategy,
    QueryPlan,
    QueryRequest,
    ScoreBreakdown,
    TrustEvidence,
    _clamp_0_1,
)
from .scoring import (
    BayesianUCBScorer,
    LinearWeightedScorer,
    ScoringStrategy,
    TOPSISScorer,
)


# ---------------------------------------------------------------------------
# Pareto dominance
# ---------------------------------------------------------------------------

def _dominates(a: DataSource, b: DataSource) -> bool:
    """Return True if *a* Pareto-dominates *b* across all four objectives.

    Source *a* dominates *b* when it is at least as good on every criterion
    (trust, latency, freshness, cost) and strictly better on at least one.
    All criteria are expressed so that *higher is better*:
    latency → 1/latency, cost → 1/cost.
    """
    ta, tb = a.effective_trust(), b.effective_trust()
    # lower latency is better → compare inverted
    la, lb = -a.latency_ms, -b.latency_ms
    fa, fb = a.freshness_score, b.freshness_score
    ca, cb = -a.cost_per_query, -b.cost_per_query

    at_least_as_good = (
        ta >= tb and la >= lb and fa >= fb and ca >= cb
    )
    strictly_better = (
        ta > tb or la > lb or fa > fb or ca > cb
    )
    return at_least_as_good and strictly_better


def compute_pareto_front(sources: List[DataSource]) -> List[DataSource]:
    """Return the Pareto-optimal (non-dominated) subset of *sources*.

    A source is non-dominated if no other source dominates it on all
    four objectives simultaneously.
    """
    front: List[DataSource] = []
    for candidate in sources:
        if not any(_dominates(other, candidate) for other in sources
                   if other.name != candidate.name):
            front.append(candidate)
    return front


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class TrustAwareQueryOptimizer:
    """Trust-aware multi-criteria query optimizer for AI-native data systems.

    Supports three interchangeable scoring strategies:

    ``"linear"``
        Weighted additive utility model — fast, interpretable baseline.
    ``"topsis"``
        TOPSIS (Hwang & Yoon, 1981) — vector-normalized MADM ranking.
    ``"bayesian-ucb"``
        Bayesian UCB — exploration-exploitation via confidence bounds.

    In addition to ranked sources, the returned :class:`~models.QueryPlan`
    carries:

    - **pareto_front** — non-dominated sources across all four objectives.
    - **confidence**   — margin-based confidence in the top-1 choice.
    - **execution_strategy** — SINGLE / FALLBACK / ENSEMBLE recommendation.
    - **breakdowns**   — per-criterion score decompositions.

    Parameters
    ----------
    strategy : str
        One of ``"linear"``, ``"topsis"``, ``"bayesian-ucb"``.
    exploration_bonus : float
        Exploration bonus weight for ``"bayesian-ucb"`` (default 0.1).
    """

    def __init__(
        self,
        strategy: str = "linear",
        exploration_bonus: float = 0.1,
    ) -> None:
        self._scorer: ScoringStrategy = self._build_scorer(
            strategy, exploration_bonus
        )
        self.strategy = strategy

    @staticmethod
    def _build_scorer(
        strategy: str, exploration_bonus: float
    ) -> ScoringStrategy:
        if strategy == "topsis":
            return TOPSISScorer()
        if strategy == "bayesian-ucb":
            return BayesianUCBScorer(exploration_bonus=exploration_bonus)
        return LinearWeightedScorer()

    # ------------------------------------------------------------------
    # Hard constraint filtering
    # ------------------------------------------------------------------

    @staticmethod
    def _filter(
        sources: Iterable[DataSource], request: QueryRequest
    ) -> List[DataSource]:
        result = []
        for source in sources:
            if request.requires_vector and not source.supports_vector:
                continue
            if source.freshness_score < request.min_freshness_score:
                continue
            if (
                request.max_latency_ms is not None
                and source.latency_ms > request.max_latency_ms
            ):
                continue
            if source.reliability < request.min_reliability:
                continue
            result.append(source)
        return result

    # ------------------------------------------------------------------
    # Plan metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _execution_strategy(
        ranked: List[Tuple[DataSource, float]],
    ) -> ExecutionStrategy:
        """Recommend an execution strategy based on score distribution."""
        if not ranked:
            return ExecutionStrategy.SINGLE
        if len(ranked) == 1:
            return ExecutionStrategy.SINGLE
        margin = ranked[0][1] - ranked[1][1]
        if margin < 0.05:
            return ExecutionStrategy.ENSEMBLE
        return ExecutionStrategy.FALLBACK

    @staticmethod
    def _plan_confidence(ranked: List[Tuple[DataSource, float]]) -> float:
        """Estimate confidence as score × (1 + margin-over-second-best)."""
        if not ranked:
            return 0.0
        best = ranked[0][1]
        if len(ranked) == 1:
            return _clamp_0_1(best)
        margin = best - ranked[1][1]
        return _clamp_0_1(best * (1.0 + margin))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def optimize(
        self,
        sources: Iterable[DataSource],
        request: QueryRequest,
    ) -> QueryPlan:
        """Produce an optimized :class:`~models.QueryPlan`.

        Parameters
        ----------
        sources : Iterable[DataSource]
            All candidate data sources to consider.
        request : QueryRequest
            Query constraints and optimization preferences.

        Returns
        -------
        QueryPlan
            Ranked sources, Pareto front, confidence, execution strategy,
            and per-criterion score breakdowns.
        """
        candidates = self._filter(sources, request)
        if not candidates:
            return QueryPlan(ranked_sources=[], confidence=0.0)

        scored: List[Tuple[DataSource, float, ScoreBreakdown]] = [
            (s, *self._scorer.score(s, request, candidates))
            for s in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        ranked: List[Tuple[DataSource, float]] = [
            (s, score) for s, score, _ in scored
        ]
        breakdowns: List[ScoreBreakdown] = [bd for _, _, bd in scored]

        return QueryPlan(
            ranked_sources=ranked,
            breakdowns=breakdowns,
            confidence=self._plan_confidence(ranked),
            execution_strategy=self._execution_strategy(ranked),
            pareto_front=compute_pareto_front(candidates),
        )
