"""Core data models for trust-aware query optimization.

Scientific contributions:
- Bayesian trust estimation via Beta-Binomial model (TrustEvidence)
- Uncertainty quantification with confidence bounds (LCB / UCB)
- Bayesian shrinkage blending static and empirical trust
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


def _clamp_0_1(value: float) -> float:
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Bayesian trust evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrustEvidence:
    """Bayesian evidence for trust estimation via Beta-Binomial model.

    Models source trust as a latent variable T ~ Beta(alpha, beta) where::

        alpha = prior_alpha + successes
        beta  = prior_beta  + failures

    A uniform (uninformative) Beta(1, 1) prior is used by default.
    As evidence accumulates, the posterior mean converges to the empirical
    success rate.  Confidence bounds support risk-aware selection::

        LCB: conservative (pessimistic) trust estimate
        UCB: optimistic trust estimate for exploration

    Reference: DeGroot (1970), *Optimal Statistical Decisions*.
    """

    successes: int = 0
    failures: int = 0
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    @property
    def alpha(self) -> float:
        """Posterior alpha parameter."""
        return self.prior_alpha + self.successes

    @property
    def beta_param(self) -> float:
        """Posterior beta parameter."""
        return self.prior_beta + self.failures

    @property
    def n(self) -> int:
        """Total number of observations."""
        return self.successes + self.failures

    @property
    def mean(self) -> float:
        """Posterior mean: E[T] = alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta_param)

    @property
    def variance(self) -> float:
        """Posterior variance of Beta(alpha, beta)."""
        a, b = self.alpha, self.beta_param
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def std(self) -> float:
        """Posterior standard deviation."""
        return math.sqrt(self.variance)

    @property
    def ucb(self) -> float:
        """Optimistic upper confidence bound (mean + 1.96 * std)."""
        return _clamp_0_1(self.mean + 1.96 * self.std)

    @property
    def lcb(self) -> float:
        """Conservative lower confidence bound (mean - 1.96 * std)."""
        return _clamp_0_1(self.mean - 1.96 * self.std)

    def update(self, success: bool) -> TrustEvidence:
        """Return updated evidence after observing a success or failure."""
        if success:
            return TrustEvidence(
                self.successes + 1, self.failures,
                self.prior_alpha, self.prior_beta,
            )
        return TrustEvidence(
            self.successes, self.failures + 1,
            self.prior_alpha, self.prior_beta,
        )

    def decay(self, factor: float = 0.95) -> TrustEvidence:
        """Temporal decay: shrink observation counts toward the prior.

        Useful for non-stationary environments where past behavior should
        gradually be discounted.
        """
        return TrustEvidence(
            max(0, int(self.successes * factor)),
            max(0, int(self.failures * factor)),
            self.prior_alpha,
            self.prior_beta,
        )


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataSource:
    """A data source that can serve queries in an AI-native system.

    Combines static metadata (trust_score, latency_ms, etc.) with
    accumulating Bayesian evidence (TrustEvidence) to yield posterior
    trust estimates via Bayesian shrinkage.
    """

    name: str
    trust_score: float
    latency_ms: float
    cost_per_query: float
    freshness_score: float
    supports_vector: bool = False
    reliability: float = 1.0
    evidence: TrustEvidence = field(default_factory=TrustEvidence)
    tags: Tuple[str, ...] = field(default_factory=tuple)

    def effective_trust(self) -> float:
        """Bayesian shrinkage estimate blending static score with posterior mean.

        Weight toward evidence grows as n increases::

            w(n) = n / (n + 10)          # shrinkage toward prior
            T_eff = (1 - w) * T_static + w * posterior_mean
        """
        n = self.evidence.n
        weight = min(1.0, n / max(1, n + 10))
        return _clamp_0_1(
            (1.0 - weight) * self.trust_score + weight * self.evidence.mean
        )

    def effective_trust_lcb(self) -> float:
        """Conservative (risk-averse) effective trust using posterior LCB."""
        n = self.evidence.n
        weight = min(1.0, n / max(1, n + 10))
        return _clamp_0_1(
            (1.0 - weight) * self.trust_score + weight * self.evidence.lcb
        )

    def with_evidence(self, evidence: TrustEvidence) -> DataSource:
        """Return a copy of this source with updated evidence."""
        return DataSource(
            name=self.name,
            trust_score=self.trust_score,
            latency_ms=self.latency_ms,
            cost_per_query=self.cost_per_query,
            freshness_score=self.freshness_score,
            supports_vector=self.supports_vector,
            reliability=self.reliability,
            evidence=evidence,
            tags=self.tags,
        )


# ---------------------------------------------------------------------------
# Query request
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryRequest:
    """A query request specifying constraints and optimization preferences."""

    requires_vector: bool = False
    max_latency_ms: float | None = None
    min_freshness_score: float = 0.0
    min_reliability: float = 0.0
    trust_weight: float = 0.5
    latency_weight: float = 0.2
    freshness_weight: float = 0.2
    cost_weight: float = 0.1
    top_k: int = 1
    use_conservative_trust: bool = False

    def validate(self) -> None:
        """Raise ValueError if criteria weights do not sum to 1.0."""
        total = (
            self.trust_weight
            + self.latency_weight
            + self.freshness_weight
            + self.cost_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Criteria weights must sum to 1.0, got {total:.6f}"
            )


# ---------------------------------------------------------------------------
# Execution strategy and score breakdown
# ---------------------------------------------------------------------------

class ExecutionStrategy(str, Enum):
    """Recommended execution strategy for a query plan."""

    SINGLE = "single"
    PARALLEL = "parallel"
    ENSEMBLE = "ensemble"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class ScoreBreakdown:
    """Per-criterion contribution to a source's composite score."""

    trust: float
    latency: float
    freshness: float
    cost: float
    composite: float


# ---------------------------------------------------------------------------
# Query plan
# ---------------------------------------------------------------------------

@dataclass
class QueryPlan:
    """An optimized execution plan produced by TrustAwareQueryOptimizer.

    Attributes:
        ranked_sources:     Sources ordered by composite score (best first).
        breakdowns:         Per-source score breakdowns aligned with
                            ranked_sources.
        confidence:         Margin-based confidence in the top-1 selection.
        execution_strategy: Recommended execution strategy.
        pareto_front:       Non-dominated sources across all objectives.
    """

    ranked_sources: List[Tuple[DataSource, float]]
    breakdowns: List[ScoreBreakdown] = field(default_factory=list)
    confidence: float = 1.0
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SINGLE
    pareto_front: List[DataSource] = field(default_factory=list)

    @property
    def primary_source(self) -> DataSource | None:
        if not self.ranked_sources:
            return None
        return self.ranked_sources[0][0]

    @property
    def fallback_chain(self) -> List[DataSource]:
        """Ordered fallback sources following the primary."""
        return [s for s, _ in self.ranked_sources[1:]]

    def top_k_sources(self, k: int) -> List[DataSource]:
        """Return up to k highest-scored sources."""
        return [s for s, _ in self.ranked_sources[:k]]
