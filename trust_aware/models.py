from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import FrozenSet, Iterable, List, Mapping, Tuple


def clamp_0_1(value: float) -> float:
    """Clamp a numeric value into the closed trust interval."""

    return max(0.0, min(1.0, float(value)))


class Capability(str, Enum):
    """Capabilities that matter for AI-native query planning."""

    STRUCTURED_SQL = "structured_sql"
    VECTOR_SEARCH = "vector_search"
    LEXICAL_SEARCH = "lexical_search"
    GRAPH_LOOKUP = "graph_lookup"
    LLM_INFERENCE = "llm_inference"
    STREAMING = "streaming"
    FRESHNESS_GUARANTEE = "freshness_guarantee"
    POLICY_AUDIT = "policy_audit"


def normalize_capability(capability: str | Capability) -> str:
    if isinstance(capability, Capability):
        return capability.value

    normalized = str(capability).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "vector": Capability.VECTOR_SEARCH.value,
        "sql": Capability.STRUCTURED_SQL.value,
        "text": Capability.LEXICAL_SEARCH.value,
        "keyword": Capability.LEXICAL_SEARCH.value,
        "llm": Capability.LLM_INFERENCE.value,
        "audit": Capability.POLICY_AUDIT.value,
    }
    return aliases.get(normalized, normalized)


def normalize_capabilities(
    capabilities: Iterable[str | Capability] | None,
) -> FrozenSet[str]:
    if capabilities is None:
        return frozenset()
    return frozenset(normalize_capability(capability) for capability in capabilities)


def z_value_for_risk_tolerance(risk_tolerance: float) -> float:
    """Approximate one-sided Gaussian z values without a scipy dependency."""

    risk_tolerance = clamp_0_1(risk_tolerance)
    if risk_tolerance <= 0.01:
        return 2.33
    if risk_tolerance <= 0.05:
        return 1.64
    if risk_tolerance <= 0.10:
        return 1.28
    if risk_tolerance <= 0.20:
        return 0.84
    if risk_tolerance <= 0.30:
        return 0.52
    return 0.0


@dataclass(frozen=True, init=False)
class TrustEvidence:
    """A calibrated evidence item for one trust dimension.

    The posterior is a beta-binomial estimate. It is intentionally small and
    dependency-free so experiments can run anywhere a paper artifact is checked
    out.

    Supports both positional API (dimension, positive, total, ...) and the
    keyword-only shorthand ``successes=`` / ``failures=``.
    """

    dimension: str
    positive: float
    total: float
    weight: float
    prior_positive: float
    prior_negative: float
    notes: str

    def __init__(
        self,
        dimension: str = "",
        positive: float = 0.0,
        total: float = 0.0,
        weight: float = 1.0,
        prior_positive: float = 2.0,
        prior_negative: float = 2.0,
        notes: str = "",
        *,
        successes: float | None = None,
        failures: float | None = None,
    ) -> None:
        if successes is not None:
            positive = float(successes)
            if failures is not None:
                total = float(successes) + float(failures)
            else:
                total = float(successes)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "positive", float(positive))
        object.__setattr__(self, "total", float(total))
        object.__setattr__(self, "weight", float(weight))
        object.__setattr__(self, "prior_positive", float(prior_positive))
        object.__setattr__(self, "prior_negative", float(prior_negative))
        object.__setattr__(self, "notes", notes)
        # validation (was in __post_init__)
        if self.total < 0:
            raise ValueError("total evidence count must be non-negative")
        if self.positive < 0:
            raise ValueError("positive evidence count must be non-negative")
        if self.positive > self.total:
            raise ValueError("positive evidence cannot exceed total evidence")
        if self.weight < 0:
            raise ValueError("evidence weight must be non-negative")
        if self.prior_positive < 0 or self.prior_negative < 0:
            raise ValueError("beta prior values must be non-negative")

    @property
    def observed_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return clamp_0_1(self.positive / self.total)

    @property
    def posterior_alpha(self) -> float:
        return self.prior_positive + self.positive

    @property
    def posterior_beta(self) -> float:
        _failures = self.total - self.positive
        return self.prior_negative + _failures

    @property
    def posterior_mean(self) -> float:
        denominator = self.posterior_alpha + self.posterior_beta
        if denominator == 0:
            return 0.5
        return clamp_0_1(self.posterior_alpha / denominator)

    @property
    def posterior_variance(self) -> float:
        alpha = self.posterior_alpha
        beta = self.posterior_beta
        denominator = (alpha + beta) ** 2 * (alpha + beta + 1.0)
        if denominator == 0:
            return 0.25
        return max(0.0, (alpha * beta) / denominator)

    @property
    def confidence(self) -> float:
        effective_n = self.total + self.prior_positive + self.prior_negative
        return clamp_0_1(effective_n / (effective_n + 20.0))

    def lower_bound(self, z_value: float) -> float:
        return clamp_0_1(
            self.posterior_mean - max(0.0, z_value) * math.sqrt(self.posterior_variance)
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def successes(self) -> float:
        return self.positive

    @property
    def failures(self) -> float:
        return self.total - self.positive

    @property
    def n(self) -> float:
        return self.total

    @property
    def mean(self) -> float:
        return self.posterior_mean

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.posterior_variance))

    @property
    def lcb(self) -> float:
        return self.lower_bound(1.96)

    @property
    def ucb(self) -> float:
        return clamp_0_1(
            self.posterior_mean + 1.96 * math.sqrt(max(0.0, self.posterior_variance))
        )

    # ------------------------------------------------------------------
    # Mutation helpers (return new frozen instances)
    # ------------------------------------------------------------------

    def update(self, success: bool) -> "TrustEvidence":
        """Return a new instance with one additional observation."""
        return TrustEvidence(
            dimension=self.dimension,
            positive=self.positive + (1.0 if success else 0.0),
            total=self.total + 1.0,
            weight=self.weight,
            prior_positive=self.prior_positive,
            prior_negative=self.prior_negative,
            notes=self.notes,
        )

    def decay(self, factor: float) -> "TrustEvidence":
        """Return a new instance with observation counts scaled by *factor*."""
        factor = clamp_0_1(factor)
        return TrustEvidence(
            dimension=self.dimension,
            positive=self.positive * factor,
            total=self.total * factor,
            weight=self.weight,
            prior_positive=self.prior_positive,
            prior_negative=self.prior_negative,
            notes=self.notes,
        )


@dataclass(frozen=True)
class DataSource:
    """A queryable source, model, index, tool, or memory shard."""

    name: str
    trust_score: float
    latency_ms: float
    cost_per_query: float
    freshness_score: float
    supports_vector: bool = False
    source_type: str = "source"
    capabilities: FrozenSet[str] | Iterable[str | Capability] = field(
        default_factory=frozenset
    )
    trust_evidence: Tuple[TrustEvidence, ...] | Iterable[TrustEvidence] = field(
        default_factory=tuple
    )
    coverage: float = 1.0
    privacy_risk: float = 0.0
    hallucination_risk: float = 0.0
    schema_reliability: float = 1.0
    metadata: Mapping[str, str] = field(default_factory=dict)
    reliability: float = 1.0
    evidence: TrustEvidence = field(default_factory=TrustEvidence)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trust_score", clamp_0_1(self.trust_score))
        object.__setattr__(self, "latency_ms", max(0.0, float(self.latency_ms)))
        object.__setattr__(
            self, "cost_per_query", max(0.0, float(self.cost_per_query))
        )
        object.__setattr__(self, "freshness_score", clamp_0_1(self.freshness_score))
        object.__setattr__(self, "coverage", clamp_0_1(self.coverage))
        object.__setattr__(self, "privacy_risk", clamp_0_1(self.privacy_risk))
        object.__setattr__(
            self, "hallucination_risk", clamp_0_1(self.hallucination_risk)
        )
        object.__setattr__(
            self, "schema_reliability", clamp_0_1(self.schema_reliability)
        )

        capabilities = set(normalize_capabilities(self.capabilities))
        if self.supports_vector:
            capabilities.add(Capability.VECTOR_SEARCH.value)
        object.__setattr__(self, "capabilities", frozenset(capabilities))
        object.__setattr__(self, "trust_evidence", tuple(self.trust_evidence))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "reliability", clamp_0_1(self.reliability))
        if not isinstance(self.evidence, TrustEvidence):
            object.__setattr__(self, "evidence", TrustEvidence())

    def supports_all(self, required_capabilities: Iterable[str | Capability]) -> bool:
        required = normalize_capabilities(required_capabilities)
        return required.issubset(self.capabilities)

    def missing_capabilities(
        self, required_capabilities: Iterable[str | Capability]
    ) -> FrozenSet[str]:
        required = normalize_capabilities(required_capabilities)
        return frozenset(required.difference(self.capabilities))

    def calibrated_trust(self) -> float:
        if not self.trust_evidence:
            return self.trust_score

        weighted_sum = self.trust_score * 0.25
        total_weight = 0.25
        for evidence in self.trust_evidence:
            weighted_sum += evidence.posterior_mean * evidence.weight
            total_weight += evidence.weight
        return clamp_0_1(weighted_sum / total_weight)

    def trust_lower_bound(self, risk_tolerance: float = 0.15) -> float:
        if not self.trust_evidence:
            return self.trust_score

        z_value = z_value_for_risk_tolerance(risk_tolerance)
        weighted_sum = self.trust_score * 0.25
        total_weight = 0.25
        for evidence in self.trust_evidence:
            weighted_sum += evidence.lower_bound(z_value) * evidence.weight
            total_weight += evidence.weight
        return clamp_0_1(weighted_sum / total_weight)

    def trust_confidence(self) -> float:
        if not self.trust_evidence:
            return 0.5

        weighted_sum = 0.0
        total_weight = 0.0
        for evidence in self.trust_evidence:
            weighted_sum += evidence.confidence * evidence.weight
            total_weight += evidence.weight
        if total_weight == 0:
            return 0.0
        return clamp_0_1(weighted_sum / total_weight)

    def with_evidence(self, evidence: TrustEvidence) -> "DataSource":
        from dataclasses import replace
        return replace(self, evidence=evidence)

    def effective_trust(self) -> float:
        n = self.evidence.n
        if n <= 0:
            return self.trust_score
        weight = min(1.0, n / (n + 20.0))
        return clamp_0_1((1.0 - weight) * self.trust_score + weight * self.evidence.mean)

    def effective_trust_lcb(self) -> float:
        n = self.evidence.n
        if n <= 0:
            return self.trust_score
        weight = min(1.0, n / (n + 20.0))
        return clamp_0_1((1.0 - weight) * self.trust_score + weight * self.evidence.lcb)


@dataclass(frozen=True)
class QueryRequest:
    """Optimization policy and hard constraints for a query."""

    requires_vector: bool = False
    max_latency_ms: float | None = None
    min_freshness_score: float = 0.0
    trust_weight: float = 0.5
    latency_weight: float = 0.2
    freshness_weight: float = 0.2
    cost_weight: float = 0.1
    query: str = ""
    required_capabilities: FrozenSet[str] | Iterable[str | Capability] = field(
        default_factory=frozenset
    )
    min_trust_score: float = 0.0
    max_cost_per_query: float | None = None
    max_privacy_risk: float | None = None
    max_hallucination_risk: float | None = None
    coverage_weight: float = 0.0
    risk_weight: float = 0.0
    uncertainty_weight: float = 0.0
    risk_tolerance: float = 0.15
    top_k: int | None = None
    use_conservative_trust: bool = False
    min_reliability: float = 0.0

    def __post_init__(self) -> None:
        capabilities = set(normalize_capabilities(self.required_capabilities))
        if self.requires_vector:
            capabilities.add(Capability.VECTOR_SEARCH.value)
        object.__setattr__(self, "required_capabilities", frozenset(capabilities))
        object.__setattr__(
            self, "min_freshness_score", clamp_0_1(self.min_freshness_score)
        )
        object.__setattr__(self, "min_trust_score", clamp_0_1(self.min_trust_score))
        object.__setattr__(self, "risk_tolerance", clamp_0_1(self.risk_tolerance))
        object.__setattr__(self, "min_reliability", clamp_0_1(self.min_reliability))
        if self.max_latency_ms is not None:
            object.__setattr__(
                self, "max_latency_ms", max(0.0, float(self.max_latency_ms))
            )
        if self.max_cost_per_query is not None:
            object.__setattr__(
                self, "max_cost_per_query", max(0.0, float(self.max_cost_per_query))
            )
        if self.max_privacy_risk is not None:
            object.__setattr__(
                self, "max_privacy_risk", clamp_0_1(self.max_privacy_risk)
            )
        if self.max_hallucination_risk is not None:
            object.__setattr__(
                self,
                "max_hallucination_risk",
                clamp_0_1(self.max_hallucination_risk),
            )
        if self.top_k is not None:
            object.__setattr__(self, "top_k", max(0, int(self.top_k)))


@dataclass(frozen=True)
class ScoreBreakdown:
    components: Mapping[str, float]
    weights: Mapping[str, float]
    weighted_score: float
    constraints: Tuple[str, ...] = ()

    def contribution(self, name: str) -> float:
        return self.components.get(name, 0.0) * self.weights.get(name, 0.0)

    @property
    def trust(self) -> float:
        return self.components.get("trust", 0.0)

    @property
    def latency(self) -> float:
        return self.components.get("latency", 0.0)

    @property
    def freshness(self) -> float:
        return self.components.get("freshness", 0.0)

    @property
    def cost(self) -> float:
        return self.components.get("cost", 0.0)


@dataclass(frozen=True)
class PlanStep:
    source: DataSource
    score: float
    breakdown: ScoreBreakdown
    is_pareto_optimal: bool
    rationale: Tuple[str, ...]


@dataclass(frozen=True)
class RejectedSource:
    source: DataSource
    reasons: Tuple[str, ...]


class ExecutionStrategy(str, Enum):
    SINGLE = "single"
    ENSEMBLE = "ensemble"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class QueryPlan:
    ranked_sources: List[Tuple[DataSource, float]]
    steps: Tuple[PlanStep, ...] = ()
    rejected_sources: Tuple[RejectedSource, ...] = ()
    objective: str = "trust_aware_expected_utility"
    confidence: float = 0.0
    pareto_front: Tuple[DataSource, ...] = ()
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SINGLE
    fallback_chain: Tuple[DataSource, ...] = ()
    breakdowns: Tuple[ScoreBreakdown, ...] = ()

    @property
    def primary_source(self) -> DataSource | None:
        if not self.ranked_sources:
            return None
        return self.ranked_sources[0][0]

    @property
    def pareto_sources(self) -> Tuple[DataSource, ...]:
        return tuple(step.source for step in self.steps if step.is_pareto_optimal)

    def explain(self, top: int = 3) -> str:
        if not self.steps:
            if self.rejected_sources:
                reasons = "; ".join(
                    f"{item.source.name}: {', '.join(item.reasons)}"
                    for item in self.rejected_sources[:top]
                )
                return f"No admissible sources. Rejections: {reasons}"
            return "No admissible sources."

        lines = [f"Objective: {self.objective}"]
        for index, step in enumerate(self.steps[:top], start=1):
            components = ", ".join(
                f"{name}={value:.3f}"
                for name, value in sorted(step.breakdown.components.items())
            )
            marker = "pareto" if step.is_pareto_optimal else "dominated"
            lines.append(
                f"{index}. {step.source.name} score={step.score:.3f} "
                f"({marker}; {components})"
            )
            if step.rationale:
                lines.append(f"   rationale: {'; '.join(step.rationale)}")
        if self.rejected_sources:
            lines.append(f"Rejected {len(self.rejected_sources)} source(s).")
        return "\n".join(lines)


def _source_dominates(a: DataSource, b: DataSource) -> bool:
    at_least_as_good = (
        a.trust_score >= b.trust_score
        and a.latency_ms <= b.latency_ms
        and a.cost_per_query <= b.cost_per_query
        and a.freshness_score >= b.freshness_score
    )
    strictly_better = (
        a.trust_score > b.trust_score
        or a.latency_ms < b.latency_ms
        or a.cost_per_query < b.cost_per_query
        or a.freshness_score > b.freshness_score
    )
    return at_least_as_good and strictly_better


def compute_pareto_front(sources: Iterable[DataSource]) -> List[DataSource]:
    """Return sources not dominated by any other on key quality axes."""
    sources_list = list(sources)
    result = []
    for candidate in sources_list:
        dominated = False
        for other in sources_list:
            if other is candidate:
                continue
            if _source_dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            result.append(candidate)
    return result
