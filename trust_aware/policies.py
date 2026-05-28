from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from .models import Capability, QueryRequest


@dataclass(frozen=True)
class TrustPolicy:
    """Named optimizer profile for repeatable experiments and deployments."""

    name: str
    description: str
    trust_weight: float
    latency_weight: float
    freshness_weight: float
    cost_weight: float
    coverage_weight: float = 0.0
    risk_weight: float = 0.0
    uncertainty_weight: float = 0.0
    min_trust_score: float = 0.0
    max_privacy_risk: float | None = None
    max_hallucination_risk: float | None = None
    risk_tolerance: float = 0.15

    def request(
        self,
        query: str = "",
        required_capabilities: Iterable[str | Capability] = (),
        *,
        requires_vector: bool = False,
        max_latency_ms: float | None = None,
        min_freshness_score: float = 0.0,
        max_cost_per_query: float | None = None,
        top_k: int | None = None,
    ) -> QueryRequest:
        return QueryRequest(
            query=query,
            required_capabilities=required_capabilities,
            requires_vector=requires_vector,
            max_latency_ms=max_latency_ms,
            min_freshness_score=min_freshness_score,
            max_cost_per_query=max_cost_per_query,
            min_trust_score=self.min_trust_score,
            max_privacy_risk=self.max_privacy_risk,
            max_hallucination_risk=self.max_hallucination_risk,
            trust_weight=self.trust_weight,
            latency_weight=self.latency_weight,
            freshness_weight=self.freshness_weight,
            cost_weight=self.cost_weight,
            coverage_weight=self.coverage_weight,
            risk_weight=self.risk_weight,
            uncertainty_weight=self.uncertainty_weight,
            risk_tolerance=self.risk_tolerance,
            top_k=top_k,
        )

    def apply(self, request: QueryRequest) -> QueryRequest:
        return replace(
            request,
            trust_weight=self.trust_weight,
            latency_weight=self.latency_weight,
            freshness_weight=self.freshness_weight,
            cost_weight=self.cost_weight,
            coverage_weight=self.coverage_weight,
            risk_weight=self.risk_weight,
            uncertainty_weight=self.uncertainty_weight,
            min_trust_score=max(request.min_trust_score, self.min_trust_score),
            max_privacy_risk=_stricter_ceiling(
                request.max_privacy_risk, self.max_privacy_risk
            ),
            max_hallucination_risk=_stricter_ceiling(
                request.max_hallucination_risk, self.max_hallucination_risk
            ),
            risk_tolerance=min(request.risk_tolerance, self.risk_tolerance),
        )


def _stricter_ceiling(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)


HIGH_ASSURANCE = TrustPolicy(
    name="high-assurance",
    description="Prioritize calibrated trust, evidence confidence, and risk control.",
    trust_weight=0.56,
    latency_weight=0.08,
    freshness_weight=0.11,
    cost_weight=0.04,
    coverage_weight=0.08,
    risk_weight=0.13,
    uncertainty_weight=0.07,
    min_trust_score=0.72,
    max_privacy_risk=0.22,
    max_hallucination_risk=0.25,
    risk_tolerance=0.05,
)

BALANCED = TrustPolicy(
    name="balanced",
    description="Balance trust, operational cost, latency, and coverage.",
    trust_weight=0.43,
    latency_weight=0.15,
    freshness_weight=0.14,
    cost_weight=0.10,
    coverage_weight=0.09,
    risk_weight=0.09,
    uncertainty_weight=0.04,
    min_trust_score=0.58,
    max_privacy_risk=0.40,
    max_hallucination_risk=0.40,
    risk_tolerance=0.10,
)

LATENCY_CRITICAL = TrustPolicy(
    name="latency-critical",
    description="Favor fast admissible sources while retaining basic trust gates.",
    trust_weight=0.22,
    latency_weight=0.50,
    freshness_weight=0.08,
    cost_weight=0.08,
    coverage_weight=0.04,
    risk_weight=0.08,
    uncertainty_weight=0.02,
    min_trust_score=0.45,
    max_privacy_risk=0.45,
    max_hallucination_risk=0.45,
    risk_tolerance=0.20,
)

COST_EFFICIENT = TrustPolicy(
    name="cost-efficient",
    description="Ablation-style profile that strongly rewards low cost.",
    trust_weight=0.12,
    latency_weight=0.18,
    freshness_weight=0.08,
    cost_weight=0.52,
    coverage_weight=0.03,
    risk_weight=0.07,
    uncertainty_weight=0.01,
    min_trust_score=0.0,
    max_privacy_risk=0.55,
    max_hallucination_risk=0.55,
    risk_tolerance=0.25,
)

TRUST_ONLY = TrustPolicy(
    name="trust-only",
    description="Ablation profile that ranks by calibrated trust with light gates.",
    trust_weight=0.90,
    latency_weight=0.00,
    freshness_weight=0.05,
    cost_weight=0.00,
    coverage_weight=0.00,
    risk_weight=0.05,
    uncertainty_weight=0.05,
    min_trust_score=0.0,
    risk_tolerance=0.05,
)

DEFAULT_POLICIES: Mapping[str, TrustPolicy] = {
    policy.name: policy
    for policy in (
        HIGH_ASSURANCE,
        BALANCED,
        LATENCY_CRITICAL,
        COST_EFFICIENT,
        TRUST_ONLY,
    )
}


def get_policy(name: str) -> TrustPolicy:
    try:
        return DEFAULT_POLICIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(DEFAULT_POLICIES))
        raise KeyError(f"unknown policy {name!r}; available policies: {available}") from exc
