from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Tuple

from .models import (
    DataSource,
    ExecutionStrategy,
    QueryPlan,
    QueryRequest,
    PlanStep,
    RejectedSource,
    ScoreBreakdown,
    clamp_0_1,
)


@dataclass(frozen=True)
class _Candidate:
    source: DataSource
    breakdown: ScoreBreakdown
    rationale: Tuple[str, ...]


class TrustAwareQueryOptimizer:
    """Constraint-aware optimizer for AI-native data systems.

    The optimizer treats trust as a first-class planning dimension. It filters
    unsafe or incompatible sources, scores the remaining candidates with a
    calibrated trust lower bound, and returns explainable rankings with Pareto
    annotations for downstream policy inspection.
    """

    def __init__(self, strategy: str = "linear") -> None:
        if strategy not in {"linear", "topsis", "bayesian-ucb"}:
            raise ValueError(
                f"unknown strategy {strategy!r}; choose linear, topsis, or bayesian-ucb"
            )
        self._strategy_name = strategy
        self._scorer = self._make_scorer(strategy)

    def _make_scorer(self, strategy: str):
        from .scoring import LinearWeightedScorer, TOPSISScorer, BayesianUCBScorer
        return {
            "linear": LinearWeightedScorer,
            "topsis": TOPSISScorer,
            "bayesian-ucb": BayesianUCBScorer,
        }[strategy]()

    def optimize(
        self, sources: Iterable[DataSource], request: QueryRequest
    ) -> QueryPlan:
        candidates: List[_Candidate] = []
        rejected: List[RejectedSource] = []

        for source in sources:
            rejection_reasons = self._reject_reasons(source, request)
            if rejection_reasons:
                rejected.append(RejectedSource(source=source, reasons=rejection_reasons))
                continue

            breakdown = self._score(source, request)
            rationale = self._rationale(source, breakdown)
            candidates.append(
                _Candidate(source=source, breakdown=breakdown, rationale=rationale)
            )

        # Apply strategy scorer to override weighted_score for ranking
        if self._strategy_name != "linear" and candidates:
            candidate_sources = [c.source for c in candidates]
            new_candidates = []
            for c in candidates:
                strat_score, _ = self._scorer.score(
                    c.source, request, all_sources=candidate_sources
                )
                new_bd = ScoreBreakdown(
                    components=c.breakdown.components,
                    weights=c.breakdown.weights,
                    weighted_score=strat_score,
                    constraints=c.breakdown.constraints,
                )
                new_candidates.append(
                    _Candidate(source=c.source, breakdown=new_bd, rationale=c.rationale)
                )
            candidates = new_candidates

        candidates.sort(
            key=lambda candidate: (
                -candidate.breakdown.weighted_score,
                -candidate.breakdown.components["trust"],
                candidate.source.latency_ms,
                candidate.source.cost_per_query,
                candidate.source.name,
            )
        )

        if request.top_k is not None:
            candidates = candidates[: request.top_k]

        pareto_names = self._pareto_optimal_names(candidates)
        steps = tuple(
            PlanStep(
                source=candidate.source,
                score=candidate.breakdown.weighted_score,
                breakdown=candidate.breakdown,
                is_pareto_optimal=candidate.source.name in pareto_names,
                rationale=candidate.rationale,
            )
            for candidate in candidates
        )
        ranked_sources = [
            (candidate.source, candidate.breakdown.weighted_score)
            for candidate in candidates
        ]

        # Compute QueryPlan metadata
        confidence = steps[0].breakdown.components.get("confidence", 0.0) if steps else 0.0
        pareto_front = tuple(c.source for c in candidates if c.source.name in pareto_names)

        if len(steps) < 2:
            exec_strategy = ExecutionStrategy.SINGLE
        elif abs(steps[0].score - steps[1].score) < 0.01:
            exec_strategy = ExecutionStrategy.ENSEMBLE
        else:
            exec_strategy = ExecutionStrategy.FALLBACK

        fallback_chain = tuple(c.source for c in candidates[1:])
        breakdowns = tuple(step.breakdown for step in steps)

        return QueryPlan(
            ranked_sources=ranked_sources,
            steps=steps,
            rejected_sources=tuple(rejected),
            confidence=confidence,
            pareto_front=pareto_front,
            execution_strategy=exec_strategy,
            fallback_chain=fallback_chain,
            breakdowns=breakdowns,
        )

    def _reject_reasons(
        self, source: DataSource, request: QueryRequest
    ) -> Tuple[str, ...]:
        reasons: List[str] = []

        missing_capabilities = source.missing_capabilities(request.required_capabilities)
        if missing_capabilities:
            missing = ", ".join(sorted(missing_capabilities))
            reasons.append(f"missing required capabilities: {missing}")

        if source.freshness_score < request.min_freshness_score:
            reasons.append(
                "freshness "
                f"{source.freshness_score:.3f} < {request.min_freshness_score:.3f}"
            )

        if (
            request.max_latency_ms is not None
            and source.latency_ms > request.max_latency_ms
        ):
            reasons.append(
                f"latency {source.latency_ms:.1f}ms > {request.max_latency_ms:.1f}ms"
            )

        trust_lower_bound = source.trust_lower_bound(request.risk_tolerance)
        if trust_lower_bound < request.min_trust_score:
            reasons.append(
                "trust lower bound "
                f"{trust_lower_bound:.3f} < {request.min_trust_score:.3f}"
            )

        if (
            request.max_cost_per_query is not None
            and source.cost_per_query > request.max_cost_per_query
        ):
            reasons.append(
                "cost "
                f"{source.cost_per_query:.3f} > {request.max_cost_per_query:.3f}"
            )

        if (
            request.max_privacy_risk is not None
            and source.privacy_risk > request.max_privacy_risk
        ):
            reasons.append(
                "privacy risk "
                f"{source.privacy_risk:.3f} > {request.max_privacy_risk:.3f}"
            )

        if (
            request.max_hallucination_risk is not None
            and source.hallucination_risk > request.max_hallucination_risk
        ):
            reasons.append(
                "hallucination risk "
                f"{source.hallucination_risk:.3f} > "
                f"{request.max_hallucination_risk:.3f}"
            )

        if (
            request.min_reliability > 0.0
            and source.reliability < request.min_reliability
        ):
            reasons.append(
                f"reliability {source.reliability:.3f} < {request.min_reliability:.3f}"
            )

        return tuple(reasons)

    def _score(self, source: DataSource, request: QueryRequest) -> ScoreBreakdown:
        components = {
            "trust": source.trust_lower_bound(request.risk_tolerance),
            "latency": self._latency_component(source, request),
            "freshness": clamp_0_1(source.freshness_score),
            "cost": 1.0 / (1.0 + source.cost_per_query),
            "coverage": clamp_0_1(source.coverage),
            "risk": 1.0 - max(source.privacy_risk, source.hallucination_risk),
            "schema": clamp_0_1(source.schema_reliability),
            "confidence": source.trust_confidence(),
        }
        components["risk"] = clamp_0_1(components["risk"])

        weights = {
            "trust": max(0.0, request.trust_weight),
            "latency": max(0.0, request.latency_weight),
            "freshness": max(0.0, request.freshness_weight),
            "cost": max(0.0, request.cost_weight),
            "coverage": max(0.0, request.coverage_weight),
            "risk": max(0.0, request.risk_weight),
            "uncertainty": max(0.0, request.uncertainty_weight),
        }

        positive_score = sum(
            weights[name] * components[name]
            for name in ("trust", "latency", "freshness", "cost", "coverage", "risk")
        )
        uncertainty_penalty = weights["uncertainty"] * (1.0 - components["confidence"])
        weighted_score = max(0.0, positive_score - uncertainty_penalty)

        constraints = tuple(self._constraint_summary(source, request))
        return ScoreBreakdown(
            components=components,
            weights=weights,
            weighted_score=weighted_score,
            constraints=constraints,
        )

    def _latency_component(self, source: DataSource, request: QueryRequest) -> float:
        if request.max_latency_ms is None or request.max_latency_ms <= 0:
            return 1.0

        latency_ratio = source.latency_ms / request.max_latency_ms
        return clamp_0_1(1.0 - latency_ratio)

    def _constraint_summary(
        self, source: DataSource, request: QueryRequest
    ) -> Iterable[str]:
        if request.required_capabilities:
            capabilities = ", ".join(sorted(request.required_capabilities))
            yield f"capabilities include {capabilities}"
        if request.max_latency_ms is not None:
            yield f"latency <= {request.max_latency_ms:.1f}ms"
        if request.min_freshness_score > 0:
            yield f"freshness >= {request.min_freshness_score:.3f}"
        if request.min_trust_score > 0:
            yield f"trust lower bound >= {request.min_trust_score:.3f}"
        if request.max_cost_per_query is not None:
            yield f"cost <= {request.max_cost_per_query:.3f}"
        if request.max_privacy_risk is not None:
            yield f"privacy risk <= {request.max_privacy_risk:.3f}"
        if request.max_hallucination_risk is not None:
            yield f"hallucination risk <= {request.max_hallucination_risk:.3f}"

    def _rationale(
        self, source: DataSource, breakdown: ScoreBreakdown
    ) -> Tuple[str, ...]:
        contributions = {
            name: breakdown.contribution(name)
            for name in ("trust", "latency", "freshness", "cost", "coverage", "risk")
        }
        leaders = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
        rationale = [
            f"{name} contribution {value:.3f}"
            for name, value in leaders[:3]
            if value > 0
        ]
        if source.trust_evidence:
            rationale.append(
                f"calibrated from {len(source.trust_evidence)} evidence stream(s)"
            )
        return tuple(rationale)

    def _pareto_optimal_names(self, candidates: Iterable[_Candidate]) -> set[str]:
        candidate_list = list(candidates)
        pareto_names: set[str] = set()

        for candidate in candidate_list:
            dominated = False
            for other in candidate_list:
                if other is candidate:
                    continue
                if self._dominates(other.breakdown.components, candidate.breakdown.components):
                    dominated = True
                    break
            if not dominated:
                pareto_names.add(candidate.source.name)

        return pareto_names

    def _dominates(
        self, left_components: Mapping[str, float], right_components: Mapping[str, float]
    ) -> bool:
        axes = ("trust", "latency", "freshness", "cost", "coverage", "risk", "schema")
        at_least_as_good = all(
            left_components[axis] >= right_components[axis] for axis in axes
        )
        strictly_better = any(
            left_components[axis] > right_components[axis] for axis in axes
        )
        return at_least_as_good and strictly_better
