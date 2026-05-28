from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Tuple

from .models import Capability, DataSource, QueryPlan, QueryRequest
from .optimizer import TrustAwareQueryOptimizer
from .policies import TrustPolicy


@dataclass(frozen=True)
class QueryStage:
    """One logical operator in an AI-native query plan."""

    name: str
    required_capabilities: Tuple[str | Capability, ...]
    max_latency_ms: float | None = None
    min_freshness_score: float = 0.0
    min_trust_score: float = 0.0
    max_privacy_risk: float | None = None
    max_hallucination_risk: float | None = None


@dataclass(frozen=True)
class PipelineStep:
    stage: QueryStage
    plan: QueryPlan

    @property
    def selected_source(self) -> DataSource | None:
        return self.plan.primary_source


@dataclass(frozen=True)
class PipelinePlan:
    query: str
    steps: Tuple[PipelineStep, ...]

    @property
    def is_complete(self) -> bool:
        return all(step.selected_source is not None for step in self.steps)

    @property
    def selected_sources(self) -> Tuple[DataSource, ...]:
        return tuple(
            step.selected_source
            for step in self.steps
            if step.selected_source is not None
        )

    @property
    def total_latency_ms(self) -> float:
        return sum(source.latency_ms for source in self.selected_sources)

    @property
    def total_cost(self) -> float:
        return sum(source.cost_per_query for source in self.selected_sources)

    @property
    def trust_floor(self) -> float:
        if not self.steps or not self.selected_sources:
            return 0.0
        return min(
            step.plan.steps[0].breakdown.components["trust"]
            for step in self.steps
            if step.plan.steps
        )

    def explain(self) -> str:
        lines = [f"Pipeline query: {self.query}"]
        for index, step in enumerate(self.steps, start=1):
            selected = step.selected_source.name if step.selected_source else "none"
            lines.append(f"{index}. {step.stage.name}: {selected}")
            if step.plan.steps:
                lines.append(
                    f"   score={step.plan.steps[0].score:.3f} "
                    f"trust_lb={step.plan.steps[0].breakdown.components['trust']:.3f}"
                )
            elif step.plan.rejected_sources:
                lines.append(
                    f"   rejected={len(step.plan.rejected_sources)} candidate(s)"
                )
        lines.append(
            "complete="
            f"{self.is_complete} latency_ms={self.total_latency_ms:.1f} "
            f"cost={self.total_cost:.3f} trust_floor={self.trust_floor:.3f}"
        )
        return "\n".join(lines)


class TrustAwarePipelinePlanner:
    """Plans multi-stage AI query workflows with a shared trust policy."""

    def __init__(self, optimizer: TrustAwareQueryOptimizer | None = None) -> None:
        self.optimizer = optimizer or TrustAwareQueryOptimizer()

    def plan(
        self,
        query: str,
        stages: Iterable[QueryStage],
        sources: Iterable[DataSource],
        policy: TrustPolicy,
    ) -> PipelinePlan:
        source_tuple = tuple(sources)
        steps = []
        for stage in stages:
            request = policy.request(
                query=f"{query} :: {stage.name}",
                required_capabilities=stage.required_capabilities,
                max_latency_ms=stage.max_latency_ms,
                min_freshness_score=stage.min_freshness_score,
            )
            request = replace(
                request,
                min_trust_score=max(request.min_trust_score, stage.min_trust_score),
                max_privacy_risk=_stricter_ceiling(
                    request.max_privacy_risk, stage.max_privacy_risk
                ),
                max_hallucination_risk=_stricter_ceiling(
                    request.max_hallucination_risk,
                    stage.max_hallucination_risk,
                ),
            )
            steps.append(
                PipelineStep(
                    stage=stage,
                    plan=self.optimizer.optimize(source_tuple, request),
                )
            )
        return PipelinePlan(query=query, steps=tuple(steps))


def _stricter_ceiling(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right)
