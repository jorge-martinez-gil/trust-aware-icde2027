from __future__ import annotations

import math
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Iterable, Sequence, Tuple

from .models import (
    DataSource,
    QueryPlan,
    QueryRequest,
    RejectedSource,
    ScoreBreakdown,
    clamp_0_1,
)
from .optimizer import TrustAwareQueryOptimizer


@dataclass(frozen=True)
class PortfolioBudget:
    """Budget and robustness controls for redundant source selection."""

    min_sources: int = 1
    max_sources: int = 3
    max_total_latency_ms: float | None = None
    max_total_cost: float | None = None
    min_portfolio_trust: float = 0.0
    top_k: int = 5
    latency_model: str = "parallel"
    correlation_penalty: float = 0.35
    source_penalty: float = 0.015
    candidate_limit: int = 18

    def __post_init__(self) -> None:
        if self.latency_model not in {"parallel", "sequential"}:
            raise ValueError("latency_model must be 'parallel' or 'sequential'")
        min_sources = max(1, int(self.min_sources))
        max_sources = max(min_sources, int(self.max_sources))
        object.__setattr__(self, "min_sources", min_sources)
        object.__setattr__(self, "max_sources", max_sources)
        object.__setattr__(self, "top_k", max(0, int(self.top_k)))
        object.__setattr__(self, "candidate_limit", max(1, int(self.candidate_limit)))
        object.__setattr__(
            self, "min_portfolio_trust", clamp_0_1(self.min_portfolio_trust)
        )
        object.__setattr__(
            self, "correlation_penalty", clamp_0_1(self.correlation_penalty)
        )
        object.__setattr__(self, "source_penalty", max(0.0, float(self.source_penalty)))
        if self.max_total_latency_ms is not None:
            object.__setattr__(
                self,
                "max_total_latency_ms",
                max(0.0, float(self.max_total_latency_ms)),
            )
        if self.max_total_cost is not None:
            object.__setattr__(
                self, "max_total_cost", max(0.0, float(self.max_total_cost))
            )


@dataclass(frozen=True)
class SourcePortfolio:
    """A selected set of sources with portfolio-level trust accounting."""

    sources: Tuple[DataSource, ...]
    source_scores: Tuple[float, ...]
    score: float
    breakdown: ScoreBreakdown
    latency_ms: float
    cost_per_query: float
    rationale: Tuple[str, ...]

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(source.name for source in self.sources)

    @property
    def trust_lower_bound(self) -> float:
        return self.breakdown.components["trust"]

    @property
    def coverage(self) -> float:
        return self.breakdown.components["coverage"]


@dataclass(frozen=True)
class PortfolioPlan:
    """Ranked redundant source portfolios for a single trust-aware request."""

    ranked_portfolios: Tuple[SourcePortfolio, ...]
    base_plan: QueryPlan
    budget: PortfolioBudget
    candidate_count: int
    evaluated_count: int
    budget_rejected_count: int
    objective: str = "trust_aware_portfolio_expected_utility"

    @property
    def primary_portfolio(self) -> SourcePortfolio | None:
        if not self.ranked_portfolios:
            return None
        return self.ranked_portfolios[0]

    @property
    def selected_sources(self) -> Tuple[DataSource, ...]:
        if self.primary_portfolio is None:
            return ()
        return self.primary_portfolio.sources

    @property
    def rejected_sources(self) -> Tuple[RejectedSource, ...]:
        return self.base_plan.rejected_sources

    def explain(self, top: int = 3) -> str:
        if not self.ranked_portfolios:
            return (
                "No admissible portfolios. "
                f"candidates={self.candidate_count} "
                f"evaluated={self.evaluated_count} "
                f"budget_rejected={self.budget_rejected_count}"
            )

        lines = [f"Objective: {self.objective}"]
        for index, portfolio in enumerate(self.ranked_portfolios[:top], start=1):
            names = " + ".join(portfolio.names)
            components = ", ".join(
                f"{name}={value:.3f}"
                for name, value in sorted(portfolio.breakdown.components.items())
            )
            lines.append(
                f"{index}. {names} score={portfolio.score:.3f} "
                f"latency_ms={portfolio.latency_ms:.1f} "
                f"cost={portfolio.cost_per_query:.3f} ({components})"
            )
            if portfolio.rationale:
                lines.append(f"   rationale: {'; '.join(portfolio.rationale)}")
        if self.budget_rejected_count:
            lines.append(
                f"Rejected {self.budget_rejected_count} portfolio(s) by budget."
            )
        if self.rejected_sources:
            lines.append(
                f"Rejected {len(self.rejected_sources)} source(s) before portfolios."
            )
        return "\n".join(lines)


class TrustAwarePortfolioPlanner:
    """Build budgeted source portfolios from trust-aware single-source rankings.

    The planner enumerates small source sets from admissible optimizer outputs and
    scores them with a redundancy-aware trust model. This captures high-stakes
    retrieval patterns where a plan should triangulate evidence rather than trust
    one source blindly.
    """

    def __init__(self, optimizer: TrustAwareQueryOptimizer | None = None) -> None:
        self.optimizer = optimizer or TrustAwareQueryOptimizer()

    def plan(
        self,
        sources: Iterable[DataSource],
        request: QueryRequest,
        budget: PortfolioBudget | None = None,
    ) -> PortfolioPlan:
        budget = budget or PortfolioBudget()
        base_plan = self.optimizer.optimize(sources, replace(request, top_k=None))
        steps = base_plan.steps[: budget.candidate_limit]

        portfolios: list[SourcePortfolio] = []
        evaluated_count = 0
        budget_rejected_count = 0
        max_size = min(budget.max_sources, len(steps))

        for size in range(budget.min_sources, max_size + 1):
            for step_tuple in combinations(steps, size):
                evaluated_count += 1
                candidate = self._build_portfolio(step_tuple, request, budget)
                if self._violates_budget(candidate, budget):
                    budget_rejected_count += 1
                    continue
                portfolios.append(candidate)

        portfolios.sort(
            key=lambda portfolio: (
                -portfolio.score,
                -portfolio.trust_lower_bound,
                portfolio.latency_ms,
                portfolio.cost_per_query,
                portfolio.names,
            )
        )
        ranked_portfolios = tuple(portfolios[: budget.top_k])
        return PortfolioPlan(
            ranked_portfolios=ranked_portfolios,
            base_plan=base_plan,
            budget=budget,
            candidate_count=len(steps),
            evaluated_count=evaluated_count,
            budget_rejected_count=budget_rejected_count,
        )

    def _build_portfolio(
        self,
        steps: Sequence,
        request: QueryRequest,
        budget: PortfolioBudget,
    ) -> SourcePortfolio:
        sources = tuple(step.source for step in steps)
        source_scores = tuple(step.score for step in steps)
        latency_ms = _portfolio_latency(sources, budget.latency_model)
        total_cost = sum(source.cost_per_query for source in sources)
        trust_values = [step.breakdown.components["trust"] for step in steps]
        confidence_values = [
            step.breakdown.components.get("confidence", source.trust_confidence())
            for step, source in zip(steps, sources)
        ]

        components = {
            "trust": _redundant_probability(
                trust_values, correlation_penalty=budget.correlation_penalty
            ),
            "latency": self._latency_component(latency_ms, request, budget),
            "freshness": _mean(source.freshness_score for source in sources),
            "cost": 1.0 / (1.0 + total_cost),
            "coverage": _redundant_probability(
                [source.coverage for source in sources],
                correlation_penalty=budget.correlation_penalty,
            ),
            "risk": 1.0
            - max(
                max(source.privacy_risk, source.hallucination_risk)
                for source in sources
            ),
            "schema": min(source.schema_reliability for source in sources),
            "confidence": _mean(confidence_values),
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
        redundancy_penalty = budget.source_penalty * max(0, len(sources) - 1)
        score = max(0.0, positive_score - uncertainty_penalty - redundancy_penalty)
        constraints = tuple(_portfolio_constraints(request, budget))
        breakdown = ScoreBreakdown(
            components=components,
            weights=weights,
            weighted_score=score,
            constraints=constraints,
        )
        rationale = self._rationale(sources, breakdown, budget)
        return SourcePortfolio(
            sources=sources,
            source_scores=source_scores,
            score=score,
            breakdown=breakdown,
            latency_ms=latency_ms,
            cost_per_query=total_cost,
            rationale=rationale,
        )

    def _latency_component(
        self,
        latency_ms: float,
        request: QueryRequest,
        budget: PortfolioBudget,
    ) -> float:
        latency_limit = budget.max_total_latency_ms
        if latency_limit is None:
            latency_limit = request.max_latency_ms
        if latency_limit is None or latency_limit <= 0:
            return 1.0
        return clamp_0_1(1.0 - latency_ms / latency_limit)

    def _violates_budget(
        self, portfolio: SourcePortfolio, budget: PortfolioBudget
    ) -> bool:
        if (
            budget.max_total_latency_ms is not None
            and portfolio.latency_ms > budget.max_total_latency_ms
        ):
            return True
        if (
            budget.max_total_cost is not None
            and portfolio.cost_per_query > budget.max_total_cost
        ):
            return True
        return portfolio.trust_lower_bound < budget.min_portfolio_trust

    def _rationale(
        self,
        sources: Tuple[DataSource, ...],
        breakdown: ScoreBreakdown,
        budget: PortfolioBudget,
    ) -> Tuple[str, ...]:
        names = ", ".join(source.name for source in sources)
        contributions = {
            name: breakdown.contribution(name)
            for name in ("trust", "latency", "freshness", "cost", "coverage", "risk")
        }
        leaders = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
        rationale = [
            f"portfolio over {len(sources)} source(s): {names}",
            (
                "trust assumes "
                f"{budget.correlation_penalty:.2f} correlated evidence penalty"
            ),
        ]
        rationale.extend(
            f"{name} contribution {value:.3f}"
            for name, value in leaders[:2]
            if value > 0
        )
        return tuple(rationale)


def _portfolio_latency(sources: Sequence[DataSource], latency_model: str) -> float:
    if latency_model == "sequential":
        return sum(source.latency_ms for source in sources)
    return max(source.latency_ms for source in sources)


def _redundant_probability(
    values: Iterable[float],
    *,
    correlation_penalty: float,
) -> float:
    probabilities = [clamp_0_1(value) for value in values]
    if not probabilities:
        return 0.0
    independent_failure = math.prod(1.0 - value for value in probabilities)
    independent_success = 1.0 - independent_failure
    best_single = max(probabilities)
    discount = 1.0 - clamp_0_1(correlation_penalty)
    return clamp_0_1(best_single + discount * (independent_success - best_single))


def _mean(values: Iterable[float]) -> float:
    values = tuple(values)
    if not values:
        return 0.0
    return clamp_0_1(sum(values) / len(values))


def _portfolio_constraints(
    request: QueryRequest, budget: PortfolioBudget
) -> Iterable[str]:
    if request.required_capabilities:
        capabilities = ", ".join(sorted(request.required_capabilities))
        yield f"source capabilities include {capabilities}"
    if budget.max_total_latency_ms is not None:
        yield f"portfolio latency <= {budget.max_total_latency_ms:.1f}ms"
    if budget.max_total_cost is not None:
        yield f"portfolio cost <= {budget.max_total_cost:.3f}"
    if budget.min_portfolio_trust > 0:
        yield f"portfolio trust lower bound >= {budget.min_portfolio_trust:.3f}"
    yield f"portfolio latency model = {budget.latency_model}"
