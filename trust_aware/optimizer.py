from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


def _clamp_0_1(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class DataSource:
    name: str
    trust_score: float
    latency_ms: float
    cost_per_query: float
    freshness_score: float
    supports_vector: bool = False


@dataclass(frozen=True)
class QueryRequest:
    requires_vector: bool = False
    max_latency_ms: float | None = None
    min_freshness_score: float = 0.0
    trust_weight: float = 0.5
    latency_weight: float = 0.2
    freshness_weight: float = 0.2
    cost_weight: float = 0.1


@dataclass(frozen=True)
class QueryPlan:
    ranked_sources: List[Tuple[DataSource, float]]

    @property
    def primary_source(self) -> DataSource | None:
        if not self.ranked_sources:
            return None
        return self.ranked_sources[0][0]


class TrustAwareQueryOptimizer:
    """Ranks candidate data sources for AI-native query execution."""

    def optimize(
        self, sources: Iterable[DataSource], request: QueryRequest
    ) -> QueryPlan:
        candidates = []
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

            trust_component = _clamp_0_1(source.trust_score)
            freshness_component = _clamp_0_1(source.freshness_score)

            if request.max_latency_ms is None or request.max_latency_ms <= 0:
                latency_component = 1.0
            else:
                latency_ratio = source.latency_ms / request.max_latency_ms
                latency_component = _clamp_0_1(1.0 - latency_ratio)

            cost_component = 1.0 / (1.0 + max(0.0, source.cost_per_query))

            score = (
                request.trust_weight * trust_component
                + request.latency_weight * latency_component
                + request.freshness_weight * freshness_component
                + request.cost_weight * cost_component
            )
            candidates.append((source, score))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return QueryPlan(ranked_sources=candidates)
