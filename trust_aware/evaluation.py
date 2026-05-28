from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Mapping, Sequence

from .benchmark import BenchmarkResult, evaluate_policy, synthetic_sources, synthetic_workload
from .models import DataSource, QueryRequest
from .policies import BALANCED, COST_EFFICIENT, HIGH_ASSURANCE, LATENCY_CRITICAL, TRUST_ONLY, TrustPolicy
from .optimizer import TrustAwareQueryOptimizer


@dataclass(frozen=True)
class StudyResult:
    policy_name: str
    seeds: tuple[int, ...]
    average_accepted: float
    average_score: float
    average_trust: float
    average_latency_ms: float
    average_cost: float
    average_rejection_rate: float

    def to_markdown_row(self) -> str:
        return (
            f"| {self.policy_name} | {len(self.seeds)} | "
            f"{self.average_accepted:.1f} | {self.average_score:.3f} | "
            f"{self.average_trust:.3f} | {self.average_latency_ms:.1f} | "
            f"{self.average_cost:.3f} | {self.average_rejection_rate:.3f} |"
        )


@dataclass(frozen=True)
class EvaluationReport:
    results: tuple[StudyResult, ...]

    def winner_by(self, metric: str) -> StudyResult:
        if metric in {"average_latency_ms", "average_cost", "average_rejection_rate"}:
            return min(self.results, key=lambda result: getattr(result, metric))
        return max(self.results, key=lambda result: getattr(result, metric))

    def to_markdown(self) -> str:
        lines = [
            "# Trust-Aware Optimization Study",
            "",
            "| policy | seeds | accepted | score | trust | latency_ms | cost | rejected |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        lines.extend(result.to_markdown_row() for result in self.results)
        lines.extend(
            [
                "",
                "## Winners",
                f"- Utility: `{self.winner_by('average_score').policy_name}`",
                f"- Trust: `{self.winner_by('average_trust').policy_name}`",
                f"- Latency: `{self.winner_by('average_latency_ms').policy_name}`",
                f"- Cost: `{self.winner_by('average_cost').policy_name}`",
            ]
        )
        return "\n".join(lines)


def compare_policies(
    sources: Sequence[DataSource],
    workload: Iterable[QueryRequest],
    policies: Iterable[TrustPolicy],
    optimizer: TrustAwareQueryOptimizer | None = None,
) -> tuple[BenchmarkResult, ...]:
    optimizer = optimizer or TrustAwareQueryOptimizer()
    workload = tuple(workload)
    results = []
    for policy in policies:
        results.append(
            evaluate_policy(
                policy.name,
                sources,
                (policy.apply(request) for request in workload),
                optimizer,
            )
        )
    return tuple(results)


def run_reproducible_study(
    seeds: Iterable[int] = (7, 17, 29, 41, 53),
    *,
    source_count: int = 48,
    workload_count: int = 96,
    policies: Iterable[TrustPolicy] = (
        HIGH_ASSURANCE,
        BALANCED,
        LATENCY_CRITICAL,
        COST_EFFICIENT,
        TRUST_ONLY,
    ),
) -> EvaluationReport:
    seeds = tuple(seeds)
    policies = tuple(policies)
    per_policy: dict[str, list[BenchmarkResult]] = {policy.name: [] for policy in policies}

    for seed in seeds:
        sources = synthetic_sources(seed=seed, count=source_count)
        workload = synthetic_workload(seed=seed + 1, count=workload_count)
        for result in compare_policies(sources, workload, policies):
            per_policy[result.policy_name].append(result)

    return EvaluationReport(
        results=tuple(
            _aggregate(policy_name, seeds, results)
            for policy_name, results in per_policy.items()
        )
    )


def _aggregate(
    policy_name: str,
    seeds: tuple[int, ...],
    results: Sequence[BenchmarkResult],
) -> StudyResult:
    return StudyResult(
        policy_name=policy_name,
        seeds=seeds,
        average_accepted=mean(result.accepted for result in results),
        average_score=mean(result.average_score for result in results),
        average_trust=mean(result.average_trust for result in results),
        average_latency_ms=mean(result.average_latency_ms for result in results),
        average_cost=mean(result.average_cost for result in results),
        average_rejection_rate=mean(result.rejection_rate for result in results),
    )


def report_delta(report: EvaluationReport, baseline: str = "cost-efficient") -> Mapping[str, float]:
    by_name = {result.policy_name: result for result in report.results}
    if baseline not in by_name or "high-assurance" not in by_name:
        return {}
    high_assurance = by_name["high-assurance"]
    baseline_result = by_name[baseline]
    return {
        "trust_lift": high_assurance.average_trust - baseline_result.average_trust,
        "cost_delta": high_assurance.average_cost - baseline_result.average_cost,
        "latency_delta_ms": (
            high_assurance.average_latency_ms - baseline_result.average_latency_ms
        ),
        "acceptance_delta": high_assurance.average_accepted
        - baseline_result.average_accepted,
    }
