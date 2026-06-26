from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import Capability, DataSource, QueryRequest, TrustEvidence
from .optimizer import TrustAwareQueryOptimizer


@dataclass(frozen=True)
class BenchmarkResult:
    policy_name: str
    workload_size: int
    accepted: int
    average_score: float
    average_trust: float
    average_latency_ms: float
    average_cost: float
    rejection_rate: float

    def to_markdown_row(self) -> str:
        return (
            f"| {self.policy_name} | {self.workload_size} | {self.accepted} | "
            f"{self.average_score:.3f} | {self.average_trust:.3f} | "
            f"{self.average_latency_ms:.1f} | {self.average_cost:.3f} | "
            f"{self.rejection_rate:.3f} |"
        )


def synthetic_sources(seed: int = 7, count: int = 32) -> tuple[DataSource, ...]:
    rng = random.Random(seed)
    sources: list[DataSource] = []
    capability_pool = [
        Capability.STRUCTURED_SQL,
        Capability.VECTOR_SEARCH,
        Capability.LEXICAL_SEARCH,
        Capability.GRAPH_LOOKUP,
        Capability.LLM_INFERENCE,
        Capability.POLICY_AUDIT,
    ]

    for index in range(count):
        capabilities = {
            capability
            for capability in capability_pool
            if rng.random() < 0.45
        }
        if not capabilities:
            capabilities.add(Capability.STRUCTURED_SQL)

        latent_quality = rng.betavariate(8, 3)
        total = rng.randint(12, 220)
        positive = min(total, max(0, round(latent_quality * total + rng.gauss(0, 3))))
        source = DataSource(
            name=f"source_{index:02d}",
            trust_score=latent_quality,
            latency_ms=rng.uniform(20, 450),
            cost_per_query=rng.uniform(0.02, 2.50),
            freshness_score=rng.betavariate(7, 2),
            supports_vector=Capability.VECTOR_SEARCH in capabilities,
            capabilities=capabilities,
            trust_evidence=(
                TrustEvidence(
                    dimension="answer_acceptance",
                    positive=positive,
                    total=total,
                    weight=0.7,
                ),
                TrustEvidence(
                    dimension="policy_compliance",
                    positive=max(0, positive - rng.randint(0, 8)),
                    total=total,
                    weight=0.3,
                ),
            ),
            coverage=rng.betavariate(6, 2),
            privacy_risk=rng.betavariate(2, 10),
            hallucination_risk=rng.betavariate(2, 8),
            schema_reliability=rng.betavariate(9, 2),
        )
        sources.append(source)

    return tuple(sources)


def synthetic_workload(seed: int = 11, count: int = 64) -> tuple[QueryRequest, ...]:
    rng = random.Random(seed)
    requests: list[QueryRequest] = []
    capability_options = [
        (Capability.STRUCTURED_SQL,),
        (Capability.VECTOR_SEARCH,),
        (Capability.LEXICAL_SEARCH,),
        (Capability.VECTOR_SEARCH, Capability.POLICY_AUDIT),
        (Capability.GRAPH_LOOKUP,),
        (Capability.LLM_INFERENCE,),
    ]

    for index in range(count):
        capabilities = capability_options[index % len(capability_options)]
        requests.append(
            QueryRequest(
                query=f"benchmark query {index}",
                required_capabilities=capabilities,
                max_latency_ms=rng.choice([120, 180, 250, 350, None]),
                min_freshness_score=rng.choice([0.0, 0.55, 0.70, 0.82]),
                min_trust_score=rng.choice([0.0, 0.55, 0.65, 0.75]),
                max_privacy_risk=rng.choice([None, 0.25, 0.40]),
                trust_weight=0.46,
                latency_weight=0.16,
                freshness_weight=0.15,
                cost_weight=0.08,
                coverage_weight=0.08,
                risk_weight=0.07,
                uncertainty_weight=0.04,
                risk_tolerance=rng.choice([0.05, 0.10, 0.15, 0.20]),
            )
        )

    return tuple(requests)


def evaluate_policy(
    policy_name: str,
    sources: Sequence[DataSource],
    requests: Iterable[QueryRequest],
    optimizer: TrustAwareQueryOptimizer | None = None,
) -> BenchmarkResult:
    optimizer = optimizer or TrustAwareQueryOptimizer()
    requests = tuple(requests)

    accepted = 0
    total_score = 0.0
    total_trust = 0.0
    total_latency = 0.0
    total_cost = 0.0
    total_rejections = 0

    for request in requests:
        plan = optimizer.optimize(sources, request)
        total_rejections += len(plan.rejected_sources)
        if plan.primary_source is None:
            continue

        source, score = plan.ranked_sources[0]
        accepted += 1
        total_score += score
        total_trust += source.trust_lower_bound(request.risk_tolerance)
        total_latency += source.latency_ms
        total_cost += source.cost_per_query

    denominator = max(1, accepted)
    rejection_denominator = max(1, len(requests) * len(sources))
    return BenchmarkResult(
        policy_name=policy_name,
        workload_size=len(requests),
        accepted=accepted,
        average_score=total_score / denominator,
        average_trust=total_trust / denominator,
        average_latency_ms=total_latency / denominator,
        average_cost=total_cost / denominator,
        rejection_rate=total_rejections / rejection_denominator,
    )


def cost_first_requests(requests: Iterable[QueryRequest]) -> tuple[QueryRequest, ...]:
    return tuple(
        QueryRequest(
            requires_vector=request.requires_vector,
            max_latency_ms=request.max_latency_ms,
            min_freshness_score=request.min_freshness_score,
            required_capabilities=request.required_capabilities,
            min_trust_score=0.0,
            max_cost_per_query=request.max_cost_per_query,
            max_privacy_risk=request.max_privacy_risk,
            max_hallucination_risk=request.max_hallucination_risk,
            trust_weight=0.10,
            latency_weight=0.20,
            freshness_weight=0.10,
            cost_weight=0.55,
            coverage_weight=0.03,
            risk_weight=0.02,
            risk_tolerance=request.risk_tolerance,
        )
        for request in requests
    )


def fast_first_requests(requests: Iterable[QueryRequest]) -> tuple[QueryRequest, ...]:
    return tuple(
        QueryRequest(
            requires_vector=request.requires_vector,
            max_latency_ms=request.max_latency_ms,
            min_freshness_score=request.min_freshness_score,
            required_capabilities=request.required_capabilities,
            min_trust_score=max(0.45, request.min_trust_score * 0.7),
            max_cost_per_query=request.max_cost_per_query,
            max_privacy_risk=request.max_privacy_risk,
            max_hallucination_risk=request.max_hallucination_risk,
            trust_weight=0.18,
            latency_weight=0.58,
            freshness_weight=0.08,
            cost_weight=0.06,
            coverage_weight=0.03,
            risk_weight=0.07,
            uncertainty_weight=0.01,
            risk_tolerance=max(request.risk_tolerance, 0.20),
        )
        for request in requests
    )


def trust_only_requests(requests: Iterable[QueryRequest]) -> tuple[QueryRequest, ...]:
    return tuple(
        QueryRequest(
            requires_vector=request.requires_vector,
            max_latency_ms=request.max_latency_ms,
            min_freshness_score=request.min_freshness_score,
            required_capabilities=request.required_capabilities,
            min_trust_score=request.min_trust_score,
            max_cost_per_query=request.max_cost_per_query,
            max_privacy_risk=request.max_privacy_risk,
            max_hallucination_risk=request.max_hallucination_risk,
            trust_weight=0.90,
            latency_weight=0.00,
            freshness_weight=0.05,
            cost_weight=0.00,
            coverage_weight=0.00,
            risk_weight=0.05,
            uncertainty_weight=0.05,
            risk_tolerance=min(request.risk_tolerance, 0.05),
        )
        for request in requests
    )


def run_ablation(seed: int = 7) -> tuple[BenchmarkResult, ...]:
    sources = synthetic_sources(seed=seed)
    workload = synthetic_workload(seed=seed + 1)
    optimizer = TrustAwareQueryOptimizer()
    return (
        evaluate_policy("trust-aware", sources, workload, optimizer),
        evaluate_policy("cost-first", sources, cost_first_requests(workload), optimizer),
        evaluate_policy(
            "fast-first", sources, fast_first_requests(workload), optimizer
        ),
        evaluate_policy(
            "trust-only", sources, trust_only_requests(workload), optimizer
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic trust-aware optimizer benchmark."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--study",
        action="store_true",
        help="run the multi-seed policy study used by the paper artifact",
    )
    parser.add_argument(
        "--plots",
        nargs="?",
        const="figures",
        help="generate publication-quality SVG figures into the given directory",
    )
    parser.add_argument(
        "--real-study",
        nargs="?",
        const="",
        metavar="WDBC_PATH",
        help=(
            "run the real-data WDBC study; optionally pass a path to wdbc.data"
        ),
    )
    parser.add_argument(
        "--federated-study",
        action="store_true",
        help=(
            "run the federated trust-aware retrieval study (sweep + multi-seed "
            "+ adversary variants) over the real IR collections, then emit "
            "figures and LaTeX tables"
        ),
    )
    parser.add_argument(
        "--federated-report",
        action="store_true",
        help="regenerate federated figures + tables from cached results JSON",
    )
    args = parser.parse_args(argv)

    if args.federated_study:
        from .federated.experiment import run_study, run_multiseed, run_variant_study
        from .federated import plots as fed_plots
        from .federated import report as fed_report

        print("Running federated robustness sweeps (seeds 7, 11, 13)...")
        for sweep_seed in (7, 11, 13):
            run_study(seed=sweep_seed)
        print("Running multi-seed headline conditions...")
        run_multiseed()
        print("Running adversary variant sensitivity...")
        run_variant_study()
        fed_report.generate_all()
        fed_plots.generate_all()
        print("Federated study complete. "
              "See results/federated/ and figures/federated/.")
        return 0

    if args.federated_report:
        from .federated import plots as fed_plots
        from .federated import report as fed_report

        fed_report.generate_all()
        fed_plots.generate_all()
        return 0

    if args.plots is not None:
        from .visualization import generate_all_figures

        for path in generate_all_figures(args.plots, seed=args.seed):
            print(path)
        return 0

    if args.study:
        from .evaluation import run_reproducible_study

        print(run_reproducible_study().to_markdown())
        return 0

    if args.real_study is not None:
        from .realdata import default_wdbc_path, run_wdbc_real_study

        path = default_wdbc_path() if args.real_study == "" else args.real_study
        print(run_wdbc_real_study(path).to_markdown())
        return 0

    results = run_ablation(seed=args.seed)
    print("| policy | workload | accepted | score | trust | latency_ms | cost | rejected |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        print(result.to_markdown_row())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
