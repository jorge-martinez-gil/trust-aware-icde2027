from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trust_aware import (
    Capability,
    DataSource,
    HIGH_ASSURANCE,
    PortfolioBudget,
    QueryRequest,
    QueryStage,
    TrustAwarePortfolioPlanner,
    TrustAwarePipelinePlanner,
    TrustAwareQueryOptimizer,
    TrustEvidence,
    certify_plan,
    certify_portfolio,
)


def main() -> int:
    sources = [
        DataSource(
            name="curated-vector-index",
            trust_score=0.90,
            latency_ms=85,
            cost_per_query=0.22,
            freshness_score=0.88,
            capabilities={Capability.VECTOR_SEARCH, Capability.POLICY_AUDIT},
            trust_evidence=(
                TrustEvidence("human_acceptance", positive=420, total=470, weight=0.6),
                TrustEvidence("citation_support", positive=310, total=350, weight=0.4),
            ),
            coverage=0.84,
            privacy_risk=0.08,
            hallucination_risk=0.10,
        ),
        DataSource(
            name="fast-raw-embedding-cache",
            trust_score=0.73,
            latency_ms=28,
            cost_per_query=0.04,
            freshness_score=0.81,
            supports_vector=True,
            trust_evidence=(
                TrustEvidence("human_acceptance", positive=92, total=130),
            ),
            coverage=0.62,
            privacy_risk=0.18,
            hallucination_risk=0.17,
        ),
        DataSource(
            name="expensive-grounded-agent",
            trust_score=0.94,
            latency_ms=260,
            cost_per_query=1.40,
            freshness_score=0.93,
            capabilities={Capability.VECTOR_SEARCH, Capability.LLM_INFERENCE},
            trust_evidence=(
                TrustEvidence("human_acceptance", positive=280, total=300, weight=0.5),
                TrustEvidence("policy_compliance", positive=294, total=300, weight=0.5),
            ),
            coverage=0.91,
            privacy_risk=0.05,
            hallucination_risk=0.07,
        ),
    ]

    request = QueryRequest(
        query="Find grounded evidence for model drift in EU support tickets.",
        required_capabilities={Capability.VECTOR_SEARCH},
        max_latency_ms=300,
        min_freshness_score=0.80,
        min_trust_score=0.70,
        max_privacy_risk=0.25,
        trust_weight=0.48,
        latency_weight=0.13,
        freshness_weight=0.12,
        cost_weight=0.07,
        coverage_weight=0.12,
        risk_weight=0.08,
        uncertainty_weight=0.03,
        risk_tolerance=0.05,
    )

    plan = TrustAwareQueryOptimizer().optimize(sources, request)
    print(plan.explain())
    print()
    certificate = certify_plan(plan, request)
    print(certificate.to_markdown())
    print()
    portfolio = TrustAwarePortfolioPlanner().plan(
        sources,
        request,
        PortfolioBudget(
            min_sources=2,
            max_sources=2,
            max_total_latency_ms=300,
            max_total_cost=2.00,
            min_portfolio_trust=0.86,
            latency_model="parallel",
        ),
    )
    print(portfolio.explain())
    print()
    print(certify_portfolio(portfolio, request).to_markdown())
    print()
    pipeline = TrustAwarePipelinePlanner().plan(
        query=request.query,
        stages=(
            QueryStage(
                name="retrieve evidence",
                required_capabilities=(Capability.VECTOR_SEARCH,),
                max_latency_ms=300,
                min_trust_score=0.70,
            ),
            QueryStage(
                name="ground answer",
                required_capabilities=(Capability.LLM_INFERENCE,),
                max_latency_ms=350,
                min_trust_score=0.72,
                max_hallucination_risk=0.12,
            ),
        ),
        sources=sources,
        policy=HIGH_ASSURANCE,
    )
    print(pipeline.explain())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
