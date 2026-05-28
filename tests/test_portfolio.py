import unittest

from trust_aware import (
    Capability,
    DataSource,
    PortfolioBudget,
    QueryRequest,
    TrustAwarePortfolioPlanner,
    TrustEvidence,
    certify_portfolio,
)


class TrustAwarePortfolioPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = [
            DataSource(
                name="primary-index",
                trust_score=0.83,
                latency_ms=70,
                cost_per_query=0.20,
                freshness_score=0.91,
                capabilities={Capability.VECTOR_SEARCH},
                trust_evidence=(TrustEvidence("acceptance", positive=80, total=100),),
                coverage=0.70,
                privacy_risk=0.08,
                hallucination_risk=0.10,
            ),
            DataSource(
                name="independent-validator",
                trust_score=0.78,
                latency_ms=90,
                cost_per_query=0.25,
                freshness_score=0.88,
                capabilities={Capability.VECTOR_SEARCH, Capability.POLICY_AUDIT},
                trust_evidence=(TrustEvidence("acceptance", positive=72, total=90),),
                coverage=0.65,
                privacy_risk=0.06,
                hallucination_risk=0.09,
            ),
            DataSource(
                name="cheap-cache",
                trust_score=0.55,
                latency_ms=30,
                cost_per_query=0.03,
                freshness_score=0.82,
                capabilities={Capability.VECTOR_SEARCH},
                coverage=0.45,
                privacy_risk=0.14,
                hallucination_risk=0.20,
            ),
        ]
        self.request = QueryRequest(
            query="triangulate safety-critical retrieval evidence",
            required_capabilities={Capability.VECTOR_SEARCH},
            max_latency_ms=140,
            trust_weight=0.55,
            latency_weight=0.10,
            freshness_weight=0.10,
            cost_weight=0.05,
            coverage_weight=0.15,
            risk_weight=0.05,
            uncertainty_weight=0.02,
            risk_tolerance=0.05,
        )

    def test_builds_budgeted_redundant_portfolio(self) -> None:
        plan = TrustAwarePortfolioPlanner().plan(
            self.sources,
            self.request,
            PortfolioBudget(
                min_sources=2,
                max_sources=2,
                max_total_latency_ms=110,
                max_total_cost=0.60,
                min_portfolio_trust=0.84,
                latency_model="parallel",
            ),
        )

        self.assertIsNotNone(plan.primary_portfolio)
        self.assertEqual(
            plan.primary_portfolio.names,
            ("primary-index", "independent-validator"),
        )
        self.assertGreater(
            plan.primary_portfolio.trust_lower_bound,
            plan.base_plan.steps[0].breakdown.components["trust"],
        )
        self.assertIn("correlated evidence penalty", plan.explain())

    def test_sequential_budget_can_reject_portfolios(self) -> None:
        plan = TrustAwarePortfolioPlanner().plan(
            self.sources,
            self.request,
            PortfolioBudget(
                min_sources=2,
                max_sources=2,
                max_total_latency_ms=80,
                min_portfolio_trust=0.84,
                latency_model="sequential",
            ),
        )

        self.assertIsNone(plan.primary_portfolio)
        self.assertGreater(plan.budget_rejected_count, 0)
        self.assertIn("No admissible portfolios", plan.explain())

    def test_portfolio_certificate_is_stable(self) -> None:
        plan = TrustAwarePortfolioPlanner().plan(
            self.sources,
            self.request,
            PortfolioBudget(min_sources=2, max_sources=2, min_portfolio_trust=0.84),
        )

        first = certify_portfolio(plan, self.request)
        second = certify_portfolio(plan, self.request)

        self.assertEqual(first.certificate_id, second.certificate_id)
        self.assertIn("Ranked Portfolios", first.to_markdown())
        self.assertIn("primary-index", first.to_json())


if __name__ == "__main__":
    unittest.main()
