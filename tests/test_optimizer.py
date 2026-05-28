import unittest

from trust_aware import DataSource, QueryRequest, TrustAwareQueryOptimizer


class TrustAwareQueryOptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.optimizer = TrustAwareQueryOptimizer()
        self.sources = [
            DataSource(
                name="high-trust-vector",
                trust_score=0.92,
                latency_ms=100,
                cost_per_query=0.3,
                freshness_score=0.95,
                supports_vector=True,
            ),
            DataSource(
                name="fast-low-trust",
                trust_score=0.55,
                latency_ms=30,
                cost_per_query=0.1,
                freshness_score=0.9,
                supports_vector=True,
            ),
            DataSource(
                name="trusted-no-vector",
                trust_score=0.97,
                latency_ms=80,
                cost_per_query=0.2,
                freshness_score=0.95,
                supports_vector=False,
            ),
        ]

    def test_prefers_high_trust_source_when_constraints_met(self) -> None:
        request = QueryRequest(requires_vector=True, max_latency_ms=150)
        plan = self.optimizer.optimize(self.sources, request)
        self.assertIsNotNone(plan.primary_source)
        self.assertEqual(plan.primary_source.name, "high-trust-vector")

    def test_filters_sources_that_do_not_meet_constraints(self) -> None:
        request = QueryRequest(
            requires_vector=True,
            max_latency_ms=40,
            min_freshness_score=0.85,
        )
        plan = self.optimizer.optimize(self.sources, request)
        self.assertEqual([s.name for s, _ in plan.ranked_sources], ["fast-low-trust"])

    def test_returns_empty_plan_when_no_candidates_pass(self) -> None:
        request = QueryRequest(
            requires_vector=True,
            max_latency_ms=20,
            min_freshness_score=0.99,
        )
        plan = self.optimizer.optimize(self.sources, request)
        self.assertIsNone(plan.primary_source)
        self.assertEqual(plan.ranked_sources, [])


if __name__ == "__main__":
    unittest.main()
