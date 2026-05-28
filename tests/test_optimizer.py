import unittest

from trust_aware import (
    DataSource,
    ExecutionStrategy,
    QueryRequest,
    TrustAwareQueryOptimizer,
    TrustEvidence,
    compute_pareto_front,
)


class TrustAwareQueryOptimizerTests(unittest.TestCase):
    """Original baseline tests — must always pass (backward compatibility)."""

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
        self.assertEqual(
            [s.name for s, _ in plan.ranked_sources], ["fast-low-trust"]
        )

    def test_returns_empty_plan_when_no_candidates_pass(self) -> None:
        request = QueryRequest(
            requires_vector=True,
            max_latency_ms=20,
            min_freshness_score=0.99,
        )
        plan = self.optimizer.optimize(self.sources, request)
        self.assertIsNone(plan.primary_source)
        self.assertEqual(plan.ranked_sources, [])


class OptimizerStrategyTests(unittest.TestCase):
    """Tests for pluggable scoring strategies."""

    def setUp(self) -> None:
        self.sources = [
            DataSource(
                name="strong",
                trust_score=0.95,
                latency_ms=40,
                cost_per_query=0.05,
                freshness_score=0.98,
                supports_vector=True,
            ),
            DataSource(
                name="weak",
                trust_score=0.30,
                latency_ms=160,
                cost_per_query=1.0,
                freshness_score=0.40,
                supports_vector=True,
            ),
        ]
        self.request = QueryRequest(max_latency_ms=200)

    def test_topsis_optimizer_ranks_strong_source_first(self) -> None:
        optimizer = TrustAwareQueryOptimizer(strategy="topsis")
        plan = optimizer.optimize(self.sources, self.request)
        self.assertEqual(plan.primary_source.name, "strong")

    def test_bayesian_ucb_optimizer_ranks_strong_source_first(self) -> None:
        optimizer = TrustAwareQueryOptimizer(strategy="bayesian-ucb")
        plan = optimizer.optimize(self.sources, self.request)
        self.assertEqual(plan.primary_source.name, "strong")

    def test_all_strategies_produce_score_breakdowns(self) -> None:
        for strategy in ("linear", "topsis", "bayesian-ucb"):
            with self.subTest(strategy=strategy):
                optimizer = TrustAwareQueryOptimizer(strategy=strategy)
                plan = optimizer.optimize(self.sources, self.request)
                self.assertEqual(len(plan.breakdowns), len(plan.ranked_sources))


class OptimizerPlanMetadataTests(unittest.TestCase):
    """Tests for QueryPlan metadata: confidence, strategy, Pareto front."""

    def _make_sources(self):
        return [
            DataSource("a", trust_score=0.9, latency_ms=50, cost_per_query=0.1,
                       freshness_score=0.95),
            DataSource("b", trust_score=0.88, latency_ms=55, cost_per_query=0.15,
                       freshness_score=0.90),
            DataSource("c", trust_score=0.30, latency_ms=180, cost_per_query=2.0,
                       freshness_score=0.30),
        ]

    def test_confidence_is_bounded(self) -> None:
        plan = TrustAwareQueryOptimizer().optimize(
            self._make_sources(), QueryRequest()
        )
        self.assertGreaterEqual(plan.confidence, 0.0)
        self.assertLessEqual(plan.confidence, 1.0)

    def test_empty_plan_has_zero_confidence(self) -> None:
        plan = TrustAwareQueryOptimizer().optimize(
            [], QueryRequest()
        )
        self.assertEqual(plan.confidence, 0.0)

    def test_plan_has_pareto_front(self) -> None:
        plan = TrustAwareQueryOptimizer().optimize(
            self._make_sources(), QueryRequest()
        )
        self.assertGreater(len(plan.pareto_front), 0)

    def test_dominated_source_excluded_from_pareto_front(self) -> None:
        """Source 'c' is dominated by 'a' and 'b' on all criteria."""
        plan = TrustAwareQueryOptimizer().optimize(
            self._make_sources(), QueryRequest()
        )
        pareto_names = {s.name for s in plan.pareto_front}
        self.assertNotIn("c", pareto_names)

    def test_execution_strategy_is_valid_enum_value(self) -> None:
        plan = TrustAwareQueryOptimizer().optimize(
            self._make_sources(), QueryRequest()
        )
        self.assertIn(plan.execution_strategy, list(ExecutionStrategy))

    def test_close_scores_recommend_ensemble(self) -> None:
        """Two nearly identical sources should trigger ENSEMBLE strategy."""
        s1 = DataSource("x", trust_score=0.80, latency_ms=50, cost_per_query=0.1,
                        freshness_score=0.85)
        s2 = DataSource("y", trust_score=0.80, latency_ms=50, cost_per_query=0.1,
                        freshness_score=0.85)
        plan = TrustAwareQueryOptimizer().optimize([s1, s2], QueryRequest())
        self.assertEqual(plan.execution_strategy, ExecutionStrategy.ENSEMBLE)

    def test_fallback_chain_excludes_primary(self) -> None:
        plan = TrustAwareQueryOptimizer().optimize(
            self._make_sources(), QueryRequest()
        )
        if plan.primary_source:
            self.assertNotIn(plan.primary_source, plan.fallback_chain)

    def test_reliability_filter_excludes_unreliable_source(self) -> None:
        unreliable = DataSource(
            "unreliable", trust_score=0.9, latency_ms=20,
            cost_per_query=0.05, freshness_score=0.99, reliability=0.4,
        )
        reliable = DataSource(
            "reliable", trust_score=0.7, latency_ms=80,
            cost_per_query=0.2, freshness_score=0.85, reliability=0.99,
        )
        request = QueryRequest(min_reliability=0.9)
        plan = TrustAwareQueryOptimizer().optimize([unreliable, reliable], request)
        names = [s.name for s, _ in plan.ranked_sources]
        self.assertNotIn("unreliable", names)
        self.assertIn("reliable", names)


class ParetoFrontTests(unittest.TestCase):
    """Tests for the standalone compute_pareto_front utility."""

    def test_perfectly_dominated_source_excluded(self) -> None:
        dominant = DataSource(
            "dom", trust_score=0.9, latency_ms=30, cost_per_query=0.1,
            freshness_score=0.95,
        )
        dominated = DataSource(
            "sub", trust_score=0.5, latency_ms=90, cost_per_query=0.5,
            freshness_score=0.60,
        )
        front = compute_pareto_front([dominant, dominated])
        self.assertIn(dominant, front)
        self.assertNotIn(dominated, front)

    def test_incomparable_sources_both_on_front(self) -> None:
        """Fast-cheap vs. high-trust: neither dominates the other."""
        fast_cheap = DataSource(
            "fc", trust_score=0.4, latency_ms=10, cost_per_query=0.01,
            freshness_score=0.5,
        )
        high_trust = DataSource(
            "ht", trust_score=0.95, latency_ms=200, cost_per_query=2.0,
            freshness_score=0.95,
        )
        front = compute_pareto_front([fast_cheap, high_trust])
        self.assertIn(fast_cheap, front)
        self.assertIn(high_trust, front)

    def test_single_source_is_on_pareto_front(self) -> None:
        src = DataSource(
            "solo", trust_score=0.7, latency_ms=80, cost_per_query=0.3,
            freshness_score=0.8,
        )
        front = compute_pareto_front([src])
        self.assertEqual(front, [src])


if __name__ == "__main__":
    unittest.main()
