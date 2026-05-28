import unittest

from trust_aware import (
    Capability,
    DataSource,
    QueryRequest,
    SourceCatalog,
    TrustAwareQueryOptimizer,
    TrustEvidence,
)
from trust_aware.benchmark import run_ablation, synthetic_sources, synthetic_workload


class TrustAwarePlanningTests(unittest.TestCase):
    def test_prefers_well_calibrated_trust_over_noisy_claim(self) -> None:
        optimizer = TrustAwareQueryOptimizer()
        sources = [
            DataSource(
                name="flashy-but-uncertain",
                trust_score=0.99,
                latency_ms=40,
                cost_per_query=0.05,
                freshness_score=0.95,
                capabilities={Capability.VECTOR_SEARCH},
                trust_evidence=(
                    TrustEvidence("acceptance", positive=2, total=2),
                ),
            ),
            DataSource(
                name="boring-and-proven",
                trust_score=0.82,
                latency_ms=70,
                cost_per_query=0.20,
                freshness_score=0.90,
                capabilities={Capability.VECTOR_SEARCH},
                trust_evidence=(
                    TrustEvidence("acceptance", positive=165, total=190),
                ),
            ),
        ]

        request = QueryRequest(
            required_capabilities={Capability.VECTOR_SEARCH},
            trust_weight=0.82,
            latency_weight=0.04,
            freshness_weight=0.08,
            cost_weight=0.06,
            risk_tolerance=0.05,
        )
        plan = optimizer.optimize(sources, request)

        self.assertEqual(plan.primary_source.name, "boring-and-proven")
        self.assertGreater(plan.ranked_sources[0][1], plan.ranked_sources[1][1])
        self.assertIn("calibrated", plan.explain())

    def test_rejections_are_explainable(self) -> None:
        optimizer = TrustAwareQueryOptimizer()
        source = DataSource(
            name="private-slow-llm",
            trust_score=0.91,
            latency_ms=900,
            cost_per_query=0.35,
            freshness_score=0.80,
            capabilities={Capability.LLM_INFERENCE},
            privacy_risk=0.75,
        )

        request = QueryRequest(
            required_capabilities={Capability.LLM_INFERENCE},
            max_latency_ms=250,
            max_privacy_risk=0.25,
        )
        plan = optimizer.optimize([source], request)

        self.assertIsNone(plan.primary_source)
        explanation = plan.explain()
        self.assertIn("latency", explanation)
        self.assertIn("privacy risk", explanation)

    def test_catalog_filters_by_capability(self) -> None:
        catalog = SourceCatalog()
        catalog.extend(
            [
                DataSource(
                    name="sql",
                    trust_score=0.8,
                    latency_ms=100,
                    cost_per_query=0.1,
                    freshness_score=0.9,
                    capabilities={Capability.STRUCTURED_SQL},
                ),
                DataSource(
                    name="vector",
                    trust_score=0.8,
                    latency_ms=100,
                    cost_per_query=0.1,
                    freshness_score=0.9,
                    capabilities={Capability.VECTOR_SEARCH},
                ),
            ]
        )

        matches = catalog.match_capabilities({Capability.VECTOR_SEARCH})
        self.assertEqual([source.name for source in matches], ["vector"])

    def test_benchmark_is_deterministic(self) -> None:
        first = run_ablation(seed=17)
        second = run_ablation(seed=17)

        self.assertEqual(first, second)
        self.assertEqual(
            [result.policy_name for result in first],
            ["trust-aware", "cost-first", "fast-first", "trust-only"],
        )
        self.assertEqual(len(synthetic_sources(seed=17)), 32)
        self.assertEqual(len(synthetic_workload(seed=18)), 64)


if __name__ == "__main__":
    unittest.main()
