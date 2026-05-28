import unittest

from trust_aware import (
    BALANCED,
    Capability,
    DataSource,
    FeedbackEvent,
    HIGH_ASSURANCE,
    QueryRequest,
    QueryStage,
    TrustAwarePipelinePlanner,
    TrustAwareQueryOptimizer,
    TrustEvidence,
    TrustLedger,
    certify_plan,
    run_reproducible_study,
)
from trust_aware.evaluation import report_delta


class AwardArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = [
            DataSource(
                name="audited-vector",
                trust_score=0.88,
                latency_ms=80,
                cost_per_query=0.20,
                freshness_score=0.90,
                capabilities={Capability.VECTOR_SEARCH, Capability.POLICY_AUDIT},
                trust_evidence=(TrustEvidence("acceptance", positive=90, total=100),),
                coverage=0.84,
                privacy_risk=0.08,
                hallucination_risk=0.10,
            ),
            DataSource(
                name="grounded-generator",
                trust_score=0.92,
                latency_ms=210,
                cost_per_query=0.95,
                freshness_score=0.88,
                capabilities={Capability.LLM_INFERENCE},
                trust_evidence=(TrustEvidence("acceptance", positive=135, total=150),),
                coverage=0.78,
                privacy_risk=0.06,
                hallucination_risk=0.08,
            ),
        ]

    def test_certificate_is_stable_and_auditable(self) -> None:
        request = BALANCED.request(
            query="audit a retrieval decision",
            required_capabilities={Capability.VECTOR_SEARCH},
            max_latency_ms=150,
        )
        plan = TrustAwareQueryOptimizer().optimize(self.sources, request)

        first = certify_plan(plan, request)
        second = certify_plan(plan, request)

        self.assertEqual(first.certificate_id, second.certificate_id)
        self.assertIn("audited-vector", first.to_json())
        self.assertIn("Trust Certificate", first.to_markdown())

    def test_pipeline_planner_builds_complete_ai_query_plan(self) -> None:
        pipeline = TrustAwarePipelinePlanner().plan(
            query="answer with retrieved evidence",
            stages=(
                QueryStage(
                    name="retrieve",
                    required_capabilities=(Capability.VECTOR_SEARCH,),
                    max_latency_ms=150,
                    min_trust_score=0.70,
                ),
                QueryStage(
                    name="generate",
                    required_capabilities=(Capability.LLM_INFERENCE,),
                    max_latency_ms=250,
                    min_trust_score=0.70,
                ),
            ),
            sources=self.sources,
            policy=HIGH_ASSURANCE,
        )

        self.assertTrue(pipeline.is_complete)
        self.assertEqual(
            [source.name for source in pipeline.selected_sources],
            ["audited-vector", "grounded-generator"],
        )
        self.assertGreater(pipeline.trust_floor, 0.70)

    def test_feedback_ledger_updates_evidence(self) -> None:
        source = DataSource(
            name="adaptive-cache",
            trust_score=0.50,
            latency_ms=30,
            cost_per_query=0.02,
            freshness_score=0.70,
            supports_vector=True,
        )
        ledger = TrustLedger(
            [
                FeedbackEvent("adaptive-cache", "acceptance", True),
                FeedbackEvent("adaptive-cache", "acceptance", True),
                FeedbackEvent("adaptive-cache", "acceptance", False),
            ]
        )

        updated = ledger.update_source(source)

        self.assertEqual(ledger.summary()["events"], 3)
        self.assertEqual(updated.trust_evidence[0].total, 3)
        self.assertGreater(updated.calibrated_trust(), source.calibrated_trust())

    def test_reproducible_study_reports_policy_deltas(self) -> None:
        report = run_reproducible_study(
            seeds=(3, 5),
            source_count=18,
            workload_count=24,
        )
        markdown = report.to_markdown()
        delta = report_delta(report)

        self.assertIn("high-assurance", markdown)
        self.assertIn("cost-efficient", markdown)
        self.assertIn("trust_lift", delta)
        self.assertGreater(delta["trust_lift"], 0.0)

    def test_policy_apply_keeps_query_constraints(self) -> None:
        request = QueryRequest(
            query="latency bounded trust request",
            required_capabilities={Capability.VECTOR_SEARCH},
            max_latency_ms=100,
            min_trust_score=0.80,
        )

        applied = BALANCED.apply(request)

        self.assertEqual(applied.max_latency_ms, 100)
        self.assertEqual(applied.min_trust_score, 0.80)
        self.assertLessEqual(applied.risk_tolerance, BALANCED.risk_tolerance)


if __name__ == "__main__":
    unittest.main()
