"""Tests for AdaptiveTrustManager: online Bayesian trust learning."""
import unittest

from trust_aware import (
    AdaptiveTrustManager,
    DataSource,
    ExecutionFeedback,
    TrustEvidence,
)


def _src(name: str = "src", trust: float = 0.7) -> DataSource:
    return DataSource(
        name=name,
        trust_score=trust,
        latency_ms=50.0,
        cost_per_query=0.1,
        freshness_score=0.9,
    )


class AdaptiveTrustManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = AdaptiveTrustManager(decay_factor=1.0)
        self.src = _src("alpha", trust=0.7)
        self.manager.register(self.src)

    def test_success_observation_increases_posterior_mean(self) -> None:
        before = self.manager.augment(self.src).evidence.mean
        self.manager.observe(ExecutionFeedback("alpha", success=True, latency_ms=45))
        after = self.manager.augment(self.src).evidence.mean
        self.assertGreater(after, before)

    def test_failure_observation_decreases_posterior_mean(self) -> None:
        # Seed with some success first to get mean above 0.5
        for _ in range(20):
            self.manager.observe(ExecutionFeedback("alpha", success=True, latency_ms=45))
        mid = self.manager.augment(self.src).evidence.mean

        for _ in range(10):
            self.manager.observe(
                ExecutionFeedback("alpha", success=False, latency_ms=45)
            )
        after = self.manager.augment(self.src).evidence.mean
        self.assertLess(after, mid)

    def test_observation_count_increases_after_feedback(self) -> None:
        self.manager.observe(ExecutionFeedback("alpha", success=True, latency_ms=45))
        ev = self.manager.augment(self.src).evidence
        self.assertEqual(ev.n, 1)

    def test_decay_reduces_observation_count(self) -> None:
        for _ in range(10):
            self.manager.observe(ExecutionFeedback("alpha", success=True, latency_ms=45))
        n_before = self.manager.augment(self.src).evidence.n

        manager_decay = AdaptiveTrustManager(decay_factor=0.5)
        manager_decay.register(self.src)
        for _ in range(10):
            manager_decay.observe(
                ExecutionFeedback("alpha", success=True, latency_ms=45)
            )
        manager_decay.decay()
        n_after = manager_decay.augment(self.src).evidence.n
        self.assertLess(n_after, n_before)

    def test_quality_filter_gates_success_confirmation(self) -> None:
        manager = AdaptiveTrustManager(quality_threshold=0.7)
        src = _src("beta")
        manager.register(src)

        # High-quality success: should be confirmed
        manager.observe(
            ExecutionFeedback("beta", success=True, latency_ms=30, quality_score=0.9)
        )
        ev_high = manager.augment(src).evidence
        self.assertEqual(ev_high.successes, 1)
        self.assertEqual(ev_high.failures, 0)

        # Low-quality "success": should be treated as failure
        manager.observe(
            ExecutionFeedback("beta", success=True, latency_ms=30, quality_score=0.4)
        )
        ev_low = manager.augment(src).evidence
        self.assertEqual(ev_low.successes, 1)
        self.assertEqual(ev_low.failures, 1)

    def test_augment_all_applies_evidence_to_all_sources(self) -> None:
        sources = [_src(f"s{i}", trust=0.5) for i in range(4)]
        manager = AdaptiveTrustManager()
        manager.register_all(sources)
        for s in sources:
            manager.observe(ExecutionFeedback(s.name, success=True, latency_ms=50))

        augmented = manager.augment_all(sources)
        for src in augmented:
            self.assertEqual(src.evidence.n, 1)

    def test_trust_summary_includes_all_registered_sources(self) -> None:
        src2 = _src("beta")
        self.manager.register(src2)
        summary = self.manager.trust_summary()
        self.assertIn("alpha", summary)
        self.assertIn("beta", summary)
        self.assertIn("mean", summary["alpha"])
        self.assertIn("n", summary["alpha"])

    def test_unregistered_source_gets_default_evidence(self) -> None:
        unknown = _src("unknown")
        augmented = self.manager.augment(unknown)
        self.assertEqual(augmented.evidence.n, 0)

    def test_invalid_decay_factor_raises(self) -> None:
        with self.assertRaises(ValueError):
            AdaptiveTrustManager(decay_factor=0.0)
        with self.assertRaises(ValueError):
            AdaptiveTrustManager(decay_factor=1.5)

    def test_posterior_mean_bounded_in_unit_interval(self) -> None:
        for _ in range(100):
            self.manager.observe(
                ExecutionFeedback("alpha", success=True, latency_ms=10)
            )
        ev = self.manager.augment(self.src).evidence
        self.assertGreaterEqual(ev.mean, 0.0)
        self.assertLessEqual(ev.mean, 1.0)


if __name__ == "__main__":
    unittest.main()
