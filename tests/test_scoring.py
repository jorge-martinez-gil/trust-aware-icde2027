"""Tests for scoring strategies: LinearWeightedScorer, TOPSISScorer,
and BayesianUCBScorer."""
import math
import unittest

from trust_aware import (
    BayesianUCBScorer,
    DataSource,
    LinearWeightedScorer,
    QueryRequest,
    TOPSISScorer,
    TrustEvidence,
)


def _make_source(
    name="src",
    trust=0.8,
    latency=50.0,
    cost=0.1,
    freshness=0.9,
    supports_vector=True,
) -> DataSource:
    return DataSource(
        name=name,
        trust_score=trust,
        latency_ms=latency,
        cost_per_query=cost,
        freshness_score=freshness,
        supports_vector=supports_vector,
    )


_DEFAULT_REQUEST = QueryRequest(max_latency_ms=200)


class LinearWeightedScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = LinearWeightedScorer()

    def test_score_is_in_unit_interval(self) -> None:
        src = _make_source()
        score, _ = self.scorer.score(src, _DEFAULT_REQUEST)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_high_trust_source_scores_higher_than_low_trust(self) -> None:
        high = _make_source(name="high", trust=0.95)
        low = _make_source(name="low", trust=0.30)
        request = QueryRequest(max_latency_ms=200)
        s_high, _ = self.scorer.score(high, request)
        s_low, _ = self.scorer.score(low, request)
        self.assertGreater(s_high, s_low)

    def test_breakdown_components_are_clamped_to_unit_interval(self) -> None:
        src = _make_source()
        _, bd = self.scorer.score(src, _DEFAULT_REQUEST)
        for component in (bd.trust, bd.latency, bd.freshness, bd.cost):
            self.assertGreaterEqual(component, 0.0)
            self.assertLessEqual(component, 1.0)

    def test_zero_cost_source_gets_maximum_cost_component(self) -> None:
        src = _make_source(cost=0.0)
        _, bd = self.scorer.score(src, _DEFAULT_REQUEST)
        self.assertAlmostEqual(bd.cost, 1.0)

    def test_latency_at_max_yields_zero_latency_component(self) -> None:
        src = _make_source(latency=200.0)
        request = QueryRequest(max_latency_ms=200)
        _, bd = self.scorer.score(src, request)
        self.assertAlmostEqual(bd.latency, 0.0)

    def test_no_max_latency_yields_full_latency_component(self) -> None:
        src = _make_source(latency=9999.0)
        request = QueryRequest(max_latency_ms=None)
        _, bd = self.scorer.score(src, request)
        self.assertAlmostEqual(bd.latency, 1.0)

    def test_conservative_trust_uses_lcb(self) -> None:
        evidence = TrustEvidence(successes=2, failures=8)  # low trust history
        src = DataSource(
            name="risky",
            trust_score=0.9,
            latency_ms=50,
            cost_per_query=0.1,
            freshness_score=0.9,
            evidence=evidence,
        )
        normal_req = QueryRequest(max_latency_ms=200, use_conservative_trust=False)
        conservative_req = QueryRequest(max_latency_ms=200, use_conservative_trust=True)
        s_normal, bd_normal = self.scorer.score(src, normal_req)
        s_conservative, bd_conservative = self.scorer.score(src, conservative_req)
        self.assertGreaterEqual(bd_normal.trust, bd_conservative.trust)


class TOPSISScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = TOPSISScorer()

    def test_single_source_scores_half(self) -> None:
        """With one candidate, d_pos == d_neg → closeness = 0.5."""
        src = _make_source()
        score, _ = self.scorer.score(src, _DEFAULT_REQUEST, all_sources=[src])
        self.assertAlmostEqual(score, 0.5, places=5)

    def test_score_in_unit_interval(self) -> None:
        sources = [
            _make_source("a", trust=0.9, latency=30, cost=0.1, freshness=0.95),
            _make_source("b", trust=0.4, latency=180, cost=0.8, freshness=0.5),
        ]
        for src in sources:
            score, _ = self.scorer.score(src, _DEFAULT_REQUEST, all_sources=sources)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_ideal_source_scores_higher_than_poor_source(self) -> None:
        ideal = _make_source("ideal", trust=1.0, latency=1.0, cost=0.0, freshness=1.0)
        poor = _make_source("poor", trust=0.1, latency=199.0, cost=5.0, freshness=0.1)
        sources = [ideal, poor]
        s_ideal, _ = self.scorer.score(ideal, _DEFAULT_REQUEST, sources)
        s_poor, _ = self.scorer.score(poor, _DEFAULT_REQUEST, sources)
        self.assertGreater(s_ideal, s_poor)

    def test_three_sources_ordering_matches_intuition(self) -> None:
        best = _make_source("best", trust=0.95, latency=20, cost=0.05, freshness=0.98)
        mid = _make_source("mid", trust=0.70, latency=100, cost=0.30, freshness=0.75)
        worst = _make_source("worst", trust=0.20, latency=180, cost=1.0, freshness=0.30)
        sources = [best, mid, worst]
        s_best, _ = self.scorer.score(best, _DEFAULT_REQUEST, sources)
        s_mid, _ = self.scorer.score(mid, _DEFAULT_REQUEST, sources)
        s_worst, _ = self.scorer.score(worst, _DEFAULT_REQUEST, sources)
        self.assertGreater(s_best, s_mid)
        self.assertGreater(s_mid, s_worst)

    def test_topsis_scores_sum_is_consistent(self) -> None:
        """For two symmetric sources the scores should sum to ~1."""
        a = _make_source("a", trust=0.9, latency=20, cost=0.1, freshness=0.9)
        b = _make_source("b", trust=0.1, latency=180, cost=0.9, freshness=0.1)
        sources = [a, b]
        request = QueryRequest(max_latency_ms=200)
        s_a, _ = self.scorer.score(a, request, sources)
        s_b, _ = self.scorer.score(b, request, sources)
        self.assertAlmostEqual(s_a + s_b, 1.0, places=5)


class BayesianUCBScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = BayesianUCBScorer(exploration_bonus=0.1)

    def test_unexplored_source_gets_higher_score_than_explored_equal(self) -> None:
        """Unexplored source (n=0) should score higher than identical observed one."""
        unexplored = _make_source(name="unexplored")
        explored = DataSource(
            name="explored",
            trust_score=0.8,
            latency_ms=50.0,
            cost_per_query=0.1,
            freshness_score=0.9,
            evidence=TrustEvidence(successes=100, failures=20),
        )
        s_unexplored, _ = self.scorer.score(unexplored, _DEFAULT_REQUEST)
        s_explored, _ = self.scorer.score(explored, _DEFAULT_REQUEST)
        self.assertGreater(s_unexplored, s_explored)

    def test_exploration_bonus_decays_with_observations(self) -> None:
        """Fewer observations → higher UCB and bonus → higher score (same success rate)."""
        # Both sources have the same empirical 50% success rate, only n differs.
        # The source with fewer observations has a wider posterior CI → higher UCB
        # and a larger exploration bonus → higher composite score.
        few_obs = DataSource(
            name="few",
            trust_score=0.8,
            latency_ms=50,
            cost_per_query=0.1,
            freshness_score=0.9,
            evidence=TrustEvidence(successes=5, failures=5),
        )
        many_obs = DataSource(
            name="many",
            trust_score=0.8,
            latency_ms=50,
            cost_per_query=0.1,
            freshness_score=0.9,
            evidence=TrustEvidence(successes=50, failures=50),
        )

        s_few, _ = self.scorer.score(few_obs, _DEFAULT_REQUEST)
        s_many, _ = self.scorer.score(many_obs, _DEFAULT_REQUEST)
        self.assertGreater(s_few, s_many)

    def test_zero_exploration_bonus_matches_linear_scorer(self) -> None:
        scorer_no_bonus = BayesianUCBScorer(exploration_bonus=0.0)
        linear = LinearWeightedScorer()
        src = _make_source()
        s_ucb, _ = scorer_no_bonus.score(src, _DEFAULT_REQUEST)
        s_linear, _ = linear.score(src, _DEFAULT_REQUEST)
        self.assertAlmostEqual(s_ucb, s_linear, places=5)

    def test_score_in_unit_interval(self) -> None:
        src = _make_source()
        score, _ = self.scorer.score(src, _DEFAULT_REQUEST)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
