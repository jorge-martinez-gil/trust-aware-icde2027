"""Tests for TrustGraph: trust propagation and augmented sources."""
import unittest

from trust_aware import DataSource, TrustGraph


def _src(name: str, trust: float, latency: float = 50.0) -> DataSource:
    return DataSource(
        name=name,
        trust_score=trust,
        latency_ms=latency,
        cost_per_query=0.1,
        freshness_score=0.9,
    )


class TrustGraphPropagationTests(unittest.TestCase):
    def test_isolated_nodes_preserve_relative_order(self) -> None:
        """Without edges, higher static trust → higher propagated trust."""
        g = TrustGraph()
        high = _src("high", trust=0.9)
        low = _src("low", trust=0.2)
        g.add_source(high)
        g.add_source(low)

        scores = g.propagated_trust()
        self.assertGreater(scores["high"], scores["low"])

    def test_propagated_trust_values_in_unit_interval(self) -> None:
        g = TrustGraph()
        for i, t in enumerate([0.1, 0.5, 0.9]):
            g.add_source(_src(f"s{i}", trust=t))
        for v in g.propagated_trust().values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_trust_flows_from_endorser_to_endorsed(self) -> None:
        """A high-trust source endorsing a low-trust source should raise the
        low-trust source's propagated score compared to isolation."""
        g_plain = TrustGraph()
        g_endorsed = TrustGraph()

        endorser = _src("endorser", trust=0.95)
        endorsed = _src("endorsed", trust=0.1)

        for g in (g_plain, g_endorsed):
            g.add_source(endorser)
            g.add_source(endorsed)

        g_endorsed.add_trust_edge("endorser", "endorsed", weight=1.0)

        plain_score = g_plain.propagated_trust()["endorsed"]
        endorsed_score = g_endorsed.propagated_trust()["endorsed"]
        self.assertGreater(endorsed_score, plain_score)

    def test_convergence_with_cycle(self) -> None:
        """Power iteration must converge even with cycles in the graph."""
        g = TrustGraph()
        a, b, c = _src("a", 0.8), _src("b", 0.5), _src("c", 0.3)
        for s in (a, b, c):
            g.add_source(s)
        g.add_trust_edge("a", "b", 0.7)
        g.add_trust_edge("b", "c", 0.6)
        g.add_trust_edge("c", "a", 0.5)  # cycle

        scores = g.propagated_trust()
        self.assertEqual(set(scores.keys()), {"a", "b", "c"})
        for v in scores.values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_empty_graph_returns_empty_dict(self) -> None:
        g = TrustGraph()
        self.assertEqual(g.propagated_trust(), {})

    def test_augmented_sources_trust_scores_in_unit_interval(self) -> None:
        g = TrustGraph()
        for i in range(3):
            g.add_source(_src(f"s{i}", trust=0.3 * i + 0.1))
        for source in g.augmented_sources():
            self.assertGreaterEqual(source.trust_score, 0.0)
            self.assertLessEqual(source.trust_score, 1.0)

    def test_augmented_sources_count_matches_nodes(self) -> None:
        g = TrustGraph()
        sources = [_src(f"s{i}", trust=0.5) for i in range(5)]
        for s in sources:
            g.add_source(s)
        self.assertEqual(len(g.augmented_sources()), 5)

    def test_invalid_damping_raises(self) -> None:
        with self.assertRaises(ValueError):
            TrustGraph(damping=0.0)
        with self.assertRaises(ValueError):
            TrustGraph(damping=1.0)

    def test_blend_parameter_controls_propagation_weight(self) -> None:
        """blend=0 should return static scores; blend=1 should use propagated."""
        g = TrustGraph()
        g.add_source(_src("a", trust=0.8))
        g.add_source(_src("b", trust=0.2))
        g.add_trust_edge("a", "b", 1.0)

        augmented_blend0 = {s.name: s.trust_score for s in g.augmented_sources(blend=0.0)}
        # With blend=0, blended = static trust (no propagation weight)
        self.assertAlmostEqual(augmented_blend0["a"], 0.8, places=5)
        self.assertAlmostEqual(augmented_blend0["b"], 0.2, places=5)


if __name__ == "__main__":
    unittest.main()
