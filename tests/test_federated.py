"""Tests for the federated trust-aware retrieval study.

These validate: (1) the real testbed parses to the expected shape, (2) BM25
retrieval is sane, (3) the resource-selection methods are well-formed, and
(4) the headline scientific claim holds -- trust-aware routing is far more
robust to untrusted duplicate sources than content-based selection.
"""

import unittest

import numpy as np

from trust_aware.federated.testbed import Collection, load_testbed
from trust_aware.federated.retrieval import build_federation, tokenize
from trust_aware.federated.selection import (
    CORISelector, CORIMeanSelector, CORIUCBSelector, ReDDESelector,
    TrustAwareSelector, build_selectors,
)
from trust_aware.federated.experiment import (
    build_federation_with_untrusted, build_query_stream, evaluate_setting,
    aggregate, experiment_metadata, make_mirror_collection,
)
from trust_aware.federated.statistics import (
    cliffs_delta, compare_methods_clustered, holm_bonferroni,
    paired_rank_biserial, paired_wilcoxon,
)
from trust_aware.federated.report import significance_analysis


class TestTestbed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = load_testbed()

    def test_five_collections(self):
        self.assertEqual(set(self.tb.collections), {
            "CACM", "MED", "NPL", "CRAN", "CISI"})

    def test_doc_counts(self):
        # Known sizes of the classic collections.
        self.assertEqual(self.tb.collections["CACM"].num_docs(), 3204)
        self.assertEqual(self.tb.collections["MED"].num_docs(), 1033)
        self.assertEqual(self.tb.collections["NPL"].num_docs(), 11429)
        self.assertEqual(self.tb.collections["CRAN"].num_docs(), 1400)

    def test_qrels_reference_real_docs(self):
        for c in self.tb.collections.values():
            for qid, rels in c.qrels.items():
                for d in rels:
                    self.assertIn(d, c.docs)

    def test_usable_queries_have_qrels(self):
        total = sum(c.num_queries() for c in self.tb.collections.values())
        self.assertGreater(total, 400)


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = load_testbed()
        cls.fed = build_federation(cls.tb)

    def test_tokenize_drops_stopwords(self):
        self.assertNotIn("the", tokenize("the cat and the dog"))

    def test_bm25_retrieves_relevant(self):
        # MED is a strong collection; mean nDCG@10 should be clearly non-trivial.
        c = self.tb.collections["MED"]
        idx = self.fed.indexes["MED"]
        ndcgs = []
        for qid in c.usable_query_ids():
            ranked = [d for d, _ in idx.search(c.queries[qid], top_k=10)]
            rel = c.qrels[qid]
            dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked) if d in rel)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(10, len(rel))))
            ndcgs.append(dcg / idcg if idcg else 0.0)
        self.assertGreater(np.mean(ndcgs), 0.5)


class TestSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tb = load_testbed()
        cls.fed = build_federation(cls.tb)

    def test_selectors_rank_all_sources(self):
        for sel in [CORISelector(self.fed), ReDDESelector(self.fed),
                    CORIMeanSelector(self.fed), CORIUCBSelector(self.fed),
                    TrustAwareSelector(self.fed)]:
            order = sel.rank_sources("computer algorithm")
            self.assertEqual(set(order), set(self.fed.source_names()))

    def test_online_reputation_demotes_on_failure(self):
        selectors = [
            (CORIMeanSelector(self.fed), "mean"),
            (CORIUCBSelector(self.fed), "upper_bound"),
            (TrustAwareSelector(self.fed), "lower_bound"),
        ]
        for sel, key in selectors:
            before = sel.trust_report()["CACM"][key]
            for _ in range(20):
                sel.observe("CACM", False)
            self.assertLess(sel.trust_report()["CACM"][key], before)

    def test_seeded_ties_are_reproducible_for_all_selectors(self):
        first = build_selectors(self.fed, {}, seed=23)
        second = build_selectors(self.fed, {}, seed=23)
        for method in ["CORI", "ReDDE", "CORI+Mean", "CORI+UCB", "Trust-Aware"]:
            self.assertEqual(
                first[method].rank_sources("term_not_present_anywhere"),
                second[method].rank_sources("term_not_present_anywhere"),
            )


class TestExperimentConstruction(unittest.TestCase):
    def test_corrupted_mirror_changes_content_and_preserves_fresh_ids(self):
        source = Collection(
            name="SOURCE",
            domain="test",
            docs={"1": "alpha beta gamma", "2": "delta epsilon"},
            queries={},
            qrels={},
        )
        mirror = make_mirror_collection("TWIN", source, seed=7, corruption=1.0)
        self.assertEqual(set(mirror.docs), {"tw_1", "tw_2"})
        self.assertNotEqual(mirror.docs["tw_1"], source.docs["1"])
        self.assertTrue(all(token.startswith("corrupt_")
                            for token in mirror.docs["tw_1"].split()))

    def test_protocol_metadata_is_explicit(self):
        metadata = experiment_metadata(seed=11, docs_each=1500)
        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["evaluation"], "prequential")
        self.assertEqual(metadata["seed"], 11)
        self.assertIn("corrupted-20", metadata["adversary_variants"])

    def test_adversary_variants_are_named(self):
        tb = load_testbed()
        tb_variant, _fed_variant = build_federation_with_untrusted(
            tb, n_untrusted=1, docs_each=1500, seed=7, adversary="partial-50"
        )
        names = set(tb_variant.collections)
        self.assertTrue(any(name.startswith("TWIN-partial-50") for name in names))


class TestStatistics(unittest.TestCase):
    def test_cliffs_delta_bounds(self):
        self.assertAlmostEqual(cliffs_delta([3, 4, 5], [0, 1, 2]), 1.0)
        self.assertAlmostEqual(cliffs_delta([0, 1, 2], [0, 1, 2]), 0.0)

    def test_wilcoxon_identical(self):
        _, p = paired_wilcoxon([1, 2, 3], [1, 2, 3])
        self.assertEqual(p, 1.0)

    def test_holm_monotone(self):
        out = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.2})
        self.assertLessEqual(out["a"]["p_holm"], out["b"]["p_holm"])
        self.assertLessEqual(out["b"]["p_holm"], out["c"]["p_holm"])

    def test_paired_rank_biserial_respects_direction(self):
        self.assertAlmostEqual(paired_rank_biserial([3, 4, 5], [0, 1, 2]), 1.0)
        self.assertAlmostEqual(paired_rank_biserial([0, 1, 2], [3, 4, 5]), -1.0)

    def test_clustered_analysis_uses_queries_not_seed_rows(self):
        records = {
            "Trust-Aware": [
                {"qid": qid, "score": score}
                for qid, score in [
                    ("q1", 1.0), ("q1", 0.8), ("q1", 0.9),
                    ("q2", 0.7), ("q2", 0.8), ("q2", 0.9),
                ]
            ],
            "CORI": [
                {"qid": qid, "score": score}
                for qid, score in [
                    ("q1", 0.2), ("q1", 0.3), ("q1", 0.4),
                    ("q2", 0.1), ("q2", 0.2), ("q2", 0.3),
                ]
            ],
        }
        row = compare_methods_clustered(
            records, "score", "Trust-Aware", ["CORI"]
        )["comparisons"]["CORI"]
        self.assertEqual(row["n_clusters"], 2)
        self.assertEqual(row["n_observations"], 6)
        self.assertEqual((row["wins"], row["ties"], row["losses"]), (2, 0, 0))
        self.assertAlmostEqual(row["rank_biserial"], 1.0)


class TestHeadlineClaim(unittest.TestCase):
    """Trust-aware routing is robust to untrusted duplicate sources."""

    def test_trust_beats_content_selection_under_twins(self):
        tb = load_testbed()
        tb6, fed6 = build_federation_with_untrusted(tb, n_untrusted=6,
                                                    docs_each=1500, seed=7)
        res = evaluate_setting(tb6, fed6, seed=7)
        agg = aggregate(res["records"])
        # With 6 evil-twins, trust-aware retains far higher routing accuracy.
        self.assertGreater(agg["Trust-Aware"]["R@1"], 0.70)
        self.assertLess(agg["CORI"]["R@1"], 0.55)
        self.assertGreater(
            agg["Trust-Aware"]["R@1"] - agg["CORI"]["R@1"], 0.25)

    def test_cached_confirmatory_family_is_query_clustered_and_significant(self):
        analysis = significance_analysis()
        self.assertEqual(analysis["independent_unit"], "query")
        self.assertEqual(len(analysis["comparisons"]), 4)
        for comparison in analysis["comparisons"].values():
            self.assertEqual(comparison["n_clusters"], 476)
            self.assertEqual(comparison["n_observations"], 1428)
            self.assertTrue(comparison["reject_H0"])
            self.assertGreater(comparison["rank_biserial"], 0.70)


if __name__ == "__main__":
    unittest.main()
