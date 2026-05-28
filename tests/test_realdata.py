import unittest

from trust_aware import load_wdbc_dataset, run_wdbc_real_study
from trust_aware.realdata import evaluate_wdbc_sources
from trust_aware.stats import paired_difference_ci, wilson_interval


class RealDataStudyTests(unittest.TestCase):
    def test_loads_public_wdbc_dataset(self) -> None:
        records = load_wdbc_dataset()

        self.assertEqual(len(records), 569)
        self.assertEqual(len(records[0].features), 30)
        self.assertEqual(sum(record.malignant for record in records), 212)

    def test_cross_validated_sources_have_intervals(self) -> None:
        records = load_wdbc_dataset()
        evaluations = evaluate_wdbc_sources(
            records,
            folds=5,
            seed=3,
            bootstrap_resamples=100,
        )

        self.assertGreaterEqual(len(evaluations), 5)
        best = max(evaluations, key=lambda item: item.counts.balanced_accuracy)
        self.assertGreater(best.counts.sensitivity, 0.85)
        self.assertGreater(best.balanced_accuracy_summary.interval.lower, 0.80)

    def test_real_study_reports_policy_decisions(self) -> None:
        report = run_wdbc_real_study(seed=5, bootstrap_resamples=100)
        markdown = report.to_markdown()

        self.assertEqual(report.row_count, 569)
        self.assertIn("Wisconsin Diagnostic Breast Cancer", markdown)
        self.assertIn("bootstrap", markdown)
        self.assertGreaterEqual(len(report.policy_decisions), 4)

    def test_statistical_intervals_are_well_ordered(self) -> None:
        interval = wilson_interval(92, 100)
        paired = paired_difference_ci([0.9, 0.8, 0.85], [0.7, 0.75, 0.8], seed=11)

        self.assertLess(interval.lower, interval.estimate)
        self.assertLess(interval.estimate, interval.upper)
        self.assertGreater(paired.estimate, 0.0)


if __name__ == "__main__":
    unittest.main()
