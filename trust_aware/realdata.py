from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .models import Capability, DataSource, QueryRequest, TrustEvidence
from .optimizer import TrustAwareQueryOptimizer
from .policies import (
    BALANCED,
    COST_EFFICIENT,
    HIGH_ASSURANCE,
    LATENCY_CRITICAL,
    TRUST_ONLY,
    TrustPolicy,
)
from .stats import (
    ConfidenceInterval,
    DistributionSummary,
    summarize_distribution,
    wilson_interval,
)


WDBC_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/wdbc.data"
)
WDBC_CITATION = (
    "W.N. Street, W.H. Wolberg and O.L. Mangasarian, Nuclear feature extraction "
    "for breast tumor diagnosis, IS&T/SPIE 1993."
)

FEATURE_NAMES = (
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
)


@dataclass(frozen=True)
class WDBCRecord:
    sample_id: str
    diagnosis: str
    features: tuple[float, ...]

    @property
    def malignant(self) -> bool:
        return self.diagnosis == "M"


@dataclass(frozen=True)
class FeatureProfile:
    name: str
    feature_names: tuple[str, ...]
    latency_ms: float
    cost_per_query: float
    privacy_risk: float


@dataclass(frozen=True)
class ConfusionCounts:
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def total(self) -> int:
        return (
            self.true_positive
            + self.true_negative
            + self.false_positive
            + self.false_negative
        )

    @property
    def accuracy(self) -> float:
        return _safe_rate(self.true_positive + self.true_negative, self.total)

    @property
    def sensitivity(self) -> float:
        return _safe_rate(self.true_positive, self.true_positive + self.false_negative)

    @property
    def specificity(self) -> float:
        return _safe_rate(self.true_negative, self.true_negative + self.false_positive)

    @property
    def balanced_accuracy(self) -> float:
        return (self.sensitivity + self.specificity) / 2.0

    @property
    def false_negative_rate(self) -> float:
        return 1.0 - self.sensitivity

    @property
    def false_positive_rate(self) -> float:
        return 1.0 - self.specificity


@dataclass(frozen=True)
class SourceEvaluation:
    profile: FeatureProfile
    counts: ConfusionCounts
    fold_balanced_accuracy: tuple[float, ...]
    accuracy_interval: ConfidenceInterval
    sensitivity_interval: ConfidenceInterval
    specificity_interval: ConfidenceInterval
    balanced_accuracy_summary: DistributionSummary

    @property
    def source_name(self) -> str:
        return self.profile.name

    def to_data_source(self) -> DataSource:
        total_positive = self.counts.true_positive + self.counts.false_negative
        total_negative = self.counts.true_negative + self.counts.false_positive
        return DataSource(
            name=self.profile.name,
            trust_score=self.counts.balanced_accuracy,
            latency_ms=self.profile.latency_ms,
            cost_per_query=self.profile.cost_per_query,
            freshness_score=1.0,
            source_type="wdbc_diagnostic_rule",
            capabilities={Capability.STRUCTURED_SQL, Capability.POLICY_AUDIT},
            trust_evidence=(
                TrustEvidence(
                    "cross_validated_accuracy",
                    positive=self.counts.true_positive + self.counts.true_negative,
                    total=self.counts.total,
                    weight=0.30,
                ),
                TrustEvidence(
                    "malignant_sensitivity",
                    positive=self.counts.true_positive,
                    total=total_positive,
                    weight=0.50,
                ),
                TrustEvidence(
                    "benign_specificity",
                    positive=self.counts.true_negative,
                    total=total_negative,
                    weight=0.20,
                ),
            ),
            coverage=1.0,
            privacy_risk=self.profile.privacy_risk,
            hallucination_risk=self.counts.false_negative_rate,
            schema_reliability=1.0,
            metadata={
                "dataset": "UCI WDBC",
                "features": ",".join(self.profile.feature_names),
                "accuracy_ci95": self.accuracy_interval.as_percent(),
                "sensitivity_ci95": self.sensitivity_interval.as_percent(),
                "specificity_ci95": self.specificity_interval.as_percent(),
            },
        )

    def to_markdown_row(self) -> str:
        balanced_interval = self.balanced_accuracy_summary.interval
        return (
            f"| {self.profile.name} | {len(self.profile.feature_names)} | "
            f"{balanced_interval.as_percent()} | "
            f"{self.sensitivity_interval.as_percent()} | "
            f"{self.specificity_interval.as_percent()} | "
            f"{self.accuracy_interval.as_percent()} | "
            f"{self.counts.false_negative_rate:.3f} |"
        )


@dataclass(frozen=True)
class RealPolicyDecision:
    policy_name: str
    selected_source: str
    score: float
    trust_lower_bound: float
    latency_ms: float
    cost_per_query: float
    sensitivity: float
    specificity: float
    false_negative_rate: float

    def to_markdown_row(self) -> str:
        return (
            f"| {self.policy_name} | {self.selected_source} | "
            f"{self.score:.3f} | {self.trust_lower_bound:.3f} | "
            f"{self.sensitivity:.3f} | {self.specificity:.3f} | "
            f"{self.false_negative_rate:.3f} | {self.latency_ms:.1f} | "
            f"{self.cost_per_query:.3f} |"
        )


@dataclass(frozen=True)
class RealDatasetReport:
    dataset_name: str
    dataset_path: Path
    row_count: int
    feature_count: int
    malignant_count: int
    benign_count: int
    folds: int
    source_evaluations: tuple[SourceEvaluation, ...]
    policy_decisions: tuple[RealPolicyDecision, ...]

    @property
    def best_source(self) -> SourceEvaluation:
        return max(
            self.source_evaluations,
            key=lambda item: (
                item.counts.balanced_accuracy,
                item.counts.sensitivity,
                -item.profile.latency_ms,
            ),
        )

    def to_markdown(self) -> str:
        lines = [
            "# Real-Data Trust Study: Wisconsin Diagnostic Breast Cancer",
            "",
            f"- Rows: {self.row_count}",
            f"- Features: {self.feature_count}",
            f"- Malignant / benign: {self.malignant_count} / {self.benign_count}",
            f"- Cross-validation folds: {self.folds}",
            f"- Source: {WDBC_CITATION}",
            "- Intervals: Wilson score for binomial metrics; bootstrap over "
            "fold-level balanced accuracy.",
            "",
            "## Source Reliability",
            "| source | features | bal_acc bootstrap 95% CI | sensitivity 95% CI | specificity 95% CI | accuracy 95% CI | fnr |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        lines.extend(
            evaluation.to_markdown_row()
            for evaluation in sorted(
                self.source_evaluations,
                key=lambda item: item.counts.balanced_accuracy,
                reverse=True,
            )
        )
        lines.extend(
            [
                "",
                "## Policy Decisions",
                "| policy | selected source | score | trust_lb | sensitivity | specificity | fnr | latency_ms | cost |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        lines.extend(decision.to_markdown_row() for decision in self.policy_decisions)
        return "\n".join(lines)


DEFAULT_FEATURE_PROFILES = (
    FeatureProfile(
        "compact-morphology",
        ("radius_mean", "texture_mean", "perimeter_mean", "area_mean"),
        latency_ms=22,
        cost_per_query=0.04,
        privacy_risk=0.05,
    ),
    FeatureProfile(
        "nuclear-shape-risk",
        ("compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean"),
        latency_ms=34,
        cost_per_query=0.07,
        privacy_risk=0.07,
    ),
    FeatureProfile(
        "worst-case-specialist",
        ("radius_worst", "texture_worst", "perimeter_worst", "area_worst"),
        latency_ms=48,
        cost_per_query=0.10,
        privacy_risk=0.10,
    ),
    FeatureProfile(
        "texture-screen",
        ("texture_mean", "texture_se", "texture_worst"),
        latency_ms=14,
        cost_per_query=0.02,
        privacy_risk=0.04,
    ),
    FeatureProfile(
        "full-cytology-panel",
        FEATURE_NAMES,
        latency_ms=86,
        cost_per_query=0.18,
        privacy_risk=0.18,
    ),
)


def default_wdbc_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "wdbc.data"


def load_wdbc_dataset(path: str | Path | None = None) -> tuple[WDBCRecord, ...]:
    data_path = Path(path) if path is not None else default_wdbc_path()
    records: list[WDBCRecord] = []
    with data_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if len(row) != 32:
                raise ValueError(f"expected 32 columns in WDBC row, got {len(row)}")
            records.append(
                WDBCRecord(
                    sample_id=row[0],
                    diagnosis=row[1],
                    features=tuple(float(value) for value in row[2:]),
                )
            )
    return tuple(records)


def evaluate_wdbc_sources(
    records: Sequence[WDBCRecord],
    *,
    profiles: Sequence[FeatureProfile] = DEFAULT_FEATURE_PROFILES,
    folds: int = 5,
    seed: int = 7,
    bootstrap_resamples: int = 1000,
) -> tuple[SourceEvaluation, ...]:
    fold_indices = stratified_folds(records, folds=folds, seed=seed)
    evaluations = []
    for profile_index, profile in enumerate(profiles):
        feature_indices = tuple(FEATURE_NAMES.index(name) for name in profile.feature_names)
        counts = ConfusionCounts()
        fold_scores = []
        for test_indices in fold_indices:
            test_set = set(test_indices)
            train_indices = [index for index in range(len(records)) if index not in test_set]
            model = _fit_threshold_model(records, train_indices, feature_indices)
            fold_counts = _evaluate_threshold_model(
                records, test_indices, feature_indices, model
            )
            counts = _add_counts(counts, fold_counts)
            fold_scores.append(fold_counts.balanced_accuracy)

        evaluations.append(
            SourceEvaluation(
                profile=profile,
                counts=counts,
                fold_balanced_accuracy=tuple(fold_scores),
                accuracy_interval=wilson_interval(
                    counts.true_positive + counts.true_negative,
                    counts.total,
                    confidence=0.95,
                ),
                sensitivity_interval=wilson_interval(
                    counts.true_positive,
                    counts.true_positive + counts.false_negative,
                    confidence=0.95,
                ),
                specificity_interval=wilson_interval(
                    counts.true_negative,
                    counts.true_negative + counts.false_positive,
                    confidence=0.95,
                ),
                balanced_accuracy_summary=summarize_distribution(
                    fold_scores,
                    confidence=0.95,
                    bootstrap_resamples=bootstrap_resamples,
                    seed=seed + profile_index,
                ),
            )
        )
    return tuple(evaluations)


def wdbc_request() -> QueryRequest:
    return QueryRequest(
        query="Route a WDBC diagnostic support query to the most trustworthy audited source.",
        required_capabilities={Capability.STRUCTURED_SQL, Capability.POLICY_AUDIT},
        max_latency_ms=100,
        min_trust_score=0.70,
        max_privacy_risk=0.25,
        max_hallucination_risk=0.20,
        trust_weight=0.52,
        latency_weight=0.10,
        freshness_weight=0.04,
        cost_weight=0.05,
        coverage_weight=0.06,
        risk_weight=0.23,
        uncertainty_weight=0.05,
        risk_tolerance=0.05,
    )


def run_wdbc_real_study(
    path: str | Path | None = None,
    *,
    folds: int = 5,
    seed: int = 7,
    bootstrap_resamples: int = 1000,
    policies: Iterable[TrustPolicy] = (
        HIGH_ASSURANCE,
        BALANCED,
        LATENCY_CRITICAL,
        COST_EFFICIENT,
        TRUST_ONLY,
    ),
) -> RealDatasetReport:
    records = load_wdbc_dataset(path)
    evaluations = evaluate_wdbc_sources(
        records,
        folds=folds,
        seed=seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    sources = tuple(evaluation.to_data_source() for evaluation in evaluations)
    by_source = {evaluation.profile.name: evaluation for evaluation in evaluations}
    request = wdbc_request()
    optimizer = TrustAwareQueryOptimizer()
    decisions = []
    for policy in policies:
        plan = optimizer.optimize(sources, policy.apply(request))
        if plan.primary_source is None or not plan.steps:
            continue
        source = plan.primary_source
        evaluation = by_source[source.name]
        decisions.append(
            RealPolicyDecision(
                policy_name=policy.name,
                selected_source=source.name,
                score=plan.steps[0].score,
                trust_lower_bound=plan.steps[0].breakdown.components["trust"],
                latency_ms=source.latency_ms,
                cost_per_query=source.cost_per_query,
                sensitivity=evaluation.counts.sensitivity,
                specificity=evaluation.counts.specificity,
                false_negative_rate=evaluation.counts.false_negative_rate,
            )
        )

    malignant_count = sum(1 for record in records if record.malignant)
    benign_count = len(records) - malignant_count
    return RealDatasetReport(
        dataset_name="Wisconsin Diagnostic Breast Cancer",
        dataset_path=Path(path) if path is not None else default_wdbc_path(),
        row_count=len(records),
        feature_count=len(FEATURE_NAMES),
        malignant_count=malignant_count,
        benign_count=benign_count,
        folds=folds,
        source_evaluations=evaluations,
        policy_decisions=tuple(decisions),
    )


def stratified_folds(
    records: Sequence[WDBCRecord],
    *,
    folds: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    folds = max(2, int(folds))
    malignant = [index for index, record in enumerate(records) if record.malignant]
    benign = [index for index, record in enumerate(records) if not record.malignant]
    rng = random.Random(seed)
    rng.shuffle(malignant)
    rng.shuffle(benign)
    buckets = [[] for _ in range(folds)]
    for group in (malignant, benign):
        for index, record_index in enumerate(group):
            buckets[index % folds].append(record_index)
    return tuple(tuple(sorted(bucket)) for bucket in buckets)


def _fit_threshold_model(
    records: Sequence[WDBCRecord],
    indices: Sequence[int],
    feature_indices: Sequence[int],
) -> tuple[tuple[float, ...], tuple[float, ...], float, bool]:
    means, scales = _feature_scaler(records, indices, feature_indices)
    scored = [
        (_score_record(records[index], feature_indices, means, scales), records[index].malignant)
        for index in indices
    ]
    unique_scores = sorted({score for score, _ in scored})
    if len(unique_scores) == 1:
        thresholds = unique_scores
    else:
        thresholds = [
            (left + right) / 2.0
            for left, right in zip(unique_scores, unique_scores[1:])
        ]
        thresholds.insert(0, unique_scores[0] - 1e-9)
        thresholds.append(unique_scores[-1] + 1e-9)

    best_key = (-1.0, -1.0, -1.0)
    best = (0.0, True)
    for high_is_malignant in (True, False):
        for threshold in thresholds:
            counts = _counts_for_scores(scored, threshold, high_is_malignant)
            key = (counts.balanced_accuracy, counts.sensitivity, -abs(threshold))
            if key > best_key:
                best_key = key
                best = (threshold, high_is_malignant)
    return means, scales, best[0], best[1]


def _evaluate_threshold_model(
    records: Sequence[WDBCRecord],
    indices: Sequence[int],
    feature_indices: Sequence[int],
    model: tuple[tuple[float, ...], tuple[float, ...], float, bool],
) -> ConfusionCounts:
    means, scales, threshold, high_is_malignant = model
    scored = [
        (_score_record(records[index], feature_indices, means, scales), records[index].malignant)
        for index in indices
    ]
    return _counts_for_scores(scored, threshold, high_is_malignant)


def _feature_scaler(
    records: Sequence[WDBCRecord],
    indices: Sequence[int],
    feature_indices: Sequence[int],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    means = []
    scales = []
    for feature_index in feature_indices:
        values = [records[index].features[feature_index] for index in indices]
        center = sum(values) / len(values)
        variance = sum((value - center) ** 2 for value in values) / max(1, len(values) - 1)
        scale = variance**0.5 or 1.0
        means.append(center)
        scales.append(scale)
    return tuple(means), tuple(scales)


def _score_record(
    record: WDBCRecord,
    feature_indices: Sequence[int],
    means: Sequence[float],
    scales: Sequence[float],
) -> float:
    standardized = [
        (record.features[feature_index] - mean) / scale
        for feature_index, mean, scale in zip(feature_indices, means, scales)
    ]
    return sum(standardized) / len(standardized)


def _counts_for_scores(
    scored: Iterable[tuple[float, bool]],
    threshold: float,
    high_is_malignant: bool,
) -> ConfusionCounts:
    tp = tn = fp = fn = 0
    for score, malignant in scored:
        predicted = score >= threshold if high_is_malignant else score <= threshold
        if predicted and malignant:
            tp += 1
        elif predicted and not malignant:
            fp += 1
        elif not predicted and malignant:
            fn += 1
        else:
            tn += 1
    return ConfusionCounts(tp, tn, fp, fn)


def _add_counts(left: ConfusionCounts, right: ConfusionCounts) -> ConfusionCounts:
    return ConfusionCounts(
        true_positive=left.true_positive + right.true_positive,
        true_negative=left.true_negative + right.true_negative,
        false_positive=left.false_positive + right.false_positive,
        false_negative=left.false_negative + right.false_negative,
    )


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator
