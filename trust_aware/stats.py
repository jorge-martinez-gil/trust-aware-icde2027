from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ConfidenceInterval:
    """A scalar estimate with a confidence interval."""

    estimate: float
    lower: float
    upper: float
    confidence: float = 0.95

    def as_percent(self) -> str:
        return (
            f"{100 * self.estimate:.1f}% "
            f"[{100 * self.lower:.1f}, {100 * self.upper:.1f}]"
        )


@dataclass(frozen=True)
class DistributionSummary:
    """Compact descriptive statistics for repeated measurements."""

    count: int
    mean: float
    stddev: float
    standard_error: float
    interval: ConfidenceInterval


def summarize_distribution(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    bootstrap_resamples: int = 1000,
    seed: int = 7,
) -> DistributionSummary:
    values = tuple(float(value) for value in values)
    if not values:
        interval = ConfidenceInterval(0.0, 0.0, 0.0, confidence)
        return DistributionSummary(0, 0.0, 0.0, 0.0, interval)

    estimate = mean(values)
    stddev = sample_stddev(values)
    standard_error = stddev / math.sqrt(len(values)) if values else 0.0
    interval = bootstrap_mean_ci(
        values,
        confidence=confidence,
        resamples=bootstrap_resamples,
        seed=seed,
    )
    return DistributionSummary(
        count=len(values),
        mean=estimate,
        stddev=stddev,
        standard_error=standard_error,
        interval=interval,
    )


def mean(values: Iterable[float]) -> float:
    values = tuple(float(value) for value in values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def sample_stddev(values: Iterable[float]) -> float:
    values = tuple(float(value) for value in values)
    if len(values) < 2:
        return 0.0
    center = mean(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def wilson_interval(
    successes: float,
    total: float,
    *,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Wilson score interval for a binomial proportion."""

    if total <= 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, confidence)

    z = z_value(confidence)
    proportion = successes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            (proportion * (1.0 - proportion) + z**2 / (4.0 * total)) / total
        )
        / denominator
    )
    return ConfidenceInterval(
        estimate=proportion,
        lower=max(0.0, center - margin),
        upper=min(1.0, center + margin),
        confidence=confidence,
    )


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 7,
) -> ConfidenceInterval:
    if not values:
        return ConfidenceInterval(0.0, 0.0, 0.0, confidence)

    values = tuple(float(value) for value in values)
    rng = random.Random(seed)
    sample_size = len(values)
    estimates = []
    for _ in range(max(1, int(resamples))):
        draw = [values[rng.randrange(sample_size)] for _ in range(sample_size)]
        estimates.append(mean(draw))
    alpha = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        estimate=mean(values),
        lower=quantile(estimates, alpha),
        upper=quantile(estimates, 1.0 - alpha),
        confidence=confidence,
    )


def paired_difference_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 7,
) -> ConfidenceInterval:
    if len(left) != len(right):
        raise ValueError("paired samples must have the same length")
    differences = tuple(float(a) - float(b) for a, b in zip(left, right))
    return bootstrap_mean_ci(
        differences,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    probability = min(1.0, max(0.0, probability))
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def z_value(confidence: float) -> float:
    if confidence >= 0.995:
        return 2.81
    if confidence >= 0.99:
        return 2.58
    if confidence >= 0.975:
        return 2.24
    if confidence >= 0.95:
        return 1.96
    if confidence >= 0.90:
        return 1.64
    return 1.28
