from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from scipy import stats

from reasonsec.config import ConfigError


@dataclass
class TestResult:
    name: str
    statistic: float
    p_value: float
    detail: dict[str, float]

    @property
    def significant_at(self) -> float:
        return self.p_value

    def as_dict(self) -> dict[str, object]:
        return {"test": self.name, "statistic": self.statistic, "p_value": self.p_value, "detail": self.detail}


def mcnemar_test(left: Sequence[int], right: Sequence[int], exact: bool, continuity_correction: bool) -> TestResult:
    if len(left) != len(right):
        raise ConfigError("McNemar's test requires paired outcome vectors of equal length")
    left_only = sum(1 for a, b in zip(left, right) if a == 1 and b == 0)
    right_only = sum(1 for a, b in zip(left, right) if a == 0 and b == 1)
    discordant = left_only + right_only
    if discordant == 0:
        return TestResult(
            name="mcnemar",
            statistic=0.0,
            p_value=1.0,
            detail={"left_only": float(left_only), "right_only": float(right_only), "discordant": 0.0},
        )
    if exact:
        p_value = float(stats.binomtest(left_only, discordant, 0.5).pvalue)
        statistic = float(min(left_only, right_only))
    else:
        numerator = abs(left_only - right_only) - (1.0 if continuity_correction else 0.0)
        statistic = float(max(numerator, 0.0) ** 2 / discordant)
        p_value = float(stats.chi2.sf(statistic, df=1))
    return TestResult(
        name="mcnemar",
        statistic=statistic,
        p_value=p_value,
        detail={
            "left_only": float(left_only),
            "right_only": float(right_only),
            "discordant": float(discordant),
        },
    )


def paired_t_test(left: Sequence[float], right: Sequence[float]) -> TestResult:
    if len(left) != len(right):
        raise ConfigError("the paired t-test requires vectors of equal length")
    if len(left) < 2:
        raise ConfigError("the paired t-test requires at least two paired observations")
    statistic, p_value = stats.ttest_rel(list(left), list(right))
    return TestResult(
        name="paired_t",
        statistic=float(statistic),
        p_value=float(p_value),
        detail={"n": float(len(left)), "mean_difference": float(sum(a - b for a, b in zip(left, right)) / len(left))},
    )


def proportion_confidence_interval(successes: int, total: int, confidence: float) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    lower, upper = stats.beta.interval(
        confidence,
        max(successes, 1e-9),
        max(total - successes, 1e-9),
    )
    return float(0.0 if successes == 0 else lower), float(1.0 if successes == total else upper)


def minimum_detectable_difference(sample_size: int, baseline_rate: float, power: float, alpha: float) -> float:
    if sample_size <= 0:
        return 0.0
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    variance = baseline_rate * (1.0 - baseline_rate)
    return float((z_alpha + z_power) * math.sqrt(2.0 * variance / sample_size))


def mean_and_standard_deviation(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(sum(values) / len(values))
    if len(values) == 1:
        return mean, 0.0
    variance = float(sum((value - mean) ** 2 for value in values) / len(values))
    return mean, float(math.sqrt(variance))


def cohens_kappa(first: Sequence[int], second: Sequence[int], categories: int) -> float:
    if len(first) != len(second) or not first:
        raise ConfigError("Cohen's kappa requires two rating vectors of equal, non-zero length")
    total = len(first)
    agreement = sum(1 for a, b in zip(first, second) if a == b) / total
    expected = 0.0
    for category in range(categories):
        first_fraction = sum(1 for value in first if value == category) / total
        second_fraction = sum(1 for value in second if value == category) / total
        expected += first_fraction * second_fraction
    if expected >= 1.0:
        return 1.0
    return float((agreement - expected) / (1.0 - expected))


def classification_scores(
    predicted_positive: Sequence[int], actual_positive: Sequence[int]
) -> dict[str, float]:
    true_positive = sum(1 for p, a in zip(predicted_positive, actual_positive) if p == 1 and a == 1)
    false_positive = sum(1 for p, a in zip(predicted_positive, actual_positive) if p == 1 and a == 0)
    false_negative = sum(1 for p, a in zip(predicted_positive, actual_positive) if p == 0 and a == 1)
    true_negative = sum(1 for p, a in zip(predicted_positive, actual_positive) if p == 0 and a == 0)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "true_negative": float(true_negative),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
