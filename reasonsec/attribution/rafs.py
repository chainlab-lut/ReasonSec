from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from reasonsec.config import Config, ConfigError
from reasonsec.attribution.activations import FeatureMatrices
from reasonsec.reasoning.vocabulary import SecurityConceptVocabulary
from reasonsec.types import FeatureSet
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


def bin_count_sturges(sample_count: int) -> int:
    return max(1, int(math.ceil(math.log2(max(sample_count, 1)) + 1)))


def histogram_distance(
    safe_values: np.ndarray,
    vulnerable_values: np.ndarray,
    bin_count: int,
) -> np.ndarray:
    feature_dimension = safe_values.shape[1]
    distances = np.zeros(feature_dimension, dtype=np.float32)
    for index in range(feature_dimension):
        safe_column = safe_values[:, index]
        vulnerable_column = vulnerable_values[:, index]
        lower = float(min(safe_column.min(), vulnerable_column.min()))
        upper = float(max(safe_column.max(), vulnerable_column.max()))
        if upper <= lower:
            distances[index] = 0.0
            continue
        edges = np.linspace(lower, upper, bin_count + 1)
        safe_histogram, _ = np.histogram(safe_column, bins=edges)
        vulnerable_histogram, _ = np.histogram(vulnerable_column, bins=edges)
        safe_total = safe_histogram.sum()
        vulnerable_total = vulnerable_histogram.sum()
        if safe_total == 0 or vulnerable_total == 0:
            distances[index] = 0.0
            continue
        safe_normalised = safe_histogram / safe_total
        vulnerable_normalised = vulnerable_histogram / vulnerable_total
        distances[index] = float(1.0 - np.minimum(safe_normalised, vulnerable_normalised).sum())
    return distances


def min_max_normalise(values: np.ndarray, epsilon: float) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    return (values - minimum) / (maximum - minimum + epsilon)


@dataclass
class CategoryScores:
    cwe: str
    distributional_distance: np.ndarray
    alignment: np.ndarray
    normalised_alignment: np.ndarray
    rafs: np.ndarray
    vulnerable_count: int
    safe_count: int


class ReasoningAnchoredFeatureScorer:
    def __init__(self, config: Config, vocabulary: SecurityConceptVocabulary) -> None:
        section = config.require_section("rafs")
        self._alpha = section.require_float("alpha")
        self._epsilon = section.require_float("normalisation_epsilon")
        self._use_alignment = section.require_bool("use_alignment_term")
        self._use_distributional = section.require_bool("use_distributional_term")
        self._bin_rule = section.require_str("histogram.bin_rule").strip().lower()
        self._fixed_bin_count = section.require_int("histogram.fixed_bin_count")
        self._minimum_vulnerable_samples = section.require_int("selection.minimum_vulnerable_samples")
        self._minimum_safe_samples = section.require_int("selection.minimum_safe_samples")
        self._use_vocabulary_size = section.require_bool("selection.use_vocabulary_size")
        self._fixed_count = section.require_int("selection.fixed_count")
        self._minimum_candidates = section.require_int("selection.minimum_candidates")
        self._maximum_candidates = section.require_int("selection.maximum_candidates")
        self._safe_quantiles = [float(item) for item in section.require_list("safe_distribution.action_bound_quantiles")]
        self._minimum_standard_deviation = section.require_float("safe_distribution.minimum_standard_deviation")
        self._quantile_grid_size = section.require_int("safe_distribution.quantile_grid_size")
        self._random_selection = section.require_bool("selection.random")
        self._random_seed = section.require_int("selection.random_seed")
        self._vocabulary = vocabulary
        if not 0.0 <= self._alpha <= 1.0:
            raise ConfigError("rafs.alpha must lie in the interval [0, 1]")
        if len(self._safe_quantiles) != 2:
            raise ConfigError("rafs.safe_distribution.action_bound_quantiles must contain exactly two values")

    @property
    def alpha(self) -> float:
        return self._alpha

    def _bin_count(self, sample_count: int) -> int:
        if self._bin_rule == "sturges":
            return bin_count_sturges(sample_count)
        if self._bin_rule == "fixed":
            return self._fixed_bin_count
        raise ConfigError(f"unsupported histogram bin rule '{self._bin_rule}'")

    def score_category(
        self,
        cwe: str,
        matrices: FeatureMatrices,
        vulnerable_rows: Sequence[int],
        safe_rows: Sequence[int],
    ) -> CategoryScores:
        vulnerable_distributional = matrices.distributional[list(vulnerable_rows)]
        safe_distributional = matrices.distributional[list(safe_rows)]
        vulnerable_planning = matrices.planning[list(vulnerable_rows)]
        safe_planning = matrices.planning[list(safe_rows)]

        if self._use_distributional:
            bin_count = self._bin_count(len(vulnerable_rows) + len(safe_rows))
            distances = histogram_distance(safe_distributional, vulnerable_distributional, bin_count)
        else:
            distances = np.zeros(matrices.feature_dimension, dtype=np.float32)

        if self._use_alignment:
            alignment = vulnerable_planning.mean(axis=0) - safe_planning.mean(axis=0)
        else:
            alignment = np.zeros(matrices.feature_dimension, dtype=np.float32)
        normalised_alignment = min_max_normalise(alignment, self._epsilon)

        rafs = self._alpha * distances + (1.0 - self._alpha) * normalised_alignment
        return CategoryScores(
            cwe=cwe,
            distributional_distance=distances,
            alignment=alignment,
            normalised_alignment=normalised_alignment,
            rafs=rafs,
            vulnerable_count=len(vulnerable_rows),
            safe_count=len(safe_rows),
        )

    def candidate_count(self, cwe: str) -> int:
        if self._use_vocabulary_size:
            requested = self._vocabulary.size(cwe)
        else:
            requested = self._fixed_count
        bounded = max(requested, self._minimum_candidates)
        if self._maximum_candidates > 0:
            bounded = min(bounded, self._maximum_candidates)
        return bounded

    def select_candidates(self, scores: CategoryScores, count: int) -> list[int]:
        if self._random_selection:
            generator = np.random.default_rng(self._random_seed + int(scores.cwe.split("-")[1]))
            return [int(index) for index in generator.choice(scores.rafs.shape[0], size=count, replace=False)]
        ordered = np.argsort(-scores.rafs, kind="stable")
        return [int(index) for index in ordered[:count]]

    def quantile_levels(self) -> list[float]:
        return [float(value) for value in np.linspace(0.0, 1.0, self._quantile_grid_size)]

    def safe_statistics(
        self,
        matrices: FeatureMatrices,
        safe_rows: Sequence[int],
        vulnerable_rows: Sequence[int],
        feature_indices: Sequence[int],
    ) -> tuple[dict[int, dict[str, float]], dict[int, list[float]], dict[int, list[float]], dict[int, list[float]]]:
        statistics: dict[int, dict[str, float]] = {}
        bounds: dict[int, list[float]] = {}
        safe_quantiles: dict[int, list[float]] = {}
        vulnerable_quantiles: dict[int, list[float]] = {}
        levels = self.quantile_levels()
        safe_values = matrices.distributional[list(safe_rows)]
        vulnerable_values = matrices.distributional[list(vulnerable_rows)]
        for feature_index in feature_indices:
            column = safe_values[:, feature_index]
            mean = float(column.mean())
            deviation = float(column.std())
            statistics[int(feature_index)] = {
                "mean": mean,
                "standard_deviation": max(deviation, self._minimum_standard_deviation),
                "minimum": float(column.min()),
                "maximum": float(column.max()),
            }
            lower = float(np.quantile(column, self._safe_quantiles[0]))
            upper = float(np.quantile(column, self._safe_quantiles[1]))
            if upper <= lower:
                upper = lower + self._minimum_standard_deviation
            bounds[int(feature_index)] = [lower, upper]
            safe_quantiles[int(feature_index)] = [float(value) for value in np.quantile(column, levels)]
            vulnerable_quantiles[int(feature_index)] = [
                float(value) for value in np.quantile(vulnerable_values[:, feature_index], levels)
            ]
        return statistics, bounds, safe_quantiles, vulnerable_quantiles

    def build_feature_set(
        self,
        cwe: str,
        matrices: FeatureMatrices,
        vulnerable_rows: Sequence[int],
        safe_rows: Sequence[int],
    ) -> tuple[FeatureSet, CategoryScores] | None:
        if len(vulnerable_rows) < self._minimum_vulnerable_samples:
            LOGGER.warning(
                "skipping %s: %d vulnerable samples available, %d required",
                cwe,
                len(vulnerable_rows),
                self._minimum_vulnerable_samples,
            )
            return None
        if len(safe_rows) < self._minimum_safe_samples:
            LOGGER.warning(
                "skipping %s: %d safe samples available, %d required", cwe, len(safe_rows), self._minimum_safe_samples
            )
            return None
        scores = self.score_category(cwe, matrices, vulnerable_rows, safe_rows)
        count = self.candidate_count(cwe)
        candidates = self.select_candidates(scores, count)
        statistics, bounds, safe_quantiles, vulnerable_quantiles = self.safe_statistics(
            matrices, safe_rows, vulnerable_rows, candidates
        )
        feature_set = FeatureSet(
            cwe=cwe,
            feature_indices=list(candidates),
            candidate_indices=list(candidates),
            scores={int(index): float(scores.rafs[index]) for index in candidates},
            distributional_distance={int(index): float(scores.distributional_distance[index]) for index in candidates},
            alignment={int(index): float(scores.normalised_alignment[index]) for index in candidates},
            vocabulary=self._vocabulary.get(cwe),
            safe_statistics=statistics,
            action_bounds=bounds,
            quantile_levels=self.quantile_levels(),
            safe_quantiles=safe_quantiles,
            vulnerable_quantiles=vulnerable_quantiles,
        )
        LOGGER.info(
            "selected %d candidate features for %s from %d vulnerable and %d safe samples",
            len(candidates),
            cwe,
            len(vulnerable_rows),
            len(safe_rows),
        )
        return feature_set, scores


def group_rows_by_category(matrices: FeatureMatrices, split: str | None) -> tuple[dict[str, list[int]], list[int]]:
    vulnerable: dict[str, list[int]] = {}
    safe: list[int] = []
    for index, label in enumerate(matrices.labels):
        if split is not None and matrices.splits[index] != split:
            continue
        if label is None:
            safe.append(index)
        else:
            vulnerable.setdefault(label, []).append(index)
    return vulnerable, safe


def feature_set_overlap(left: Mapping[str, FeatureSet], right: Mapping[str, FeatureSet]) -> dict[str, float]:
    overlap: dict[str, float] = {}
    for cwe, feature_set in left.items():
        other = right.get(cwe)
        if other is None:
            continue
        left_indices = set(feature_set.feature_indices)
        right_indices = set(other.feature_indices)
        union = left_indices | right_indices
        overlap[cwe] = len(left_indices & right_indices) / len(union) if union else 0.0
    return overlap
