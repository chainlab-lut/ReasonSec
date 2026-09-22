from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from reasonsec.config import Config
from reasonsec.data.cwe_catalog import normalise_phrase
from reasonsec.models.embedder import SentenceEmbedder
from reasonsec.models.language_model import LanguageModel
from reasonsec.reasoning.parsing import select_matching_sentences
from reasonsec.reasoning.vocabulary import SecurityConceptVocabulary
from reasonsec.types import FeatureSet, OracleVerdict
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class RewardTerms:
    oracle: float = 0.0
    coherence: float = 0.0
    fidelity: float = 0.0
    deviation: float = 0.0
    proxy: float = 0.0
    composite: float = 0.0
    effective: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "oracle": self.oracle,
            "coherence": self.coherence,
            "fidelity": self.fidelity,
            "deviation": self.deviation,
            "proxy": self.proxy,
            "composite": self.composite,
            "effective": self.effective,
        }


@dataclass
class RewardWeights:
    oracle: float
    coherence: float
    fidelity: float
    deviation: float
    proxy_blend: float

    @classmethod
    def from_config(cls, config: Config) -> "RewardWeights":
        section = config.require_section("intervention.reward")
        return cls(
            oracle=section.require_float("beta_oracle"),
            coherence=section.require_float("beta_coherence"),
            fidelity=section.require_float("beta_fidelity"),
            deviation=section.require_float("beta_deviation"),
            proxy_blend=section.require_float("proxy_blend"),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "beta_oracle": self.oracle,
            "beta_coherence": self.coherence,
            "beta_fidelity": self.fidelity,
            "beta_deviation": self.deviation,
            "proxy_blend": self.proxy_blend,
        }


class ReasoningCoherenceReward:
    def __init__(
        self,
        config: Config,
        embedder: SentenceEmbedder,
        language_model: LanguageModel,
        vocabulary: SecurityConceptVocabulary,
        feature_sets: Mapping[str, FeatureSet],
        weights: RewardWeights | None = None,
    ) -> None:
        section = config.require_section("intervention.reward")
        self._weights = weights or RewardWeights.from_config(config)
        self._embedder = embedder
        self._language_model = language_model
        self._vocabulary = vocabulary
        self._feature_sets = dict(feature_sets)
        self._code_prefix_tokens = section.require_int("code_prefix_tokens")
        self._fidelity_centering_constant = section.require_float("fidelity_centering_constant")
        self._fidelity_scale = section.require_float("fidelity_scale")
        self._sentence_pattern = re.compile(section.require_str("prevention.sentence_pattern"))
        self._fallback_to_segment = section.require_bool("prevention.fallback_to_segment")
        self._include_cwe_identifier_sentences = section.require_bool("prevention.include_cwe_identifier_sentences")

    @property
    def weights(self) -> RewardWeights:
        return self._weights

    def extract_prevention(self, planning_segment: str, cwe: str) -> list[str]:
        phrases = list(self._vocabulary.get(cwe))
        if self._include_cwe_identifier_sentences:
            phrases.append(cwe)
        selected = select_matching_sentences(planning_segment, phrases, self._sentence_pattern, normalise_phrase)
        if not selected and self._fallback_to_segment:
            selected = [planning_segment] if planning_segment.strip() else []
        return selected

    def code_prefix(self, code: str) -> str:
        tokenizer = self._language_model.tokenizer
        token_ids = tokenizer.encode(code, add_special_tokens=False)[: self._code_prefix_tokens]
        return tokenizer.decode(token_ids, skip_special_tokens=True)

    def coherence(self, planning_segment: str, code: str, cwe: str) -> float:
        prevention = self.extract_prevention(planning_segment, cwe)
        prefix = self.code_prefix(code)
        if not prevention or not prefix.strip():
            return 0.5
        similarity = self._embedder.similarity(prevention, [prefix])
        return float(similarity / 2.0 + 0.5)

    def fidelity(self, replacement_values: Mapping[int, float], cwe: str) -> float:
        feature_set = self._feature_sets.get(cwe)
        if feature_set is None or not replacement_values:
            return 0.5
        log_likelihoods: list[float] = []
        for feature_index, value in replacement_values.items():
            statistics = feature_set.safe_statistics.get(int(feature_index))
            if statistics is None:
                continue
            mean = float(statistics["mean"])
            deviation = float(statistics["standard_deviation"])
            log_likelihoods.append(
                -0.5 * math.log(2.0 * math.pi * deviation**2) - ((value - mean) ** 2) / (2.0 * deviation**2)
            )
        if not log_likelihoods:
            return 0.5
        mean_log_likelihood = float(np.mean(log_likelihoods))
        scaled = self._fidelity_scale * (mean_log_likelihood + self._fidelity_centering_constant)
        return float(1.0 / (1.0 + math.exp(-scaled)))

    @staticmethod
    def deviation(original: np.ndarray, modified: np.ndarray) -> float:
        return float(np.linalg.norm(modified - original))

    def compute(
        self,
        verdict: OracleVerdict,
        planning_segment: str,
        code: str,
        cwe: str,
        replacement_values: Mapping[int, float],
        original_features: np.ndarray,
        modified_features: np.ndarray,
        proxy_probability: float | None = None,
    ) -> RewardTerms:
        oracle_term = float(1 - verdict.indicator)
        coherence_term = self.coherence(planning_segment, code, cwe)
        fidelity_term = self.fidelity(replacement_values, cwe)
        deviation_term = self.deviation(original_features, modified_features)
        composite = (
            self._weights.oracle * oracle_term
            + self._weights.coherence * coherence_term
            + self._weights.fidelity * fidelity_term
            - self._weights.deviation * deviation_term
        )
        terms = RewardTerms(
            oracle=oracle_term,
            coherence=coherence_term,
            fidelity=fidelity_term,
            deviation=deviation_term,
            proxy=float(proxy_probability) if proxy_probability is not None else 0.0,
            composite=float(composite),
        )
        if proxy_probability is None or self._weights.proxy_blend <= 0.0:
            terms.effective = float(composite)
        else:
            terms.effective = float(
                (1.0 - self._weights.proxy_blend) * composite
                + self._weights.proxy_blend * (1.0 - float(proxy_probability))
            )
        return terms


def aggregate_reward_terms(terms: Sequence[RewardTerms]) -> dict[str, float]:
    if not terms:
        return {}
    keys = terms[0].as_dict().keys()
    return {key: float(np.mean([getattr(item, key) for item in terms])) for key in keys}
