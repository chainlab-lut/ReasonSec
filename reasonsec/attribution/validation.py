from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from tqdm import tqdm

from reasonsec.attribution.activations import top_activating_positions
from reasonsec.config import Config
from reasonsec.models.language_model import LanguageModel
from reasonsec.reasoning.templates import TemplateLibrary
from reasonsec.reasoning.vocabulary import SecurityConceptVocabulary, phrase_overlap
from reasonsec.sae.model import SparseAutoencoder
from reasonsec.sae.store import ActivationStore
from reasonsec.types import CorpusSample, FeatureSet
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ValidationOutcome:
    accepted: list[int]
    rejected: list[int]
    descriptions: dict[int, str]
    matched_phrases: dict[int, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": list(self.accepted),
            "rejected": list(self.rejected),
            "descriptions": {str(key): value for key, value in self.descriptions.items()},
            "matched_phrases": {str(key): value for key, value in self.matched_phrases.items()},
        }


class SemanticValidator:
    def __init__(
        self,
        config: Config,
        model: LanguageModel,
        store: ActivationStore,
        autoencoder: SparseAutoencoder,
        corpus: Sequence[CorpusSample],
        templates: TemplateLibrary | None = None,
    ) -> None:
        section = config.require_section("rafs.semantic_validation")
        self._enabled = section.require_bool("enabled")
        self._top_n = section.require_int("top_activating_contexts")
        self._context_characters = section.require_int("context_characters")
        self._max_new_tokens = section.require_int("max_new_tokens")
        self._token_overlap = section.require_float("minimum_token_overlap")
        self._no_content_marker = section.require_str("no_content_marker").strip().lower()
        self._batch_size = section.require_int("encode_batch_size")
        self._device = torch.device(section.require_str("device"))
        self._keep_all_when_none_accepted = section.require_bool("keep_candidates_when_none_accepted")
        self._model = model
        self._store = store
        self._autoencoder = autoencoder
        self._templates = templates or TemplateLibrary(config)
        self._corpus = {sample.identifier: sample for sample in corpus}
        self._sample_lookup = store.samples

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _locate_token(self, global_position: int) -> tuple[str, tuple[int, int]] | None:
        for sample in self._sample_lookup:
            if sample.start <= global_position < sample.end:
                offset = global_position - sample.start
                if offset < len(sample.offsets):
                    return sample.identifier, sample.offsets[offset]
                return sample.identifier, (0, 0)
        return None

    def _context_window(self, identifier: str, span: tuple[int, int]) -> str:
        sample = self._corpus.get(identifier)
        if sample is None:
            return ""
        text = sample.full_text
        start = max(0, span[0] - self._context_characters)
        end = min(len(text), span[1] + self._context_characters)
        return " ".join(text[start:end].split())

    def describe_features(self, feature_indices: Sequence[int]) -> dict[int, str]:
        positions = top_activating_positions(
            self._store,
            self._autoencoder,
            feature_indices,
            batch_size=self._batch_size,
            top_n=self._top_n,
            device=self._device,
        )
        descriptions: dict[int, str] = {}
        for feature_index in tqdm(feature_indices, desc="describing candidate features"):
            contexts: list[str] = []
            for global_position, _value in positions.get(int(feature_index), []):
                located = self._locate_token(global_position)
                if located is None:
                    continue
                window = self._context_window(located[0], located[1])
                if window and window not in contexts:
                    contexts.append(window)
            if not contexts:
                descriptions[int(feature_index)] = ""
                continue
            rendered = "\n".join(f"- {context}" for context in contexts)
            instruction = self._templates.feature_description.render(top_activating_contexts=rendered)
            response = self._model.generate(instruction, overrides={"max_new_tokens": self._max_new_tokens})
            descriptions[int(feature_index)] = " ".join(response.split())
        return descriptions

    def validate(self, feature_set: FeatureSet, vocabulary: SecurityConceptVocabulary) -> ValidationOutcome:
        if not self._enabled:
            return ValidationOutcome(
                accepted=list(feature_set.candidate_indices), rejected=[], descriptions={}, matched_phrases={}
            )
        phrases = vocabulary.get(feature_set.cwe)
        descriptions = self.describe_features(feature_set.candidate_indices)
        accepted: list[int] = []
        rejected: list[int] = []
        matched: dict[int, str] = {}
        for feature_index in feature_set.candidate_indices:
            description = descriptions.get(int(feature_index), "")
            normalised = description.strip().lower()
            if not normalised or normalised.startswith(self._no_content_marker):
                rejected.append(int(feature_index))
                continue
            is_match, phrase = phrase_overlap(description, phrases, self._token_overlap)
            if is_match:
                accepted.append(int(feature_index))
                if phrase:
                    matched[int(feature_index)] = phrase
            else:
                rejected.append(int(feature_index))
        if not accepted and self._keep_all_when_none_accepted:
            LOGGER.warning(
                "semantic validation rejected every candidate for %s; retaining the ranked candidate set",
                feature_set.cwe,
            )
            accepted = list(feature_set.candidate_indices)
            rejected = []
        LOGGER.info(
            "semantic validation for %s accepted %d of %d candidates",
            feature_set.cwe,
            len(accepted),
            len(feature_set.candidate_indices),
        )
        return ValidationOutcome(
            accepted=accepted, rejected=rejected, descriptions=descriptions, matched_phrases=matched
        )


def apply_validation(feature_set: FeatureSet, outcome: ValidationOutcome) -> FeatureSet:
    feature_set.feature_indices = list(outcome.accepted)
    feature_set.descriptions = dict(outcome.descriptions)
    return feature_set


def export_feature_labels(feature_sets: Mapping[str, FeatureSet]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for cwe, feature_set in feature_sets.items():
        for feature_index in feature_set.feature_indices:
            rows.append(
                {
                    "cwe": cwe,
                    "feature_index": feature_index,
                    "rafs_score": feature_set.scores.get(feature_index),
                    "distributional_distance": feature_set.distributional_distance.get(feature_index),
                    "alignment": feature_set.alignment.get(feature_index),
                    "description": feature_set.descriptions.get(feature_index, ""),
                }
            )
    return rows
