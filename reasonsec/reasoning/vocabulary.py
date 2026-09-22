from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from tqdm import tqdm

from reasonsec.config import Config
from reasonsec.data.cwe_catalog import normalise_phrase
from reasonsec.models.language_model import LanguageModel
from reasonsec.reasoning.templates import TemplateLibrary
from reasonsec.types import CorpusSample
from reasonsec.utils.io import read_json, write_json
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class SecurityConceptVocabulary:
    phrases: dict[str, list[str]] = field(default_factory=dict)

    def __contains__(self, cwe: str) -> bool:
        return cwe in self.phrases

    def __len__(self) -> int:
        return len(self.phrases)

    def size(self, cwe: str) -> int:
        return len(self.phrases.get(cwe, []))

    def get(self, cwe: str) -> list[str]:
        return list(self.phrases.get(cwe, []))

    def categories(self) -> list[str]:
        return sorted(self.phrases, key=lambda item: int(item.split("-")[1]))

    def matched_categories(self, text: str, minimum_matches: int = 1) -> dict[str, int]:
        normalised_text = f" {normalise_phrase(text)} "
        matches: dict[str, int] = {}
        for cwe, phrases in self.phrases.items():
            count = sum(1 for phrase in phrases if f" {normalise_phrase(phrase)} " in normalised_text)
            if count >= minimum_matches:
                matches[cwe] = count
        return matches

    def save(self, path: str | Path) -> Path:
        return write_json(path, {"phrases": self.phrases})

    @classmethod
    def load(cls, path: str | Path) -> "SecurityConceptVocabulary":
        payload = read_json(path)
        return cls(phrases={str(key): [str(item) for item in value] for key, value in payload["phrases"].items()})


class ConceptExtractor:
    def __init__(
        self,
        config: Config,
        model: LanguageModel,
        templates: TemplateLibrary | None = None,
    ) -> None:
        section = config.require_section("reasoning.concept_extraction")
        self._model = model
        self._templates = templates or TemplateLibrary(config)
        self._max_phrase_words = section.require_int("max_phrase_words")
        self._min_phrase_words = section.require_int("min_phrase_words")
        self._max_phrases_per_category = section.require_int("max_phrases_per_category")
        self._max_new_tokens = section.require_int("max_new_tokens")
        self._rejected_markers = [normalise_phrase(str(item)) for item in section.require_list("rejected_markers")]

    def _parse_response(self, response: str) -> list[str]:
        phrases: list[str] = []
        for line in response.splitlines():
            candidate = line.strip().strip("-*•").strip()
            if not candidate:
                continue
            normalised = normalise_phrase(candidate)
            if not normalised or normalised in self._rejected_markers:
                continue
            words = normalised.split()
            if len(words) < self._min_phrase_words or len(words) > self._max_phrase_words:
                continue
            phrases.append(" ".join(words))
        return phrases

    def extract(self, sample: CorpusSample, cwe: str) -> list[str]:
        instruction = self._templates.concept_extraction.render(
            cwe=cwe,
            security_planning_segment=sample.chain.planning_segment,
        )
        response = self._model.generate(instruction, overrides={"max_new_tokens": self._max_new_tokens})
        return self._parse_response(response)

    def build(self, vulnerable_samples: Mapping[str, Sequence[CorpusSample]]) -> SecurityConceptVocabulary:
        vocabulary = SecurityConceptVocabulary()
        for cwe in tqdm(sorted(vulnerable_samples), desc="extracting security concepts"):
            collected: list[str] = []
            for sample in vulnerable_samples[cwe]:
                for phrase in self.extract(sample, cwe):
                    if phrase not in collected:
                        collected.append(phrase)
                if len(collected) >= self._max_phrases_per_category:
                    break
            vocabulary.phrases[cwe] = collected[: self._max_phrases_per_category]
            LOGGER.info("vocabulary for %s contains %d phrases", cwe, len(vocabulary.phrases[cwe]))
        return vocabulary


def phrase_overlap(description: str, phrases: Iterable[str], minimum_token_overlap: float) -> tuple[bool, str | None]:
    normalised_description = normalise_phrase(description)
    if not normalised_description:
        return False, None
    description_tokens = set(normalised_description.split())
    for phrase in phrases:
        normalised_phrase = normalise_phrase(phrase)
        if not normalised_phrase:
            continue
        if f" {normalised_phrase} " in f" {normalised_description} ":
            return True, phrase
        phrase_tokens = set(normalised_phrase.split())
        if not phrase_tokens:
            continue
        overlap = len(phrase_tokens & description_tokens) / len(phrase_tokens)
        if overlap >= minimum_token_overlap:
            return True, phrase
    return False, None
