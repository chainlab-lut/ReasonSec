from __future__ import annotations

import re
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from reasonsec.config import Config, ConfigError
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)

CWE_IDENTIFIER_PATTERN = re.compile(r"\bCWE[\s\-_]?(\d{1,5})\b", re.IGNORECASE)
_QUOTED_ALIAS_PATTERN = re.compile(r"['‘’\"“”]([^'‘’\"“”]{3,})['‘’\"“”]")
_NON_WORD_PATTERN = re.compile(r"[^a-z0-9]+")


def normalise_phrase(text: str) -> str:
    return _NON_WORD_PATTERN.sub(" ", text.lower()).strip()


def canonical_cwe(identifier: str | int | None) -> str | None:
    if identifier is None:
        return None
    text = str(identifier).strip()
    if not text:
        return None
    match = CWE_IDENTIFIER_PATTERN.search(text)
    if match:
        return f"CWE-{int(match.group(1))}"
    if text.isdigit():
        return f"CWE-{int(text)}"
    return None


@dataclass
class CweEntry:
    identifier: str
    name: str
    description: str
    alternate_terms: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)

    @property
    def phrases(self) -> list[str]:
        ordered: list[str] = []
        for phrase in [self.name, *self.alternate_terms, *self.aliases]:
            normalised = normalise_phrase(phrase)
            if normalised and normalised not in ordered:
                ordered.append(normalised)
        return ordered


class CweCatalog:
    def __init__(
        self,
        entries: Mapping[str, CweEntry],
        min_phrase_words: int,
        match_mode: str = "exact",
        minimum_token_coverage: float = 1.0,
        minimum_matched_tokens: int = 2,
    ) -> None:
        self._entries = dict(entries)
        self._min_phrase_words = min_phrase_words
        self._match_mode = match_mode.strip().lower()
        self._minimum_token_coverage = float(minimum_token_coverage)
        self._minimum_matched_tokens = int(minimum_matched_tokens)
        self._phrase_index = self._build_phrase_index()

    @classmethod
    def from_config(cls, config: Config) -> "CweCatalog":
        section = config.require_section("cwe_catalog")
        catalog_path = section.require_path("path")
        min_phrase_words = section.require_int("min_phrase_words")
        match_mode = section.require_str("match.mode")
        minimum_token_coverage = section.require_float("match.minimum_token_coverage")
        minimum_matched_tokens = section.require_int("match.minimum_matched_tokens")
        entries = _parse_catalog(catalog_path)
        if section.has("supplementary_terms_path"):
            supplementary_path = section.optional("supplementary_terms_path")
            if supplementary_path:
                entries = _apply_supplementary_terms(entries, Path(str(supplementary_path)).expanduser())
        if not entries:
            raise ConfigError(f"no CWE entries parsed from {catalog_path}")
        LOGGER.info("loaded %d CWE entries from %s", len(entries), catalog_path)
        return cls(entries, min_phrase_words, match_mode, minimum_token_coverage, minimum_matched_tokens)

    def _build_phrase_index(self) -> dict[str, str]:
        index: dict[str, str] = {}
        for entry in self._entries.values():
            for phrase in entry.phrases:
                if len(phrase.split()) < self._min_phrase_words:
                    continue
                index.setdefault(phrase, entry.identifier)
        return index

    def __contains__(self, identifier: str) -> bool:
        return canonical_cwe(identifier) in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def identifiers(self) -> list[str]:
        return sorted(self._entries, key=lambda item: int(item.split("-")[1]))

    def entry(self, identifier: str) -> CweEntry | None:
        canonical = canonical_cwe(identifier)
        return self._entries.get(canonical) if canonical else None

    def name(self, identifier: str) -> str:
        entry = self.entry(identifier)
        return entry.name if entry else ""

    def phrases_for(self, identifier: str) -> list[str]:
        entry = self.entry(identifier)
        return entry.phrases if entry else []

    def explicit_mentions(self, text: str) -> list[str]:
        found: list[str] = []
        for match in CWE_IDENTIFIER_PATTERN.finditer(text or ""):
            identifier = f"CWE-{int(match.group(1))}"
            if identifier in self._entries and identifier not in found:
                found.append(identifier)
        return found

    def _phrase_matches(self, phrase: str, normalised_text: str, text_tokens: set[str]) -> bool:
        if f" {phrase} " in f" {normalised_text} ":
            return True
        if self._match_mode != "coverage":
            return False
        phrase_tokens = phrase.split()
        if not phrase_tokens:
            return False
        matched = sum(1 for token in phrase_tokens if token in text_tokens)
        if matched < self._minimum_matched_tokens:
            return False
        return matched / len(phrase_tokens) >= self._minimum_token_coverage

    def phrase_mentions(self, text: str, restrict_to: Iterable[str] | None = None) -> list[str]:
        normalised_text = normalise_phrase(text or "")
        if not normalised_text:
            return []
        allowed = {canonical_cwe(item) for item in restrict_to} if restrict_to is not None else None
        text_tokens = set(normalised_text.split())
        found: list[str] = []
        for phrase, identifier in self._phrase_index.items():
            if allowed is not None and identifier not in allowed:
                continue
            if identifier in found:
                continue
            if self._phrase_matches(phrase, normalised_text, text_tokens):
                found.append(identifier)
        return found

    def mentions(self, text: str, restrict_to: Iterable[str] | None = None) -> list[str]:
        explicit = self.explicit_mentions(text)
        if restrict_to is not None:
            allowed = {canonical_cwe(item) for item in restrict_to}
            explicit = [item for item in explicit if item in allowed]
        phrase_based = self.phrase_mentions(text, restrict_to=restrict_to)
        merged = list(explicit)
        for identifier in phrase_based:
            if identifier not in merged:
                merged.append(identifier)
        return merged


def _strip_namespace(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _element_text(element: ElementTree.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _parse_catalog(path: Path) -> dict[str, CweEntry]:
    tree = ElementTree.parse(path)
    entries: dict[str, CweEntry] = {}
    for element in tree.iter():
        if _strip_namespace(element.tag) != "Weakness":
            continue
        raw_identifier = element.attrib.get("ID")
        identifier = canonical_cwe(raw_identifier)
        if identifier is None:
            continue
        name = element.attrib.get("Name", "").strip()
        description = ""
        alternate_terms: list[str] = []
        for child in element:
            tag = _strip_namespace(child.tag)
            if tag == "Description" and not description:
                description = _element_text(child)
            elif tag == "Alternate_Terms":
                for term_element in child:
                    for term_child in term_element:
                        if _strip_namespace(term_child.tag) == "Term":
                            term = _element_text(term_child)
                            if term:
                                alternate_terms.append(term)
        aliases = [alias.strip() for alias in _QUOTED_ALIAS_PATTERN.findall(name)]
        base_name = _QUOTED_ALIAS_PATTERN.sub("", name).replace("()", "").strip(" ()")
        if base_name and base_name != name:
            aliases.append(base_name)
        entries[identifier] = CweEntry(
            identifier=identifier,
            name=name,
            description=description,
            alternate_terms=alternate_terms,
            aliases=aliases,
        )
    return entries


def _apply_supplementary_terms(entries: dict[str, CweEntry], path: Path) -> dict[str, CweEntry]:
    if not path.is_file():
        raise ConfigError(f"supplementary CWE term file not found: {path}")
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ConfigError(f"supplementary CWE term file {path} must contain a mapping of CWE identifier to term list")
    for raw_identifier, terms in payload.items():
        identifier = canonical_cwe(raw_identifier)
        if identifier is None or identifier not in entries:
            continue
        for term in terms or []:
            text = str(term).strip()
            if text and text not in entries[identifier].alternate_terms:
                entries[identifier].alternate_terms.append(text)
    return entries
