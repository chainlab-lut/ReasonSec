from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from reasonsec.config import Config, ConfigError
from reasonsec.types import ReasoningChain


@dataclass
class PhaseDefinition:
    key: str
    heading: str


class ChainParser:
    def __init__(self, config: Config) -> None:
        section = config.require_section("reasoning")
        self._phases = [
            PhaseDefinition(key=str(item["key"]), heading=str(item["heading"]))
            for item in section.require_list("phases")
        ]
        self._planning_key = section.require_str("planning_phase_key")
        self._implementation_key = section.require_str("implementation_phase_key")
        heading_pattern = section.require_str("heading_pattern")
        self._heading_patterns = {
            phase.key: re.compile(heading_pattern.replace("{heading}", re.escape(phase.heading)), re.MULTILINE | re.IGNORECASE)
            for phase in self._phases
        }
        self._code_block_pattern = re.compile(section.require_str("code_block_pattern"), re.DOTALL)
        self._planning_fallback = section.require_str("planning_fallback").strip().lower()
        known_keys = {phase.key for phase in self._phases}
        for key in (self._planning_key, self._implementation_key):
            if key not in known_keys:
                raise ConfigError(f"reasoning phase key '{key}' is not declared in reasoning.phases")

    @property
    def planning_key(self) -> str:
        return self._planning_key

    @property
    def phase_keys(self) -> list[str]:
        return [phase.key for phase in self._phases]

    def _heading_positions(self, text: str) -> list[tuple[int, int, str]]:
        positions: list[tuple[int, int, str]] = []
        for key, pattern in self._heading_patterns.items():
            match = pattern.search(text)
            if match is not None:
                positions.append((match.start(), match.end(), key))
        return sorted(positions, key=lambda item: item[0])

    def extract_code(self, text: str) -> tuple[str, tuple[int, int] | None]:
        matches = list(self._code_block_pattern.finditer(text))
        if not matches:
            return "", None
        group_index = 1 if matches[0].groups() else 0
        best = max(matches, key=lambda item: len(item.group(group_index)))
        return best.group(group_index).strip("\n"), (best.start(group_index), best.end(group_index))

    def parse(self, text: str) -> ReasoningChain:
        positions = self._heading_positions(text)
        phases: dict[str, str] = {}
        spans: dict[str, tuple[int, int]] = {}
        for index, (_start, end, key) in enumerate(positions):
            stop = positions[index + 1][0] if index + 1 < len(positions) else len(text)
            phases[key] = text[end:stop].strip()
            spans[key] = (end, stop)
        planning_segment = phases.get(self._planning_key, "")
        planning_span = spans.get(self._planning_key)
        implementation_text = phases.get(self._implementation_key, "")
        implementation_span = spans.get(self._implementation_key)
        if implementation_text:
            code, relative_span = self.extract_code(implementation_text)
            if code and relative_span and implementation_span:
                code_span = (implementation_span[0] + relative_span[0], implementation_span[0] + relative_span[1])
            elif implementation_span:
                code, code_span = implementation_text.strip(), implementation_span
            else:
                code, code_span = code, None
        else:
            code, code_span = self.extract_code(text)
        if not planning_segment.strip():
            planning_segment, planning_span = self._fallback_planning(text, code_span)
        return ReasoningChain(
            raw_text=text,
            phases=phases,
            planning_segment=planning_segment,
            planning_char_span=planning_span,
            code=code,
            code_char_span=code_span,
        )

    def _fallback_planning(self, text: str, code_span: tuple[int, int] | None) -> tuple[str, tuple[int, int] | None]:
        if self._planning_fallback == "none":
            return "", None
        if self._planning_fallback == "prefix_before_code":
            end = code_span[0] if code_span else len(text)
            return text[:end].strip(), (0, end)
        if self._planning_fallback == "full_text":
            return text.strip(), (0, len(text))
        raise ConfigError(f"unsupported reasoning.planning_fallback value '{self._planning_fallback}'")

    def parse_code_only(self, text: str) -> ReasoningChain:
        code, code_span = self.extract_code(text)
        if not code:
            code, code_span = text.strip(), (0, len(text))
        return ReasoningChain(
            raw_text=text,
            phases={},
            planning_segment="",
            planning_char_span=None,
            code=code,
            code_char_span=code_span,
        )


def sentences(text: str, pattern: re.Pattern[str]) -> list[str]:
    return [segment.strip() for segment in pattern.split(text) if segment.strip()]


def select_matching_sentences(
    text: str,
    phrases: Sequence[str],
    sentence_pattern: re.Pattern[str],
    normaliser,
) -> list[str]:
    selected: list[str] = []
    normalised_phrases = [normaliser(phrase) for phrase in phrases if normaliser(phrase)]
    for sentence in sentences(text, sentence_pattern):
        normalised_sentence = normaliser(sentence)
        if any(phrase in normalised_sentence for phrase in normalised_phrases):
            selected.append(sentence)
    return selected
