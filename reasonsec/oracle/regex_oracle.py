from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from reasonsec.config import Config, ConfigError
from reasonsec.data.cwe_catalog import canonical_cwe
from reasonsec.oracle.base import SecurityOracle
from reasonsec.types import OracleFinding
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class RegexRule:
    identifier: str
    pattern: re.Pattern[str]
    cwe: str | None
    languages: set[str]
    message: str


def _load_documents(path: Path) -> list[Any]:
    targets = sorted(path.rglob("*")) if path.is_dir() else [path]
    documents: list[Any] = []
    for target in targets:
        if target.is_dir():
            continue
        suffix = target.suffix.lower()
        if suffix == ".json":
            with target.open("r", encoding="utf-8") as handle:
                documents.append(json.load(handle))
        elif suffix in {".yaml", ".yml"}:
            with target.open("r", encoding="utf-8") as handle:
                documents.append(yaml.safe_load(handle))
        elif suffix in {".jsonl", ".ndjson"}:
            records = []
            with target.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        records.append(json.loads(stripped))
            documents.append(records)
    return documents


def _iter_rule_records(document: Any) -> list[Mapping[str, Any]]:
    if isinstance(document, list):
        return [item for item in document if isinstance(item, Mapping)]
    if isinstance(document, Mapping):
        for key in ("rules", "patterns", "data"):
            if isinstance(document.get(key), list):
                return [item for item in document[key] if isinstance(item, Mapping)]
        return [document]
    return []


class RegexOracle(SecurityOracle):
    name = "regex"

    def __init__(self, config: Config) -> None:
        section = config.require_section("oracle.regex")
        self._selection_strategy = config.require_str("oracle.cwe_selection_strategy")
        field_map = section.require_section("fields")
        pattern_field = field_map.require_str("pattern")
        cwe_field = field_map.require_str("cwe")
        language_field = field_map.optional("language")
        identifier_field = field_map.optional("identifier")
        message_field = field_map.optional("message")
        case_sensitive = section.require_bool("case_sensitive")
        flags = 0 if case_sensitive else re.IGNORECASE
        self._rules: list[RegexRule] = []
        for path in section.require_paths("rule_paths"):
            for document in _load_documents(path):
                for index, record in enumerate(_iter_rule_records(document)):
                    raw_pattern = record.get(pattern_field)
                    if not raw_pattern:
                        continue
                    languages_value = record.get(language_field) if language_field else None
                    if isinstance(languages_value, str):
                        languages = {languages_value.strip().lower()}
                    elif isinstance(languages_value, Sequence):
                        languages = {str(item).strip().lower() for item in languages_value}
                    else:
                        languages = set()
                    try:
                        compiled = re.compile(str(raw_pattern), flags)
                    except re.error as error:
                        LOGGER.warning("skipping invalid regex rule in %s: %s", path, error)
                        continue
                    identifier = str(record.get(identifier_field, f"{path.stem}-{index}")) if identifier_field else f"{path.stem}-{index}"
                    self._rules.append(
                        RegexRule(
                            identifier=identifier,
                            pattern=compiled,
                            cwe=canonical_cwe(record.get(cwe_field)),
                            languages=languages,
                            message=str(record.get(message_field, "")) if message_field else "",
                        )
                    )
        if not self._rules:
            raise ConfigError("oracle.regex.rule_paths produced no usable rules")
        LOGGER.info("loaded %d regular expression oracle rules", len(self._rules))

    def analyse(self, code: str, language: str) -> list[OracleFinding]:
        normalised_language = language.strip().lower()
        findings: list[OracleFinding] = []
        for rule in self._rules:
            if rule.languages and normalised_language not in rule.languages:
                continue
            match = rule.pattern.search(code)
            if match is None:
                continue
            line = code.count("\n", 0, match.start()) + 1
            findings.append(
                OracleFinding(
                    rule_identifier=rule.identifier,
                    cwe=rule.cwe,
                    message=rule.message,
                    line=line,
                    tool=self.name,
                )
            )
        return findings
