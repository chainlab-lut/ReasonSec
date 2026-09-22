from __future__ import annotations

import hashlib
import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from reasonsec.config import Config, ConfigError
from reasonsec.data.cwe_catalog import canonical_cwe
from reasonsec.types import OracleFinding, OracleVerdict
from reasonsec.utils.io import ensure_directory
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


def select_reported_cwe(findings: Sequence[OracleFinding], strategy: str, expected: str | None) -> str | None:
    identified = [finding.cwe for finding in findings if finding.cwe]
    if not identified:
        return None
    normalised = str(strategy).strip().lower()
    if normalised == "expected_first" and expected is not None and expected in identified:
        return expected
    if normalised == "lowest_identifier":
        return sorted(identified, key=lambda item: int(item.split("-")[1]))[0]
    if normalised == "most_frequent":
        counts: dict[str, int] = {}
        for identifier in identified:
            counts[identifier] = counts.get(identifier, 0) + 1
        return max(sorted(counts), key=lambda item: counts[item])
    return identified[0]


class SecurityOracle(ABC):
    name: str = "oracle"

    @abstractmethod
    def analyse(self, code: str, language: str) -> list[OracleFinding]:
        raise NotImplementedError

    def evaluate(self, code: str, language: str, expected_cwe: str | None = None) -> OracleVerdict:
        findings = self.analyse(code, language) if code.strip() else []
        cwe = select_reported_cwe(findings, self.selection_strategy, expected_cwe)
        return OracleVerdict(vulnerable=bool(findings), cwe=cwe, findings=list(findings), tool=self.name)

    def evaluate_many(
        self, items: Sequence[tuple[str, str, str | None]]
    ) -> list[OracleVerdict]:
        return [self.evaluate(code, language, expected) for code, language, expected in items]

    @property
    def selection_strategy(self) -> str:
        return getattr(self, "_selection_strategy", "expected_first")


class CachedOracle(SecurityOracle):
    def __init__(self, inner: SecurityOracle, cache_path: Path) -> None:
        self._inner = inner
        self.name = inner.name
        self._selection_strategy = inner.selection_strategy
        self._cache_path = cache_path
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        if cache_path.is_file():
            with cache_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if stripped:
                        record = json.loads(stripped)
                        self._cache[record["key"]] = record["verdict"]
            LOGGER.info("loaded %d cached oracle verdicts from %s", len(self._cache), cache_path)
        else:
            ensure_directory(cache_path.parent)

    @staticmethod
    def _key(code: str, language: str, tool: str) -> str:
        digest = hashlib.sha256()
        digest.update(tool.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(language.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(code.encode("utf-8"))
        return digest.hexdigest()

    def analyse(self, code: str, language: str) -> list[OracleFinding]:
        return self._inner.analyse(code, language)

    def evaluate(self, code: str, language: str, expected_cwe: str | None = None) -> OracleVerdict:
        key = self._key(code, language, self.name)
        with self._lock:
            cached = self._cache.get(key)
        if cached is not None:
            findings = [OracleFinding(**finding) for finding in cached["findings"]]
            return OracleVerdict(
                vulnerable=bool(cached["vulnerable"]),
                cwe=select_reported_cwe(findings, self.selection_strategy, expected_cwe),
                findings=findings,
                tool=self.name,
            )
        verdict = self._inner.evaluate(code, language, expected_cwe)
        with self._lock:
            self._cache[key] = verdict.as_dict()
            with self._cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "verdict": verdict.as_dict()}, ensure_ascii=False))
                handle.write("\n")
        return verdict


class CompositeOracle(SecurityOracle):
    def __init__(self, oracles: Sequence[SecurityOracle], name: str, selection_strategy: str) -> None:
        if not oracles:
            raise ConfigError("composite oracle requires at least one component oracle")
        self._oracles = list(oracles)
        self.name = name
        self._selection_strategy = selection_strategy

    def analyse(self, code: str, language: str) -> list[OracleFinding]:
        findings: list[OracleFinding] = []
        for oracle in self._oracles:
            findings.extend(oracle.analyse(code, language))
        return findings


def language_extension(config: Config, language: str) -> str:
    mapping: Mapping[str, str] = config.require_mapping("oracle.language_extensions")
    key = language.strip().lower()
    if key not in mapping:
        raise ConfigError(f"no file extension configured under oracle.language_extensions for language '{language}'")
    return str(mapping[key])


def normalise_finding_cwe(values: Iterable[object]) -> str | None:
    for value in values:
        identifier = canonical_cwe(str(value))
        if identifier:
            return identifier
    return None
