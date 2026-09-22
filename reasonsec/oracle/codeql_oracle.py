from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

from reasonsec.config import Config, ConfigError
from reasonsec.oracle.base import SecurityOracle, language_extension, normalise_finding_cwe, select_reported_cwe
from reasonsec.types import OracleFinding, OracleVerdict
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


class CodeQLOracle(SecurityOracle):
    name = "codeql"

    def __init__(self, config: Config) -> None:
        section = config.require_section("oracle.codeql")
        self._binary = section.require_str("binary")
        self._language = section.require_str("language")
        self._query_suite = section.require_str("query_suite")
        self._timeout_seconds = section.require_int("timeout_seconds")
        self._threads = section.require_int("threads")
        self._selection_strategy = config.require_str("oracle.cwe_selection_strategy")
        self._config = config
        self._working_directory = config.require_path("oracle.working_directory", must_exist=False)
        self._working_directory.mkdir(parents=True, exist_ok=True)

    def analyse(self, code: str, language: str) -> list[OracleFinding]:
        grouped = self._analyse_sources([("candidate", code, language)])
        return grouped.get("candidate", [])

    def evaluate_many(self, items: Sequence[tuple[str, str, str | None]]) -> list[OracleVerdict]:
        sources = [(f"candidate_{index}", code, language) for index, (code, language, _) in enumerate(items)]
        grouped = self._analyse_sources(sources)
        verdicts: list[OracleVerdict] = []
        for index, (_code, _language, expected) in enumerate(items):
            findings = grouped.get(f"candidate_{index}", [])
            verdicts.append(
                OracleVerdict(
                    vulnerable=bool(findings),
                    cwe=select_reported_cwe(findings, self._selection_strategy, expected),
                    findings=findings,
                    tool=self.name,
                )
            )
        return verdicts

    def _analyse_sources(self, sources: Sequence[tuple[str, str, str]]) -> dict[str, list[OracleFinding]]:
        if not sources:
            return {}
        with tempfile.TemporaryDirectory(dir=self._working_directory) as directory:
            root = Path(directory)
            source_root = root / "source"
            source_root.mkdir(parents=True, exist_ok=True)
            for name, code, language in sources:
                extension = language_extension(self._config, language)
                (source_root / f"{name}{extension}").write_text(code, encoding="utf-8")
            database = root / "database"
            sarif_path = root / "results.sarif"
            self._run(
                [
                    self._binary,
                    "database",
                    "create",
                    str(database),
                    f"--language={self._language}",
                    f"--source-root={source_root}",
                    f"--threads={self._threads}",
                    "--overwrite",
                ]
            )
            self._run(
                [
                    self._binary,
                    "database",
                    "analyze",
                    str(database),
                    self._query_suite,
                    "--format=sarif-latest",
                    f"--output={sarif_path}",
                    f"--threads={self._threads}",
                ]
            )
            if not sarif_path.is_file():
                return {}
            return self._parse_sarif(sarif_path)

    def _run(self, command: Sequence[str]) -> None:
        if shutil.which(self._binary) is None and not Path(self._binary).exists():
            raise ConfigError(f"codeql binary '{self._binary}' was not found on this system")
        completed = subprocess.run(list(command), capture_output=True, text=True, timeout=self._timeout_seconds, check=False)
        if completed.returncode != 0:
            LOGGER.warning("codeql command failed with code %d: %s", completed.returncode, completed.stderr.strip()[:500])

    def _parse_sarif(self, path: Path) -> dict[str, list[OracleFinding]]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        grouped: dict[str, list[OracleFinding]] = {}
        for run in payload.get("runs", []):
            rules_index: dict[str, dict[str, Any]] = {}
            driver = run.get("tool", {}).get("driver", {})
            for rule in driver.get("rules", []):
                rules_index[str(rule.get("id"))] = rule
            for result in run.get("results", []):
                rule_identifier = str(result.get("ruleId", ""))
                rule = rules_index.get(rule_identifier, {})
                tags = rule.get("properties", {}).get("tags", []) if isinstance(rule.get("properties"), dict) else []
                for location in result.get("locations", []):
                    artifact = location.get("physicalLocation", {}).get("artifactLocation", {})
                    file_name = Path(str(artifact.get("uri", ""))).stem
                    region = location.get("physicalLocation", {}).get("region", {})
                    grouped.setdefault(file_name, []).append(
                        OracleFinding(
                            rule_identifier=rule_identifier,
                            cwe=normalise_finding_cwe([*tags, rule_identifier]),
                            message=str(result.get("message", {}).get("text", "")),
                            line=int(region["startLine"]) if "startLine" in region else None,
                            tool=self.name,
                        )
                    )
        return grouped
