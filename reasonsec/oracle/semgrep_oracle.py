from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

from reasonsec.config import Config, ConfigError
from reasonsec.oracle.base import SecurityOracle, language_extension, normalise_finding_cwe
from reasonsec.types import OracleFinding
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


class SemgrepOracle(SecurityOracle):
    name = "semgrep"

    def __init__(self, config: Config) -> None:
        section = config.require_section("oracle.semgrep")
        self._binary = section.require_str("binary")
        self._rule_sources = [str(item) for item in section.require_list("rule_sources")]
        self._timeout_seconds = section.require_int("timeout_seconds")
        self._extra_arguments = [str(item) for item in section.require_list("extra_arguments")]
        self._metadata_cwe_keys = [str(item) for item in section.require_list("metadata_cwe_keys")]
        self._selection_strategy = config.require_str("oracle.cwe_selection_strategy")
        self._config = config
        self._working_directory = config.require_path("oracle.working_directory", must_exist=False)
        self._working_directory.mkdir(parents=True, exist_ok=True)
        if not self._rule_sources:
            raise ConfigError("oracle.semgrep.rule_sources must list at least one Semgrep rule source")

    def _command(self, target: Path) -> list[str]:
        command = [self._binary, "--json", "--quiet", "--disable-version-check", "--metrics", "off"]
        for source in self._rule_sources:
            command.extend(["--config", source])
        command.extend(self._extra_arguments)
        command.append(str(target))
        return command

    def analyse(self, code: str, language: str) -> list[OracleFinding]:
        extension = language_extension(self._config, language)
        with tempfile.TemporaryDirectory(dir=self._working_directory) as directory:
            target = Path(directory) / f"candidate{extension}"
            target.write_text(code, encoding="utf-8")
            return self._run(self._command(target), {target.name: None})

    def analyse_directory(self, directory: Path) -> dict[str, list[OracleFinding]]:
        results = self._run_raw(self._command(directory))
        grouped: dict[str, list[OracleFinding]] = {}
        for item in results:
            path = Path(str(item.get("path", ""))).name
            grouped.setdefault(path, []).append(self._to_finding(item))
        return grouped

    def _run(self, command: Sequence[str], _targets: dict[str, Any]) -> list[OracleFinding]:
        return [self._to_finding(item) for item in self._run_raw(command)]

    def _run_raw(self, command: Sequence[str]) -> list[dict[str, Any]]:
        try:
            completed = subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
            )
        except FileNotFoundError as error:
            raise ConfigError(f"semgrep binary '{self._binary}' was not found on this system") from error
        except subprocess.TimeoutExpired:
            LOGGER.warning("semgrep timed out after %ds", self._timeout_seconds)
            return []
        if not completed.stdout.strip():
            if completed.returncode not in (0, 1):
                LOGGER.warning("semgrep exited with code %d: %s", completed.returncode, completed.stderr.strip())
            return []
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            LOGGER.warning("semgrep produced output that is not valid JSON")
            return []
        return [item for item in payload.get("results", []) if isinstance(item, dict)]

    def _to_finding(self, item: dict[str, Any]) -> OracleFinding:
        extra = item.get("extra", {}) if isinstance(item.get("extra"), dict) else {}
        metadata = extra.get("metadata", {}) if isinstance(extra.get("metadata"), dict) else {}
        candidates: list[object] = []
        for key in self._metadata_cwe_keys:
            value = metadata.get(key)
            if isinstance(value, list):
                candidates.extend(value)
            elif value is not None:
                candidates.append(value)
        candidates.append(item.get("check_id", ""))
        start = item.get("start", {}) if isinstance(item.get("start"), dict) else {}
        return OracleFinding(
            rule_identifier=str(item.get("check_id", "")),
            cwe=normalise_finding_cwe(candidates),
            message=str(extra.get("message", "")),
            line=int(start["line"]) if "line" in start else None,
            tool=self.name,
        )
