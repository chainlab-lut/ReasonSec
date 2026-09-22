from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from reasonsec.config import Config, ConfigError
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ExecutionResult:
    passed: bool
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "return_code": self.return_code,
            "stdout": self.stdout[-2000:],
            "stderr": self.stderr[-2000:],
            "timed_out": self.timed_out,
        }


class ProgramRunner:
    def __init__(self, config: Config) -> None:
        section = config.require_section("evaluation.utility.execution")
        self._enabled = section.require_bool("enabled")
        self._interpreter = section.require_str("interpreter")
        self._timeout_seconds = section.require_int("timeout_seconds")
        self._sandbox_command = [str(item) for item in section.require_list("sandbox_command")]
        self._working_directory = Path(section.require_str("working_directory")).expanduser()
        self._working_directory.mkdir(parents=True, exist_ok=True)
        self._environment_overrides = {str(key): str(value) for key, value in section.require_mapping("environment").items()}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def run(self, program: str) -> ExecutionResult:
        if not self._enabled:
            raise ConfigError(
                "functional evaluation requires evaluation.utility.execution.enabled to be set to true, "
                "because it executes model-generated programs"
            )
        with tempfile.TemporaryDirectory(dir=self._working_directory) as directory:
            script = Path(directory) / "candidate.py"
            script.write_text(program, encoding="utf-8")
            command: Sequence[str] = [*self._sandbox_command, self._interpreter, str(script)]
            try:
                completed = subprocess.run(
                    list(command),
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_seconds,
                    cwd=directory,
                    env=self._environment_overrides or None,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return ExecutionResult(passed=False, return_code=-1, stdout="", stderr="", timed_out=True)
            return ExecutionResult(
                passed=completed.returncode == 0,
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                timed_out=False,
            )
