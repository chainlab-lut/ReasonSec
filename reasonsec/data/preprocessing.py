from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

from reasonsec.config import Config, ConfigError
from reasonsec.types import BenchmarkPrompt, CorpusSample
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class PipelineStage:
    index: int
    name: str
    removed: int
    remaining: int

    def as_dict(self) -> dict[str, int | str]:
        return {"index": self.index, "stage": self.name, "removed": self.removed, "remaining": self.remaining}


@dataclass
class PreprocessingReport:
    stages: list[PipelineStage] = field(default_factory=list)

    def add(self, name: str, removed: int, remaining: int) -> None:
        self.stages.append(PipelineStage(index=len(self.stages) + 1, name=name, removed=removed, remaining=remaining))

    @property
    def total_removed(self) -> int:
        return sum(stage.removed for stage in self.stages)

    def as_dict(self) -> dict[str, object]:
        return {
            "stages": [stage.as_dict() for stage in self.stages],
            "total_removed": self.total_removed,
            "final_count": self.stages[-1].remaining if self.stages else 0,
        }


def compile_patterns(patterns: Sequence[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.MULTILINE) for pattern in patterns]


def cyclomatic_complexity(code: str, decision_patterns: Sequence[re.Pattern[str]], base: int) -> int:
    if not code.strip():
        return base
    total = base
    for pattern in decision_patterns:
        total += len(pattern.findall(code))
    return total


def normalise_code(code: str, comment_patterns: Sequence[re.Pattern[str]], collapse_blank_lines: bool) -> str:
    normalised = code.replace("\r\n", "\n").replace("\r", "\n")
    for pattern in comment_patterns:
        normalised = pattern.sub("", normalised)
    lines = [line.rstrip() for line in normalised.split("\n")]
    if collapse_blank_lines:
        collapsed: list[str] = []
        previous_blank = False
        for line in lines:
            blank = not line.strip()
            if blank and previous_blank:
                continue
            collapsed.append(line)
            previous_blank = blank
        lines = collapsed
    return "\n".join(lines).strip("\n")


def extract_reference_code(prompt: BenchmarkPrompt, metadata_fields: Sequence[str], fenced_pattern: re.Pattern[str]) -> str:
    for field_name in metadata_fields:
        value = prompt.metadata.get(field_name)
        if isinstance(value, str) and value.strip():
            return value
    blocks = fenced_pattern.findall(prompt.prompt)
    if blocks:
        return "\n".join(block if isinstance(block, str) else block[-1] for block in blocks)
    return prompt.prompt


class PromptPreprocessor:
    def __init__(self, config: Config, token_counter: Callable[[str], int]) -> None:
        section = config.require_section("preprocessing")
        self._token_counter = token_counter
        self._max_prompt_tokens = section.require_int("max_prompt_tokens")
        self._max_cyclomatic_complexity = section.require_int("cyclomatic_complexity.maximum")
        self._complexity_base = section.require_int("cyclomatic_complexity.base")
        self._decision_patterns = compile_patterns([str(item) for item in section.require_list("cyclomatic_complexity.decision_patterns")])
        self._reference_code_fields = [str(item) for item in section.require_list("cyclomatic_complexity.reference_code_metadata_fields")]
        self._fenced_pattern = re.compile(section.require_str("code_block_pattern"), re.DOTALL)
        self._languages = [str(item).lower() for item in section.require_list("languages")]
        self._comment_patterns = {
            str(language): compile_patterns([str(pattern) for pattern in patterns])
            for language, patterns in section.require_mapping("normalisation.comment_patterns").items()
        }
        self._collapse_blank_lines = section.require_bool("normalisation.collapse_blank_lines")
        self._require_cwe_label = section.require_bool("require_cwe_label")

    def comment_patterns_for(self, language: str) -> list[re.Pattern[str]]:
        return self._comment_patterns.get(language.lower(), self._comment_patterns.get("default", []))

    def normalise(self, code: str, language: str) -> str:
        return normalise_code(code, self.comment_patterns_for(language), self._collapse_blank_lines)

    def run(self, prompts: Sequence[BenchmarkPrompt]) -> tuple[list[BenchmarkPrompt], PreprocessingReport]:
        report = PreprocessingReport()
        current = list(prompts)
        report.add("raw ingestion", 0, len(current))

        language_filtered = [prompt for prompt in current if prompt.language.lower() in self._languages]
        report.add("language filter", len(current) - len(language_filtered), len(language_filtered))
        current = language_filtered

        if self._require_cwe_label:
            labelled = [prompt for prompt in current if prompt.cwe is not None]
            report.add("cwe label filter", len(current) - len(labelled), len(labelled))
            current = labelled

        length_filtered = [prompt for prompt in current if self._token_counter(prompt.prompt) <= self._max_prompt_tokens]
        report.add(f"length filter (>{self._max_prompt_tokens} tokens)", len(current) - len(length_filtered), len(length_filtered))
        current = length_filtered

        complexity_filtered: list[BenchmarkPrompt] = []
        for prompt in current:
            reference = extract_reference_code(prompt, self._reference_code_fields, self._fenced_pattern)
            complexity = cyclomatic_complexity(reference, self._decision_patterns, self._complexity_base)
            prompt.metadata["cyclomatic_complexity"] = complexity
            if complexity <= self._max_cyclomatic_complexity:
                complexity_filtered.append(prompt)
        report.add(
            f"cyclomatic complexity (>{self._max_cyclomatic_complexity})",
            len(current) - len(complexity_filtered),
            len(complexity_filtered),
        )
        current = complexity_filtered

        for prompt in current:
            reference = extract_reference_code(prompt, self._reference_code_fields, self._fenced_pattern)
            prompt.metadata["normalised_reference_code"] = self.normalise(reference, prompt.language)
        report.add("code normalisation", 0, len(current))

        if not current:
            raise ConfigError("preprocessing removed every prompt; review the preprocessing thresholds in the configuration")
        return current, report


def stratified_split(
    samples: Sequence[CorpusSample],
    fractions: Mapping[str, float],
    stratify_by_cwe: bool,
    seed: int,
) -> dict[str, list[CorpusSample]]:
    names = list(fractions.keys())
    total_fraction = sum(float(fractions[name]) for name in names)
    if abs(total_fraction - 1.0) > 1e-6:
        raise ConfigError(f"split fractions must sum to 1.0, received {total_fraction}")
    rng = random.Random(seed)
    groups: dict[str, list[CorpusSample]] = {}
    for sample in samples:
        key = (sample.label or "safe") if stratify_by_cwe else "all"
        groups.setdefault(key, []).append(sample)
    result: dict[str, list[CorpusSample]] = {name: [] for name in names}
    for key in sorted(groups):
        bucket = list(groups[key])
        rng.shuffle(bucket)
        start = 0
        for position, name in enumerate(names):
            if position == len(names) - 1:
                chunk = bucket[start:]
            else:
                count = int(round(len(bucket) * float(fractions[name])))
                chunk = bucket[start : start + count]
                start += count
            for sample in chunk:
                sample.split = name
            result[name].extend(chunk)
    for name in names:
        rng.shuffle(result[name])
    LOGGER.info("split corpus into %s", {name: len(items) for name, items in result.items()})
    return result


def partition_by_cwe(samples: Iterable[CorpusSample]) -> tuple[dict[str, list[CorpusSample]], list[CorpusSample]]:
    vulnerable: dict[str, list[CorpusSample]] = {}
    safe: list[CorpusSample] = []
    for sample in samples:
        if sample.verdict.vulnerable and sample.verdict.cwe:
            vulnerable.setdefault(sample.verdict.cwe, []).append(sample)
        elif not sample.verdict.vulnerable:
            safe.append(sample)
    return vulnerable, safe
