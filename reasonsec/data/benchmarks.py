from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml

from reasonsec.config import Config, ConfigError
from reasonsec.data.cwe_catalog import canonical_cwe
from reasonsec.types import BenchmarkPrompt, FunctionalTask, MultipleChoiceQuestion
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _iter_records(paths: Sequence[Path], file_format: str) -> Iterator[Mapping[str, Any]]:
    normalised_format = file_format.strip().lower()
    for path in paths:
        scanning_directory = path.is_dir()
        targets = sorted(path.rglob("*")) if scanning_directory else [path]
        for target in targets:
            if target.is_dir():
                continue
            if normalised_format == "json":
                if scanning_directory and target.suffix.lower() != ".json":
                    continue
                with target.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                records = payload if isinstance(payload, list) else payload.get("prompts", payload.get("data", []))
                for record in records:
                    if isinstance(record, Mapping):
                        yield record
            elif normalised_format == "jsonl":
                if scanning_directory and target.suffix.lower() not in {".jsonl", ".ndjson"}:
                    continue
                with target.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        stripped = line.strip()
                        if stripped:
                            record = json.loads(stripped)
                            if isinstance(record, Mapping):
                                yield record
            elif normalised_format == "yaml":
                if scanning_directory and target.suffix.lower() not in {".yaml", ".yml"}:
                    continue
                with target.open("r", encoding="utf-8") as handle:
                    payload = yaml.safe_load(handle) or []
                records = payload if isinstance(payload, list) else [payload]
                for record in records:
                    if isinstance(record, Mapping):
                        yield record
            else:
                raise ConfigError(f"unsupported dataset format '{file_format}'")


def _lookup(record: Mapping[str, Any], field_name: str) -> Any:
    node: Any = record
    for part in str(field_name).split("."):
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        else:
            return None
    return node


def _require_field(record: Mapping[str, Any], field_name: str, context: str) -> Any:
    value = _lookup(record, field_name)
    if value is None:
        raise ConfigError(f"field '{field_name}' configured for {context} is absent from a record: {sorted(record)}")
    return value


def load_security_prompts(config: Config, dataset_key: str) -> list[BenchmarkPrompt]:
    section = config.require_section(f"datasets.{dataset_key}")
    paths = section.require_paths("path")
    file_format = section.require_str("format")
    fields = section.require_section("fields")
    identifier_field = fields.optional("identifier")
    prompt_field = fields.require_str("prompt")
    language_field = fields.optional("language")
    cwe_field = fields.optional("cwe")
    default_language = section.optional("default_language")
    cwe_pattern = section.optional("cwe_from_identifier_pattern")
    language_allowlist = section.optional("languages")
    metadata_fields = section.optional("metadata_fields") or []
    context_field = fields.optional("context")

    prompts: list[BenchmarkPrompt] = []
    seen: set[str] = set()
    for index, record in enumerate(_iter_records(paths, file_format)):
        prompt_text = _require_field(record, prompt_field, dataset_key)
        if context_field:
            context_value = _lookup(record, context_field)
            if context_value:
                prompt_text = f"{context_value}\n{prompt_text}"
        raw_identifier = _lookup(record, identifier_field) if identifier_field else None
        identifier = str(raw_identifier) if raw_identifier is not None else f"{dataset_key}-{index}"
        if identifier in seen:
            identifier = f"{identifier}-{index}"
        seen.add(identifier)
        language = str(_lookup(record, language_field)) if language_field and _lookup(record, language_field) else None
        if language is None:
            if default_language is None:
                raise ConfigError(
                    f"record {identifier} of {dataset_key} has no language and datasets.{dataset_key}.default_language is unset"
                )
            language = str(default_language)
        language = language.strip().lower()
        if language_allowlist and language not in {str(item).lower() for item in language_allowlist}:
            continue
        cwe_value = _lookup(record, cwe_field) if cwe_field else None
        cwe = canonical_cwe(cwe_value) if cwe_value is not None else None
        if cwe is None and cwe_pattern:
            match = re.search(str(cwe_pattern), identifier)
            if match:
                cwe = canonical_cwe(match.group(match.lastindex or 0))
        metadata = {name: _lookup(record, name) for name in metadata_fields}
        prompts.append(
            BenchmarkPrompt(
                identifier=identifier,
                prompt=str(prompt_text),
                language=language,
                benchmark=dataset_key,
                cwe=cwe,
                metadata={key: value for key, value in metadata.items() if value is not None},
            )
        )
    if not prompts:
        raise ConfigError(f"dataset '{dataset_key}' produced no prompts from {[str(path) for path in paths]}")
    LOGGER.info("loaded %d prompts from dataset '%s'", len(prompts), dataset_key)
    return prompts


def load_functional_tasks(config: Config, dataset_key: str) -> list[FunctionalTask]:
    section = config.require_section(f"datasets.{dataset_key}")
    paths = section.require_paths("path")
    file_format = section.require_str("format")
    fields = section.require_section("fields")
    task_field = fields.require_str("task_id")
    instruction_field = fields.require_str("instruction")
    context_field = fields.optional("context")
    test_field = fields.require_str("test")
    entry_point_field = fields.require_str("entry_point")

    tasks: list[FunctionalTask] = []
    for record in _iter_records(paths, file_format):
        context_value = _lookup(record, context_field) if context_field else ""
        tasks.append(
            FunctionalTask(
                task_id=str(_require_field(record, task_field, dataset_key)),
                instruction=str(_require_field(record, instruction_field, dataset_key)),
                context=str(context_value or ""),
                test=str(_require_field(record, test_field, dataset_key)),
                entry_point=str(_require_field(record, entry_point_field, dataset_key)),
                metadata={},
            )
        )
    if not tasks:
        raise ConfigError(f"dataset '{dataset_key}' produced no functional tasks")
    LOGGER.info("loaded %d functional tasks from dataset '%s'", len(tasks), dataset_key)
    return tasks


def load_multiple_choice_questions(config: Config, dataset_key: str, split: str) -> list[MultipleChoiceQuestion]:
    section = config.require_section(f"datasets.{dataset_key}")
    split_directories = section.require_mapping("splits")
    if split not in split_directories:
        raise ConfigError(f"split '{split}' is not configured under datasets.{dataset_key}.splits")
    directory = Path(str(split_directories[split])).expanduser()
    if not directory.exists():
        raise ConfigError(f"split directory for datasets.{dataset_key}.splits.{split} does not exist: {directory}")
    file_format = section.require_str("format")
    choice_count = section.require_int("choice_count")
    subject_pattern = section.require_str("subject_from_filename_pattern")
    answer_labels = [str(label) for label in section.require_list("answer_labels")]

    questions: list[MultipleChoiceQuestion] = []
    files = sorted(directory.rglob("*")) if directory.is_dir() else [directory]
    for path in files:
        if path.is_dir():
            continue
        if file_format == "csv" and path.suffix.lower() != ".csv":
            continue
        match = re.search(subject_pattern, path.name)
        subject = match.group(1) if match and match.groups() else path.stem
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row_index, row in enumerate(csv.reader(handle)):
                if len(row) < choice_count + 2:
                    continue
                question_text = row[0].strip()
                choices = [cell.strip() for cell in row[1 : choice_count + 1]]
                answer_label = row[choice_count + 1].strip()
                if answer_label not in answer_labels:
                    continue
                questions.append(
                    MultipleChoiceQuestion(
                        identifier=f"{subject}-{row_index}",
                        subject=subject,
                        question=question_text,
                        choices=choices,
                        answer_index=answer_labels.index(answer_label),
                    )
                )
    if not questions:
        raise ConfigError(f"no questions parsed for dataset '{dataset_key}' split '{split}' from {directory}")
    LOGGER.info("loaded %d %s questions for split '%s'", len(questions), dataset_key, split)
    return questions


def group_by_cwe(prompts: Iterable[BenchmarkPrompt]) -> dict[str, list[BenchmarkPrompt]]:
    grouped: dict[str, list[BenchmarkPrompt]] = {}
    for prompt in prompts:
        if prompt.cwe is None:
            continue
        grouped.setdefault(prompt.cwe, []).append(prompt)
    return grouped
