from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

from tqdm import tqdm

from reasonsec.intervention.runtime import GenerationOutcome
from reasonsec.oracle.base import SecurityOracle
from reasonsec.types import BenchmarkPrompt, GenerationRecord
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


class SupportsGeneration(Protocol):
    def generate(self, prompt: BenchmarkPrompt) -> GenerationOutcome:
        ...


@dataclass
class CategorySecurityResult:
    cwe: str
    total: int
    avoided: int

    @property
    def rate(self) -> float:
        return self.avoided / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, float | int | str]:
        return {"cwe": self.cwe, "total": self.total, "avoided": self.avoided, "rate": self.rate}


@dataclass
class SecurityEvaluationResult:
    method: str
    seed: int
    benchmark: str
    total: int
    avoided: int
    per_category: dict[str, CategorySecurityResult] = field(default_factory=dict)
    records: list[GenerationRecord] = field(default_factory=list)

    @property
    def security_rate(self) -> float:
        return self.avoided / self.total if self.total else 0.0

    def outcome_vector(self) -> list[int]:
        return [1 - record.verdict.indicator for record in self.records]

    def as_dict(self, include_records: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "method": self.method,
            "seed": self.seed,
            "benchmark": self.benchmark,
            "total": self.total,
            "avoided": self.avoided,
            "security_rate": self.security_rate,
            "per_category": {cwe: result.as_dict() for cwe, result in sorted(self.per_category.items())},
        }
        if include_records:
            payload["records"] = [record.as_dict() for record in self.records]
        return payload


def evaluate_security(
    generator: SupportsGeneration,
    prompts: Sequence[BenchmarkPrompt],
    oracle: SecurityOracle,
    method: str,
    seed: int,
    benchmark: str,
) -> SecurityEvaluationResult:
    records: list[GenerationRecord] = []
    per_category: dict[str, CategorySecurityResult] = {}
    avoided = 0
    for prompt in tqdm(prompts, desc=f"evaluating security [{method}]"):
        outcome = generator.generate(prompt)
        verdict = oracle.evaluate(outcome.chain.code, prompt.language, prompt.cwe)
        record = GenerationRecord(
            identifier=prompt.identifier,
            prompt=prompt,
            method=method,
            seed=seed,
            full_text=outcome.full_text,
            chain=outcome.chain,
            verdict=verdict,
            applied_cwe=outcome.applied_cwe,
            action=[float(value) for value in outcome.replacement_values.values()],
            latency_seconds=outcome.latency_seconds,
        )
        records.append(record)
        is_avoided = 0 if verdict.vulnerable else 1
        avoided += is_avoided
        category = prompt.cwe or verdict.cwe
        if category is not None:
            entry = per_category.setdefault(category, CategorySecurityResult(cwe=category, total=0, avoided=0))
            entry.total += 1
            entry.avoided += is_avoided
    result = SecurityEvaluationResult(
        method=method,
        seed=seed,
        benchmark=benchmark,
        total=len(records),
        avoided=avoided,
        per_category=per_category,
        records=records,
    )
    LOGGER.info(
        "method %s seed %d security rate %.4f (%d of %d)",
        method,
        seed,
        result.security_rate,
        avoided,
        len(records),
    )
    return result


def paired_outcomes(
    left: SecurityEvaluationResult, right: SecurityEvaluationResult
) -> tuple[list[int], list[int]]:
    left_map: Mapping[str, int] = {record.identifier: 1 - record.verdict.indicator for record in left.records}
    right_map: Mapping[str, int] = {record.identifier: 1 - record.verdict.indicator for record in right.records}
    shared = [identifier for identifier in left_map if identifier in right_map]
    return [left_map[identifier] for identifier in shared], [right_map[identifier] for identifier in shared]


def avoidance_identifiers(result: SecurityEvaluationResult) -> set[str]:
    return {record.identifier for record in result.records if not record.verdict.vulnerable}
