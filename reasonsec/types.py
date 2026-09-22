from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class BenchmarkPrompt:
    identifier: str
    prompt: str
    language: str
    benchmark: str
    cwe: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BenchmarkPrompt":
        return cls(
            identifier=str(payload["identifier"]),
            prompt=str(payload["prompt"]),
            language=str(payload["language"]),
            benchmark=str(payload["benchmark"]),
            cwe=payload.get("cwe"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class OracleFinding:
    rule_identifier: str
    cwe: str | None
    message: str
    line: int | None = None
    tool: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OracleVerdict:
    vulnerable: bool
    cwe: str | None
    findings: list[OracleFinding] = field(default_factory=list)
    tool: str = ""

    @property
    def indicator(self) -> int:
        return 1 if self.vulnerable else 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "vulnerable": self.vulnerable,
            "cwe": self.cwe,
            "tool": self.tool,
            "findings": [finding.as_dict() for finding in self.findings],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OracleVerdict":
        return cls(
            vulnerable=bool(payload["vulnerable"]),
            cwe=payload.get("cwe"),
            tool=str(payload.get("tool", "")),
            findings=[OracleFinding(**finding) for finding in payload.get("findings", [])],
        )


@dataclass
class ReasoningChain:
    raw_text: str
    phases: dict[str, str]
    planning_segment: str
    planning_char_span: tuple[int, int] | None
    code: str
    code_char_span: tuple[int, int] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "phases": self.phases,
            "planning_segment": self.planning_segment,
            "planning_char_span": list(self.planning_char_span) if self.planning_char_span else None,
            "code": self.code,
            "code_char_span": list(self.code_char_span) if self.code_char_span else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReasoningChain":
        planning_span = payload.get("planning_char_span")
        code_span = payload.get("code_char_span")
        return cls(
            raw_text=str(payload["raw_text"]),
            phases=dict(payload.get("phases", {})),
            planning_segment=str(payload.get("planning_segment", "")),
            planning_char_span=tuple(planning_span) if planning_span else None,
            code=str(payload.get("code", "")),
            code_char_span=tuple(code_span) if code_span else None,
        )


@dataclass
class CorpusSample:
    identifier: str
    prompt: BenchmarkPrompt
    full_text: str
    chain: ReasoningChain
    verdict: OracleVerdict
    mentioned_cwes: list[str] = field(default_factory=list)
    split: str = ""

    @property
    def label(self) -> str | None:
        return self.verdict.cwe if self.verdict.vulnerable else None

    @property
    def code(self) -> str:
        return self.chain.code

    @property
    def planning_segment(self) -> str:
        return self.chain.planning_segment

    def as_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "prompt": self.prompt.as_dict(),
            "full_text": self.full_text,
            "chain": self.chain.as_dict(),
            "verdict": self.verdict.as_dict(),
            "mentioned_cwes": list(self.mentioned_cwes),
            "split": self.split,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CorpusSample":
        return cls(
            identifier=str(payload["identifier"]),
            prompt=BenchmarkPrompt.from_dict(payload["prompt"]),
            full_text=str(payload["full_text"]),
            chain=ReasoningChain.from_dict(payload["chain"]),
            verdict=OracleVerdict.from_dict(payload["verdict"]),
            mentioned_cwes=list(payload.get("mentioned_cwes", [])),
            split=str(payload.get("split", "")),
        )


@dataclass
class FunctionalTask:
    task_id: str
    instruction: str
    context: str
    test: str
    entry_point: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MultipleChoiceQuestion:
    identifier: str
    subject: str
    question: str
    choices: list[str]
    answer_index: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationRecord:
    identifier: str
    prompt: BenchmarkPrompt
    method: str
    seed: int
    full_text: str
    chain: ReasoningChain
    verdict: OracleVerdict
    applied_cwe: str | None = None
    action: list[float] = field(default_factory=list)
    reward_terms: dict[str, float] = field(default_factory=dict)
    latency_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "prompt": self.prompt.as_dict(),
            "method": self.method,
            "seed": self.seed,
            "full_text": self.full_text,
            "chain": self.chain.as_dict(),
            "verdict": self.verdict.as_dict(),
            "applied_cwe": self.applied_cwe,
            "action": list(self.action),
            "reward_terms": dict(self.reward_terms),
            "latency_seconds": self.latency_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GenerationRecord":
        return cls(
            identifier=str(payload["identifier"]),
            prompt=BenchmarkPrompt.from_dict(payload["prompt"]),
            method=str(payload["method"]),
            seed=int(payload["seed"]),
            full_text=str(payload["full_text"]),
            chain=ReasoningChain.from_dict(payload["chain"]),
            verdict=OracleVerdict.from_dict(payload["verdict"]),
            applied_cwe=payload.get("applied_cwe"),
            action=list(payload.get("action", [])),
            reward_terms=dict(payload.get("reward_terms", {})),
            latency_seconds=float(payload.get("latency_seconds", 0.0)),
        )


@dataclass
class FeatureSet:
    cwe: str
    feature_indices: list[int]
    candidate_indices: list[int]
    scores: dict[int, float]
    distributional_distance: dict[int, float]
    alignment: dict[int, float]
    descriptions: dict[int, str] = field(default_factory=dict)
    vocabulary: list[str] = field(default_factory=list)
    safe_statistics: dict[int, dict[str, float]] = field(default_factory=dict)
    action_bounds: dict[int, list[float]] = field(default_factory=dict)
    quantile_levels: list[float] = field(default_factory=list)
    safe_quantiles: dict[int, list[float]] = field(default_factory=dict)
    vulnerable_quantiles: dict[int, list[float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cwe": self.cwe,
            "feature_indices": list(self.feature_indices),
            "candidate_indices": list(self.candidate_indices),
            "scores": {str(key): value for key, value in self.scores.items()},
            "distributional_distance": {str(key): value for key, value in self.distributional_distance.items()},
            "alignment": {str(key): value for key, value in self.alignment.items()},
            "descriptions": {str(key): value for key, value in self.descriptions.items()},
            "vocabulary": list(self.vocabulary),
            "safe_statistics": {str(key): value for key, value in self.safe_statistics.items()},
            "action_bounds": {str(key): list(value) for key, value in self.action_bounds.items()},
            "quantile_levels": list(self.quantile_levels),
            "safe_quantiles": {str(key): list(value) for key, value in self.safe_quantiles.items()},
            "vulnerable_quantiles": {str(key): list(value) for key, value in self.vulnerable_quantiles.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureSet":
        return cls(
            cwe=str(payload["cwe"]),
            feature_indices=[int(index) for index in payload["feature_indices"]],
            candidate_indices=[int(index) for index in payload.get("candidate_indices", [])],
            scores={int(key): float(value) for key, value in payload.get("scores", {}).items()},
            distributional_distance={
                int(key): float(value) for key, value in payload.get("distributional_distance", {}).items()
            },
            alignment={int(key): float(value) for key, value in payload.get("alignment", {}).items()},
            descriptions={int(key): str(value) for key, value in payload.get("descriptions", {}).items()},
            vocabulary=list(payload.get("vocabulary", [])),
            safe_statistics={
                int(key): {name: float(item) for name, item in value.items()}
                for key, value in payload.get("safe_statistics", {}).items()
            },
            action_bounds={
                int(key): [float(item) for item in value] for key, value in payload.get("action_bounds", {}).items()
            },
            quantile_levels=[float(item) for item in payload.get("quantile_levels", [])],
            safe_quantiles={
                int(key): [float(item) for item in value] for key, value in payload.get("safe_quantiles", {}).items()
            },
            vulnerable_quantiles={
                int(key): [float(item) for item in value]
                for key, value in payload.get("vulnerable_quantiles", {}).items()
            },
        )


def sequence_to_list(values: Sequence[Any]) -> list[Any]:
    return [value for value in values]
