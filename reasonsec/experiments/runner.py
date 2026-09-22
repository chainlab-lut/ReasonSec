from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from reasonsec.config import Config, ConfigError
from reasonsec.data.benchmarks import load_functional_tasks, load_multiple_choice_questions
from reasonsec.evaluation.security import SecurityEvaluationResult, evaluate_security
from reasonsec.evaluation.statistics import mean_and_standard_deviation
from reasonsec.evaluation.utility import (
    FunctionalEvaluationResult,
    FunctionalEvaluator,
    KnowledgeEvaluationResult,
    KnowledgeEvaluator,
)
from reasonsec.pipeline import ReasonSecPipeline
from reasonsec.types import BenchmarkPrompt
from reasonsec.utils.io import ensure_directory, read_json, write_json
from reasonsec.utils.logging import get_logger
from reasonsec.utils.seeding import seed_everything

LOGGER = get_logger(__name__)


@dataclass
class MethodEvaluation:
    method: str
    security: dict[int, SecurityEvaluationResult] = field(default_factory=dict)
    secondary_security: dict[int, SecurityEvaluationResult] = field(default_factory=dict)
    functional: dict[int, FunctionalEvaluationResult] = field(default_factory=dict)
    knowledge: dict[int, KnowledgeEvaluationResult] = field(default_factory=dict)

    def security_rates(self) -> list[float]:
        return [result.security_rate for _seed, result in sorted(self.security.items())]

    def secondary_security_rates(self) -> list[float]:
        return [result.security_rate for _seed, result in sorted(self.secondary_security.items())]

    def pass_at_one_values(self) -> list[float]:
        return [result.pass_at_one for _seed, result in sorted(self.functional.items())]

    def knowledge_values(self) -> list[float]:
        return [result.accuracy for _seed, result in sorted(self.knowledge.items())]

    def aggregate(self) -> dict[str, Any]:
        security_mean, security_deviation = mean_and_standard_deviation(self.security_rates())
        secondary_mean, secondary_deviation = mean_and_standard_deviation(self.secondary_security_rates())
        functional_mean, functional_deviation = mean_and_standard_deviation(self.pass_at_one_values())
        knowledge_mean, knowledge_deviation = mean_and_standard_deviation(self.knowledge_values())
        return {
            "method": self.method,
            "security_rate_mean": security_mean,
            "security_rate_standard_deviation": security_deviation,
            "secondary_security_rate_mean": secondary_mean,
            "secondary_security_rate_standard_deviation": secondary_deviation,
            "pass_at_1_mean": functional_mean,
            "pass_at_1_standard_deviation": functional_deviation,
            "knowledge_accuracy_mean": knowledge_mean,
            "knowledge_accuracy_standard_deviation": knowledge_deviation,
            "seeds": sorted(self.security.keys()),
        }

    def per_category(self) -> dict[str, dict[str, float]]:
        totals: dict[str, dict[str, float]] = {}
        for result in self.security.values():
            for cwe, entry in result.per_category.items():
                record = totals.setdefault(cwe, {"total": 0.0, "avoided": 0.0, "rate": 0.0})
                record["total"] += entry.total
                record["avoided"] += entry.avoided
        seed_count = max(len(self.security), 1)
        for record in totals.values():
            record["total"] = record["total"] / seed_count
            record["avoided"] = record["avoided"] / seed_count
            record["rate"] = record["avoided"] / record["total"] if record["total"] else 0.0
        return totals


class ExperimentRunner:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.seeds = [int(seed) for seed in config.require_list("run.seeds")]
        self._root = ensure_directory(config.require_path("run.output_directory", must_exist=False))
        self._seed_scoped = config.require_bool("run.seed_scoped_outputs")
        self._pipelines: dict[int, ReasonSecPipeline] = {}
        self._evaluations: dict[str, MethodEvaluation] = {}
        self._functional_evaluator: FunctionalEvaluator | None = None
        self._knowledge_evaluator: KnowledgeEvaluator | None = None
        self._evaluate_functional = config.require_bool("evaluation.utility.enabled")
        self._evaluate_knowledge = config.require_bool("evaluation.knowledge.enabled")
        self._evaluate_secondary = config.require_bool("evaluation.secondary_security.enabled")
        self._primary_dataset = config.require_str("pipeline.security_dataset")
        self._secondary_dataset = config.require_str("evaluation.secondary_security.dataset")
        self._cache_results = config.require_bool("evaluation.cache_results")
        if self._evaluate_functional and not config.require_bool("evaluation.utility.execution.enabled"):
            raise ConfigError(
                "evaluation.utility.enabled is true while evaluation.utility.execution.enabled is false; "
                "functional correctness requires executing generated programs, so enable execution in a "
                "sandboxed environment or disable the functional evaluation"
            )

    @property
    def root(self) -> Path:
        return self._root

    def pipeline(self, seed: int) -> ReasonSecPipeline:
        if seed in self._pipelines:
            return self._pipelines[seed]
        overrides: dict[str, Any] = {}
        if self._seed_scoped:
            overrides["run.output_directory"] = str(self._root / f"seed_{seed}")
        pipeline = ReasonSecPipeline(self.config.derive(overrides), seed)
        if self._pipelines:
            reference = next(iter(self._pipelines.values()))
            pipeline.attach(
                model=reference.model,
                embedder=reference.embedder if self._requires_embedder() else None,
                oracle=reference.oracle,
                catalog=reference.catalog,
                templates=reference.templates,
            )
        self._pipelines[seed] = pipeline
        return pipeline

    def _requires_embedder(self) -> bool:
        return self.config.require_bool("evaluation.share_embedder_across_seeds")

    def functional_evaluator(self) -> FunctionalEvaluator:
        if self._functional_evaluator is None:
            self._functional_evaluator = FunctionalEvaluator(self.config)
        return self._functional_evaluator

    def knowledge_evaluator(self, seed: int) -> KnowledgeEvaluator:
        if self._knowledge_evaluator is None:
            self._knowledge_evaluator = KnowledgeEvaluator(self.config, self.pipeline(seed).model)
        return self._knowledge_evaluator

    def _result_path(self, method: str, seed: int, name: str) -> Path:
        return self._root / "results" / method / f"seed_{seed}" / f"{name}.json"

    def evaluate_method(self, method: str, seeds: Sequence[int] | None = None) -> MethodEvaluation:
        selected_seeds = [int(seed) for seed in (seeds or self.seeds)]
        evaluation = self._evaluations.setdefault(method, MethodEvaluation(method=method))
        for seed in selected_seeds:
            if seed in evaluation.security:
                continue
            seed_everything(seed, self.config.require_bool("run.deterministic_algorithms"))
            pipeline = self.pipeline(seed)
            generator = pipeline.build_generator(method)
            evaluation_prompts = pipeline.evaluation_prompts()
            evaluation.security[seed] = self._security_stage(
                method, seed, generator, evaluation_prompts, self._primary_dataset, "security"
            )
            if self._evaluate_secondary:
                secondary_prompts = pipeline.security_prompts(self._secondary_dataset)
                evaluation.secondary_security[seed] = self._security_stage(
                    method, seed, generator, secondary_prompts, self._secondary_dataset, "secondary_security"
                )
            if self._evaluate_functional:
                evaluation.functional[seed] = self._functional_stage(method, seed, generator)
            if self._evaluate_knowledge:
                evaluation.knowledge[seed] = self._knowledge_stage(method, seed)
            pipeline.release_fine_tuning()
        return evaluation

    def _security_stage(
        self,
        method: str,
        seed: int,
        generator: Any,
        prompts: Sequence[BenchmarkPrompt],
        benchmark: str,
        name: str,
    ) -> SecurityEvaluationResult:
        path = self._result_path(method, seed, name)
        if self._cache_results and path.is_file():
            payload = read_json(path)
            return _security_from_payload(payload)
        pipeline = self.pipeline(seed)
        result = evaluate_security(generator, prompts, pipeline.oracle, method, seed, benchmark)
        write_json(path, result.as_dict(include_records=True))
        return result

    def _functional_stage(self, method: str, seed: int, generator: Any) -> FunctionalEvaluationResult:
        path = self._result_path(method, seed, "functional")
        if self._cache_results and path.is_file():
            payload = read_json(path)
            return FunctionalEvaluationResult(
                method=payload["method"],
                seed=int(payload["seed"]),
                total=int(payload["total"]),
                passed=int(payload["passed"]),
                per_task={key: bool(value) for key, value in payload.get("per_task", {}).items()},
            )
        tasks = load_functional_tasks(self.config, self.config.require_str("evaluation.utility.dataset"))
        result = self.functional_evaluator().evaluate(generator, tasks, method, seed)
        write_json(path, result.as_dict())
        return result

    def _knowledge_stage(self, method: str, seed: int) -> KnowledgeEvaluationResult:
        path = self._result_path(method, seed, "knowledge")
        if self._cache_results and path.is_file():
            payload = read_json(path)
            return KnowledgeEvaluationResult(
                method=payload["method"],
                seed=int(payload["seed"]),
                total=int(payload["total"]),
                correct=int(payload["correct"]),
                per_subject={
                    subject: {"total": int(counts["total"]), "correct": int(counts["correct"])}
                    for subject, counts in payload.get("per_subject", {}).items()
                },
            )
        dataset_key = self.config.require_str("evaluation.knowledge.dataset")
        questions = load_multiple_choice_questions(
            self.config, dataset_key, self.config.require_str("evaluation.knowledge.evaluation_split")
        )
        few_shot = load_multiple_choice_questions(
            self.config, dataset_key, self.config.require_str("evaluation.knowledge.few_shot_split")
        )
        result = self.knowledge_evaluator(seed).evaluate(questions, few_shot, method, seed)
        write_json(path, result.as_dict())
        return result

    def prepare(self, methods: Sequence[str]) -> None:
        for seed in self.seeds:
            self.pipeline(seed).prepare_offline(methods)

    def evaluations(self) -> dict[str, MethodEvaluation]:
        return dict(self._evaluations)


def _security_from_payload(payload: Mapping[str, Any]) -> SecurityEvaluationResult:
    from reasonsec.evaluation.security import CategorySecurityResult
    from reasonsec.types import GenerationRecord

    return SecurityEvaluationResult(
        method=str(payload["method"]),
        seed=int(payload["seed"]),
        benchmark=str(payload["benchmark"]),
        total=int(payload["total"]),
        avoided=int(payload["avoided"]),
        per_category={
            cwe: CategorySecurityResult(cwe=cwe, total=int(entry["total"]), avoided=int(entry["avoided"]))
            for cwe, entry in payload.get("per_category", {}).items()
        },
        records=[GenerationRecord.from_dict(record) for record in payload.get("records", [])],
    )
