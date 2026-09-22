from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from tqdm import tqdm

from reasonsec.config import Config
from reasonsec.evaluation.execution import ProgramRunner
from reasonsec.models.language_model import LanguageModel
from reasonsec.reasoning.templates import TemplateLibrary
from reasonsec.types import BenchmarkPrompt, FunctionalTask, MultipleChoiceQuestion
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


class SupportsGeneration(Protocol):
    def generate(self, prompt: BenchmarkPrompt):
        ...


@dataclass
class FunctionalEvaluationResult:
    method: str
    seed: int
    total: int
    passed: int
    per_task: dict[str, bool] = field(default_factory=dict)

    @property
    def pass_at_one(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "seed": self.seed,
            "total": self.total,
            "passed": self.passed,
            "pass_at_1": self.pass_at_one,
            "per_task": dict(self.per_task),
        }


@dataclass
class KnowledgeEvaluationResult:
    method: str
    seed: int
    total: int
    correct: int
    per_subject: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "seed": self.seed,
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "per_subject": {
                subject: {**counts, "accuracy": counts["correct"] / counts["total"] if counts["total"] else 0.0}
                for subject, counts in sorted(self.per_subject.items())
            },
        }


class FunctionalEvaluator:
    def __init__(self, config: Config, templates: TemplateLibrary | None = None) -> None:
        section = config.require_section("evaluation.utility")
        self._runner = ProgramRunner(config)
        self._templates = templates or TemplateLibrary(config)
        self._program_template = section.require_str("program_template")
        self._language = section.require_str("language")

    def build_prompt(self, task: FunctionalTask) -> BenchmarkPrompt:
        instruction = self._templates.functional_completion.render(
            instruction=task.instruction, context=task.context, entry_point=task.entry_point
        )
        return BenchmarkPrompt(
            identifier=task.task_id,
            prompt=instruction,
            language=self._language,
            benchmark="functional",
            cwe=None,
            metadata={"entry_point": task.entry_point},
        )

    def build_program(self, task: FunctionalTask, code: str) -> str:
        return (
            self._program_template.replace("{context}", task.context)
            .replace("{completion}", code)
            .replace("{test}", task.test)
            .replace("{entry_point}", task.entry_point)
        )

    def evaluate(
        self,
        generator: SupportsGeneration,
        tasks: Sequence[FunctionalTask],
        method: str,
        seed: int,
    ) -> FunctionalEvaluationResult:
        passed = 0
        per_task: dict[str, bool] = {}
        for task in tqdm(tasks, desc=f"evaluating pass@1 [{method}]"):
            prompt = self.build_prompt(task)
            outcome = generator.generate(prompt)
            program = self.build_program(task, outcome.chain.code)
            result = self._runner.run(program)
            per_task[task.task_id] = result.passed
            passed += int(result.passed)
        evaluation = FunctionalEvaluationResult(method=method, seed=seed, total=len(tasks), passed=passed, per_task=per_task)
        LOGGER.info("method %s seed %d pass@1 %.4f (%d of %d)", method, seed, evaluation.pass_at_one, passed, len(tasks))
        return evaluation


class KnowledgeEvaluator:
    def __init__(self, config: Config, model: LanguageModel) -> None:
        section = config.require_section("evaluation.knowledge")
        self._model = model
        self._question_template = section.require_str("question_template")
        self._choice_template = section.require_str("choice_template")
        self._answer_template = section.require_str("answer_template")
        self._answer_labels = [str(label) for label in section.require_list("answer_labels")]
        self._few_shot_count = section.require_int("few_shot_count")
        self._length_normalise = section.require_bool("length_normalise")

    def _render_question(self, question: MultipleChoiceQuestion, include_answer: bool) -> str:
        choices = "".join(
            self._choice_template.replace("{label}", self._answer_labels[index]).replace("{choice}", choice)
            for index, choice in enumerate(question.choices)
        )
        text = self._question_template.replace("{question}", question.question).replace("{choices}", choices)
        if include_answer:
            text += self._answer_template.replace("{answer}", self._answer_labels[question.answer_index])
        return text

    def evaluate(
        self,
        questions: Sequence[MultipleChoiceQuestion],
        few_shot_pool: Sequence[MultipleChoiceQuestion],
        method: str,
        seed: int,
    ) -> KnowledgeEvaluationResult:
        by_subject: dict[str, list[MultipleChoiceQuestion]] = {}
        for item in few_shot_pool:
            by_subject.setdefault(item.subject, []).append(item)
        correct = 0
        per_subject: dict[str, dict[str, int]] = {}
        for question in tqdm(questions, desc=f"evaluating knowledge [{method}]"):
            examples = by_subject.get(question.subject, [])[: self._few_shot_count]
            prefix = "".join(self._render_question(example, include_answer=True) for example in examples)
            prefix += self._render_question(question, include_answer=False)
            scores: list[float] = []
            for label in self._answer_labels[: len(question.choices)]:
                continuation = self._answer_template.replace("{answer}", label)
                score = self._model.continuation_log_likelihood(prefix, continuation)
                if self._length_normalise and continuation:
                    score = score / max(len(continuation), 1)
                scores.append(score)
            prediction = max(range(len(scores)), key=lambda index: scores[index])
            counts = per_subject.setdefault(question.subject, {"total": 0, "correct": 0})
            counts["total"] += 1
            if prediction == question.answer_index:
                counts["correct"] += 1
                correct += 1
        evaluation = KnowledgeEvaluationResult(
            method=method, seed=seed, total=len(questions), correct=correct, per_subject=per_subject
        )
        LOGGER.info("method %s seed %d knowledge accuracy %.4f", method, seed, evaluation.accuracy)
        return evaluation
