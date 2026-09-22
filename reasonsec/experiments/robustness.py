from __future__ import annotations

import random
import time
from typing import Any, Sequence

from tqdm import tqdm

from reasonsec.config import Config, ConfigError
from reasonsec.data.cwe_catalog import canonical_cwe
from reasonsec.evaluation.report import render_markdown_table, write_csv, write_report
from reasonsec.evaluation.security import evaluate_security
from reasonsec.evaluation.statistics import classification_scores, cohens_kappa
from reasonsec.experiments.runner import ExperimentRunner
from reasonsec.intervention.runtime import GenerationOutcome, ReasonSecGenerator
from reasonsec.oracle import build_oracle
from reasonsec.pipeline import MethodDefinition
from reasonsec.types import BenchmarkPrompt
from reasonsec.utils.io import read_json, write_json
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


class NoisyTraceGenerator:
    def __init__(
        self,
        generator: ReasonSecGenerator,
        model,
        noise_rate: float,
        seed: int,
    ) -> None:
        self._generator = generator
        self._model = model
        self._noise_rate = float(noise_rate)
        self._random = random.Random(seed)

    @property
    def parser(self):
        return self._generator.parser

    def _corrupt(self, segment: str) -> str:
        if self._noise_rate <= 0.0 or not segment.strip():
            return segment
        tokenizer = self._model.tokenizer
        token_ids = tokenizer.encode(segment, add_special_tokens=False)
        vocabulary_size = int(tokenizer.vocab_size)
        corrupted = [
            self._random.randrange(vocabulary_size) if self._random.random() < self._noise_rate else token_id
            for token_id in token_ids
        ]
        return tokenizer.decode(corrupted, skip_special_tokens=True)

    def generate(self, prompt: BenchmarkPrompt) -> GenerationOutcome:
        started = time.perf_counter()
        instruction = self._generator.instruction_for(prompt)
        trace_text = self._generator.generate_trace(instruction)
        trace_chain = self._generator.parser.parse(trace_text)
        span = trace_chain.planning_char_span
        if span is not None:
            corrupted_segment = self._corrupt(trace_text[span[0] : span[1]])
            trace_text = trace_text[: span[0]] + corrupted_segment + trace_text[span[1] :]
            trace_chain = self._generator.parser.parse(trace_text)
        state = self._generator.build_state(instruction, trace_chain, trace_text)
        applied_cwe, replacements = self._generator.select_and_compute(state)
        self._generator.apply_replacements(replacements)
        try:
            completion = self._generator.complete(instruction, trace_text)
        finally:
            self._generator.clear_replacements()
        full_text = trace_text + completion
        return GenerationOutcome(
            trace_text=trace_text,
            completion_text=completion,
            full_text=full_text,
            chain=self._generator.parser.parse(full_text),
            state=state,
            applied_cwe=applied_cwe,
            replacement_values=replacements,
            modified_features=None,
            latency_seconds=time.perf_counter() - started,
        )


class ForcedCategoryGenerator:
    def __init__(self, generator: ReasonSecGenerator, template, categories: Sequence[str]) -> None:
        self._generator = generator
        self._template = template
        self._categories = sorted(categories, key=lambda item: int(item.split("-")[1]))

    @property
    def parser(self):
        return self._generator.parser

    def _forced_category(self, prompt: BenchmarkPrompt) -> str:
        if not self._categories:
            raise ConfigError("the forced-category experiment requires at least one attributed CWE category")
        target = canonical_cwe(prompt.cwe)
        for category in self._categories:
            if category != target:
                return category
        return self._categories[0]

    def generate(self, prompt: BenchmarkPrompt) -> GenerationOutcome:
        started = time.perf_counter()
        forced = self._forced_category(prompt)
        values = {"prompt": prompt.prompt, "language": prompt.language, "forced_cwe": forced}
        instruction = self._template.render(**{key: values[key] for key in self._template.placeholders})
        trace_text = self._generator.generate_trace(instruction)
        trace_chain = self._generator.parser.parse(trace_text)
        state = self._generator.build_state(instruction, trace_chain, trace_text)
        applied_cwe, replacements = self._generator.select_and_compute(state)
        self._generator.apply_replacements(replacements)
        try:
            completion = self._generator.complete(instruction, trace_text)
        finally:
            self._generator.clear_replacements()
        full_text = trace_text + completion
        return GenerationOutcome(
            trace_text=trace_text,
            completion_text=completion,
            full_text=full_text,
            chain=self._generator.parser.parse(full_text),
            state=state,
            applied_cwe=applied_cwe,
            replacement_values=replacements,
            modified_features=None,
            latency_seconds=time.perf_counter() - started,
        )


def run_reasoning_noise(runner: ExperimentRunner, name: str) -> dict[str, Any]:
    config = runner.config
    section = config.require_section(f"experiments.{name}")
    method = section.require_str("method")
    noise_rates = [float(rate) for rate in section.require_list("noise_rates")]
    include_forced_category = section.require_bool("include_forced_category")
    runner.prepare([method])
    seed = runner.seeds[0]
    pipeline = runner.pipeline(seed)
    prompts = pipeline.evaluation_prompts()
    rows: list[list[Any]] = []
    payload: dict[str, Any] = {"name": name, "method": method, "conditions": []}

    for rate in noise_rates:
        generator = NoisyTraceGenerator(pipeline.build_generator(method), pipeline.model, rate, seed)
        result = evaluate_security(
            generator, prompts, pipeline.oracle, f"{method}_noise_{rate}", seed, config.require_str("pipeline.security_dataset")
        )
        payload["conditions"].append({"condition": f"noise_{rate}", "security_rate": result.security_rate})
        rows.append([f"{rate * 100.0:.0f}% token corruption", f"{result.security_rate * 100.0:.1f}"])

    if include_forced_category:
        base_generator = pipeline.build_generator(method)
        template = pipeline.templates.forced_category
        definition = MethodDefinition.from_config(config, method)
        categories = sorted(
            pipeline.build_feature_sets(definition.attribution_key, pipeline.method_config(definition))
        )
        generator = ForcedCategoryGenerator(base_generator, template, categories)
        result = evaluate_security(
            generator, prompts, pipeline.oracle, f"{method}_forced_category", seed, config.require_str("pipeline.security_dataset")
        )
        payload["conditions"].append({"condition": "forced_category", "security_rate": result.security_rate})
        rows.append(["forced incorrect CWE plan", f"{result.security_rate * 100.0:.1f}"])

    headers = ["Condition", "Security rate (%)"]
    directory = runner.root / "experiments" / name
    write_csv(directory / "reasoning_noise.csv", headers, rows)
    write_json(directory / "results.json", payload)
    write_report(directory / "report.md", f"Experiment: {name}", [("Reasoning robustness", render_markdown_table(headers, rows))])
    return payload


def run_oracle_reliability(runner: ExperimentRunner, name: str) -> dict[str, Any]:
    config = runner.config
    section = config.require_section(f"experiments.{name}")
    methods = [str(item) for item in section.require_list("methods")]
    alternative_oracle_name = section.require_str("alternative_oracle")
    alternative = build_oracle(config, alternative_oracle_name)
    seed = runner.seeds[0]
    payload: dict[str, Any] = {"name": name, "alternative_oracle": alternative_oracle_name, "methods": {}}
    rows: list[list[Any]] = []

    for method in methods:
        evaluation = runner.evaluate_method(method, [seed])
        records = evaluation.security[seed].records
        primary_flags = [record.verdict.indicator for record in records]
        alternative_flags: list[int] = []
        for record in tqdm(records, desc=f"re-scoring [{method}] with {alternative_oracle_name}"):
            verdict = alternative.evaluate(record.chain.code, record.prompt.language, record.prompt.cwe)
            alternative_flags.append(verdict.indicator)
        agreement = sum(1 for left, right in zip(primary_flags, alternative_flags) if left == right)
        scores = classification_scores(primary_flags, alternative_flags)
        alternative_rate = 1.0 - (sum(alternative_flags) / len(alternative_flags) if alternative_flags else 0.0)
        payload["methods"][method] = {
            "primary_security_rate": evaluation.security[seed].security_rate,
            "alternative_security_rate": alternative_rate,
            "agreement": agreement / len(records) if records else 0.0,
            "agreement_count": agreement,
            "total": len(records),
            "scores": scores,
        }
        rows.append(
            [
                method,
                f"{evaluation.security[seed].security_rate * 100.0:.1f}",
                f"{alternative_rate * 100.0:.1f}",
                f"{(agreement / len(records) if records else 0.0) * 100.0:.1f}",
            ]
        )

    headers = ["Method", "Primary oracle security rate (%)", "Alternative oracle security rate (%)", "Agreement (%)"]
    directory = runner.root / "experiments" / name
    write_csv(directory / "oracle_reliability.csv", headers, rows)
    write_json(directory / "results.json", payload)
    write_report(directory / "report.md", f"Experiment: {name}", [("Oracle reliability", render_markdown_table(headers, rows))])
    return payload


def run_hyperparameter_sensitivity(config: Config, name: str) -> dict[str, Any]:
    section = config.require_section(f"experiments.{name}")
    method = section.require_str("method")
    sweeps = section.require_mapping("sweeps")
    payload: dict[str, Any] = {"name": name, "method": method, "sweeps": {}}
    rows: list[list[Any]] = []

    for parameter, values in sweeps.items():
        parameter_results: list[dict[str, float]] = []
        for value in values:
            label = f"{parameter}={value}"
            variant = f"{method}__{parameter.replace('.', '_')}_{value}"
            overrides = {
                parameter: value,
                "run.output_directory": str(
                    config.require_path("run.output_directory", must_exist=False) / "sensitivity" / variant
                ),
            }
            variant_config = config.derive(overrides)
            runner = ExperimentRunner(variant_config)
            runner.prepare([method])
            evaluation = runner.evaluate_method(method)
            aggregate = evaluation.aggregate()
            parameter_results.append(
                {
                    "value": float(value),
                    "security_rate": aggregate["security_rate_mean"],
                    "pass_at_1": aggregate["pass_at_1_mean"],
                }
            )
            rows.append(
                [
                    label,
                    f"{aggregate['security_rate_mean'] * 100.0:.1f}",
                    f"{aggregate['pass_at_1_mean'] * 100.0:.1f}",
                ]
            )
        payload["sweeps"][parameter] = parameter_results

    headers = ["Setting", "Security rate (%)", "pass@1 (%)"]
    directory = config.require_path("run.output_directory", must_exist=False) / "experiments" / name
    write_csv(directory / "sensitivity.csv", headers, rows)
    write_json(directory / "results.json", payload)
    write_report(directory / "report.md", f"Experiment: {name}", [("Hyperparameter sensitivity", render_markdown_table(headers, rows))])
    return payload


def run_feature_label_agreement(runner: ExperimentRunner, name: str) -> dict[str, Any]:
    config = runner.config
    section = config.require_section(f"experiments.{name}")
    method = section.require_str("method")
    ratings_path = section.optional("human_ratings_path")
    category_count = section.require_int("rating_category_count")
    seed = runner.seeds[0]
    pipeline = runner.pipeline(seed)
    definition = MethodDefinition.from_config(config, method)
    feature_sets = pipeline.build_feature_sets(definition.attribution_key, pipeline.method_config(definition))
    labels = [
        {"cwe": cwe, "feature_index": index, "description": feature_set.descriptions.get(index, "")}
        for cwe, feature_set in feature_sets.items()
        for index in feature_set.feature_indices
    ]
    directory = runner.root / "experiments" / name
    write_json(directory / "feature_labels.json", labels)
    payload: dict[str, Any] = {"name": name, "label_count": len(labels)}

    if ratings_path:
        ratings = read_json(ratings_path)
        automatic: list[int] = []
        human: list[int] = []
        for entry in ratings:
            if "automatic_rating" in entry and "human_rating" in entry:
                automatic.append(int(entry["automatic_rating"]))
                human.append(int(entry["human_rating"]))
        if automatic:
            payload["agreement"] = sum(1 for a, h in zip(automatic, human) if a == h) / len(automatic)
            payload["cohens_kappa"] = cohens_kappa(automatic, human, category_count)
            payload["rated_count"] = len(automatic)
    write_json(directory / "results.json", payload)
    return payload
