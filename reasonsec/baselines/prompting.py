from __future__ import annotations

import time
from typing import Sequence

from reasonsec.config import Config, ConfigError
from reasonsec.intervention.runtime import GenerationOutcome
from reasonsec.models.language_model import LanguageModel
from reasonsec.reasoning.parsing import ChainParser
from reasonsec.reasoning.templates import PromptTemplate, TemplateLibrary
from reasonsec.types import BenchmarkPrompt, CorpusSample
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


class PromptingGenerator:
    def __init__(
        self,
        config: Config,
        model: LanguageModel,
        template: PromptTemplate,
        structured: bool,
        max_new_tokens: int,
        exemplar_pool: Sequence[CorpusSample] | None = None,
        parser: ChainParser | None = None,
    ) -> None:
        self._model = model
        self._template = template
        self._structured = structured
        self._max_new_tokens = max_new_tokens
        self._parser = parser or ChainParser(config)
        self._exemplar_pool = list(exemplar_pool or [])

    @property
    def parser(self) -> ChainParser:
        return self._parser

    def _exemplar_for(self, prompt: BenchmarkPrompt) -> CorpusSample:
        candidates = [
            sample
            for sample in self._exemplar_pool
            if sample.prompt.language == prompt.language
            and not sample.verdict.vulnerable
            and sample.identifier != prompt.identifier
            and sample.chain.code.strip()
        ]
        if not candidates:
            raise ConfigError(
                f"no safe exemplar is available in the curated corpus for language '{prompt.language}'"
            )
        return sorted(candidates, key=lambda sample: sample.identifier)[0]

    def instruction_for(self, prompt: BenchmarkPrompt) -> str:
        values = {"prompt": prompt.prompt, "language": prompt.language}
        if "example_task" in self._template.placeholders or "example_solution" in self._template.placeholders:
            exemplar = self._exemplar_for(prompt)
            values["example_task"] = exemplar.prompt.prompt
            values["example_solution"] = exemplar.chain.code
        return self._template.render(**{key: values[key] for key in self._template.placeholders})

    def generate(self, prompt: BenchmarkPrompt) -> GenerationOutcome:
        started = time.perf_counter()
        instruction = self.instruction_for(prompt)
        completion = self._model.generate(instruction, overrides={"max_new_tokens": self._max_new_tokens})
        chain = self._parser.parse(completion) if self._structured else self._parser.parse_code_only(completion)
        return GenerationOutcome(
            trace_text="",
            completion_text=completion,
            full_text=completion,
            chain=chain,
            state=None,
            applied_cwe=None,
            replacement_values={},
            modified_features=None,
            latency_seconds=time.perf_counter() - started,
        )


def build_prompting_generator(
    config: Config,
    model: LanguageModel,
    template_name: str,
    exemplar_pool: Sequence[CorpusSample] | None = None,
    templates: TemplateLibrary | None = None,
    parser: ChainParser | None = None,
) -> PromptingGenerator:
    library = templates or TemplateLibrary(config)
    available = {
        "unstructured": library.unstructured,
        "elicitation": library.elicitation,
        "zero_shot_security": library.zero_shot_security,
        "one_shot_security": library.one_shot_security,
        "chain_of_thought_security": library.chain_of_thought_security,
        "functional_completion": library.functional_completion,
    }
    if template_name not in available:
        raise ConfigError(f"unknown prompt template '{template_name}'; available templates are {sorted(available)}")
    structured_templates = {str(item) for item in config.require_list("reasoning.structured_templates")}
    return PromptingGenerator(
        config=config,
        model=model,
        template=available[template_name],
        structured=template_name in structured_templates,
        max_new_tokens=config.require_int("baselines.prompting.max_new_tokens"),
        exemplar_pool=exemplar_pool,
        parser=parser,
    )
