from __future__ import annotations

from typing import Sequence

from tqdm import tqdm

from reasonsec.config import Config
from reasonsec.data.cwe_catalog import CweCatalog
from reasonsec.models.language_model import LanguageModel
from reasonsec.oracle.base import SecurityOracle
from reasonsec.reasoning.parsing import ChainParser
from reasonsec.reasoning.templates import TemplateLibrary
from reasonsec.types import BenchmarkPrompt, CorpusSample
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


class ReasoningElicitor:
    def __init__(
        self,
        config: Config,
        model: LanguageModel,
        oracle: SecurityOracle,
        catalog: CweCatalog,
        templates: TemplateLibrary | None = None,
        parser: ChainParser | None = None,
    ) -> None:
        self._config = config
        self._model = model
        self._oracle = oracle
        self._catalog = catalog
        self._templates = templates or TemplateLibrary(config)
        self._parser = parser or ChainParser(config)
        self._batch_size = config.require_int("reasoning.elicitation_batch_size")
        self._restrict_universe = config.require_bool("reasoning.restrict_mentions_to_benchmark_cwes")

    @property
    def parser(self) -> ChainParser:
        return self._parser

    def build_instruction(self, prompt: BenchmarkPrompt) -> str:
        return self._templates.elicitation.render(prompt=prompt.prompt, language=prompt.language)

    def elicit(self, prompts: Sequence[BenchmarkPrompt], universe: Sequence[str] | None = None) -> list[CorpusSample]:
        restrict_to = list(universe) if (universe and self._restrict_universe) else None
        samples: list[CorpusSample] = []
        for start in tqdm(range(0, len(prompts), self._batch_size), desc="eliciting reasoning chains"):
            batch = list(prompts[start : start + self._batch_size])
            instructions = [self.build_instruction(prompt) for prompt in batch]
            completions = (
                self._model.generate_batch(instructions)
                if len(instructions) > 1
                else [self._model.generate(instructions[0])]
            )
            for prompt, completion in zip(batch, completions):
                samples.append(self._build_sample(prompt, completion, restrict_to))
        LOGGER.info("elicited %d reasoning chains", len(samples))
        return samples

    def _build_sample(
        self,
        prompt: BenchmarkPrompt,
        completion: str,
        restrict_to: Sequence[str] | None,
    ) -> CorpusSample:
        chain = self._parser.parse(completion)
        verdict = self._oracle.evaluate(chain.code, prompt.language, prompt.cwe)
        mentioned = self._catalog.mentions(chain.planning_segment, restrict_to=restrict_to)
        return CorpusSample(
            identifier=prompt.identifier,
            prompt=prompt,
            full_text=completion,
            chain=chain,
            verdict=verdict,
            mentioned_cwes=mentioned,
        )
