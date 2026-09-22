from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

from reasonsec.config import Config
from reasonsec.intervention.proxy import ProxyVulnerabilityClassifier
from reasonsec.intervention.reward import ReasoningCoherenceReward, RewardTerms
from reasonsec.intervention.runtime import InterventionState, ReasonSecGenerator
from reasonsec.oracle.base import SecurityOracle
from reasonsec.types import BenchmarkPrompt, FeatureSet, OracleVerdict
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class EpisodeContext:
    prompt: BenchmarkPrompt
    instruction: str
    trace_text: str
    planning_segment: str
    state: InterventionState

    def observation(self, feature_indices: Sequence[int], include_embedding: bool, include_suspicion: bool) -> np.ndarray:
        return self.state.policy_input(feature_indices, include_embedding, include_suspicion)


@dataclass
class StepOutcome:
    verdict: OracleVerdict
    code: str
    reward: RewardTerms
    replacement_values: dict[int, float] = field(default_factory=dict)


class RecalibrationEnvironment:
    def __init__(
        self,
        config: Config,
        generator: ReasonSecGenerator,
        oracle: SecurityOracle,
        reward: ReasoningCoherenceReward,
        feature_set: FeatureSet,
        proxy: ProxyVulnerabilityClassifier | None = None,
    ) -> None:
        section = config.require_section("intervention.environment")
        self._generator = generator
        self._oracle = oracle
        self._reward = reward
        self._feature_set = feature_set
        self._proxy = proxy
        self._proxy_device = torch.device(section.require_str("proxy_device"))
        self._contexts: list[EpisodeContext] = []

    @property
    def feature_set(self) -> FeatureSet:
        return self._feature_set

    @property
    def contexts(self) -> list[EpisodeContext]:
        return list(self._contexts)

    def prepare(self, prompts: Sequence[BenchmarkPrompt]) -> list[EpisodeContext]:
        contexts: list[EpisodeContext] = []
        for prompt in tqdm(prompts, desc=f"preparing episodes for {self._feature_set.cwe}"):
            instruction = self._generator.instruction_for(prompt)
            trace_text = self._generator.generate_trace(instruction)
            trace_chain = self._generator.parser.parse(trace_text)
            if not trace_chain.planning_segment.strip():
                continue
            state = self._generator.build_state(instruction, trace_chain, trace_text)
            contexts.append(
                EpisodeContext(
                    prompt=prompt,
                    instruction=instruction,
                    trace_text=trace_text,
                    planning_segment=trace_chain.planning_segment,
                    state=state,
                )
            )
        self._contexts = contexts
        LOGGER.info("prepared %d episodes for %s", len(contexts), self._feature_set.cwe)
        return contexts

    def _proxy_probability(self, modified_features: np.ndarray) -> float | None:
        if self._proxy is None:
            return None
        tensor = torch.tensor(modified_features, dtype=torch.float32, device=self._proxy_device).unsqueeze(0)
        with torch.no_grad():
            return float(self._proxy.to(self._proxy_device).probability(tensor).squeeze(0).item())

    def step(self, context: EpisodeContext, replacement_values: Mapping[int, float]) -> StepOutcome:
        self._generator.apply_replacements(replacement_values)
        try:
            completion = self._generator.complete(context.instruction, context.trace_text)
        finally:
            self._generator.clear_replacements()
        chain = self._generator.parser.parse(context.trace_text + completion)
        code = chain.code
        verdict = self._oracle.evaluate(code, context.prompt.language, self._feature_set.cwe)
        modified_features = context.state.features.copy()
        for index, value in replacement_values.items():
            modified_features[int(index)] = float(value)
        reward = self._reward.compute(
            verdict=verdict,
            planning_segment=context.planning_segment,
            code=code,
            cwe=self._feature_set.cwe,
            replacement_values=dict(replacement_values),
            original_features=context.state.features,
            modified_features=modified_features,
            proxy_probability=self._proxy_probability(modified_features),
        )
        return StepOutcome(
            verdict=verdict, code=code, reward=reward, replacement_values=dict(replacement_values)
        )
