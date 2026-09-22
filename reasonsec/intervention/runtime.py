from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from reasonsec.config import Config, ConfigError
from reasonsec.data.cwe_catalog import CWE_IDENTIFIER_PATTERN
from reasonsec.models.language_model import LanguageModel
from reasonsec.reasoning.parsing import ChainParser
from reasonsec.reasoning.templates import TemplateLibrary
from reasonsec.reasoning.vocabulary import SecurityConceptVocabulary
from reasonsec.sae.model import SparseAutoencoder
from reasonsec.types import BenchmarkPrompt, FeatureSet, ReasoningChain
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


class FeatureRecalibrator:
    def __init__(
        self,
        autoencoder: SparseAutoencoder,
        preserve_reconstruction_error: bool,
        apply_positions: str,
    ) -> None:
        self._autoencoder = autoencoder
        self._preserve_reconstruction_error = preserve_reconstruction_error
        self._apply_positions = apply_positions.strip().lower()
        self._indices: torch.Tensor | None = None
        self._values: torch.Tensor | None = None

    def set_targets(self, indices: Sequence[int], values: Sequence[float], device: torch.device) -> None:
        if not indices:
            self.clear()
            return
        self._indices = torch.tensor(list(indices), dtype=torch.long, device=device)
        self._values = torch.tensor(list(values), dtype=torch.float32, device=device)

    def clear(self) -> None:
        self._indices = None
        self._values = None

    @property
    def active(self) -> bool:
        return self._indices is not None and self._values is not None

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor | None:
        if not self.active:
            return None
        original_dtype = hidden.dtype
        working = hidden.to(torch.float32)
        features = self._autoencoder.encode(working)
        reconstruction = self._autoencoder.decode(features)
        modified = features.clone()
        if self._apply_positions == "last":
            modified[:, -1:, self._indices] = self._values
        elif self._apply_positions == "all":
            modified[..., self._indices] = self._values
        else:
            raise ConfigError(f"unsupported intervention position mode '{self._apply_positions}'")
        edited = self._autoencoder.decode(modified)
        if self._preserve_reconstruction_error:
            edited = edited + (working - reconstruction)
        return edited.to(original_dtype)


class HeadingStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, heading_markers: Sequence[str], prompt_length: int, lookback_characters: int) -> None:
        self._tokenizer = tokenizer
        self._markers = [marker.lower() for marker in heading_markers]
        self._prompt_length = prompt_length
        self._lookback_characters = lookback_characters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated = input_ids[0, self._prompt_length :]
        if generated.shape[0] == 0:
            return False
        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        window = text[-self._lookback_characters :].lower()
        return any(marker in window for marker in self._markers)


@dataclass
class InterventionState:
    features: np.ndarray
    reasoning_embedding: np.ndarray
    suspicion: np.ndarray
    matched_counts: dict[str, int]
    planning_segment: str
    trace_text: str
    selected_cwe: str | None = None

    def policy_input(self, feature_indices: Sequence[int], include_embedding: bool, include_suspicion: bool) -> np.ndarray:
        components: list[np.ndarray] = [self.features[list(feature_indices)] if feature_indices else np.zeros(0, dtype=np.float32)]
        if include_embedding:
            components.append(self.reasoning_embedding)
        if include_suspicion:
            components.append(self.suspicion)
        return np.concatenate(components).astype(np.float32)


class StateBuilder:
    def __init__(
        self,
        config: Config,
        model: LanguageModel,
        autoencoder: SparseAutoencoder,
        vocabulary: SecurityConceptVocabulary,
        categories: Sequence[str],
    ) -> None:
        section = config.require_section("intervention.state")
        self._model = model
        self._autoencoder = autoencoder
        self._vocabulary = vocabulary
        self._categories = list(categories)
        self._feature_position = section.require_str("feature_position").strip().lower()
        self._minimum_phrase_matches = section.require_int("minimum_phrase_matches")
        self._include_embedding = section.require_bool("include_reasoning_embedding")
        self._include_suspicion = section.require_bool("include_suspicion_vector")
        self._include_explicit_mentions = section.require_bool("include_explicit_cwe_mentions")
        self._explicit_mention_weight = section.require_int("explicit_mention_weight")
        self._device = torch.device(section.require_str("device"))

    @property
    def categories(self) -> list[str]:
        return list(self._categories)

    @property
    def include_embedding(self) -> bool:
        return self._include_embedding

    @property
    def include_suspicion(self) -> bool:
        return self._include_suspicion

    def state_dimension(self, feature_count: int) -> int:
        dimension = feature_count
        if self._include_embedding:
            dimension += self._model.hidden_size
        if self._include_suspicion:
            dimension += len(self._categories)
        return dimension

    def build(self, context_text: str, planning_segment: str, planning_span: tuple[int, int] | None) -> InterventionState:
        captured = self._model.capture_residual(context_text)
        hidden = captured.hidden_states.to(self._device)
        autoencoder = self._autoencoder.to(self._device)
        planning_positions = captured.span_indices(*planning_span) if planning_span else []
        with torch.no_grad():
            if self._feature_position == "last":
                features = autoencoder.encode(hidden[-1:, :]).squeeze(0)
            elif self._feature_position == "planning_mean" and planning_positions:
                selector = torch.tensor(planning_positions, dtype=torch.long, device=self._device)
                features = autoencoder.encode(hidden.index_select(0, selector)).mean(dim=0)
            elif self._feature_position == "sequence_mean":
                features = autoencoder.encode(hidden).mean(dim=0)
            else:
                features = autoencoder.encode(hidden[-1:, :]).squeeze(0)
        if planning_positions:
            selector = torch.tensor(planning_positions, dtype=torch.long, device=self._device)
            reasoning_embedding = hidden.index_select(0, selector).mean(dim=0)
        else:
            reasoning_embedding = hidden.mean(dim=0)
        matched = self._vocabulary.matched_categories(planning_segment, self._minimum_phrase_matches)
        if self._include_explicit_mentions:
            for match in CWE_IDENTIFIER_PATTERN.finditer(planning_segment or ""):
                identifier = f"CWE-{int(match.group(1))}"
                if identifier in self._categories:
                    matched[identifier] = matched.get(identifier, 0) + self._explicit_mention_weight
        suspicion = np.zeros(len(self._categories), dtype=np.float32)
        for position, category in enumerate(self._categories):
            if category in matched:
                suspicion[position] = 1.0
        return InterventionState(
            features=features.detach().cpu().numpy().astype(np.float32),
            reasoning_embedding=reasoning_embedding.detach().cpu().numpy().astype(np.float32),
            suspicion=suspicion,
            matched_counts=matched,
            planning_segment=planning_segment,
            trace_text=context_text,
        )


def select_category(
    state: InterventionState,
    available: Sequence[str],
    fallback: str | None,
) -> str | None:
    candidates = [category for category in available if state.matched_counts.get(category, 0) > 0]
    if not candidates:
        return fallback
    return sorted(candidates, key=lambda item: (-state.matched_counts[item], int(item.split("-")[1])))[0]


class ReplacementStrategy(ABC):
    name = "strategy"

    @abstractmethod
    def values(self, state: InterventionState, feature_set: FeatureSet) -> dict[int, float]:
        raise NotImplementedError


class ZeroAblationStrategy(ReplacementStrategy):
    name = "zero"

    def values(self, state: InterventionState, feature_set: FeatureSet) -> dict[int, float]:
        return {int(index): 0.0 for index in feature_set.feature_indices}


class SafeMeanStrategy(ReplacementStrategy):
    name = "safe_mean"

    def values(self, state: InterventionState, feature_set: FeatureSet) -> dict[int, float]:
        return {
            int(index): float(feature_set.safe_statistics.get(int(index), {}).get("mean", 0.0))
            for index in feature_set.feature_indices
        }


class QuantileMappingStrategy(ReplacementStrategy):
    name = "quantile_mapping"

    def values(self, state: InterventionState, feature_set: FeatureSet) -> dict[int, float]:
        levels = np.asarray(feature_set.quantile_levels, dtype=np.float64)
        replacements: dict[int, float] = {}
        for index in feature_set.feature_indices:
            key = int(index)
            vulnerable = np.asarray(feature_set.vulnerable_quantiles.get(key, []), dtype=np.float64)
            safe = np.asarray(feature_set.safe_quantiles.get(key, []), dtype=np.float64)
            if vulnerable.size == 0 or safe.size == 0 or levels.size != vulnerable.size:
                replacements[key] = float(feature_set.safe_statistics.get(key, {}).get("mean", 0.0))
                continue
            current = float(state.features[key])
            level = float(np.interp(current, vulnerable, levels))
            replacements[key] = float(np.interp(level, levels, safe))
        return replacements


class PolicyStrategy(ReplacementStrategy):
    name = "policy"

    def __init__(
        self,
        policies: Mapping[str, object],
        state_builder: StateBuilder,
        deterministic: bool,
        device: torch.device,
    ) -> None:
        self._policies = dict(policies)
        self._state_builder = state_builder
        self._deterministic = deterministic
        self._device = device
        self.last_action: np.ndarray | None = None

    def values(self, state: InterventionState, feature_set: FeatureSet) -> dict[int, float]:
        policy = self._policies.get(feature_set.cwe)
        if policy is None:
            raise ConfigError(f"no trained policy available for {feature_set.cwe}")
        observation = state.policy_input(
            feature_set.feature_indices,
            self._state_builder.include_embedding,
            self._state_builder.include_suspicion,
        )
        tensor = torch.tensor(observation, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            output = policy.act(tensor, deterministic=self._deterministic)
        action = output.action.squeeze(0).detach().cpu().numpy()
        self.last_action = action
        return {int(index): float(action[position]) for position, index in enumerate(feature_set.feature_indices)}


@dataclass
class GenerationOutcome:
    trace_text: str
    completion_text: str
    full_text: str
    chain: ReasoningChain
    state: InterventionState | None
    applied_cwe: str | None
    replacement_values: dict[int, float] = field(default_factory=dict)
    modified_features: np.ndarray | None = None
    latency_seconds: float = 0.0


class ReasonSecGenerator:
    def __init__(
        self,
        config: Config,
        model: LanguageModel,
        autoencoder: SparseAutoencoder | None,
        feature_sets: Mapping[str, FeatureSet],
        state_builder: StateBuilder | None,
        strategy: ReplacementStrategy | None,
        parser: ChainParser | None = None,
        templates: TemplateLibrary | None = None,
        elicitation_template=None,
    ) -> None:
        section = config.require_section("intervention.runtime")
        self._config = config
        self._model = model
        self._autoencoder = autoencoder
        self._feature_sets = dict(feature_sets)
        self._state_builder = state_builder
        self._strategy = strategy
        self._parser = parser or ChainParser(config)
        self._templates = templates or TemplateLibrary(config)
        self._elicitation_template = elicitation_template or self._templates.elicitation
        self._trace_max_new_tokens = section.require_int("trace_max_new_tokens")
        self._code_max_new_tokens = section.require_int("code_max_new_tokens")
        self._stop_markers = [str(marker) for marker in section.require_list("trace_stop_markers")]
        self._stop_lookback_characters = section.require_int("stop_lookback_characters")
        self._fallback_cwe = section.optional("fallback_cwe")
        self._recalibrator = (
            FeatureRecalibrator(
                autoencoder,
                section.require_bool("preserve_reconstruction_error"),
                section.require_str("apply_positions"),
            )
            if autoencoder is not None
            else None
        )

    @property
    def parser(self) -> ChainParser:
        return self._parser

    @property
    def feature_sets(self) -> dict[str, FeatureSet]:
        return dict(self._feature_sets)

    @property
    def strategy(self) -> ReplacementStrategy | None:
        return self._strategy

    @property
    def fallback_cwe(self) -> str | None:
        return self._fallback_cwe

    def select_and_compute(self, state: InterventionState) -> tuple[str | None, dict[int, float]]:
        applied_cwe = select_category(state, list(self._feature_sets), self._fallback_cwe)
        state.selected_cwe = applied_cwe
        if applied_cwe is None or self._strategy is None or applied_cwe not in self._feature_sets:
            return applied_cwe, {}
        return applied_cwe, self._strategy.values(state, self._feature_sets[applied_cwe])

    def instruction_for(self, prompt: BenchmarkPrompt) -> str:
        values = {"prompt": prompt.prompt, "language": prompt.language}
        return self._elicitation_template.render(
            **{key: values[key] for key in self._elicitation_template.placeholders}
        )

    def generate_trace(self, instruction: str) -> str:
        formatted = self._model.format_prompt(instruction)
        encoded = self._model.tokenizer(formatted, return_tensors="pt", add_special_tokens=False)
        prompt_length = int(encoded["input_ids"].shape[1])
        criteria = StoppingCriteriaList(
            [
                HeadingStoppingCriteria(
                    self._model.tokenizer, self._stop_markers, prompt_length, self._stop_lookback_characters
                )
            ]
        )
        return self._model.generate(
            formatted,
            overrides={"max_new_tokens": self._trace_max_new_tokens, "stopping_criteria": criteria},
            preformatted=True,
        )

    def build_state(self, instruction: str, trace_chain: ReasoningChain, trace_text: str) -> InterventionState:
        if self._state_builder is None:
            raise ConfigError("state construction requires a configured state builder")
        context = self._model.format_prompt(instruction) + trace_text
        planning_span = trace_chain.planning_char_span
        offset = len(context) - len(trace_text)
        shifted_span = (planning_span[0] + offset, planning_span[1] + offset) if planning_span is not None else None
        return self._state_builder.build(context, trace_chain.planning_segment, shifted_span)

    def apply_replacements(self, replacements: Mapping[int, float]) -> None:
        if self._recalibrator is None:
            return
        if replacements:
            self._recalibrator.set_targets(
                list(replacements.keys()), list(replacements.values()), self._model.device
            )
        else:
            self._recalibrator.clear()

    def clear_replacements(self) -> None:
        if self._recalibrator is not None:
            self._recalibrator.clear()

    def complete(self, instruction: str, trace_text: str) -> str:
        return self._generate_completion(instruction, trace_text)

    def _generate_completion(self, instruction: str, trace_text: str) -> str:
        context = self._model.format_prompt(instruction) + trace_text
        if self._recalibrator is not None and self._recalibrator.active:
            with self._model.residual_editor(self._recalibrator):
                return self._model.generate(
                    context, overrides={"max_new_tokens": self._code_max_new_tokens}, preformatted=True
                )
        return self._model.generate(
            context, overrides={"max_new_tokens": self._code_max_new_tokens}, preformatted=True
        )

    def generate(self, prompt: BenchmarkPrompt) -> GenerationOutcome:
        started = time.perf_counter()
        instruction = self.instruction_for(prompt)
        trace_text = self.generate_trace(instruction)
        trace_chain = self._parser.parse(trace_text)
        state: InterventionState | None = None
        applied_cwe: str | None = None
        replacements: dict[int, float] = {}
        modified_features: np.ndarray | None = None

        if self._strategy is not None and self._state_builder is not None and self._recalibrator is not None:
            state = self.build_state(instruction, trace_chain, trace_text)
            applied_cwe = select_category(state, list(self._feature_sets), self._fallback_cwe)
            state.selected_cwe = applied_cwe
            if applied_cwe is not None and applied_cwe in self._feature_sets:
                feature_set = self._feature_sets[applied_cwe]
                replacements = self._strategy.values(state, feature_set)
                self._recalibrator.set_targets(
                    list(replacements.keys()), list(replacements.values()), self._model.device
                )
                modified_features = state.features.copy()
                for index, value in replacements.items():
                    modified_features[index] = value
            else:
                self._recalibrator.clear()
        elif self._recalibrator is not None:
            self._recalibrator.clear()

        completion = self._generate_completion(instruction, trace_text)
        if self._recalibrator is not None:
            self._recalibrator.clear()
        full_text = trace_text + completion
        chain = self._parser.parse(full_text)
        return GenerationOutcome(
            trace_text=trace_text,
            completion_text=completion,
            full_text=full_text,
            chain=chain,
            state=state,
            applied_cwe=applied_cwe,
            replacement_values=replacements,
            modified_features=modified_features,
            latency_seconds=time.perf_counter() - started,
        )
