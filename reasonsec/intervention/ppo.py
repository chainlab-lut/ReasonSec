from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from reasonsec.config import Config
from reasonsec.intervention.environment import EpisodeContext, RecalibrationEnvironment
from reasonsec.intervention.policy import RecalibrationPolicy
from reasonsec.intervention.reward import RewardTerms, aggregate_reward_terms
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    pre_squash_actions: torch.Tensor
    log_probabilities: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    terms: list[RewardTerms] = field(default_factory=list)


@dataclass
class PpoTrainingStatistics:
    cwe: str
    epochs: list[dict[str, float]] = field(default_factory=list)
    final_reward_mean: float = 0.0
    final_reward_standard_deviation: float = 0.0
    final_terms: dict[str, float] = field(default_factory=dict)
    episode_count: int = 0
    policy_parameter_count: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "cwe": self.cwe,
            "epochs": self.epochs,
            "final_reward_mean": self.final_reward_mean,
            "final_reward_standard_deviation": self.final_reward_standard_deviation,
            "final_terms": self.final_terms,
            "episode_count": self.episode_count,
            "policy_parameter_count": self.policy_parameter_count,
        }


class PpoTrainer:
    def __init__(self, config: Config, policy: RecalibrationPolicy, environment: RecalibrationEnvironment) -> None:
        section = config.require_section("intervention.ppo")
        self._policy = policy
        self._environment = environment
        self._epochs = section.require_int("epochs")
        self._rollout_size = section.require_int("rollout_size")
        self._minibatch_size = section.require_int("minibatch_size")
        self._update_iterations = section.require_int("update_iterations")
        self._clip_epsilon = section.require_float("clip_epsilon")
        self._value_coefficient = section.require_float("value_coefficient")
        self._entropy_coefficient = section.require_float("entropy_coefficient")
        self._max_gradient_norm = section.require_float("max_gradient_norm")
        self._normalise_advantages = section.require_bool("normalise_advantages")
        self._device = torch.device(section.require_str("device"))
        self._policy.to(self._device)
        self._optimiser = torch.optim.Adam(self._policy.parameters(), lr=section.require_float("learning_rate"))
        self._include_embedding = section.require_bool("observation_includes_embedding")
        self._include_suspicion = section.require_bool("observation_includes_suspicion")

    def _collect(self, contexts: Sequence[EpisodeContext], generator: random.Random) -> RolloutBatch:
        feature_indices = self._environment.feature_set.feature_indices
        observations: list[np.ndarray] = []
        pre_squash: list[torch.Tensor] = []
        log_probabilities: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[float] = []
        terms: list[RewardTerms] = []
        sampled = [contexts[generator.randrange(len(contexts))] for _ in range(min(self._rollout_size, len(contexts) * 4))]
        for context in tqdm(sampled, desc=f"rollout {self._environment.feature_set.cwe}", leave=False):
            observation = context.observation(feature_indices, self._include_embedding, self._include_suspicion)
            tensor = torch.tensor(observation, dtype=torch.float32, device=self._device).unsqueeze(0)
            with torch.no_grad():
                output = self._policy.act(tensor, deterministic=False)
            action = output.action.squeeze(0).cpu().numpy()
            replacements = {int(index): float(action[position]) for position, index in enumerate(feature_indices)}
            outcome = self._environment.step(context, replacements)
            observations.append(observation)
            pre_squash.append(output.pre_squash.squeeze(0).cpu())
            log_probabilities.append(output.log_probability.squeeze(0).cpu())
            values.append(output.value.squeeze(0).cpu())
            rewards.append(outcome.reward.effective)
            terms.append(outcome.reward)
        return RolloutBatch(
            observations=torch.tensor(np.stack(observations), dtype=torch.float32),
            pre_squash_actions=torch.stack(pre_squash),
            log_probabilities=torch.stack(log_probabilities),
            values=torch.stack(values),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terms=terms,
        )

    def _update(self, batch: RolloutBatch) -> dict[str, float]:
        observations = batch.observations.to(self._device)
        pre_squash = batch.pre_squash_actions.to(self._device)
        old_log_probabilities = batch.log_probabilities.to(self._device)
        rewards = batch.rewards.to(self._device)
        advantages = rewards - batch.values.to(self._device)
        if self._normalise_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        updates = 0
        for _iteration in range(self._update_iterations):
            permutation = torch.randperm(observations.shape[0], device=self._device)
            for start in range(0, observations.shape[0], self._minibatch_size):
                indices = permutation[start : start + self._minibatch_size]
                output = self._policy.evaluate(observations[indices], pre_squash[indices])
                ratio = torch.exp(output.log_probability - old_log_probabilities[indices])
                surrogate = ratio * advantages[indices]
                clipped = torch.clamp(ratio, 1.0 - self._clip_epsilon, 1.0 + self._clip_epsilon) * advantages[indices]
                policy_loss = -torch.min(surrogate, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(output.value, rewards[indices])
                entropy = output.entropy.mean()
                loss = policy_loss + self._value_coefficient * value_loss - self._entropy_coefficient * entropy
                self._optimiser.zero_grad(set_to_none=True)
                loss.backward()
                if self._max_gradient_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self._max_gradient_norm)
                self._optimiser.step()
                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                updates += 1
        divisor = max(updates, 1)
        return {
            "policy_loss": total_policy_loss / divisor,
            "value_loss": total_value_loss / divisor,
            "entropy": total_entropy / divisor,
        }

    def train(self, seed: int, checkpoint_path: str | Path | None = None) -> PpoTrainingStatistics:
        contexts = self._environment.contexts
        statistics = PpoTrainingStatistics(
            cwe=self._environment.feature_set.cwe,
            episode_count=len(contexts),
            policy_parameter_count=self._policy.parameter_count(),
        )
        if not contexts:
            LOGGER.warning("no episodes available for %s; policy left untrained", self._environment.feature_set.cwe)
            return statistics
        generator = random.Random(seed)
        last_batch: RolloutBatch | None = None
        for epoch in range(self._epochs):
            batch = self._collect(contexts, generator)
            losses = self._update(batch)
            epoch_record = {
                "epoch": epoch + 1,
                "reward_mean": float(batch.rewards.mean().item()),
                "reward_standard_deviation": float(batch.rewards.std(unbiased=False).item()),
                **losses,
                **aggregate_reward_terms(batch.terms),
            }
            statistics.epochs.append(epoch_record)
            LOGGER.info(
                "%s ppo epoch %d reward %.4f policy loss %.4f",
                self._environment.feature_set.cwe,
                epoch + 1,
                epoch_record["reward_mean"],
                losses["policy_loss"],
            )
            last_batch = batch
        if last_batch is not None:
            statistics.final_reward_mean = float(last_batch.rewards.mean().item())
            statistics.final_reward_standard_deviation = float(last_batch.rewards.std(unbiased=False).item())
            statistics.final_terms = aggregate_reward_terms(last_batch.terms)
        if checkpoint_path is not None:
            self._policy.save(checkpoint_path, metadata={"statistics": statistics.as_dict()})
        return statistics
