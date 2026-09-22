from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from reasonsec.config import Config, ConfigError
from reasonsec.utils.io import ensure_directory

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
}


def build_mlp(input_dimension: int, hidden_sizes: Sequence[int], output_dimension: int, activation: str) -> nn.Sequential:
    key = str(activation).strip().lower()
    if key not in _ACTIVATIONS:
        raise ConfigError(f"unsupported activation '{activation}'; available options are {sorted(_ACTIVATIONS)}")
    layers: list[nn.Module] = []
    previous = input_dimension
    for size in hidden_sizes:
        layers.append(nn.Linear(previous, int(size)))
        layers.append(_ACTIVATIONS[key]())
        previous = int(size)
    layers.append(nn.Linear(previous, output_dimension))
    return nn.Sequential(*layers)


@dataclass
class PolicyOutput:
    action: torch.Tensor
    log_probability: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    pre_squash: torch.Tensor


class RecalibrationPolicy(nn.Module):
    def __init__(
        self,
        state_dimension: int,
        action_dimension: int,
        hidden_sizes: Sequence[int],
        activation: str,
        log_standard_deviation_initial: float,
        log_standard_deviation_bounds: Sequence[float],
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> None:
        super().__init__()
        self.state_dimension = int(state_dimension)
        self.action_dimension = int(action_dimension)
        self.actor = build_mlp(state_dimension, hidden_sizes, action_dimension * 2, activation)
        self.critic = build_mlp(state_dimension, hidden_sizes, 1, activation)
        self.register_buffer("lower_bounds", lower_bounds.to(torch.float32))
        self.register_buffer("upper_bounds", upper_bounds.to(torch.float32))
        self._log_standard_deviation_initial = float(log_standard_deviation_initial)
        self._log_standard_deviation_bounds = (
            float(log_standard_deviation_bounds[0]),
            float(log_standard_deviation_bounds[1]),
        )
        with torch.no_grad():
            final_layer = self.actor[-1]
            final_layer.bias[action_dimension:] = self._log_standard_deviation_initial

    @classmethod
    def from_config(
        cls,
        config: Config,
        state_dimension: int,
        lower_bounds: torch.Tensor,
        upper_bounds: torch.Tensor,
    ) -> "RecalibrationPolicy":
        section = config.require_section("intervention.policy")
        return cls(
            state_dimension=state_dimension,
            action_dimension=int(lower_bounds.shape[0]),
            hidden_sizes=[int(size) for size in section.require_list("hidden_sizes")],
            activation=section.require_str("activation"),
            log_standard_deviation_initial=section.require_float("log_standard_deviation_initial"),
            log_standard_deviation_bounds=[
                float(value) for value in section.require_list("log_standard_deviation_bounds")
            ],
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    def distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        outputs = self.actor(state)
        mean, log_standard_deviation = outputs.chunk(2, dim=-1)
        log_standard_deviation = log_standard_deviation.clamp(*self._log_standard_deviation_bounds)
        return torch.distributions.Normal(mean, log_standard_deviation.exp())

    def squash(self, pre_squash: torch.Tensor) -> torch.Tensor:
        scaled = torch.tanh(pre_squash)
        return self.lower_bounds + (self.upper_bounds - self.lower_bounds) * (scaled + 1.0) / 2.0

    def unsquash(self, action: torch.Tensor) -> torch.Tensor:
        span = (self.upper_bounds - self.lower_bounds).clamp(min=1e-8)
        normalised = ((action - self.lower_bounds) / span) * 2.0 - 1.0
        clamped = normalised.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        return torch.atanh(clamped)

    def log_probability(self, distribution: torch.distributions.Normal, pre_squash: torch.Tensor) -> torch.Tensor:
        base = distribution.log_prob(pre_squash).sum(dim=-1)
        span = (self.upper_bounds - self.lower_bounds).clamp(min=1e-8)
        jacobian = torch.log(1.0 - torch.tanh(pre_squash).pow(2) + 1e-6) + torch.log(span / 2.0)
        return base - jacobian.sum(dim=-1)

    def act(self, state: torch.Tensor, deterministic: bool = False) -> PolicyOutput:
        distribution = self.distribution(state)
        pre_squash = distribution.mean if deterministic else distribution.rsample()
        action = self.squash(pre_squash)
        return PolicyOutput(
            action=action,
            log_probability=self.log_probability(distribution, pre_squash),
            value=self.critic(state).squeeze(-1),
            entropy=distribution.entropy().sum(dim=-1),
            pre_squash=pre_squash,
        )

    def evaluate(self, state: torch.Tensor, pre_squash: torch.Tensor) -> PolicyOutput:
        distribution = self.distribution(state)
        return PolicyOutput(
            action=self.squash(pre_squash),
            log_probability=self.log_probability(distribution, pre_squash),
            value=self.critic(state).squeeze(-1),
            entropy=distribution.entropy().sum(dim=-1),
            pre_squash=pre_squash,
        )

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        target = Path(path).expanduser()
        ensure_directory(target.parent)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "state_dimension": self.state_dimension,
                "action_dimension": self.action_dimension,
                "hidden_sizes": [layer.out_features for layer in self.actor if isinstance(layer, nn.Linear)][:-1],
                "activation": type(self.actor[1]).__name__.lower(),
                "log_standard_deviation_initial": self._log_standard_deviation_initial,
                "log_standard_deviation_bounds": list(self._log_standard_deviation_bounds),
                "lower_bounds": self.lower_bounds.cpu(),
                "upper_bounds": self.upper_bounds.cpu(),
                "metadata": metadata or {},
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> tuple["RecalibrationPolicy", dict[str, Any]]:
        payload = torch.load(Path(path).expanduser(), map_location=map_location, weights_only=False)
        module = cls(
            state_dimension=int(payload["state_dimension"]),
            action_dimension=int(payload["action_dimension"]),
            hidden_sizes=[int(size) for size in payload["hidden_sizes"]],
            activation=str(payload["activation"]),
            log_standard_deviation_initial=float(payload["log_standard_deviation_initial"]),
            log_standard_deviation_bounds=[float(value) for value in payload["log_standard_deviation_bounds"]],
            lower_bounds=payload["lower_bounds"],
            upper_bounds=payload["upper_bounds"],
        )
        module.load_state_dict(payload["state_dict"])
        module.eval()
        return module, dict(payload.get("metadata", {}))
