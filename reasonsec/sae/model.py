from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from reasonsec.config import Config
from reasonsec.utils.io import ensure_directory


class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        expansion_factor: int,
        subtract_decoder_bias: bool,
        normalise_decoder: bool,
    ) -> None:
        super().__init__()
        self.input_dimension = int(input_dimension)
        self.expansion_factor = int(expansion_factor)
        self.feature_dimension = self.input_dimension * self.expansion_factor
        self.subtract_decoder_bias = bool(subtract_decoder_bias)
        self.normalise_decoder = bool(normalise_decoder)
        self.encoder_weight = nn.Parameter(torch.empty(self.feature_dimension, self.input_dimension))
        self.encoder_bias = nn.Parameter(torch.zeros(self.feature_dimension))
        self.decoder_weight = nn.Parameter(torch.empty(self.input_dimension, self.feature_dimension))
        self.decoder_bias = nn.Parameter(torch.zeros(self.input_dimension))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.decoder_weight, a=5**0.5)
        with torch.no_grad():
            self.decoder_weight.data = nn.functional.normalize(self.decoder_weight.data, p=2, dim=0)
            self.encoder_weight.data = self.decoder_weight.data.t().clone()
            self.encoder_bias.data.zero_()
            self.decoder_bias.data.zero_()

    @classmethod
    def from_config(cls, config: Config, input_dimension: int) -> "SparseAutoencoder":
        section = config.require_section("sae")
        return cls(
            input_dimension=input_dimension,
            expansion_factor=section.require_int("expansion_factor"),
            subtract_decoder_bias=section.require_bool("subtract_decoder_bias"),
            normalise_decoder=section.require_bool("normalise_decoder"),
        )

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        centred = activations - self.decoder_bias if self.subtract_decoder_bias else activations
        return torch.relu(centred @ self.encoder_weight.t() + self.encoder_bias)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.decoder_weight.t() + self.decoder_bias

    def forward(self, activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(activations)
        reconstruction = self.decode(features)
        return features, reconstruction

    def reconstruction_error(self, activations: torch.Tensor) -> torch.Tensor:
        features = self.encode(activations)
        return activations - self.decode(features)

    def loss(self, activations: torch.Tensor, l1_coefficient: float) -> dict[str, torch.Tensor]:
        features, reconstruction = self.forward(activations)
        reconstruction_loss = (activations - reconstruction).pow(2).sum(dim=-1).mean()
        sparsity_loss = features.abs().sum(dim=-1).mean()
        return {
            "loss": reconstruction_loss + l1_coefficient * sparsity_loss,
            "reconstruction": reconstruction_loss,
            "sparsity": sparsity_loss,
            "l0": (features > 0).float().sum(dim=-1).mean(),
            "features": features,
        }

    @torch.no_grad()
    def constrain_decoder(self) -> None:
        if self.normalise_decoder:
            self.decoder_weight.data = nn.functional.normalize(self.decoder_weight.data, p=2, dim=0)

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Path:
        target = Path(path).expanduser()
        ensure_directory(target.parent)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "input_dimension": self.input_dimension,
                "expansion_factor": self.expansion_factor,
                "subtract_decoder_bias": self.subtract_decoder_bias,
                "normalise_decoder": self.normalise_decoder,
                "metadata": metadata or {},
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> tuple["SparseAutoencoder", dict[str, Any]]:
        payload = torch.load(Path(path).expanduser(), map_location=map_location, weights_only=False)
        module = cls(
            input_dimension=int(payload["input_dimension"]),
            expansion_factor=int(payload["expansion_factor"]),
            subtract_decoder_bias=bool(payload["subtract_decoder_bias"]),
            normalise_decoder=bool(payload["normalise_decoder"]),
        )
        module.load_state_dict(payload["state_dict"])
        module.eval()
        return module, dict(payload.get("metadata", {}))
