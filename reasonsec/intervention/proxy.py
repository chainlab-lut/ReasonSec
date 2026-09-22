from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from reasonsec.config import Config
from reasonsec.intervention.policy import build_mlp
from reasonsec.utils.io import ensure_directory
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ProxyTrainingStatistics:
    epochs: list[dict[str, float]] = field(default_factory=list)
    validation_accuracy: float = 0.0
    positive_count: int = 0
    negative_count: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "validation_accuracy": self.validation_accuracy,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
        }


class ProxyVulnerabilityClassifier(nn.Module):
    def __init__(self, input_dimension: int, hidden_sizes: Sequence[int], activation: str, feature_indices: Sequence[int] | None) -> None:
        super().__init__()
        self.input_dimension = int(input_dimension)
        self.network = build_mlp(input_dimension, hidden_sizes, 1, activation)
        self.register_buffer(
            "feature_indices",
            torch.tensor(list(feature_indices), dtype=torch.long) if feature_indices else torch.empty(0, dtype=torch.long),
        )

    def select(self, features: torch.Tensor) -> torch.Tensor:
        if self.feature_indices.numel() == 0:
            return features
        return features.index_select(-1, self.feature_indices.to(features.device))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(self.select(features)).squeeze(-1)

    @torch.no_grad()
    def probability(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(features))

    def save(self, path: str | Path, metadata: dict | None = None) -> Path:
        target = Path(path).expanduser()
        ensure_directory(target.parent)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "input_dimension": self.input_dimension,
                "hidden_sizes": [layer.out_features for layer in self.network if isinstance(layer, nn.Linear)][:-1],
                "activation": type(self.network[1]).__name__.lower(),
                "feature_indices": self.feature_indices.cpu(),
                "metadata": metadata or {},
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "ProxyVulnerabilityClassifier":
        payload = torch.load(Path(path).expanduser(), map_location=map_location, weights_only=False)
        indices = payload["feature_indices"].tolist()
        module = cls(
            input_dimension=int(payload["input_dimension"]),
            hidden_sizes=[int(size) for size in payload["hidden_sizes"]],
            activation=str(payload["activation"]),
            feature_indices=indices if indices else None,
        )
        module.load_state_dict(payload["state_dict"])
        module.eval()
        return module


def train_proxy_classifier(
    config: Config,
    vulnerable_features: np.ndarray,
    safe_features: np.ndarray,
    feature_indices: Sequence[int] | None,
    seed: int,
) -> tuple[ProxyVulnerabilityClassifier, ProxyTrainingStatistics]:
    section = config.require_section("intervention.proxy_classifier")
    hidden_sizes = [int(size) for size in section.require_list("hidden_sizes")]
    activation = section.require_str("activation")
    epochs = section.require_int("epochs")
    batch_size = section.require_int("batch_size")
    learning_rate = section.require_float("learning_rate")
    weight_decay = section.require_float("weight_decay")
    validation_fraction = section.require_float("validation_fraction")
    device = torch.device(section.require_str("device"))
    restrict = section.require_bool("restrict_to_selected_features")

    selected = list(feature_indices) if (restrict and feature_indices) else None
    input_dimension = len(selected) if selected else int(vulnerable_features.shape[1])
    model = ProxyVulnerabilityClassifier(input_dimension, hidden_sizes, activation, selected).to(device)

    inputs = np.concatenate([vulnerable_features, safe_features], axis=0).astype(np.float32)
    targets = np.concatenate(
        [np.ones(len(vulnerable_features), dtype=np.float32), np.zeros(len(safe_features), dtype=np.float32)]
    )
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(inputs))
    inputs, targets = inputs[order], targets[order]
    validation_size = int(round(len(inputs) * validation_fraction))
    validation_inputs = torch.tensor(inputs[:validation_size], device=device)
    validation_targets = torch.tensor(targets[:validation_size], device=device)
    training_inputs = torch.tensor(inputs[validation_size:], device=device)
    training_targets = torch.tensor(targets[validation_size:], device=device)

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    statistics = ProxyTrainingStatistics(
        positive_count=int(len(vulnerable_features)), negative_count=int(len(safe_features))
    )

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(training_inputs.shape[0], device=device)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, training_inputs.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            logits = model(training_inputs[indices])
            loss = criterion(logits, training_targets[indices])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item())
            batches += 1
        statistics.epochs.append({"epoch": epoch + 1, "loss": epoch_loss / max(batches, 1)})

    model.eval()
    if validation_inputs.shape[0] > 0:
        with torch.no_grad():
            predictions = (torch.sigmoid(model(validation_inputs)) >= 0.5).float()
            statistics.validation_accuracy = float((predictions == validation_targets).float().mean().item())
    LOGGER.info(
        "proxy classifier trained on %d vulnerable and %d safe samples, validation accuracy %.4f",
        statistics.positive_count,
        statistics.negative_count,
        statistics.validation_accuracy,
    )
    return model, statistics
