from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from reasonsec.config import Config
from reasonsec.sae.model import SparseAutoencoder
from reasonsec.sae.store import ActivationStore
from reasonsec.utils.io import write_json
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class SaeTrainingStatistics:
    epochs: list[dict[str, float]] = field(default_factory=list)
    mean_l0: float = 0.0
    explained_variance: float = 0.0
    dead_feature_fraction: float = 0.0
    held_out_tokens: int = 0
    training_tokens: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "mean_l0": self.mean_l0,
            "explained_variance": self.explained_variance,
            "dead_feature_fraction": self.dead_feature_fraction,
            "held_out_tokens": self.held_out_tokens,
            "training_tokens": self.training_tokens,
        }


def _split_indices(token_count: int, held_out_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(token_count)
    held_out_size = int(round(token_count * held_out_fraction))
    return permutation[held_out_size:], permutation[:held_out_size]


@torch.no_grad()
def evaluate_sae(
    autoencoder: SparseAutoencoder,
    memmap: np.memmap,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float, float]:
    total_l0 = 0.0
    residual_sum = 0.0
    variance_sum = 0.0
    token_total = 0
    active_features = torch.zeros(autoencoder.feature_dimension, dtype=torch.bool, device=device)
    mean_vector = torch.tensor(
        np.asarray(memmap[np.sort(indices)], dtype=np.float32).mean(axis=0), device=device
    )
    for start in range(0, len(indices), batch_size):
        batch_indices = np.sort(indices[start : start + batch_size])
        batch = torch.tensor(np.asarray(memmap[batch_indices], dtype=np.float32), device=device)
        features, reconstruction = autoencoder(batch)
        residual_sum += float((batch - reconstruction).pow(2).sum().item())
        variance_sum += float((batch - mean_vector).pow(2).sum().item())
        total_l0 += float((features > 0).float().sum(dim=-1).sum().item())
        active_features |= (features > 0).any(dim=0)
        token_total += batch.shape[0]
    if token_total == 0:
        return 0.0, 0.0, 1.0
    explained_variance = 1.0 - residual_sum / variance_sum if variance_sum > 0 else 0.0
    dead_fraction = 1.0 - float(active_features.sum().item()) / autoencoder.feature_dimension
    return total_l0 / token_total, explained_variance, dead_fraction


def train_sparse_autoencoder(
    config: Config,
    store: ActivationStore,
    output_path: str | Path,
    seed: int,
) -> tuple[SparseAutoencoder, SaeTrainingStatistics]:
    section = config.require_section("sae")
    epochs = section.require_int("epochs")
    batch_size = section.require_int("batch_size")
    learning_rate = section.require_float("learning_rate")
    l1_coefficient = section.require_float("l1_coefficient")
    held_out_fraction = section.require_float("held_out_fraction")
    device = torch.device(section.require_str("device"))
    gradient_clipping = section.require_float("gradient_clipping")

    autoencoder = SparseAutoencoder.from_config(config, store.hidden_size).to(device)
    optimiser = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    memmap = store.open_memmap()
    training_indices, held_out_indices = _split_indices(store.token_count, held_out_fraction, seed)
    statistics = SaeTrainingStatistics(
        held_out_tokens=int(len(held_out_indices)), training_tokens=int(len(training_indices))
    )
    generator = np.random.default_rng(seed)

    for epoch in range(epochs):
        autoencoder.train()
        order = training_indices.copy()
        generator.shuffle(order)
        epoch_loss = 0.0
        epoch_reconstruction = 0.0
        epoch_sparsity = 0.0
        epoch_l0 = 0.0
        batches = 0
        for start in tqdm(range(0, len(order), batch_size), desc=f"sae epoch {epoch + 1}/{epochs}"):
            batch_indices = np.sort(order[start : start + batch_size])
            batch = torch.tensor(np.asarray(memmap[batch_indices], dtype=np.float32), device=device)
            outputs = autoencoder.loss(batch, l1_coefficient)
            optimiser.zero_grad(set_to_none=True)
            outputs["loss"].backward()
            if gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), gradient_clipping)
            optimiser.step()
            autoencoder.constrain_decoder()
            epoch_loss += float(outputs["loss"].item())
            epoch_reconstruction += float(outputs["reconstruction"].item())
            epoch_sparsity += float(outputs["sparsity"].item())
            epoch_l0 += float(outputs["l0"].item())
            batches += 1
        divisor = max(batches, 1)
        statistics.epochs.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss / divisor,
                "reconstruction": epoch_reconstruction / divisor,
                "sparsity": epoch_sparsity / divisor,
                "l0": epoch_l0 / divisor,
            }
        )
        LOGGER.info("sae epoch %d loss %.4f l0 %.2f", epoch + 1, epoch_loss / divisor, epoch_l0 / divisor)

    autoencoder.eval()
    evaluation_indices = held_out_indices if len(held_out_indices) > 0 else training_indices
    mean_l0, explained_variance, dead_fraction = evaluate_sae(
        autoencoder, memmap, evaluation_indices, batch_size, device
    )
    statistics.mean_l0 = mean_l0
    statistics.explained_variance = explained_variance
    statistics.dead_feature_fraction = dead_fraction
    LOGGER.info(
        "sae evaluation: l0 %.2f explained variance %.4f dead features %.4f",
        mean_l0,
        explained_variance,
        dead_fraction,
    )

    target = Path(output_path).expanduser()
    autoencoder.save(target, metadata={"statistics": statistics.as_dict(), "layer": config.require_int("model.layer")})
    write_json(target.with_suffix(".statistics.json"), statistics.as_dict())
    return autoencoder, statistics
