from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from reasonsec.config import Config, ConfigError
from reasonsec.sae.model import SparseAutoencoder
from reasonsec.sae.store import ActivationStore, SampleIndex
from reasonsec.utils.io import ensure_directory
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class FeatureMatrices:
    identifiers: list[str]
    labels: list[str | None]
    splits: list[str]
    planning: np.ndarray
    distributional: np.ndarray

    @property
    def feature_dimension(self) -> int:
        return int(self.planning.shape[1])

    def indices_for(self, label: str | None, split: str | None = None) -> list[int]:
        selected: list[int] = []
        for index, item in enumerate(self.labels):
            if item != label:
                continue
            if split is not None and self.splits[index] != split:
                continue
            selected.append(index)
        return selected

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        ensure_directory(target.parent)
        np.savez_compressed(
            target,
            identifiers=np.array(self.identifiers, dtype=object),
            labels=np.array([label if label is not None else "" for label in self.labels], dtype=object),
            splits=np.array(self.splits, dtype=object),
            planning=self.planning,
            distributional=self.distributional,
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> "FeatureMatrices":
        payload = np.load(Path(path).expanduser(), allow_pickle=True)
        labels = [str(item) if str(item) else None for item in payload["labels"].tolist()]
        return cls(
            identifiers=[str(item) for item in payload["identifiers"].tolist()],
            labels=labels,
            splits=[str(item) for item in payload["splits"].tolist()],
            planning=payload["planning"],
            distributional=payload["distributional"],
        )


def _span_bounds(sample: SampleIndex, span: str) -> tuple[int, int]:
    if span == "planning":
        return (sample.planning_start, sample.planning_end) if sample.has_planning_span else (sample.start, sample.end)
    if span == "code":
        return (sample.code_start, sample.code_end) if sample.has_code_span else (sample.start, sample.end)
    if span == "all":
        return sample.start, sample.end
    raise ConfigError(f"unsupported activation span '{span}'")


@torch.no_grad()
def compute_feature_matrices(
    config: Config,
    store: ActivationStore,
    autoencoder: SparseAutoencoder,
) -> FeatureMatrices:
    section = config.require_section("rafs.aggregation")
    planning_span = section.require_str("alignment_span")
    distributional_span = section.require_str("distributional_span")
    batch_size = section.require_int("encode_batch_size")
    device = torch.device(section.require_str("device"))
    reduction = section.require_str("reduction").strip().lower()
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    memmap = store.open_memmap()
    feature_dimension = autoencoder.feature_dimension
    planning_matrix = np.zeros((len(store.samples), feature_dimension), dtype=np.float32)
    distributional_matrix = np.zeros((len(store.samples), feature_dimension), dtype=np.float32)

    for row, sample in enumerate(tqdm(store.samples, desc="encoding sample activations")):
        for target_matrix, span in ((planning_matrix, planning_span), (distributional_matrix, distributional_span)):
            start, end = _span_bounds(sample, span)
            if end <= start:
                continue
            accumulator = torch.zeros(feature_dimension, dtype=torch.float32, device=device)
            maximum = torch.zeros(feature_dimension, dtype=torch.float32, device=device)
            token_total = 0
            for chunk_start in range(start, end, batch_size):
                chunk_end = min(chunk_start + batch_size, end)
                chunk = torch.tensor(
                    np.asarray(memmap[chunk_start:chunk_end], dtype=np.float32), device=device
                )
                features = autoencoder.encode(chunk)
                accumulator += features.sum(dim=0)
                maximum = torch.maximum(maximum, features.max(dim=0).values)
                token_total += features.shape[0]
            if token_total == 0:
                continue
            if reduction == "mean":
                target_matrix[row] = (accumulator / token_total).cpu().numpy()
            elif reduction == "max":
                target_matrix[row] = maximum.cpu().numpy()
            else:
                raise ConfigError(f"unsupported activation reduction '{reduction}'")

    matrices = FeatureMatrices(
        identifiers=[sample.identifier for sample in store.samples],
        labels=[sample.label for sample in store.samples],
        splits=[sample.split for sample in store.samples],
        planning=planning_matrix,
        distributional=distributional_matrix,
    )
    LOGGER.info(
        "computed feature matrices for %d samples across %d features", len(store.samples), feature_dimension
    )
    return matrices


@torch.no_grad()
def top_activating_positions(
    store: ActivationStore,
    autoencoder: SparseAutoencoder,
    feature_indices: Sequence[int],
    batch_size: int,
    top_n: int,
    device: torch.device,
) -> dict[int, list[tuple[int, float]]]:
    if not feature_indices:
        return {}
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    selector = torch.tensor(list(feature_indices), dtype=torch.long, device=device)
    memmap = store.open_memmap()
    best_values = torch.full((len(feature_indices), top_n), float("-inf"), device=device)
    best_positions = torch.full((len(feature_indices), top_n), -1, dtype=torch.long, device=device)
    for start in range(0, store.token_count, batch_size):
        end = min(start + batch_size, store.token_count)
        chunk = torch.tensor(np.asarray(memmap[start:end], dtype=np.float32), device=device)
        features = autoencoder.encode(chunk).index_select(1, selector).t()
        positions = torch.arange(start, end, device=device).unsqueeze(0).expand_as(features)
        merged_values = torch.cat([best_values, features], dim=1)
        merged_positions = torch.cat([best_positions, positions], dim=1)
        ordered = merged_values.topk(k=min(top_n, merged_values.shape[1]), dim=1)
        best_values = ordered.values
        best_positions = merged_positions.gather(1, ordered.indices)
    result: dict[int, list[tuple[int, float]]] = {}
    for row, feature_index in enumerate(feature_indices):
        entries: list[tuple[int, float]] = []
        for column in range(best_values.shape[1]):
            position = int(best_positions[row, column].item())
            value = float(best_values[row, column].item())
            if position >= 0 and value > 0.0:
                entries.append((position, value))
        result[int(feature_index)] = entries
    return result
