from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from tqdm import tqdm

from reasonsec.config import Config, ConfigError
from reasonsec.models.language_model import LanguageModel
from reasonsec.types import CorpusSample
from reasonsec.utils.io import ensure_directory, read_json, write_json
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class SampleIndex:
    identifier: str
    start: int
    end: int
    planning_start: int
    planning_end: int
    code_start: int
    code_end: int
    label: str | None
    split: str
    language: str
    offsets: list[tuple[int, int]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "start": self.start,
            "end": self.end,
            "planning_start": self.planning_start,
            "planning_end": self.planning_end,
            "code_start": self.code_start,
            "code_end": self.code_end,
            "label": self.label,
            "split": self.split,
            "language": self.language,
            "offsets": [list(pair) for pair in self.offsets],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "SampleIndex":
        return cls(
            identifier=str(payload["identifier"]),
            start=int(payload["start"]),
            end=int(payload["end"]),
            planning_start=int(payload["planning_start"]),
            planning_end=int(payload["planning_end"]),
            code_start=int(payload["code_start"]),
            code_end=int(payload["code_end"]),
            label=payload.get("label"),
            split=str(payload.get("split", "")),
            language=str(payload.get("language", "")),
            offsets=[(int(pair[0]), int(pair[1])) for pair in payload.get("offsets", [])],
        )

    @property
    def has_planning_span(self) -> bool:
        return self.planning_end > self.planning_start

    @property
    def has_code_span(self) -> bool:
        return self.code_end > self.code_start


@dataclass
class ActivationStore:
    directory: Path
    hidden_size: int
    dtype: str
    samples: list[SampleIndex] = field(default_factory=list)
    token_count: int = 0

    @property
    def activation_path(self) -> Path:
        return self.directory / "activations.npy"

    @property
    def index_path(self) -> Path:
        return self.directory / "index.json"

    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    def open_memmap(self, mode: str = "r") -> np.memmap:
        return np.memmap(
            self.activation_path,
            dtype=self.numpy_dtype(),
            mode=mode,
            shape=(self.token_count, self.hidden_size),
        )

    def save_index(self) -> Path:
        return write_json(
            self.index_path,
            {
                "hidden_size": self.hidden_size,
                "dtype": self.dtype,
                "token_count": self.token_count,
                "samples": [sample.as_dict() for sample in self.samples],
            },
        )

    @classmethod
    def load(cls, directory: str | Path) -> "ActivationStore":
        resolved = Path(directory).expanduser()
        payload = read_json(resolved / "index.json")
        store = cls(
            directory=resolved,
            hidden_size=int(payload["hidden_size"]),
            dtype=str(payload["dtype"]),
            samples=[SampleIndex.from_dict(item) for item in payload["samples"]],
            token_count=int(payload["token_count"]),
        )
        if not store.activation_path.is_file():
            raise ConfigError(f"activation file missing at {store.activation_path}")
        return store

    def sample_by_identifier(self) -> dict[str, SampleIndex]:
        return {sample.identifier: sample for sample in self.samples}

    def iter_batches(self, batch_size: int, shuffle: bool, seed: int) -> Iterator[np.ndarray]:
        memmap = self.open_memmap()
        order = np.arange(self.token_count)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, self.token_count, batch_size):
            indices = np.sort(order[start : start + batch_size])
            yield np.asarray(memmap[indices], dtype=np.float32)


class ActivationCollector:
    def __init__(self, config: Config, model: LanguageModel) -> None:
        section = config.require_section("sae.activation_store")
        self._model = model
        self._directory = ensure_directory(section.require_path("directory", must_exist=False))
        self._dtype = section.require_str("dtype")
        self._retain = [str(item).lower() for item in section.require_list("retain_spans")]
        self._max_tokens_per_sample = section.require_int("max_tokens_per_sample")
        self._overwrite = section.require_bool("overwrite")

    def collect(self, samples: Sequence[CorpusSample]) -> ActivationStore:
        if self._directory.joinpath("index.json").is_file() and not self._overwrite:
            store = ActivationStore.load(self._directory)
            LOGGER.info("reusing existing activation store with %d tokens", store.token_count)
            return store

        buffers: list[np.ndarray] = []
        indices: list[SampleIndex] = []
        cursor = 0
        numpy_dtype = np.dtype(self._dtype)
        for sample in tqdm(samples, desc="collecting residual activations"):
            captured = self._model.capture_residual(sample.full_text)
            planning_positions = (
                captured.span_indices(*sample.chain.planning_char_span) if sample.chain.planning_char_span else []
            )
            code_positions = captured.span_indices(*sample.chain.code_char_span) if sample.chain.code_char_span else []
            selected = self._select_positions(captured.hidden_states.shape[0], planning_positions, code_positions)
            if not selected:
                continue
            position_map = {position: offset for offset, position in enumerate(selected)}
            block = captured.hidden_states[torch.tensor(selected, dtype=torch.long)].numpy().astype(numpy_dtype)
            buffers.append(block)
            planning_offsets = [position_map[item] for item in planning_positions if item in position_map]
            code_offsets = [position_map[item] for item in code_positions if item in position_map]
            indices.append(
                SampleIndex(
                    identifier=sample.identifier,
                    start=cursor,
                    end=cursor + block.shape[0],
                    planning_start=cursor + min(planning_offsets) if planning_offsets else cursor,
                    planning_end=cursor + max(planning_offsets) + 1 if planning_offsets else cursor,
                    code_start=cursor + min(code_offsets) if code_offsets else cursor,
                    code_end=cursor + max(code_offsets) + 1 if code_offsets else cursor,
                    label=sample.label,
                    split=sample.split,
                    language=sample.prompt.language,
                    offsets=[captured.offsets[position] for position in selected],
                )
            )
            cursor += block.shape[0]

        if cursor == 0:
            raise ConfigError("activation collection produced no tokens; check the reasoning corpus and span configuration")
        store = ActivationStore(
            directory=self._directory,
            hidden_size=self._model.hidden_size,
            dtype=self._dtype,
            samples=indices,
            token_count=cursor,
        )
        memmap = np.memmap(store.activation_path, dtype=numpy_dtype, mode="w+", shape=(cursor, self._model.hidden_size))
        offset = 0
        for block in buffers:
            memmap[offset : offset + block.shape[0]] = block
            offset += block.shape[0]
        memmap.flush()
        del memmap
        store.save_index()
        LOGGER.info("stored %d activation vectors of dimension %d", cursor, self._model.hidden_size)
        return store

    def _select_positions(
        self,
        sequence_length: int,
        planning_positions: Sequence[int],
        code_positions: Sequence[int],
    ) -> list[int]:
        selected: set[int] = set()
        if "all" in self._retain:
            selected.update(range(sequence_length))
        if "planning" in self._retain:
            selected.update(planning_positions)
        if "code" in self._retain:
            selected.update(code_positions)
        ordered = sorted(selected)
        if self._max_tokens_per_sample > 0 and len(ordered) > self._max_tokens_per_sample:
            stride = len(ordered) / self._max_tokens_per_sample
            ordered = [ordered[int(index * stride)] for index in range(self._max_tokens_per_sample)]
        return ordered
