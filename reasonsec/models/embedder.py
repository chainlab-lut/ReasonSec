from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from reasonsec.config import Config, ConfigError
from reasonsec.models.language_model import resolve_dtype
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


class SentenceEmbedder:
    def __init__(self, config: Config) -> None:
        section = config.require_section("embedder")
        self.name_or_path = section.require_str("name_or_path")
        self._max_length = section.require_int("max_length")
        self._batch_size = section.require_int("batch_size")
        self._pooling = section.require_str("pooling").strip().lower()
        self._normalise = section.require_bool("normalise")
        self._device = section.require_str("device")
        dtype = resolve_dtype(section.require_str("dtype"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        self.model = AutoModel.from_pretrained(self.name_or_path, torch_dtype=dtype)
        self.model.to(self._device)
        self.model.eval()
        LOGGER.info("loaded sentence encoder %s", self.name_or_path)

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, int(self.model.config.hidden_size)), dtype=torch.float32)
        embeddings: list[torch.Tensor] = []
        for start in range(0, len(texts), self._batch_size):
            batch = [text if text.strip() else " " for text in texts[start : start + self._batch_size]]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self._max_length,
            ).to(self._device)
            outputs = self.model(**encoded)
            hidden = outputs.last_hidden_state.to(torch.float32)
            mask = encoded["attention_mask"].unsqueeze(-1).to(torch.float32)
            if self._pooling == "mean":
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            elif self._pooling == "cls":
                pooled = hidden[:, 0, :]
            elif self._pooling == "max":
                pooled = (hidden - (1.0 - mask) * 1e9).max(dim=1).values
            else:
                raise ConfigError(f"unsupported pooling strategy '{self._pooling}'")
            if self._normalise:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            embeddings.append(pooled.cpu())
        return torch.cat(embeddings, dim=0)

    def similarity(self, left: Sequence[str], right: Sequence[str]) -> float:
        left_embedding = self.encode(list(left))
        right_embedding = self.encode(list(right))
        if left_embedding.shape[0] == 0 or right_embedding.shape[0] == 0:
            return 0.0
        left_mean = torch.nn.functional.normalize(left_embedding.mean(dim=0, keepdim=True), p=2, dim=-1)
        right_mean = torch.nn.functional.normalize(right_embedding.mean(dim=0, keepdim=True), p=2, dim=-1)
        return float((left_mean @ right_mean.T).squeeze().item())
