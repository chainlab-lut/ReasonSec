from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from reasonsec.config import Config, ConfigError
from reasonsec.models.language_model import LanguageModel
from reasonsec.reasoning.templates import TemplateLibrary
from reasonsec.types import CorpusSample
from reasonsec.utils.io import ensure_directory, write_json
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class SupervisedTrainingStatistics:
    epochs: list[dict[str, float]] = field(default_factory=list)
    example_count: int = 0
    trainable_parameters: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "epochs": self.epochs,
            "example_count": self.example_count,
            "trainable_parameters": self.trainable_parameters,
        }


class SecureCodeDataset(Dataset):
    def __init__(self, tokenizer, examples: Sequence[tuple[str, str]], max_length: int) -> None:
        self._tokenizer = tokenizer
        self._examples = list(examples)
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        instruction, completion = self._examples[index]
        prompt_ids = self._tokenizer(instruction, add_special_tokens=True)["input_ids"]
        completion_ids = self._tokenizer(completion, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + completion_ids)[: self._max_length]
        labels = ([-100] * len(prompt_ids) + completion_ids)[: self._max_length]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch: Sequence[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    length = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.full((len(batch), length), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), length), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), length), dtype=torch.long)
    for position, item in enumerate(batch):
        size = item["input_ids"].shape[0]
        input_ids[position, :size] = item["input_ids"]
        labels[position, :size] = item["labels"]
        attention_mask[position, :size] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def build_secure_examples(
    config: Config,
    samples: Sequence[CorpusSample],
    templates: TemplateLibrary,
    model: LanguageModel,
) -> list[tuple[str, str]]:
    section = config.require_section("baselines.supervised_fine_tuning")
    maximum = section.require_int("max_examples")
    completion_template = section.require_str("completion_template")
    examples: list[tuple[str, str]] = []
    for sample in samples:
        if sample.verdict.vulnerable or not sample.chain.code.strip():
            continue
        instruction = model.format_prompt(
            templates.unstructured.render(prompt=sample.prompt.prompt, language=sample.prompt.language)
        )
        completion = completion_template.replace("{code}", sample.chain.code).replace(
            "{language}", sample.prompt.language
        )
        examples.append((instruction, completion))
        if 0 < maximum <= len(examples):
            break
    if not examples:
        raise ConfigError("no safe samples are available in the curated corpus for supervised fine-tuning")
    LOGGER.info("assembled %d secure fine-tuning examples", len(examples))
    return examples


def train_lora_adapter(
    config: Config,
    model: LanguageModel,
    examples: Sequence[tuple[str, str]],
    output_directory: str | Path,
    seed: int,
) -> SupervisedTrainingStatistics:
    from peft import LoraConfig, get_peft_model

    section = config.require_section("baselines.supervised_fine_tuning")
    lora_configuration = LoraConfig(
        r=section.require_int("lora.rank"),
        lora_alpha=section.require_int("lora.alpha"),
        lora_dropout=section.require_float("lora.dropout"),
        bias=section.require_str("lora.bias"),
        task_type=section.require_str("lora.task_type"),
        target_modules=[str(item) for item in section.require_list("lora.target_modules")],
    )
    epochs = section.require_int("epochs")
    batch_size = section.require_int("batch_size")
    learning_rate = section.require_float("learning_rate")
    max_length = section.require_int("max_sequence_length")
    gradient_accumulation = section.require_int("gradient_accumulation_steps")

    torch.manual_seed(seed)
    adapted = get_peft_model(model.model, lora_configuration)
    adapted.train()
    dataset = SecureCodeDataset(model.tokenizer, examples, max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, model.tokenizer.pad_token_id),
    )
    optimiser = torch.optim.AdamW(
        [parameter for parameter in adapted.parameters() if parameter.requires_grad], lr=learning_rate
    )
    statistics = SupervisedTrainingStatistics(
        example_count=len(examples),
        trainable_parameters=int(
            sum(parameter.numel() for parameter in adapted.parameters() if parameter.requires_grad)
        ),
    )
    device = model.device
    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0
        optimiser.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = adapted(**batch)
            loss = outputs.loss / gradient_accumulation
            loss.backward()
            if (index + 1) % gradient_accumulation == 0:
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
            total_loss += float(outputs.loss.item())
            steps += 1
        statistics.epochs.append({"epoch": epoch + 1, "loss": total_loss / max(steps, 1)})
        LOGGER.info("supervised fine-tuning epoch %d loss %.4f", epoch + 1, total_loss / max(steps, 1))
    adapted.eval()
    target = ensure_directory(output_directory)
    adapted.save_pretrained(str(target))
    write_json(target / "training_statistics.json", statistics.as_dict())
    model.model = adapted
    return statistics


def load_lora_adapter(model: LanguageModel, adapter_directory: str | Path) -> None:
    from peft import PeftModel

    path = Path(adapter_directory).expanduser()
    if not path.exists():
        raise ConfigError(f"fine-tuned adapter directory not found: {path}")
    model.model = PeftModel.from_pretrained(model.model, str(path))
    model.model.eval()


def unload_lora_adapter(model: LanguageModel) -> None:
    inner = model.model
    if hasattr(inner, "unload"):
        model.model = inner.unload()
    elif hasattr(inner, "get_base_model"):
        model.model = inner.get_base_model()
    model.model.eval()
