from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasonsec.config import Config, ConfigError
from reasonsec.utils.logging import get_logger

LOGGER = get_logger(__name__)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key not in _DTYPES:
        raise ConfigError(f"unsupported dtype '{name}'; available options are {sorted(_DTYPES)}")
    return _DTYPES[key]


def resolve_module(root: torch.nn.Module, path: str) -> torch.nn.Module:
    module: Any = root
    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]
        elif hasattr(module, part):
            module = getattr(module, part)
        else:
            raise ConfigError(f"module path '{path}' could not be resolved at component '{part}'")
    if not isinstance(module, torch.nn.Module):
        raise ConfigError(f"module path '{path}' does not resolve to a torch module")
    return module


@dataclass
class CapturedActivations:
    hidden_states: torch.Tensor
    offsets: list[tuple[int, int]]
    input_ids: torch.Tensor

    def span_indices(self, start_char: int, end_char: int) -> list[int]:
        indices: list[int] = []
        for position, (token_start, token_end) in enumerate(self.offsets):
            if token_end <= token_start:
                continue
            if token_start >= end_char or token_end <= start_char:
                continue
            indices.append(position)
        return indices


class LanguageModel:
    def __init__(self, config: Config) -> None:
        section = config.require_section("model")
        self._config = section
        self.name_or_path = section.require_str("name_or_path")
        self.layer = section.require_int("layer")
        self._layer_module_path = section.require_str("layer_module_path").format(layer=self.layer)
        self._hook_output_index = section.require_int("hook_output_index")
        self._max_new_tokens = section.require_int("max_new_tokens")
        self._max_input_tokens = section.require_int("max_input_tokens")
        self._generation_arguments = section.require_mapping("generation")
        self._use_chat_template = section.require_bool("chat_template.enabled")
        self._system_prompt = section.optional("chat_template.system_prompt")
        self._trust_remote_code = section.require_bool("trust_remote_code")
        self._preformatted_add_special_tokens = section.require_bool("preformatted_add_special_tokens")
        self._device = section.require_str("device")
        self._dtype = resolve_dtype(section.require_str("dtype"))

        tokenizer_path = section.optional("tokenizer_name_or_path") or self.name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=self._trust_remote_code,
            revision=section.optional("revision"),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_arguments: dict[str, Any] = {
            "trust_remote_code": self._trust_remote_code,
            "torch_dtype": self._dtype,
        }
        revision = section.optional("revision")
        if revision:
            load_arguments["revision"] = revision
        device_map = section.optional("device_map")
        if device_map:
            load_arguments["device_map"] = device_map
        if section.require_bool("quantization.enabled"):
            load_arguments["quantization_config"] = self._build_quantization_config(section)

        self.model = AutoModelForCausalLM.from_pretrained(self.name_or_path, **load_arguments)
        if not device_map and not section.require_bool("quantization.enabled"):
            self.model.to(self._device)
        self.model.eval()
        self.model.config.use_cache = section.require_bool("use_cache")
        self._layer_module = resolve_module(self.model, self._layer_module_path)
        self.hidden_size = int(self.model.config.hidden_size)
        LOGGER.info(
            "loaded model %s with hidden size %d, intervening at module %s",
            self.name_or_path,
            self.hidden_size,
            self._layer_module_path,
        )

    @staticmethod
    def _build_quantization_config(section: Config) -> Any:
        from transformers import BitsAndBytesConfig

        bits = section.require_int("quantization.bits")
        compute_dtype = resolve_dtype(section.require_str("quantization.compute_dtype"))
        quant_type = section.require_str("quantization.quant_type")
        double_quant = section.require_bool("quantization.double_quantization")
        if bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=double_quant,
            )
        if bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True)
        raise ConfigError(f"unsupported quantization bit width {bits}")

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def format_prompt(self, instruction: str) -> str:
        if not self._use_chat_template:
            return instruction
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": str(self._system_prompt)})
        messages.append({"role": "user", "content": instruction})
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _add_special_tokens(self, preformatted: bool) -> bool:
        if preformatted:
            return self._preformatted_add_special_tokens
        return not self._use_chat_template

    def _generation_kwargs(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        arguments = dict(self._generation_arguments)
        arguments["max_new_tokens"] = self._max_new_tokens
        if overrides:
            arguments.update(overrides)
        arguments["pad_token_id"] = self.tokenizer.pad_token_id
        return arguments

    @torch.no_grad()
    def generate(self, instruction: str, overrides: dict[str, Any] | None = None, preformatted: bool = False) -> str:
        text = instruction if preformatted else self.format_prompt(instruction)
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_input_tokens,
            add_special_tokens=self._add_special_tokens(preformatted),
        ).to(self.device)
        outputs = self.model.generate(**encoded, **self._generation_kwargs(overrides))
        completion = outputs[0][encoded["input_ids"].shape[1] :]
        return self.tokenizer.decode(completion, skip_special_tokens=True)

    @torch.no_grad()
    def generate_batch(
        self,
        instructions: Sequence[str],
        overrides: dict[str, Any] | None = None,
        preformatted: bool = False,
    ) -> list[str]:
        texts = [instruction if preformatted else self.format_prompt(instruction) for instruction in instructions]
        previous_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_input_tokens,
            add_special_tokens=self._add_special_tokens(preformatted),
        ).to(self.device)
        self.tokenizer.padding_side = previous_side
        outputs = self.model.generate(**encoded, **self._generation_kwargs(overrides))
        prompt_length = encoded["input_ids"].shape[1]
        return [
            self.tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True) for sequence in outputs
        ]

    @torch.no_grad()
    def capture_residual(self, text: str) -> CapturedActivations:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_input_tokens,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        offsets = [tuple(int(value) for value in pair) for pair in encoded.pop("offset_mapping")[0].tolist()]
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        captured: dict[str, torch.Tensor] = {}

        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            tensor = output[self._hook_output_index] if isinstance(output, tuple) else output
            captured["hidden"] = tensor.detach()

        handle = self._layer_module.register_forward_hook(hook)
        try:
            self.model(**encoded)
        finally:
            handle.remove()
        if "hidden" not in captured:
            raise RuntimeError(f"no activation captured at {self._layer_module_path}")
        hidden = captured["hidden"][0].to(torch.float32).cpu()
        return CapturedActivations(hidden_states=hidden, offsets=offsets, input_ids=encoded["input_ids"][0].cpu())

    @contextmanager
    def residual_editor(self, editor: Callable[[torch.Tensor], torch.Tensor]) -> Iterator[None]:
        def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> Any:
            if isinstance(output, tuple):
                tensor = output[self._hook_output_index]
                edited = editor(tensor)
                if edited is None:
                    return output
                modified = list(output)
                modified[self._hook_output_index] = edited
                return tuple(modified)
            edited = editor(output)
            return output if edited is None else edited

        handle = self._layer_module.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def continuation_log_likelihood(self, prefix: str, continuation: str) -> float:
        prefix_ids = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=True)["input_ids"]
        full_ids = self.tokenizer(prefix + continuation, return_tensors="pt", add_special_tokens=True)["input_ids"]
        if full_ids.shape[1] <= prefix_ids.shape[1]:
            return float("-inf")
        full_ids = full_ids[:, -self._max_input_tokens :].to(self.device)
        logits = self.model(input_ids=full_ids).logits.to(torch.float32)
        log_probabilities = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = full_ids[:, 1:]
        gathered = log_probabilities.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
        continuation_length = full_ids.shape[1] - prefix_ids.shape[1]
        return float(gathered[-continuation_length:].sum().item())
