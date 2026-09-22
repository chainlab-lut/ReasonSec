from __future__ import annotations

import re
from pathlib import Path

from reasonsec.config import Config, ConfigError
from reasonsec.utils.io import read_text

_PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


class PromptTemplate:
    def __init__(self, text: str, source: str) -> None:
        self._text = text
        self._source = source
        self.placeholders = sorted(set(_PLACEHOLDER_PATTERN.findall(text)))

    @classmethod
    def from_file(cls, path: str | Path) -> "PromptTemplate":
        resolved = Path(path).expanduser()
        if not resolved.is_file():
            raise ConfigError(f"prompt template file not found: {resolved}")
        return cls(read_text(resolved), str(resolved))

    def render(self, **values: str) -> str:
        missing = [name for name in self.placeholders if name not in values]
        if missing:
            raise ConfigError(f"prompt template {self._source} requires placeholders {missing}")
        rendered = self._text
        for name, value in values.items():
            rendered = rendered.replace("{" + name + "}", str(value))
        return rendered

    @property
    def text(self) -> str:
        return self._text


class TemplateLibrary:
    def __init__(self, config: Config) -> None:
        section = config.require_section("reasoning.templates")
        self.elicitation = PromptTemplate.from_file(section.require_path("elicitation"))
        self.concept_extraction = PromptTemplate.from_file(section.require_path("concept_extraction"))
        self.feature_description = PromptTemplate.from_file(section.require_path("feature_description"))
        self.unstructured = PromptTemplate.from_file(section.require_path("unstructured"))
        self.zero_shot_security = PromptTemplate.from_file(section.require_path("zero_shot_security"))
        self.one_shot_security = PromptTemplate.from_file(section.require_path("one_shot_security"))
        self.chain_of_thought_security = PromptTemplate.from_file(section.require_path("chain_of_thought_security"))
        self.functional_completion = PromptTemplate.from_file(section.require_path("functional_completion"))
        self.forced_category = PromptTemplate.from_file(section.require_path("forced_category"))
