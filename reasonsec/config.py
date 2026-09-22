from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

import yaml

_ENV_PATTERN = re.compile(r"\$\{env:([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


class ConfigError(Exception):
    pass


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in override.items():
        if key in base and isinstance(base[key], MutableMapping) and isinstance(value, Mapping):
            _deep_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        def _substitute(match: re.Match[str]) -> str:
            name, fallback = match.group(1), match.group(2)
            resolved = os.environ.get(name)
            if resolved is None:
                if fallback is None:
                    raise ConfigError(f"environment variable '{name}' referenced by the configuration is not set")
                return fallback
            return resolved

        return _ENV_PATTERN.sub(_substitute, value)
    if isinstance(value, Mapping):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    return value


def _parse_scalar(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


class Config:
    def __init__(
        self,
        data: Mapping[str, Any],
        source: str | None = None,
        base_directory: Path | None = None,
    ) -> None:
        if not isinstance(data, Mapping):
            raise ConfigError("configuration root must be a mapping")
        self._data: dict[str, Any] = copy.deepcopy(dict(data))
        self._source = source
        self._base_directory = base_directory

    @classmethod
    def load(cls, path: str | os.PathLike[str], overrides: Sequence[str] | None = None) -> "Config":
        resolved = Path(path).expanduser().resolve()
        merged = cls._load_with_includes(resolved, set())
        merged = _expand_environment(merged)
        config = cls(merged, source=str(resolved), base_directory=resolved.parent)
        for assignment in overrides or []:
            if "=" not in assignment:
                raise ConfigError(f"override '{assignment}' must use the form key.path=value")
            key, raw_value = assignment.split("=", 1)
            config.set(key.strip(), _parse_scalar(raw_value.strip()))
        return config

    @classmethod
    def _load_with_includes(cls, path: Path, seen: set[Path]) -> dict[str, Any]:
        if path in seen:
            raise ConfigError(f"circular include detected at {path}")
        if not path.is_file():
            raise ConfigError(f"configuration file not found: {path}")
        seen.add(path)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            raise ConfigError(f"configuration file {path} must contain a mapping at the top level")
        document = dict(loaded)
        includes = document.pop("include", [])
        if isinstance(includes, str):
            includes = [includes]
        merged: dict[str, Any] = {}
        for include in includes:
            include_path = (path.parent / str(include)).expanduser().resolve()
            _deep_merge(merged, cls._load_with_includes(include_path, seen))
        _deep_merge(merged, document)
        return merged

    @property
    def source(self) -> str | None:
        return self._source

    @property
    def base_directory(self) -> Path | None:
        return self._base_directory

    def _traverse(self, key: str) -> tuple[Any, bool]:
        node: Any = self._data
        for part in key.split("."):
            if isinstance(node, Mapping) and part in node:
                node = node[part]
            else:
                return None, False
        return node, True

    def _describe(self, key: str) -> str:
        if self._source and ":" in self._source:
            return f"{self._source.split(':', 1)[1]}.{key}"
        return key

    def has(self, key: str) -> bool:
        return self._traverse(key)[1]

    def require(self, key: str) -> Any:
        value, found = self._traverse(key)
        if not found:
            raise ConfigError(f"required configuration key '{key}' is missing from {self._source or 'configuration'}")
        if value is None:
            raise ConfigError(f"configuration key '{key}' is null and must be provided")
        return value

    def optional(self, key: str) -> Any:
        value, found = self._traverse(key)
        return value if found else None

    def resolve_path(self, raw: Any) -> Path:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute() and self._base_directory is not None:
            return self._base_directory / path
        return path

    def require_path(self, key: str, must_exist: bool = True) -> Path:
        path = self.resolve_path(self.require(key))
        if must_exist and not path.exists():
            raise ConfigError(f"path configured at '{self._describe(key)}' does not exist: {path}")
        return path

    def require_paths(self, key: str, must_exist: bool = True) -> list[Path]:
        raw = self.require(key)
        if isinstance(raw, (str, os.PathLike)):
            raw = [raw]
        if not isinstance(raw, Sequence):
            raise ConfigError(f"configuration key '{key}' must be a path or a list of paths")
        paths: list[Path] = []
        for item in raw:
            path = self.resolve_path(item)
            if must_exist and not path.exists():
                raise ConfigError(f"path configured at '{self._describe(key)}' does not exist: {path}")
            paths.append(path)
        return paths

    def require_section(self, key: str) -> "Config":
        value = self.require(key)
        if not isinstance(value, Mapping):
            raise ConfigError(f"configuration key '{key}' must be a mapping")
        return Config(
            value,
            source=f"{self._source}:{key}" if self._source else key,
            base_directory=self._base_directory,
        )

    def require_mapping(self, key: str) -> dict[str, Any]:
        value = self.require(key)
        if not isinstance(value, Mapping):
            raise ConfigError(f"configuration key '{key}' must be a mapping")
        return dict(value)

    def require_list(self, key: str) -> list[Any]:
        value = self.require(key)
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ConfigError(f"configuration key '{key}' must be a list")
        return list(value)

    def require_int(self, key: str) -> int:
        value = self.require(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise ConfigError(f"configuration key '{key}' must be an integer, got {value!r}")
        return int(value)

    def require_float(self, key: str) -> float:
        value = self.require(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"configuration key '{key}' must be a number, got {value!r}")
        return float(value)

    def require_bool(self, key: str) -> bool:
        value = self.require(key)
        if not isinstance(value, bool):
            raise ConfigError(f"configuration key '{key}' must be a boolean, got {value!r}")
        return value

    def require_str(self, key: str) -> str:
        value = self.require(key)
        if not isinstance(value, str):
            raise ConfigError(f"configuration key '{key}' must be a string, got {value!r}")
        return value

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        node: dict[str, Any] = self._data
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value

    def derive(self, overrides: Mapping[str, Any] | None = None) -> "Config":
        derived = Config(self.as_dict(), source=self._source, base_directory=self._base_directory)
        for key, value in (overrides or {}).items():
            derived.set(str(key), copy.deepcopy(value))
        return derived

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def keys(self) -> Iterator[str]:
        return iter(self._data.keys())

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __repr__(self) -> str:
        return f"Config(source={self._source!r}, keys={sorted(self._data.keys())})"
