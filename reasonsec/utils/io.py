from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_directory(path: str | os.PathLike[str]) -> Path:
    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_text(path: str | os.PathLike[str], encoding: str = "utf-8") -> str:
    return Path(path).expanduser().read_text(encoding=encoding)


def write_text(path: str | os.PathLike[str], content: str, encoding: str = "utf-8") -> Path:
    target = Path(path).expanduser()
    ensure_directory(target.parent)
    target.write_text(content, encoding=encoding)
    return target


def read_json(path: str | os.PathLike[str], encoding: str = "utf-8") -> Any:
    with Path(path).expanduser().open("r", encoding=encoding) as handle:
        return json.load(handle)


def write_json(path: str | os.PathLike[str], payload: Any, encoding: str = "utf-8", indent: int = 2) -> Path:
    target = Path(path).expanduser()
    ensure_directory(target.parent)
    with target.open("w", encoding=encoding) as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False, default=_fallback_serialiser)
        handle.write("\n")
    return target


def read_jsonl(path: str | os.PathLike[str], encoding: str = "utf-8") -> Iterator[Any]:
    with Path(path).expanduser().open("r", encoding=encoding) as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def write_jsonl(path: str | os.PathLike[str], records: Iterable[Any], encoding: str = "utf-8") -> Path:
    target = Path(path).expanduser()
    ensure_directory(target.parent)
    with target.open("w", encoding=encoding) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_fallback_serialiser))
            handle.write("\n")
    return target


def _fallback_serialiser(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "as_dict"):
        return value.as_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serialisable")
