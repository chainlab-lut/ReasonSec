from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO", log_file: str | os.PathLike[str] | None = None) -> None:
    global _CONFIGURED
    root = logging.getLogger("reasonsec")
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    root.propagate = False
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = logging.Formatter(_FORMAT)
    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    if log_file is not None:
        target = Path(log_file).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(target, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    if name.startswith("reasonsec"):
        return logging.getLogger(name)
    return logging.getLogger(f"reasonsec.{name}")
