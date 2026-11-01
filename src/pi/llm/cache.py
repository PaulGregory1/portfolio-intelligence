"""On-disk response cache: makes evaluation runs reproducible and saves the free API quota."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(payload: dict[str, Any]) -> str:
    """Stable hash of everything that determines a completion."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class DiskCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def set(self, key: str, value: dict[str, Any]) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic: concurrent workers never read a half-written file
