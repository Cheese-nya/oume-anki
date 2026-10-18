"""Atomic JSON persistence for a local study profile."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class JsonStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        with self.path.open(encoding="utf-8") as handle:
            return json.load(handle)

    def save(self, data: dict[str, Any]) -> None:
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".oume-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
