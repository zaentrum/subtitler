"""The local test configuration (local.toml, untracked) and small helpers."""

from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WINDOW = 600.0  # seconds of each clip


@dataclass
class Title:
    name: str
    package: Path
    audio: str
    subs: dict[str, Path] = field(default_factory=dict)
    sdh: set[Path] = field(default_factory=set)
    wav: Path | None = None
    offset: float | None = None

    @property
    def window(self) -> tuple[float, float] | None:
        if self.offset is None:
            return None
        return self.offset, self.offset + WINDOW


@dataclass
class Config:
    scratch: Path
    models: dict[str, Path]
    titles: dict[str, Title]

    def out(self, *parts: str) -> Path:
        p = self.scratch.joinpath("results", *parts)
        p.mkdir(parents=True, exist_ok=True)
        return p


def load(path: str | None = None) -> Config:
    p = Path(path or os.environ.get("SUBTITLER_LOCAL", ROOT / "local.toml"))
    data = tomllib.loads(p.read_text())
    scratch = Path(data["scratch"])
    models = {k: scratch / v for k, v in data.get("models", {}).items()}
    titles = {}
    for name, t in data.get("titles", {}).items():
        pkg = scratch / t["package"]
        titles[name] = Title(
            name=name, package=pkg, audio=t["audio"],
            subs={k: pkg / v for k, v in t.get("subs", {}).items()},
            sdh={pkg / v for v in t.get("sdh", [])},
            wav=scratch / t["wav"] if "wav" in t else None,
            offset=t.get("offset"))
    return Config(scratch, models, titles)
