"""WebVTT in and out: cues with times in seconds and text with line breaks.

The reader takes what the packager writes (ffmpeg's WebVTT) and what other
tools write: an optional BOM, a header line with text after WEBVTT, NOTE,
STYLE and REGION blocks, cue identifiers, cue settings after the times, and
times with or without hours.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})"
_TIMING = re.compile(rf"^\s*{_TIME}\s+-->\s+{_TIME}(.*)$")
_TAG = re.compile(r"</?(?:[ibu]|c|v|lang|ruby|rt)(?:[.\s][^>]*)?>|<\d{1,2}:\d{2}[:.\d]*>", re.IGNORECASE)


@dataclass
class Cue:
    start: float
    end: float
    text: str  # payload lines joined with "\n"
    settings: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def lines(self) -> list[str]:
        return self.text.split("\n")

    def with_text(self, text: str) -> "Cue":
        return replace(self, text=text, meta=dict(self.meta))


def _seconds(h: str | None, m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse(text: str) -> list[Cue]:
    """Parse WebVTT text into cues (in file order)."""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    for block in re.split(r"\n{2,}", text):
        lines = block.split("\n")
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            continue
        head = lines[0].lstrip()
        if head.startswith("WEBVTT") or head.startswith("NOTE") or head in ("STYLE", "REGION"):
            continue
        for i, line in enumerate(lines[:2]):
            m = _TIMING.match(line)
            if m:
                g = m.groups()
                start = _seconds(g[0], g[1], g[2], g[3])
                end = _seconds(g[4], g[5], g[6], g[7])
                payload = "\n".join(l for l in lines[i + 1:]).strip("\n")
                cues.append(Cue(start, end, payload, settings=g[8].strip()))
                break
    return cues


def read(path: str | Path) -> list[Cue]:
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252")
    return parse(text)


def timestamp(t: float) -> str:
    t = max(0.0, t)
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _safe_payload(text: str) -> str:
    # A payload may not contain "-->" nor a blank line.
    text = text.replace("-->", "->")
    return "\n".join(l for l in text.split("\n") if l.strip())


def dumps(cues: list[Cue], notes: list[str] | None = None) -> str:
    out = ["WEBVTT", ""]
    for note in notes or []:
        body = note.replace("-->", "->").strip()
        out += ["NOTE " + body, ""]
    for c in cues:
        payload = _safe_payload(c.text)
        if not payload:
            continue
        timing = f"{timestamp(c.start)} --> {timestamp(c.end)}"
        if c.settings:
            timing += " " + c.settings
        out += [timing, payload, ""]
    return "\n".join(out)


def write(path: str | Path, cues: list[Cue], notes: list[str] | None = None) -> None:
    Path(path).write_text(dumps(cues, notes), encoding="utf-8")


def plain(text: str) -> str:
    """Payload without markup: tags removed, entities decoded, bidi marks dropped."""
    text = _TAG.sub("", text)
    text = html.unescape(text)
    return text.replace("‎", "").replace("‏", "").replace("‪", "").replace("‬", "")


def window(cues: list[Cue], start: float, end: float) -> list[Cue]:
    """The cues whose midpoint lies in [start, end)."""
    return [c for c in cues if start <= (c.start + c.end) / 2 < end]


def shift(cues: list[Cue], offset: float) -> list[Cue]:
    return [replace(c, start=c.start + offset, end=c.end + offset, meta=dict(c.meta)) for c in cues]
