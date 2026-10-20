"""What a package holds, as the subtitles step sees it.

Reads package.json (the packager's manifest of a version folder):
`renditions.audio` (stereo renditions, the main one `default`) and
`subtitles` (language, title, forced, purpose, format, path). Each subtitle
gets an ISO 639-1 language, its Chinese script and region where it is
Chinese, and whether its file is on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import script, vtt
from .langs import iso1

TEXT_FORMATS = {"webvtt", "vtt", "srt", "subrip", "ass", "ssa", "mov_text", "ttml", "text"}
# Image subtitles a client can draw: they count as present in their language,
# but they are no source for a translation (they would need OCR first).
SHOWN_IMAGE_FORMATS = {"pgs", "hdmv_pgs_subtitle", "sup"}


@dataclass
class Subtitle:
    id: str
    path: str
    language: str | None  # ISO 639-1, "yue" for Cantonese, None if unknown
    title: str
    forced: bool
    purpose: str  # dialogue | sdh | forced | commentary | ...
    format: str
    visible: bool
    exists: bool
    script: str | None = None  # Hans | Hant for Chinese
    region: str | None = None  # HK | TW for Chinese
    script_from: str | None = None  # title | text
    bilingual: bool = False
    cues: int | None = None  # text tracks: how many cues
    partial: bool = False  # a text track too sparse to be the dialogue (signs only)
    meta: dict = field(default_factory=dict)

    @property
    def is_text(self) -> bool:
        return self.format.lower() in TEXT_FORMATS

    @property
    def is_full(self) -> bool:
        """A complete subtitle: not forced-only, not commentary, not signs only."""
        return not self.forced and self.purpose not in ("forced", "commentary") and not self.partial

    @property
    def shown(self) -> bool:
        """A viewer can turn it on: text, or an image format the clients draw."""
        return self.exists and (self.is_text or self.format.lower() in SHOWN_IMAGE_FORMATS)

    @property
    def is_sdh(self) -> bool:
        return self.purpose == "sdh"


@dataclass
class Audio:
    id: str
    dir: str
    language: str | None
    default: bool
    purpose: str


@dataclass
class Package:
    root: Path
    audio: list[Audio]
    subtitles: list[Subtitle]

    @property
    def main_audio(self) -> Audio | None:
        mains = [a for a in self.audio if a.purpose in ("main", "", None)]
        for a in mains:
            if a.default:
                return a
        return mains[0] if mains else (self.audio[0] if self.audio else None)


def _detect_chinese(sub: Subtitle, root: Path) -> None:
    h = script.hints(sub.title)
    sub.region, sub.bilingual = h["region"], h["bilingual"]
    if h["cantonese"]:
        sub.language = "yue"
    if sub.exists and sub.is_text:
        try:
            text = " ".join(vtt.plain(c.text) for c in vtt.read(root / sub.path)[:400])
        except (OSError, UnicodeDecodeError):
            text = ""
        if text and sub.language != "yue" and script.looks_cantonese(text):
            sub.language = "yue"
        found, conf = script.detect_script(text) if text else (None, 0.0)
        if found and conf >= 0.02:
            sub.script, sub.script_from = found, "text"
            return
    if h["script"]:
        sub.script, sub.script_from = h["script"], "title"


# Dialogue subtitles run at 8-20 cues a minute; a track of signs and songs
# that is not flagged forced at well under one, and at a small share of the
# title's dialogue tracks.
MIN_CUES_PER_MINUTE = 1.0
MIN_SHARE_OF_FULLEST = 0.3


def _count_cues(path: Path) -> int:
    try:
        return path.read_bytes().count(b"-->")
    except OSError:
        return 0


def load(root: str | Path) -> Package:
    root = Path(root)
    data = json.loads((root / "package.json").read_text(encoding="utf-8"))
    minutes = (data.get("durationMs") or 0) / 60000
    audio = [Audio(a.get("id", ""), a.get("dir", ""), iso1(a.get("language")), bool(a.get("default")),
                   a.get("purpose") or "main")
             for a in (data.get("renditions") or {}).get("audio") or []]
    subs = []
    for s in data.get("subtitles") or []:
        path = s.get("path") or ""
        sub = Subtitle(
            id=s.get("id", path), path=path, language=iso1(s.get("language")), title=s.get("title") or "",
            forced=bool(s.get("forced")), purpose=s.get("purpose") or ("forced" if s.get("forced") else "dialogue"),
            format=(s.get("format") or Path(path).suffix.lstrip(".") or "").lower(),
            visible=bool(s.get("visible", True)), exists=bool(path) and (root / path).is_file())
        if sub.exists and sub.is_text:
            sub.cues = _count_cues(root / path)
        if sub.language in ("zh", "yue"):
            _detect_chinese(sub, root)
        subs.append(sub)
    # A text track far sparser than the title's fullest one, or under a cue a
    # minute, is signs and songs: no subtitle of its language.
    counts = [s.cues for s in subs if s.cues and not s.forced and s.purpose != "commentary"]
    fullest = max(counts) if counts else 0
    for s in subs:
        if s.cues is not None and minutes >= 10:
            s.partial = s.cues < MIN_SHARE_OF_FULLEST * fullest or s.cues < MIN_CUES_PER_MINUTE * minutes
    return Package(root, audio, subs)
