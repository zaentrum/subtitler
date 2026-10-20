"""Hearing-impaired annotations out of a subtitle's text.

SDH tracks carry sound descriptions ("[door slams]", "(laughs)"), speaker
labels ("MAN:", "JOHN:"), and song lyrics between music notes. A dialogue
subtitle made from one leaves those out; a cue that is nothing but an
annotation goes.
"""

from __future__ import annotations

import re

from . import vtt

_BRACKETS = re.compile(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）|【[^】]*】")
# An upper-case speaker label at the start of a line: "MAN:", "DR. KIM:", "WOMAN 2:".
_SPEAKER = re.compile(r"^\s*-?\s*(?:[A-Z][A-Z0-9 .'\-]{1,30}|[A-Z][a-z]+ ?\d?):\s+")
_NOTES = "♪♫♬#"
_DASH = re.compile(r"^\s*[-‐–—]\s*")


def is_lyric(line: str) -> bool:
    s = line.strip()
    return bool(s) and (s[0] in _NOTES or s[-1:] in _NOTES)


def clean_line(line: str, drop_lyrics: bool) -> str:
    if drop_lyrics and is_lyric(line):
        return ""
    s = _BRACKETS.sub("", line)
    s = _SPEAKER.sub(lambda m: "-" if m.group(0).lstrip().startswith("-") else "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    # A line left with only a dash or punctuation is empty.
    if not re.search(r"\w", s):
        return ""
    return s


def clean_text(text: str, drop_lyrics: bool = True) -> str:
    lines = [clean_line(l, drop_lyrics) for l in vtt.plain(text).split("\n")]
    lines = [l for l in lines if l]
    dashed = [l for l in lines if _DASH.match(l)]
    # Two speakers become one: the dialogue dash goes with its partner.
    if len(lines) == 1 and dashed:
        lines = [_DASH.sub("", lines[0])]
    return "\n".join(lines)


def clean(cues: list[vtt.Cue], drop_lyrics: bool = True) -> list[vtt.Cue]:
    out = []
    for c in cues:
        t = clean_text(c.text, drop_lyrics)
        if t:
            out.append(c.with_text(t))
    return out


def annotation_share(cues: list[vtt.Cue]) -> float:
    """Share of cues with a bracketed annotation: SDH tracks run well above 10 %."""
    if not cues:
        return 0.0
    return sum(bool(_BRACKETS.search(vtt.plain(c.text))) for c in cues) / len(cues)
