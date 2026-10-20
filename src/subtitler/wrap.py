"""Line limits per language and re-wrapping a cue's text to them.

Limits follow common streaming practice for adult content: 42 characters a
line for languages written with spaces, 16 for Chinese and Korean, 13 for
Japanese, two lines a cue. Reading speed limits are characters per second:
English 20, German 17, Chinese 9.

Chinese subtitles in the house style the library's human tracks use: no
full stop at the end of a line, a full-width space where a comma would
break a line, question and exclamation marks kept.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .langs import UNSPACED


@dataclass(frozen=True)
class Limits:
    chars_per_line: int
    lines: int = 2
    cps: float = 17.0  # characters per second a viewer can read

    @property
    def chars_per_cue(self) -> int:
        return self.chars_per_line * self.lines


LIMITS = {
    "en": Limits(42, 2, 20.0),
    "de": Limits(42, 2, 17.0),
    "zh": Limits(16, 2, 9.0),
    "yue": Limits(16, 2, 9.0),
    "ja": Limits(13, 2, 4.0),
    "ko": Limits(16, 2, 12.0),
}
DEFAULT = Limits(42, 2, 17.0)


def limits(code: str) -> Limits:
    return LIMITS.get(code, DEFAULT)


def width(s: str) -> int:
    """Display length in characters (a CJK character counts one, as the limits do)."""
    return sum(0 if unicodedata.combining(ch) else 1 for ch in s)


_BREAK_AFTER = set(",.;:!?…，。；：！？、")
_WEAK_END = {  # words a line should not end with
    "en": {"a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or", "but", "with", "my", "your", "his", "her", "our", "their", "i", "is", "are"},
    "de": {"der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer", "und", "oder", "aber", "mit", "zu", "in", "im", "am", "an", "auf", "für", "von", "vom", "ich", "du", "er", "sie", "es", "wir", "ihr"},
}


def _split_spaced(text: str, lim: Limits, lang: str) -> list[str]:
    words = text.split()
    if width(text) <= lim.chars_per_line or len(words) < 2:
        return [text]
    weak = _WEAK_END.get(lang, set())
    best, best_score = None, None
    total = width(text)
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        wa, wb = width(a), width(b)
        over = max(0, wa - lim.chars_per_line) + max(0, wb - lim.chars_per_line)
        score = over * 100 + abs(wa - wb) * 1.0
        if a[-1] in _BREAK_AFTER:
            score -= 20
        if words[i - 1].lower().strip(",.") in weak:
            score += 10
        # A bottom-heavy pyramid reads better than a top-heavy one.
        if wa > wb:
            score += 2
        if best_score is None or score < best_score:
            best, best_score = [a, b], score
    if best and max(width(x) for x in best) > lim.chars_per_line and total > lim.chars_per_cue:
        # Too long for two lines: keep two, the overflow is reported by stats().
        return best
    return best or [text]


def zh_style(text: str) -> str:
    """House style for Chinese lines: comma breaks become spaces, no final full stop."""
    t = re.sub(r"\s+", " ", text.strip())
    t = re.sub(r"，\s*|,(?!\d)\s*", " ", t)  # not the comma of 1,000
    t = re.sub(r"。\s*(?=\S)|(?<!\d)\.\s+(?=\S)", " ", t)  # not the point of 3.5
    t = re.sub(r"[。]+$|(?<![.\d])\.$", "", t)  # "..." stays
    return re.sub(r" {2,}", " ", t).strip()


def _split_unspaced(text: str, lim: Limits) -> list[str]:
    t = text.strip()
    n = width(t)
    if n <= lim.chars_per_line:
        return [t]
    mid = n / 2
    candidates = []
    for i, ch in enumerate(t):
        if i == 0 or i >= len(t) - 1:
            continue
        if ch == " " or t[i - 1] in _BREAK_AFTER:
            candidates.append(i)
    best = None
    for i in candidates:
        a, b = t[:i].strip(), t[i:].strip()
        over = max(0, width(a) - lim.chars_per_line) + max(0, width(b) - lim.chars_per_line)
        score = over * 100 + abs(width(a) - width(b))
        if best is None or score < best[0]:
            best = (score, [a, b])
    if best and best[0] < 100:
        return best[1]
    # No break point fits: split in the middle, not inside a Latin word or number.
    i = int(round(mid))
    while 0 < i < len(t) and t[i - 1].isascii() and t[i].isascii() and t[i].isalnum():
        i += 1
    return [t[:i].strip(), t[i:].strip()]


def rewrap(text: str, lang: str) -> str:
    """Re-break a cue's text for `lang`; dialogue lines starting with a dash stay one per line."""
    lim = limits(lang)
    raw_lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not raw_lines:
        return ""
    dashed = len(raw_lines) >= 2 and all(re.match(r"^[-‐–—]", l) for l in raw_lines)
    unspaced = lang in UNSPACED
    if dashed:
        lines = raw_lines
        if unspaced:
            lines = ["-" + zh_style(re.sub(r"^[-‐–—]\s*", "", l)) for l in lines] if lang in ("zh", "yue") else lines
        return "\n".join(lines)
    if unspaced:
        joined = "".join(raw_lines) if all(not l.isascii() for l in raw_lines) else " ".join(raw_lines)
        if lang in ("zh", "yue"):
            joined = zh_style(joined)
        return "\n".join(_split_unspaced(joined, lim))
    joined = " ".join(raw_lines)
    return "\n".join(_split_spaced(joined, lim, lang))


@dataclass
class Stats:
    cues: int = 0
    over_line: int = 0  # cues with a line over the limit
    over_lines: int = 0  # cues with more lines than allowed
    over_cps: int = 0  # cues read faster than the limit

    def as_dict(self) -> dict:
        n = max(1, self.cues)
        return {"cues": self.cues, "over_line_pct": round(100 * self.over_line / n, 1),
                "over_lines_pct": round(100 * self.over_lines / n, 1),
                "over_cps_pct": round(100 * self.over_cps / n, 1)}


def stats(cues, lang: str) -> Stats:
    lim = limits(lang)
    s = Stats()
    for c in cues:
        lines = [l for l in c.text.split("\n") if l.strip()]
        if not lines:
            continue
        s.cues += 1
        if max(width(l) for l in lines) > lim.chars_per_line:
            s.over_line += 1
        if len(lines) > lim.lines:
            s.over_lines += 1
        chars = sum(width(l.replace(" ", "")) if lang in UNSPACED else width(l) for l in lines)
        if c.duration > 0 and chars / c.duration > lim.cps:
            s.over_cps += 1
    return s
