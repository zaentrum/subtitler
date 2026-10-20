"""Translating a subtitle track: the same preparation for every translator.

Before: markup out (a cue in italics is remembered), SDH annotations out for
an SDH source, a cue's lines joined into one; two speakers ("-Yes.\n-No.")
are joined with a <br> marker the translators keep.
After: the marker back into dialogue lines, the target script enforced
(Simplified stays Simplified), lines re-wrapped to the target's limits,
italics restored. Times are the source cue's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .. import script, sdh, vtt, wrap
from ..langs import Lang

_DASH = re.compile(r"^\s*[-‐–—]\s*")


@dataclass
class Prepared:
    cues: list[vtt.Cue]
    texts: list[str]
    italic: list[bool]


def prepare(cues: list[vtt.Cue], is_sdh: bool = False) -> Prepared:
    out_cues, texts, italic = [], [], []
    for c in cues:
        raw = c.text.strip()
        it = raw.startswith("<i>") and raw.endswith("</i>") and raw.count("<i>") == 1
        text = sdh.clean_text(raw) if is_sdh else vtt.plain(raw)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            continue
        if len(lines) >= 2 and all(_DASH.match(l) for l in lines):
            joined = " <br> ".join("-" + _DASH.sub("", l) for l in lines)
        else:
            joined = " ".join(lines)
        out_cues.append(c)
        texts.append(re.sub(r"\s{2,}", " ", joined))
        italic.append(it)
    return Prepared(out_cues, texts, italic)


def finish(text: str, tgt: Lang, italic: bool) -> str:
    t = text.strip()
    if not t:
        return ""
    if tgt.code == "zh":
        t = script.convert(t, "t2s" if tgt.script != "Hant" else "s2t")
    if "<br>" in t:
        parts = [p.strip() for p in t.split("<br>") if p.strip()]
        lines = ["-" + _DASH.sub("", p) for p in parts]
        if tgt.code in ("zh", "yue"):
            lines = ["-" + wrap.zh_style(_DASH.sub("", l)) for l in lines]
        body = "\n".join(lines)
    else:
        body = wrap.rewrap(t, tgt.code)
    return f"<i>{body}</i>" if italic else body


def _fits_one_cue(text: str, tgt: Lang, seconds: float) -> bool:
    lim = wrap.limits(tgt.code)
    if "<br>" in text or seconds > 7.0:
        return False
    lines = wrap.rewrap(text, tgt.code).split("\n")
    return len(lines) <= lim.lines and all(wrap.width(l) <= lim.chars_per_line for l in lines)


def translate_cues(cues: list[vtt.Cue], src: Lang, tgt: Lang, translator, is_sdh: bool = False):
    """(translated cues, usage, failures).

    A translator answers per source cue with a text, or with (text, span) where
    span > 1 covers that many cues (a sentence translated as a whole): it
    becomes one cue over their time when it fits one, else it is split back
    over them by their share of the source text.
    """
    from .madlad import split_by_share

    prep = prepare(cues, is_sdh)
    if getattr(translator, "uses_durations", False):
        outs, usage = translator.translate(prep.texts, src, tgt, durations=[c.duration for c in prep.cues])
    else:
        outs, usage = translator.translate(prep.texts, src, tgt)
    result, failures, merged = [], 0, 0
    unspaced = tgt.code in ("zh", "yue", "ja")
    i, n = 0, len(prep.cues)
    while i < n:
        text, span = outs[i] if isinstance(outs[i], tuple) else (outs[i], 1)
        if span == 0:  # taken in by an earlier answer
            i += 1
            continue
        span = max(1, min(span, n - i))
        group, texts = prep.cues[i:i + span], prep.texts[i:i + span]
        italic = all(prep.italic[i:i + span])
        if span == 1:
            pieces = [(group[0], text)]
        else:
            merged += 1
            start, end = group[0].start, group[-1].end
            if _fits_one_cue(text, tgt, end - start):
                pieces = [(vtt.Cue(start, end, ""), text)]
            else:
                parts = split_by_share(text.replace("<br>", " "), [max(1, len(t)) for t in texts], unspaced)
                pieces = list(zip(group, parts))
        for c, p in pieces:
            body = finish(p, tgt, italic)
            if not body:
                failures += 1
                continue
            result.append(vtt.Cue(c.start, c.end, body, meta={"source": " ".join(texts)}))
        i += span
    if hasattr(usage, "merged"):
        usage.merged = merged
    return result, usage, failures
