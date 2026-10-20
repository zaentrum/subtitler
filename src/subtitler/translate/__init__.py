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


def translate_cues(cues: list[vtt.Cue], src: Lang, tgt: Lang, translator, is_sdh: bool = False):
    """(translated cues, usage, failures): one cue per source cue that has text."""
    prep = prepare(cues, is_sdh)
    outs, usage = translator.translate(prep.texts, src, tgt)
    result, failures = [], 0
    for c, src_text, o, it in zip(prep.cues, prep.texts, outs, prep.italic):
        body = finish(o, tgt, it)
        if not body:
            failures += 1
            continue
        nc = vtt.Cue(c.start, c.end, body, meta={"source": src_text})
        result.append(nc)
    return result, usage, failures
