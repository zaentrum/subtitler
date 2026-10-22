"""Checks of a title's made subtitles, before anyone attaches them.

What a person spot-checks is laid out for them; what can be counted is
counted:

- timing against the audio: the share of the speech Silero VAD finds that
  cues cover, cues with no speech under them (a model's invention, or a
  cue in the wrong place) and stretches of speech no cue covers;
- the transcript against a second model: windows of the audio transcribed
  again by another speech model, and the word error rate between the two
  (agreement, not accuracy: where both are wrong in the same way it says
  nothing);
- each language: cues, empty or missing translations, cues left in the
  source language, reading speed, line limits, overlaps, cue lengths;
- samples: cues spread evenly over the title, the source text beside each
  translation, for a person to read against the audio.
"""

from __future__ import annotations

import re
import unicodedata

from . import vtt, wrap
from .langs import UNSPACED


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def timing(cues: list[vtt.Cue], speech: list[tuple[float, float]], offset: float = 0.0) -> dict:
    """Cues against the speech VAD found (speech times on the WAV's timeline, cues on the title's)."""
    sp = [(s + offset, e + offset) for s, e in speech]
    total = sum(e - s for s, e in sp) or 1.0
    covered = 0.0
    for s, e in sp:
        for c in cues:
            if c.end <= s:
                continue
            if c.start >= e:
                break
            covered += _overlap(s, e, c.start, c.end)
    no_speech = []
    for c in cues:
        under = sum(_overlap(c.start, c.end, s, e) for s, e in sp if s < c.end and e > c.start)
        if under < 0.2 * max(0.3, c.duration):
            no_speech.append({"at": vtt.timestamp(c.start), "text": c.text.replace("\n", " ")[:80]})
    uncovered = []
    for s, e in sp:
        if e - s < 2.0:
            continue
        hit = sum(_overlap(s, e, c.start, c.end) for c in cues if c.start < e and c.end > s)
        if hit < 0.25 * (e - s):
            uncovered.append({"at": vtt.timestamp(s), "seconds": round(e - s, 1)})
    return {"speech_seconds": round(total, 1), "speech_covered_pct": round(100 * min(covered, total) / total, 1),
            "cues_without_speech": len(no_speech), "cues_without_speech_examples": no_speech[:8],
            "speech_without_cues": len(uncovered), "speech_without_cues_examples": uncovered[:8]}


def _words(text: str, lang: str) -> list[str]:
    t = unicodedata.normalize("NFKC", vtt.plain(text)).lower()
    t = re.sub(r"[^\w\s']", " ", t)
    if lang in UNSPACED:
        return [ch for ch in t if not ch.isspace()]
    return t.split()


def wer(ref: list[str], hyp: list[str]) -> float:
    """Word error rate of hyp against ref (Levenshtein over words)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for k, h in enumerate(hyp, 1):
            cur[k] = min(prev[k] + 1, cur[k - 1] + 1, prev[k - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def agreement(words: list[dict], windows: list[dict], lang: str, offset: float = 0.0) -> dict:
    """The transcript's words in each check window against the second model's text of it."""
    rows = []
    for w in windows:
        s, e = w["start"], w["end"]
        mine = " ".join(x["text"] for x in words if s <= x["start"] < e)
        a, b = _words(w["text"], lang), _words(mine, lang)
        rows.append({"at": vtt.timestamp(s + offset), "wer": round(wer(a, b), 3), "words": len(a),
                     "check": w["text"][:160], "transcript": mine.strip()[:160]})
    scored = [r for r in rows if r["words"] >= 5]
    mean = sum(r["wer"] for r in scored) / len(scored) if scored else None
    return {"windows": len(rows), "mean_wer": round(mean, 3) if mean is not None else None, "rows": rows}


_LATIN = re.compile(r"[A-Za-z]")
_HAN = re.compile(r"[一-鿿]")


def language(cues: list[vtt.Cue], lang: str, source: list[vtt.Cue] | None = None) -> dict:
    """Counts for one language's track."""
    st = wrap.stats(cues, lang).as_dict()
    durations = [c.duration for c in cues]
    overlaps = sum(1 for a, b in zip(cues, cues[1:]) if b.start < a.end - 0.001)
    empty = sum(1 for c in cues if not vtt.plain(c.text).strip())
    left = 0
    latin_words = 0
    if lang == "zh":
        left = sum(1 for c in cues if not _HAN.search(c.text) and _LATIN.search(c.text))
        latin_words = sum(1 for c in cues if _HAN.search(c.text) and re.search(r"[A-Za-z]{3,}", vtt.plain(c.text)))
    elif source:
        src_text = {vtt.plain(c.text).strip().lower() for c in source if len(vtt.plain(c.text).strip()) > 12}
        left = sum(1 for c in cues if vtt.plain(c.text).strip().lower() in src_text)
    st.update({"overlaps": overlaps, "empty": empty, "left_in_source_language": left, "latin_words": latin_words,
               "shortest_s": round(min(durations), 2) if durations else None,
               "longest_s": round(max(durations), 2) if durations else None,
               "lines_max": max((len([l for l in c.text.split("\n") if l.strip()]) for c in cues), default=0)})
    return st


def samples(tracks: dict[str, list[vtt.Cue]], n: int = 12, first: str = "en") -> list[dict]:
    """n moments spread evenly over the title: each track's cue at that moment."""
    base = tracks.get(first) or next(iter(tracks.values()), [])
    if not base:
        return []
    picks = [base[int(len(base) * (i + 0.5) / n)] for i in range(min(n, len(base)))]
    out = []
    for c in picks:
        mid = (c.start + c.end) / 2
        row = {"at": vtt.timestamp(c.start), "until": vtt.timestamp(c.end)}
        for tag, cs in tracks.items():
            hit = next((x for x in cs if x.start <= mid < x.end), None)
            if hit is None:  # the translated cue nearest in time
                hit = min(cs, key=lambda x: abs((x.start + x.end) / 2 - mid), default=None)
            row[tag] = hit.text.replace("\n", " / ") if hit else ""
        out.append(row)
    return out


def alignment(track: list[vtt.Cue], source: list[vtt.Cue], window: int = 16, step: int = 8) -> dict:
    """Stretches where a translation looks shifted against its source: windows of `window` cues
    (paired by start time, as translations keep their source cue's start) whose lengths follow the
    source one or two cues away far better than their own (translate.llm.misaligned's test)."""
    from .translate.llm import misaligned

    by_start = {round(c.start, 2): c for c in source}
    pairs = [(by_start[round(c.start, 2)], c) for c in track if round(c.start, 2) in by_start]
    unspaced = any(_HAN.search(c.text) for c in track[:50])
    flagged = []
    for k in range(0, max(0, len(pairs) - window + 1), step):
        seg = pairs[k:k + window]
        if misaligned([vtt.plain(s.text) for s, _ in seg], [vtt.plain(t.text) for _, t in seg], unspaced):
            flagged.append(vtt.timestamp(seg[0][0].start))
    return {"paired": len(pairs), "of": len(track), "shifted_windows": len(flagged), "at": flagged[:12]}
