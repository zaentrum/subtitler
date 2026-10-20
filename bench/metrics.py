"""Scores: speech recognition against a title's own subtitle, translation against a human one.

Speech recognition. A subtitle is not a transcript: it drops fillers,
repetitions and asides, and rephrases to fit the reading speed. Against it,
the word error rate counts every word the recogniser heard but the subtitle
left out (insertions) and every rephrasing (substitutions). Three numbers:
- WER: (S + D + I) / N, the strict measure, an upper bound on the true error;
- SD rate: (S + D) / N, insertions left out: how much of what the subtitle
  says the recognition missed or got wrong; condensation costs it nothing,
  rephrasing still does;
- local recall: the share of the subtitle's words found among the words
  recognised within 2 s of the cue, in any order: content and timing
  together, the loosest.
Chinese and Japanese are scored by characters (CER), Chinese in Simplified.

Translation. Human subtitles in two languages are cut differently. Units of
alignment are the connected groups of cues that overlap in time (hypothesis
cues keep the source's times); chrF and BLEU are computed over the units
that have both sides.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass

from subtitler import script, sdh, vtt
from subtitler.cues import Word
from subtitler.langs import UNSPACED

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)


def _normalizer(lang: str):
    from whisper_normalizer.basic import BasicTextNormalizer
    from whisper_normalizer.english import EnglishTextNormalizer

    if lang == "en":
        return EnglishTextNormalizer()
    return BasicTextNormalizer()


_NORM_CACHE: dict = {}


def tokens(text: str, lang: str) -> list[str]:
    """Normalised words (characters for unspaced languages)."""
    t = unicodedata.normalize("NFKC", vtt.plain(text)).replace("’", "'").replace("\n", " ")
    if lang in UNSPACED:
        if lang in ("zh", "yue"):
            t = script.convert(t, "t2s")
        return [ch for ch in t if ch.isalnum()]
    n = _NORM_CACHE.get(lang)
    if n is None:
        n = _NORM_CACHE[lang] = _normalizer(lang)
    return [w for w in n(t).split() if w]


def reference_text(cue: vtt.Cue) -> str:
    """A subtitle cue as speech: SDH annotations, speaker labels and lyrics out."""
    return sdh.clean_text(cue.text, drop_lyrics=True).replace("\n", " ")


# --- speech recognition --------------------------------------------------------

@dataclass
class AsrScore:
    n_ref: int
    wer: float
    sd_rate: float
    sub: int
    dele: int
    ins: int
    local_recall: float
    coverage: float  # share of the subtitle's cue time that recognised cues cover
    precision_time: float  # share of recognised cue time inside subtitle cues
    median_start_offset: float  # recognised cue start minus subtitle cue start, s

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def _intervals_union(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _overlap_len(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            tot += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def score_asr(ref_cues: list[vtt.Cue], words: list[Word], hyp_cues: list[vtt.Cue], lang: str,
              start: float, end: float, tolerance: float = 2.0) -> AsrScore:
    import jiwer

    ref = [c for c in vtt.window(ref_cues, start, end)]
    ref_tok = [(c, tokens(reference_text(c), lang)) for c in ref]
    ref_tok = [(c, t) for c, t in ref_tok if t]
    hyp_words = [w for w in words if start <= (w.start + w.end) / 2 < end]
    hyp_tok: list[tuple[float, str]] = []
    for w in hyp_words:
        for t in tokens(w.text, lang):
            hyp_tok.append(((w.start + w.end) / 2, t))
    r_all = [t for _, ts in ref_tok for t in ts]
    h_all = [t for _, t in hyp_tok]
    sep = "" if lang in UNSPACED else " "
    if lang in UNSPACED:
        out = jiwer.process_characters(sep.join(r_all), sep.join(h_all))
    else:
        out = jiwer.process_words(" ".join(r_all), " ".join(h_all))
    n = len(r_all) or 1
    s, d, i = out.substitutions, out.deletions, out.insertions

    # Local recall: each subtitle cue's words among the recognised words near it.
    times = [t for t, _ in hyp_tok]
    used = [False] * len(hyp_tok)
    found = 0
    for c, ts in ref_tok:
        lo, hi = bisect_left(times, c.start - tolerance), bisect_right(times, c.end + tolerance)
        pool: dict[str, list[int]] = {}
        for k in range(lo, hi):
            if not used[k]:
                pool.setdefault(hyp_tok[k][1], []).append(k)
        for t in ts:
            if pool.get(t):
                used[pool[t].pop(0)] = True
                found += 1

    # Timing: the time covered by cues on either side, and the start offsets.
    r_iv = _intervals_union([(c.start, c.end) for c, _ in ref_tok])
    h_cues = vtt.window(hyp_cues, start, end)
    h_iv = _intervals_union([(c.start, c.end) for c in h_cues])
    ov = _overlap_len(r_iv, h_iv)
    r_len = sum(e - s for s, e in r_iv) or 1.0
    h_len = sum(e - s for s, e in h_iv) or 1.0
    offsets = []
    for c, _ in ref_tok:
        best = None
        for h in h_cues:
            o = min(c.end, h.end) - max(c.start, h.start)
            if o > 0 and (best is None or o > best[0]):
                best = (o, h.start - c.start)
        if best:
            offsets.append(best[1])
    offsets.sort()
    med = offsets[len(offsets) // 2] if offsets else 0.0
    return AsrScore(n_ref=len(r_all), wer=(s + d + i) / n, sd_rate=(s + d) / n, sub=s, dele=d, ins=i,
                    local_recall=found / n, coverage=ov / r_len, precision_time=ov / h_len,
                    median_start_offset=med)


# --- translation ---------------------------------------------------------------

@dataclass
class Unit:
    hyp: list[int]
    ref: list[int]


def align(hyp: list[vtt.Cue], ref: list[vtt.Cue], min_share: float = 0.2) -> list[Unit]:
    """Connected groups of cues overlapping in time (by at least min_share of the shorter one)."""
    parent = list(range(len(hyp) + len(ref)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    r_sorted = sorted(range(len(ref)), key=lambda k: ref[k].start)
    starts = [ref[k].start for k in r_sorted]
    for i, h in enumerate(hyp):
        lo = bisect_left(starts, h.start - 30)
        for pos in range(lo, len(r_sorted)):
            k = r_sorted[pos]
            r = ref[k]
            if r.start > h.end:
                break
            o = min(h.end, r.end) - max(h.start, r.start)
            if o > min_share * max(1e-3, min(h.duration, r.duration)):
                a, b = find(i), find(len(hyp) + k)
                if a != b:
                    parent[a] = b
    groups: dict[int, Unit] = {}
    for i in range(len(hyp)):
        groups.setdefault(find(i), Unit([], [])).hyp.append(i)
    for k in range(len(ref)):
        groups.setdefault(find(len(hyp) + k), Unit([], [])).ref.append(k)
    return sorted(groups.values(), key=lambda u: min([hyp[i].start for i in u.hyp] + [ref[k].start for k in u.ref]))


def unit_text(cues: list[vtt.Cue], idx: list[int], lang: str) -> str:
    parts = [vtt.plain(cues[i].text).replace("\n", " ").strip() for i in sorted(idx, key=lambda i: cues[i].start)]
    joiner = "" if lang in ("zh", "yue", "ja") else " "
    t = joiner.join(p for p in parts if p)
    if lang in ("zh", "yue"):
        t = script.convert(t, "t2s")
    return re.sub(r"\s{2,}", " ", t).strip()


@dataclass
class MtScore:
    units: int
    hyp_only: int
    ref_only: int
    chrf: float
    bleu: float
    len_ratio: float  # hypothesis characters / reference characters

    def as_dict(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def score_mt(hyp: list[vtt.Cue], ref: list[vtt.Cue], lang: str) -> tuple[MtScore, list[tuple[str, str, Unit]]]:
    import sacrebleu

    units = align(hyp, ref)
    pairs = [(unit_text(hyp, u.hyp, lang), unit_text(ref, u.ref, lang), u) for u in units if u.hyp and u.ref]
    hyps = [h for h, _, _ in pairs]
    refs = [r for _, r, _ in pairs]
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score if pairs else 0.0
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh" if lang in ("zh", "yue") else "13a").score if pairs else 0.0
    hc = sum(len(h.replace(" ", "")) for h in hyps)
    rc = sum(len(r.replace(" ", "")) for r in refs) or 1
    return (MtScore(units=len(pairs), hyp_only=sum(1 for u in units if u.hyp and not u.ref),
                    ref_only=sum(1 for u in units if u.ref and not u.hyp), chrf=chrf, bleu=bleu,
                    len_ratio=hc / rc), pairs)


def displacement(hyp: list[vtt.Cue], ref: list[vtt.Cue], lang: str, margin: float = 10.0) -> dict:
    """How often a translated cue reads like the human subtitle of its neighbour rather than its own.

    For each cue: sentence chrF against the human text over its own time,
    over the next cue's time and over the previous cue's. "late" counts the
    cues that match the next cue's text better by `margin` points (the
    translation runs one cue behind the speech), "early" the previous one's.
    An aligned translation sits at the noise floor; the two together say how
    much text a viewer sees at the wrong time.
    """
    import sacrebleu

    def text_at(a: float, b: float) -> str:
        return unit_text(ref, [k for k, r in enumerate(ref) if min(r.end, b) - max(r.start, a) > 0.15], lang)

    late = early = n = 0
    for i in range(1, len(hyp) - 1):
        h = unit_text(hyp, [i], lang)
        own = text_at(hyp[i].start, hyp[i].end)
        nxt = text_at(hyp[i + 1].start, hyp[i + 1].end)
        prv = text_at(hyp[i - 1].start, hyp[i - 1].end)
        if not h or not own or own in (nxt, prv):
            continue
        n += 1
        s_own = sacrebleu.sentence_chrf(h, [own]).score
        if nxt and sacrebleu.sentence_chrf(h, [nxt]).score > s_own + margin:
            late += 1
        if prv and sacrebleu.sentence_chrf(h, [prv]).score > s_own + margin:
            early += 1
    return {"late_pct": round(100 * late / max(1, n), 1), "early_pct": round(100 * early / max(1, n), 1),
            "cues_checked": n}


_DIGITS = re.compile(r"\d+")


def anchors(text: str) -> set[str]:
    """What survives translation unchanged: question marks and numbers."""
    t = unicodedata.normalize("NFKC", text)
    out = {"?"} if "?" in t else set()
    return out | set(_DIGITS.findall(t))


def anchor_agreement(src: list[vtt.Cue], hyp: list[vtt.Cue]) -> dict:
    """Share of question marks and numbers found in the translated cue of the source cue that has them.

    Over the time of each translated cue, the source cues there: an anchor in
    the source should be in the translation at the same time, and one in the
    translation should come from the source there. A translation that runs
    a cue behind loses both.
    """
    hit = total_src = total_hyp = hit_hyp = 0
    for h in hyp:
        s_text = " ".join(vtt.plain(s.text) for s in src if min(s.end, h.end) - max(s.start, h.start) > 0.1)
        a_src, a_hyp = anchors(s_text), anchors(vtt.plain(h.text))
        total_src += len(a_src)
        hit += len(a_src & a_hyp)
        total_hyp += len(a_hyp)
        hit_hyp += len(a_hyp & a_src)
    return {"anchor_recall_pct": round(100 * hit / max(1, total_src), 1),
            "anchor_precision_pct": round(100 * hit_hyp / max(1, total_hyp), 1), "anchors": total_src}


def char_counter(text: str) -> Counter:
    return Counter(ch for ch in text if not ch.isspace())
