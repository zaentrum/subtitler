"""Timed cues from speech recognition words.

Whisper gives segments of up to 30 s with word timestamps. A subtitle cue is
at most two lines (84 characters, 32 in Chinese) and 7 s, holds one sentence
where it fits, and is cut where a subtitler would cut:

1. Words are cleaned: a word Whisper pinned far from the words it belongs
   to (an orphan before a long gap, typically the first word after silence)
   is moved next to them.
2. The word stream is cut hard at pauses of 1 s and more, then into
   sentences at sentence-final punctuation.
3. A sentence too long for one cue is cut near its middle, preferring a
   comma, a pause, then a conjunction; recursively.
4. Very short sentences next to each other share a cue, one per line.
5. Cue ends get a lead-out (the text stays a little after the voice), never
   into the next cue; a cue lasts at least 1 s where the gap allows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import vtt, wrap
from .langs import UNSPACED


@dataclass
class Word:
    start: float
    end: float
    text: str  # as the model spells it, with its leading space in spaced languages
    probability: float = 1.0


_SENTENCE_END = re.compile(r"[.?!…。？！]['\"”’」』)]*$")
_CLAUSE_END = re.compile(r"[,;:，、；：—–-]['\"”’]*$")
# Abbreviations whose full stop ends no sentence.
_ABBREV = {"mr.", "mrs.", "ms.", "dr.", "st.", "jr.", "sr.", "vs.", "prof.", "gen.", "col.", "sgt.", "lt.",
           "capt.", "mt.", "no.", "u.s.", "u.k.", "d.c.", "l.a.", "a.m.", "p.m.", "e.g.", "i.e.",
           "hr.", "fr.", "frl.", "nr.", "bzw.", "z.b.", "u.a.", "ca.", "sig.", "dott."}
# Hesitations a dialogue subtitle leaves out.
_FILLERS = {
    "en": {"uh", "um", "er", "erm", "hmm", "mm", "mm-hmm", "mhm", "uh-huh", "uh-uh", "hm"},
    "de": {"äh", "ähm", "öhm", "hm", "hmm", "mhm", "mm"},
    "it": {"ehm", "mmh", "uhm"},
}


def _ends_sentence(word: str) -> bool:
    w = word.strip()
    return bool(_SENTENCE_END.search(w)) and w.lower() not in _ABBREV


def drop_fillers(words: list[Word], lang: str) -> list[Word]:
    """Hesitations out; a sentence that started with one starts with the next word, capitalised."""
    fillers = _FILLERS.get(lang)
    if not fillers:
        return words
    out: list[Word] = []
    cap = False
    for w in words:
        bare = re.sub(r"[^\w-]", "", w.text.strip().lower()).strip("-")  # Whisper splits "Mm-hmm" in two
        if bare in fillers:
            if not out or _ends_sentence(out[-1].text):
                cap = True
            if _ends_sentence(w.text) and out and not _ends_sentence(out[-1].text):
                out[-1] = Word(out[-1].start, out[-1].end, out[-1].text.rstrip(",;") + w.text.strip()[-1],
                               out[-1].probability)
            continue
        if cap:
            t = w.text
            lead = len(t) - len(t.lstrip())
            w = Word(w.start, w.end, t[:lead] + t[lead:lead + 1].upper() + t[lead + 1:], w.probability)
            cap = False
        out.append(w)
    return out
_CONJ = {
    "en": {"and", "but", "or", "so", "because", "that", "which", "who", "when", "if", "while", "where",
           "then", "until", "before", "after", "since", "though", "although", "unless"},
    "de": {"und", "aber", "oder", "denn", "weil", "dass", "wenn", "als", "ob", "damit", "sondern",
           "bevor", "nachdem", "obwohl", "während", "die", "der", "das", "wo", "was"},
    "it": {"e", "ma", "o", "perché", "che", "quando", "se", "mentre", "dove", "poi"},
    "fr": {"et", "mais", "ou", "parce", "que", "qui", "quand", "si", "pendant", "où", "donc"},
    "es": {"y", "pero", "o", "porque", "que", "cuando", "si", "mientras", "donde"},
}

# Phrases Whisper is known to produce over music or silence (training-data
# credits); a cue that is nothing but one of them is dropped.
_HALLUCINATIONS = re.compile(
    r"^(?:thanks? (?:you )?for watching[.!]?|subtitles? by.*|amara\.org.*|"
    r"untertitel (?:im auftrag|von|der amara).*|untertitelung.*|"
    r"sous-titr.*|字幕.*(?:提供|制作).*|请不吝点赞.*|明镜.*点点.*|ご視聴ありがとうございました.*|"
    r"시청해 ?주셔서 감사합니다.*|\.+|[♪♫\s]+)$",
    re.IGNORECASE,
)


def _join(words: list[Word], lang: str) -> str:
    if lang in UNSPACED:
        return "".join(w.text.strip() for w in words).strip()
    return "".join(w.text for w in words).strip()


def _w(words: list[Word], lang: str) -> int:
    return wrap.width(_join(words, lang))


def clean_words(words: list[Word], orphan_gap: float = 2.0) -> list[Word]:
    """Drop empty words; move an orphan next to the words it belongs to."""
    ws = [Word(w.start, w.end, w.text, w.probability) for w in words if w.text.strip()]
    for i in range(len(ws) - 1):
        w, nxt = ws[i], ws[i + 1]
        prev_gap = w.start - ws[i - 1].end if i > 0 else float("inf")
        gap = nxt.start - w.end
        # A lone word that ends no sentence, far from what came before and
        # from what follows: Whisper pinned it to the start of the stretch.
        if gap > orphan_gap and prev_gap > orphan_gap and not _ends_sentence(w.text):
            d = min(0.6, max(0.2, w.end - w.start))
            w.end = nxt.start - 0.05
            w.start = w.end - d
        if w.end - w.start > 3.0:  # a word does not last 3 s
            w.start = w.end - 1.0
    return ws


def _split_score(words: list[Word], k: int, lang: str) -> float:
    """Cost of cutting before words[k]: balance, then the kind of break."""
    a, b = _w(words[:k], lang), _w(words[k:], lang)
    cost = abs(a - b) / max(1, a + b) * 100
    prev = words[k - 1].text.strip()
    gap = words[k].start - words[k - 1].end
    if _ends_sentence(prev):
        cost -= 40
    elif _CLAUSE_END.search(prev):
        cost -= 30
    cost -= min(25.0, gap * 50)
    if words[k].text.strip().lower().strip(",.") in _CONJ.get(lang, set()):
        cost -= 12
    return cost


def _text_fits(text: str, lang: str) -> bool:
    lim = wrap.limits(lang)
    lines = text.split("\n")
    return len(lines) <= lim.lines and all(wrap.width(l) <= lim.chars_per_line for l in lines)


def _cue_text(parts: list[str], lang: str) -> str:
    """Two sentences that each fit a line get a line each; otherwise one wrapped text."""
    cpl = wrap.limits(lang).chars_per_line
    if len(parts) == 2 and all(wrap.width(p) <= cpl for p in parts):
        return "\n".join(parts)
    return wrap.rewrap(("" if lang in UNSPACED else " ").join(parts), lang)


def _fits(words: list[Word], lang: str, max_chars: int, max_duration: float) -> bool:
    if _w(words, lang) > max_chars or words[-1].end - words[0].start > max_duration:
        return False
    return _text_fits(wrap.rewrap(_join(words, lang), lang), lang)


def _split_long(words: list[Word], lang: str, max_chars: int, max_duration: float) -> list[list[Word]]:
    if len(words) < 2 or _fits(words, lang, max_chars, max_duration):
        return [words]
    k = min(range(1, len(words)), key=lambda i: _split_score(words, i, lang))
    return (_split_long(words[:k], lang, max_chars, max_duration)
            + _split_long(words[k:], lang, max_chars, max_duration))


def _chunks(words: list[Word], hard_pause: float) -> list[list[Word]]:
    out, cur = [], []
    for w in words:
        if cur and w.start - cur[-1].end >= hard_pause:
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def _sentences(words: list[Word]) -> list[list[Word]]:
    out, cur = [], []
    for w in words:
        cur.append(w)
        if _ends_sentence(w.text):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def build(words: list[Word], lang: str, *, max_duration: float = 7.0, min_duration: float = 1.0,
          hard_pause: float = 1.0, lead_out: float = 0.5, min_gap: float = 0.083) -> list[vtt.Cue]:
    lim = wrap.limits(lang)
    max_chars = lim.chars_per_cue
    units: list[list[Word]] = []
    for chunk in _chunks(drop_fillers(clean_words(words), lang), hard_pause):
        for sent in _sentences(chunk):
            units += _split_long(sent, lang, max_chars, max_duration)

    # Short sentences next to each other share a cue while it still fits.
    def short(u: list[Word]) -> bool:
        return _w(u, lang) <= lim.chars_per_line // 2 or u[-1].end - u[0].start < 1.2

    groups: list[list[list[Word]]] = []
    for u in units:
        if groups:
            last = groups[-1]
            ws = [w for part in last for w in part] + u
            if (short(last[-1]) and short(u) and u[0].start - last[-1][-1].end < 0.5
                    and ws[-1].end - ws[0].start <= 4.0
                    and _text_fits(_cue_text([_join(p, lang) for p in last] + [_join(u, lang)], lang), lang)):
                last.append(u)
                continue
        groups.append([u])

    cues: list[vtt.Cue] = []
    for g in groups:
        texts = [collapse_repeats(_join(part, lang), lang) for part in g]
        texts = [t for t in texts if t and not _HALLUCINATIONS.match(t)]
        if not texts:
            continue
        ws = [w for part in g for w in part]
        prob = sum(w.probability for w in ws) / len(ws)
        cues.append(vtt.Cue(ws[0].start, ws[-1].end, _cue_text(texts, lang), meta={"p": round(prob, 3)}))

    cues = drop_loops(cues)
    for i, c in enumerate(cues):
        nxt = cues[i + 1].start if i + 1 < len(cues) else None
        end = max(c.end + lead_out, c.start + min_duration)
        if nxt is not None:
            end = min(end, nxt - min_gap)
        c.end = max(end, c.start + 0.3)
    return cues


_RUN_UNSPACED = re.compile(r"(.{1,6}?)\1{4,}")
_RUN_SPACED = re.compile(r"\b((?:\w+[,.!?]?\s+){1,3}?)\1{4,}", re.UNICODE)


def collapse_repeats(text: str, lang: str) -> str:
    """A unit repeated five times and more inside a cue (Whisper looping) is kept twice."""
    if lang in UNSPACED:
        return _RUN_UNSPACED.sub(lambda m: m.group(1) * 2, text)
    return _RUN_SPACED.sub(lambda m: m.group(1) * 2, text + " ").strip()


def drop_loops(cues: list[vtt.Cue], repeats: int = 3) -> list[vtt.Cue]:
    """Whisper's failure mode: the same line again and again. Keep its first two."""
    out: list[vtt.Cue] = []
    run = 0
    for c in cues:
        if out and _norm(c.text) == _norm(out[-1].text):
            run += 1
            if run >= repeats - 1:
                continue
        else:
            run = 0
        out.append(c)
    return out


def _norm(s: str) -> str:
    return re.sub(r"\W+", "", s.lower())
