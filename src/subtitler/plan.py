"""What to do per instance language, in ADR-0014's order.

First the text copies (OCR, ocr.py): an image subtitle (PGS, VobSub) in one
of the languages gets one where the title has no text subtitle of that
language and kind (full, SDH, forced; signs for a sparse track), the
fullest of a language and kind; a forced one too where a full text subtitle
of the language is there. A copy is the title's own subtitle in text: it
counts as present, and as a source for the steps after it.

Then for each language the instance names:
1. keep: the source brings a complete text subtitle in it (and in its
   script, where the language names one);
2. recognize: the text copy of a complete image subtitle in it;
3. convert: the same language in the other script (Traditional <-> Simplified
   Chinese) as text, a copy's too: a dictionary conversion;
4. translate: the best complete text subtitle the title has, copies among
   them: the original language first, then English, then the other languages
   (the instance's languages, then widely translated ones), a text subtitle
   before a copy of the same rank;
5. transcribe: the title has no complete text subtitle at all: speech
   recognition of the main audio, which is the subtitle when the audio is in
   the language, and the source of 4 otherwise.

Forced and commentary tracks are no subtitle of a language. Without OCR
(ocr=False) image tracks a client draws (PGS) count as present but are no
source. A Cantonese track is not Chinese of another script. A Chinese copy is
in the script reading finds (ocr.read_pictures), whatever its track's title
names: a copy titled in one script is a candidate of both, the language's own
where it reads in its script, a source to convert where it reads in the other.

Each language keeps its ordered candidates: a viewer's report on a made
subtitle excludes its source, and the next candidate makes it again; a copy
that is not attached (ocr.py: its confidence too low) leaves its candidates
to the next ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .langs import Lang, iso1
from .package import Package, Subtitle

# Widely translated languages, a source's rank after the instance's own.
_SOURCE_PREFERENCE = ["fr", "es", "de", "it", "pt", "nl", "ja", "ko", "zh", "ru", "sv", "da", "no", "pl"]

TRANSCRIPT = "transcript"
COPY = "ocr:"  # a text copy's source id: "ocr:<subtitle id>"


@dataclass
class Candidate:
    action: str  # keep | recognize | convert | translate | transcribe
    source: str  # subtitle id, a text copy's ("ocr:<subtitle id>"), or "audio:<id>"
    source_language: str | None
    detail: dict = field(default_factory=dict)

    @property
    def origin(self) -> str | None:
        return {"recognize": "recognized", "convert": "converted", "translate": "translated",
                "transcribe": "transcribed"}.get(self.action)

    def describe(self) -> str:
        if self.action == "keep":
            return f"keep {self.source}"
        if self.action == "recognize":
            return f"recognize {self.source.removeprefix(COPY)} ({self.detail.get('format')}, OCR)"
        if self.action == "convert":
            return f"convert {self.source} ({self.detail.get('config')})"
        if self.action == "translate":
            src = "the transcript" if self.source == TRANSCRIPT else self.source
            return f"translate {src} ({self.source_language})"
        return f"transcribe {self.source} ({self.source_language or 'detect'})"


@dataclass
class LanguagePlan:
    language: Lang
    candidates: list[Candidate]
    excluded: list[str] = field(default_factory=list)

    @property
    def choice(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def needs_transcript(self) -> bool:
        c = self.choice
        return c is not None and (c.action == "transcribe" or (c.action == "translate" and c.source == TRANSCRIPT))


def _matches(sub: Subtitle, lang: Lang) -> bool:
    if sub.language != lang.code:
        return False
    if lang.script is None:
        return True
    return sub.script == lang.script


def kind(sub: Subtitle) -> str:
    """What a subtitle is for: forced, signs (a sparse track, signs and songs), sdh or full."""
    if sub.forced or sub.purpose == "forced":
        return "forced"
    if sub.partial:
        return "signs"
    return "sdh" if sub.is_sdh else "full"


@dataclass
class Copy:
    """A text copy of an image subtitle to make (OCR; its origin `recognized`)."""

    sub: Subtitle
    lang: Lang  # Chinese with the script its track's title names (reading may find the other), else none
    kind: str  # full | sdh | forced | signs

    @property
    def id(self) -> str:
        return COPY + self.sub.id

    @property
    def counts(self) -> bool:
        """A complete subtitle of its language, and a source for the others."""
        return self.kind in ("full", "sdh") and not self.sub.bilingual


def copies(pkg: Package, languages: list[Lang]) -> list[Copy]:
    """The image subtitles to make a text copy of (OCR): each in one of the languages (Chinese of
    either script, not Cantonese), not commentary, of a kind the title has no text subtitle of in
    that language (and script); of several of a language and kind the fullest. A forced one is
    copied where a full text subtitle is there, as it is a kind of its own."""
    codes = {l.code for l in languages}
    have = {(s.language, s.script if s.language == "zh" else None, kind(s)) for s in pkg.subtitles
            if s.exists and s.is_text and s.purpose != "commentary"}
    out: dict[tuple, Copy] = {}
    for s in sorted((s for s in pkg.subtitles if s.exists and s.is_bitmap), key=lambda s: -(s.cues or 0)):
        if s.purpose == "commentary" or s.language not in codes or s.cues == 0:
            continue
        k = kind(s)
        script = s.script if s.language == "zh" else None
        if s.language == "zh":
            covered = any(code == "zh" and kk == k and (script is None or sc in (script, None))
                          for code, sc, kk in have)
        else:
            covered = (s.language, None, k) in have
        if covered or (s.language, script, k) in out:
            continue
        out[(s.language, script, k)] = Copy(s, Lang(s.language, script), k)
    return list(out.values())


# Under source_order="english" a complete English track goes first: the spike
# found translation from English as good as or better than from the original.
# Complete means it leaves no scene of the original-language track without
# English: no run of MAX_GAP_RUN original cues without an English cue near
# them, and under MAX_UNCOVERED of them overall. An English subtitle made for
# viewers who hear the film's English lines leaves those scenes out.
MAX_GAP_RUN = 8
MAX_UNCOVERED = 0.08


def coverage_gaps(a: list, b: list, tolerance: float = 1.0) -> tuple[int, float]:
    """(longest run of a's cues without a b cue within `tolerance` s, share of such cues)."""
    from bisect import bisect_left

    spans = sorted((c.start, c.end) for c in b)
    starts = [s for s, _ in spans]
    run = longest = missing = 0
    for c in a:
        i = bisect_left(starts, c.start - 30)
        covered = any(s <= c.end + tolerance and e >= c.start - tolerance for s, e in spans[i:i + 60])
        run = 0 if covered else run + 1
        missing += not covered
        longest = max(longest, run)
    return longest, missing / max(1, len(a))


def _english_complete(pkg: Package, english: Subtitle, original: Subtitle) -> bool:
    from . import vtt

    try:
        run, share = coverage_gaps(vtt.read(pkg.root / original.path), vtt.read(pkg.root / english.path))
    except OSError:
        return False
    return run < MAX_GAP_RUN and share < MAX_UNCOVERED


def _source_rank(sub: Subtitle, original: str | None, langs: list[Lang], english_first: bool = False) -> tuple:
    instance = [l.code for l in langs]
    lang = sub.language or ""
    if english_first and lang == "en":
        tier = -1
    elif original and lang == original:
        tier = 0
    elif lang == "en":
        tier = 1
    elif lang in instance:
        tier = 2 + instance.index(lang)
    elif lang in _SOURCE_PREFERENCE:
        tier = 20 + _SOURCE_PREFERENCE.index(lang)
    elif lang == "yue" or not lang:
        tier = 90
    else:
        tier = 50
    return (tier, sub.is_sdh, not sub.visible, -(sub.cues or 0))


def _recognize(c: Copy, lang: Lang) -> Candidate:
    return Candidate("recognize", c.id, lang.code, {"format": c.sub.format, "kind": c.kind, "script": c.lang.script})


def _convert(source: str, script: str | None, region: str | None, lang: Lang) -> Candidate:
    if lang.script == "Hans":
        config = "hk2s" if region == "HK" else "t2s"
    else:
        config = "s2hk" if region == "HK" else "s2t"
    return Candidate("convert", source, "zh", {"config": config, "from_script": script})


def plan(pkg: Package, languages: list[Lang], original_language: str | None = None,
         exclude: tuple[str, ...] | list[str] = (), source_order: str = "original", ocr: bool = True,
         made: list[Copy] | None = None) -> list[LanguagePlan]:
    """The candidates per language, best first; `exclude` drops sources (reported ones).

    source_order "original" is ADR-0014's: the original language, then English.
    "english" puts a complete English track first (ENGLISH_COMPLETE).
    ocr: the text copies of image subtitles (`made`, else copies()) are made first and count as
    the title's own text; without it an image subtitle a client draws counts as present.
    """
    audio = pkg.main_audio
    audio_lang = audio.language if audio else None
    original = iso1(original_language) if original_language else audio_lang
    subs = [s for s in pkg.subtitles if s.exists and s.is_full]
    text_sources = [s for s in subs if s.is_text and not s.bilingual]
    excl = set(exclude)
    made = [c for c in (made if made is not None else copies(pkg, languages) if ocr else [])
            if c.counts and c.id not in excl]
    english_first = False
    if source_order == "english":
        fullest = lambda code: max((s for s in text_sources if s.language == code), key=lambda s: s.cues or 0,
                                   default=None)
        en = fullest("en")
        orig = fullest(original) if original and original != "en" else None
        english_first = en is not None and (orig is None or _english_complete(pkg, en, orig))

    plans = []
    for lang in languages:
        kept = [s for s in subs if (s.is_text or (not ocr and s.shown)) and _matches(s, lang)]
        if kept:
            kept.sort(key=lambda s: (not s.is_text, s.is_sdh, not s.visible, -(s.cues or 0)))
            plans.append(LanguagePlan(lang, [Candidate("keep", kept[0].id, kept[0].language)]))
            continue
        cands: list[Candidate] = []
        # 2. The text copy of an image subtitle in it: a Chinese one whose script
        # its title does not tell is the language's if reading finds it in it.
        own = [c for c in made if c.lang.code == lang.code
               and (lang.script is None or c.lang.script in (lang.script, None))]
        for c in sorted(own, key=lambda c: (c.kind == "sdh", c.lang.script is None)):
            cands.append(_recognize(c, lang))
        # 3. The same language in the other script. A Chinese text track whose
        # script is unknown converts too: t2s leaves Simplified text as it is.
        if lang.code == "zh" and lang.script in ("Hans", "Hant"):
            other = "Hant" if lang.script == "Hans" else "Hans"
            chinese = [(s.id, s.script, s.region, s.is_sdh) for s in text_sources if s.language == "zh"]
            chinese += [(c.id, c.lang.script, c.sub.region, c.kind == "sdh") for c in made if c.lang.code == "zh"]
            for sid, script, region, _ in sorted(chinese, key=lambda x: (x[1] is None, x[3], x[0].startswith(COPY))):
                if script in (other, None):
                    cands.append(_convert(sid, script, region, lang))
            # Reading tells a copy's script, its title may be wrong (a Simplified track
            # labelled Traditional): one titled in this script converts where it reads
            # in the other, one titled in the other is this language's where it reads in it.
            titled = [c for c in made if c.lang.code == "zh" and c.lang.script in ("Hans", "Hant")]
            for c in titled:
                if c.lang.script == lang.script:
                    cands.append(_convert(c.id, c.lang.script, c.sub.region, lang))
            for c in titled:
                if c.lang.script == other:
                    cands.append(_recognize(c, lang))
        # 4. A translation of the best text subtitle in another language, a copy
        # after a text subtitle of the same rank.
        ranked = [(_source_rank(s, original, languages, english_first), False, s.id, s.language, s.is_sdh)
                  for s in text_sources if s.language != lang.code]
        ranked += [(_source_rank(c.sub, original, languages, english_first), True, c.id, c.lang.code, c.kind == "sdh")
                   for c in made if c.lang.code != lang.code]
        for rank, _copy, sid, code, sdh in sorted(ranked, key=lambda x: (x[0][0], x[1], *x[0][1:])):
            cands.append(Candidate("translate", sid, code, {"sdh": sdh}))
        # 4. Speech recognition of the main audio: the last candidate, so it is
        # the choice only when the title has no text subtitle to work from.
        if audio is not None:
            if audio_lang == lang.code:
                cands.append(Candidate("transcribe", f"audio:{audio.id}", audio_lang,
                                       {"script": lang.script} if lang.script else {}))
            else:
                cands.append(Candidate("translate", TRANSCRIPT, audio_lang, {"via": f"audio:{audio.id}"}))
        plans.append(LanguagePlan(lang, [c for c in cands if c.source not in excl], list(exclude)))
    return plans


def transcription_needed(plans: list[LanguagePlan]) -> bool:
    return any(p.needs_transcript for p in plans)
