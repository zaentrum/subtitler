"""What to do per instance language, in ADR-0014's order.

For each language the instance names:
1. keep: the source brings a complete subtitle in it (and in its script,
   where the language names one) that a viewer can turn on;
2. convert: the same language in the other script (Traditional <-> Simplified
   Chinese) as text: a dictionary conversion;
3. translate: the best complete text subtitle the title has: the original
   language first, then English, then the other languages (the instance's
   languages, then widely translated ones);
4. transcribe: the title has no complete text subtitle at all: speech
   recognition of the main audio, which is the subtitle when the audio is in
   the language, and the source of 3 otherwise.

Forced and commentary tracks are no subtitle of a language. Image tracks a
client draws (PGS) count as present but are no source: their text would
need OCR. A Cantonese track is not Chinese of another script.

Each language keeps its ordered candidates: a viewer's report on a made
subtitle excludes its source, and the next candidate makes it again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .langs import Lang, iso1
from .package import Package, Subtitle

# Widely translated languages, a source's rank after the instance's own.
_SOURCE_PREFERENCE = ["fr", "es", "de", "it", "pt", "nl", "ja", "ko", "zh", "ru", "sv", "da", "no", "pl"]

TRANSCRIPT = "transcript"


@dataclass
class Candidate:
    action: str  # keep | convert | translate | transcribe
    source: str  # subtitle id, or "audio:<id>"
    source_language: str | None
    detail: dict = field(default_factory=dict)

    @property
    def origin(self) -> str | None:
        return {"convert": "converted", "translate": "translated", "transcribe": "transcribed"}.get(self.action)

    def describe(self) -> str:
        if self.action == "keep":
            return f"keep {self.source}"
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


def _source_rank(sub: Subtitle, original: str | None, langs: list[Lang]) -> tuple:
    instance = [l.code for l in langs]
    lang = sub.language or ""
    if original and lang == original:
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


def plan(pkg: Package, languages: list[Lang], original_language: str | None = None,
         exclude: tuple[str, ...] | list[str] = ()) -> list[LanguagePlan]:
    """The candidates per language, best first; `exclude` drops sources (reported ones)."""
    audio = pkg.main_audio
    audio_lang = audio.language if audio else None
    original = iso1(original_language) if original_language else audio_lang
    subs = [s for s in pkg.subtitles if s.exists and s.is_full]
    text_sources = [s for s in subs if s.is_text and not s.bilingual]

    plans = []
    for lang in languages:
        kept = [s for s in subs if s.shown and _matches(s, lang)]
        if kept:
            kept.sort(key=lambda s: (not s.is_text, s.is_sdh, not s.visible, -(s.cues or 0)))
            plans.append(LanguagePlan(lang, [Candidate("keep", kept[0].id, kept[0].language)]))
            continue
        cands: list[Candidate] = []
        # 2. The same language in the other script. A Chinese text track whose
        # script is unknown converts too: t2s leaves Simplified text as it is.
        if lang.code == "zh" and lang.script in ("Hans", "Hant"):
            other = "Hant" if lang.script == "Hans" else "Hans"
            for s in sorted(text_sources, key=lambda s: (s.script is None, s.is_sdh)):
                if s.language == "zh" and s.script in (other, None):
                    if lang.script == "Hans":
                        config = "hk2s" if s.region == "HK" else "t2s"
                    else:
                        config = "s2hk" if s.region == "HK" else "s2t"
                    cands.append(Candidate("convert", s.id, "zh", {"config": config, "from_script": s.script}))
        # 3. A translation of the best text subtitle in another language.
        sources = sorted((s for s in text_sources if s.language != lang.code),
                         key=lambda s: _source_rank(s, original, languages))
        for s in sources:
            cands.append(Candidate("translate", s.id, s.language, {"sdh": s.is_sdh}))
        # 4. Speech recognition of the main audio: the last candidate, so it is
        # the choice only when the title has no text subtitle to work from.
        if audio is not None:
            if audio_lang == lang.code:
                cands.append(Candidate("transcribe", f"audio:{audio.id}", audio_lang,
                                       {"script": lang.script} if lang.script else {}))
            else:
                cands.append(Candidate("translate", TRANSCRIPT, audio_lang, {"via": f"audio:{audio.id}"}))
        excl = set(exclude)
        plans.append(LanguagePlan(lang, [c for c in cands if c.source not in excl], list(exclude)))
    return plans


def transcription_needed(plans: list[LanguagePlan]) -> bool:
    return any(p.needs_transcript for p in plans)
