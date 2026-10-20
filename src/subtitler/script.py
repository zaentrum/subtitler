"""Chinese scripts: which one a track is written in, and dictionary conversion.

A package tags every Chinese track "chi"; the script is in the track title
at best ("Simplified", "繁體", "cht", "Hong Kong"), often not at all ("",
"iTunes"). The text decides: OpenCC's Traditional->Simplified conversion
changes many characters of a Traditional text and next to none of a
Simplified one, and the other way round.

Written Cantonese (粵語 subtitles) is another language, not another script:
it is never converted into Simplified Mandarin.
"""

from __future__ import annotations

import re
from functools import lru_cache

_HANS_HINTS = re.compile(r"simplified|简体|簡體|简|chs\b|zh[-_]?hans|zh[-_]?cn|\bgb\b|mainland|大陆", re.I)
_HANT_HINTS = re.compile(r"traditional|繁體|繁体|繁|cht\b|zh[-_]?hant|zh[-_]?tw|zh[-_]?hk|taiwan|台灣|台湾|hong\s*kong|香港|\bbig5\b", re.I)
_HK_HINTS = re.compile(r"hong\s*kong|香港|zh[-_]?hk|\bhk\b", re.I)
_TW_HINTS = re.compile(r"taiwan|台灣|台湾|zh[-_]?tw|\btw\b", re.I)
_YUE_HINTS = re.compile(r"cantonese|粵語|粤语|廣東話|广东话|\byue\b", re.I)
_BILINGUAL_HINTS = re.compile(r"&\s*english|english\s*&|中英|双语|雙語|bilingual", re.I)

# Characters of written Cantonese that standard written Chinese hardly uses.
_YUE_CHARS = set("嘅咗喺冇佢哋嘢啲嚟嗰噉乜咩睇搵畀攞諗嚿嘥揾")

_HAN = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


@lru_cache(maxsize=None)
def _cc(config: str):
    import opencc  # imported lazily: the planner works without it for labelled tracks

    return opencc.OpenCC(config)


def convert(text: str, config: str = "t2s") -> str:
    return _cc(config).convert(text)


@lru_cache(maxsize=1)
def _jieba():
    try:
        import logging

        import jieba

        jieba.setLogLevel(logging.WARNING)
        return jieba
    except ImportError:
        return None


def zh_words(text: str) -> list[str]:
    """Chinese text cut into words (jieba, MIT), so lines break between words;
    single characters when jieba is not installed."""
    j = _jieba()
    return [w for w in j.cut(text)] if j else list(text)


def han_count(text: str) -> int:
    return len(_HAN.findall(text))


def detect_script(text: str) -> tuple[str | None, float]:
    """("Hans" | "Hant" | None, confidence) from the characters themselves."""
    han = [ch for ch in text if _HAN.match(ch)]
    if len(han) < 20:
        return None, 0.0
    sample = "".join(han[:4000])
    changed_by_t2s = sum(a != b for a, b in zip(sample, convert(sample, "t2s")))
    changed_by_s2t = sum(a != b for a, b in zip(sample, convert(sample, "s2t")))
    n = len(sample)
    if changed_by_t2s == changed_by_s2t:
        return None, 0.0
    if changed_by_t2s > changed_by_s2t:
        return "Hant", (changed_by_t2s - changed_by_s2t) / n
    return "Hans", (changed_by_s2t - changed_by_t2s) / n


def cantonese_ratio(text: str) -> float:
    """Share of Han characters that are typical of written Cantonese."""
    han = _HAN.findall(text)
    if not han:
        return 0.0
    return sum(ch in _YUE_CHARS for ch in han) / len(han)


def looks_cantonese(text: str) -> bool:
    # Written Cantonese runs at several percent of these characters; standard
    # written Chinese well under one in a thousand.
    return cantonese_ratio(text) >= 0.01


def hints(title: str | None) -> dict:
    """What a track title says about script, region, Cantonese and bilingual tracks."""
    t = title or ""
    script = None
    if _HANS_HINTS.search(t) and not _HANT_HINTS.search(t):
        script = "Hans"
    elif _HANT_HINTS.search(t) and not _HANS_HINTS.search(t):
        script = "Hant"
    region = "HK" if _HK_HINTS.search(t) else "TW" if _TW_HINTS.search(t) else None
    return {
        "script": script,
        "region": region,
        "cantonese": bool(_YUE_HINTS.search(t)),
        "bilingual": bool(_BILINGUAL_HINTS.search(t)),
    }


def to_simplified_config(region: str | None) -> str:
    """OpenCC configuration for Traditional -> Simplified by the track's region.

    hk2s for Hong Kong variants; t2s otherwise. Taiwan phrase conversion
    (tw2sp) also rewrites vocabulary, which reads more mainland but changes
    the translator's words; the spike measures both.
    """
    return "hk2s" if region == "HK" else "t2s"
