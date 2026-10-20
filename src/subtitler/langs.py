"""Language tags: the package's ISO 639-2 codes, the instance's BCP 47 tags.

The packager tags tracks with what the source says, mostly ISO 639-2/B
("ger", "chi", "fre"), sometimes /T ("deu", "zho") or 639-1 ("de"). The
instance names its languages as BCP 47 tags with a script where it matters
("en", "de", "zh-Hans"). Everything here compares on ISO 639-1 where there
is one.
"""

from __future__ import annotations

from dataclasses import dataclass

# ISO 639-2 (B and T) -> ISO 639-1, for the languages a library is likely to
# carry and the speech and translation models know.
_ISO2_TO_1 = {
    "afr": "af", "alb": "sq", "sqi": "sq", "amh": "am", "ara": "ar", "arm": "hy", "hye": "hy",
    "aze": "az", "baq": "eu", "eus": "eu", "bel": "be", "ben": "bn", "bos": "bs", "bul": "bg",
    "bur": "my", "mya": "my", "cat": "ca", "chi": "zh", "zho": "zh", "cmn": "zh", "cze": "cs",
    "ces": "cs", "dan": "da", "dut": "nl", "nld": "nl", "eng": "en", "est": "et", "fil": "tl",
    "tgl": "tl", "fin": "fi", "fre": "fr", "fra": "fr", "geo": "ka", "kat": "ka", "ger": "de",
    "deu": "de", "gle": "ga", "glg": "gl", "gre": "el", "ell": "el", "guj": "gu", "heb": "he",
    "hin": "hi", "hrv": "hr", "hun": "hu", "ice": "is", "isl": "is", "ind": "id", "ita": "it",
    "jpn": "ja", "kan": "kn", "kaz": "kk", "khm": "km", "kor": "ko", "lao": "lo", "lat": "la",
    "lav": "lv", "lit": "lt", "mac": "mk", "mkd": "mk", "mal": "ml", "mar": "mr", "may": "ms",
    "msa": "ms", "mon": "mn", "nep": "ne", "nor": "no", "nob": "no", "nno": "nn", "pan": "pa",
    "per": "fa", "fas": "fa", "pol": "pl", "por": "pt", "rum": "ro", "ron": "ro", "rus": "ru",
    "sin": "si", "slo": "sk", "slk": "sk", "slv": "sl", "spa": "es", "srp": "sr", "swa": "sw",
    "swe": "sv", "tam": "ta", "tel": "te", "tha": "th", "tib": "bo", "bod": "bo", "tur": "tr",
    "ukr": "uk", "urd": "ur", "uzb": "uz", "vie": "vi", "wel": "cy", "cym": "cy", "yue": "yue",
}
_SPECIAL = {"und", "zxx", "mul", "mis", ""}

NAMES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "az": "Azerbaijani", "be": "Belarusian",
    "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan", "bs": "Bosnian", "ca": "Catalan",
    "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish",
    "fr": "French", "ga": "Irish", "gl": "Galician", "gu": "Gujarati", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian",
    "is": "Icelandic", "it": "Italian", "ja": "Japanese", "ka": "Georgian", "kk": "Kazakh",
    "km": "Khmer", "kn": "Kannada", "ko": "Korean", "la": "Latin", "lo": "Lao", "lt": "Lithuanian",
    "lv": "Latvian", "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi",
    "ms": "Malay", "my": "Burmese", "ne": "Nepali", "nl": "Dutch", "nn": "Norwegian Nynorsk",
    "no": "Norwegian", "pa": "Punjabi", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "sq": "Albanian",
    "sr": "Serbian", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
    "th": "Thai", "tl": "Filipino", "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu",
    "uz": "Uzbek", "vi": "Vietnamese", "yue": "Cantonese", "zh": "Chinese",
}
SCRIPT_NAMES = {"Hans": "Simplified", "Hant": "Traditional"}

# Languages written without spaces between words: cue splitting, wrapping and
# error rates work on characters.
UNSPACED = {"zh", "yue", "ja", "th", "lo", "km", "my", "bo"}


def iso1(code: str | None) -> str | None:
    """ISO 639-1 (or the 639-2/3 code where there is none); None for und/zxx/mul."""
    if not code:
        return None
    c = code.strip().replace("_", "-").split("-")[0].lower()
    if c in _SPECIAL:
        return None
    if len(c) == 2:
        return c
    return _ISO2_TO_1.get(c, c)


@dataclass(frozen=True)
class Lang:
    """An instance language: ISO 639-1 code plus an optional script (BCP 47)."""

    code: str
    script: str | None = None

    @classmethod
    def parse(cls, tag: str) -> "Lang":
        parts = tag.strip().replace("_", "-").split("-")
        code = iso1(parts[0]) or parts[0].lower()
        script = None
        for p in parts[1:]:
            if len(p) == 4 and p.isalpha():
                script = p.title()
            elif p.upper() in ("CN", "SG") and code == "zh":
                script = script or "Hans"
            elif p.upper() in ("TW", "HK", "MO") and code == "zh":
                script = script or "Hant"
        return cls(code, script)

    @property
    def tag(self) -> str:
        return f"{self.code}-{self.script}" if self.script else self.code

    @property
    def name(self) -> str:
        base = NAMES.get(self.code, self.code)
        if self.script in SCRIPT_NAMES:
            return f"{base} ({SCRIPT_NAMES[self.script]})"
        return base


def parse_list(csv: str) -> list[Lang]:
    return [Lang.parse(t) for t in csv.split(",") if t.strip()]


def name(code: str, script: str | None = None) -> str:
    return Lang(code, script).name


ORIGIN_MARK = {"transcribed": "AI", "translated": "AI", "converted": "Converted"}


def made_label(lang: Lang, origin: str) -> str:
    """The label a made subtitle carries: "German · AI", "Chinese (Simplified) · Converted"."""
    return f"{lang.name} · {ORIGIN_MARK[origin]}"
