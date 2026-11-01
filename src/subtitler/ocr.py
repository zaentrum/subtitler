"""Text copies of image subtitles (OCR): a bitmap track's lines, read, as WebVTT.

Every app is to show a title's subtitles, and an image subtitle (PGS of a
Blu-ray, VobSub of a DVD) is pictures of its lines that only some clients
draw. Its text copy is made here, picture by picture, by Tesseract
(Apache-2.0, its LSTM engine) with the model of the track's language
(tessdata_fast, Apache-2.0):

1. a picture's objects (pgs.Picture: what a display set shows, from that set
   to the one that clears or replaces it) are laid onto black by their
   alpha; objects that stand apart (a line at the top, one at the bottom)
   are read one by one, the upper first, and each of their lines on its own
   (the runs of rows with ink: an accent goes with its line, two lines that
   touch are cut where they have the least ink);
2. each line is inverted to dark on light, scaled so that it is about
   LINE_HEIGHT pixels high, thresholded between the letters' fill and their
   outline (Otsu over the pixels drawn) and given a margin;
3. Tesseract reads the lines of a whole track in `jobs` processes, page by
   page, each a single line (`--psm 7`; as a block of lines, `--psm 6`, it
   leaves parts of Chinese lines out): its words with their confidence; a
   line read less surely than SECOND_LOOK is read again in grey (Tesseract
   thresholds it), and the surer reading is kept;
4. a line whose strokes lean as italics do is set in <i>: its slant is
   measured on its pixels;
5. a cue per picture, with the picture's times; pictures in a row that read
   the same are one cue.

A track whose confidence (Tesseract's, per character, over the track) is
under MIN_CONFIDENCE is not attached: its copy is written aside and the
report says why.
"""

from __future__ import annotations

import math
import os
import re
import resource
import shutil
import subprocess
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import vtt
from .langs import UNSPACED, Lang

# Tesseract's model per language (ISO 639-1; "yue" written Cantonese). Chinese
# takes its script's model; a language not here is not read (and said so).
MODELS = {
    "en": "eng", "de": "deu", "fr": "fra", "es": "spa", "it": "ita", "pt": "por", "nl": "nld",
    "sv": "swe", "da": "dan", "no": "nor", "nb": "nor", "nn": "nor", "fi": "fin", "is": "isl",
    "pl": "pol", "cs": "ces", "sk": "slk", "sl": "slv", "hr": "hrv", "bs": "bos", "sr": "srp_latn",
    "hu": "hun", "ro": "ron", "bg": "bul", "ru": "rus", "uk": "ukr", "el": "ell", "tr": "tur",
    "et": "est", "lv": "lav", "lt": "lit", "ca": "cat", "eu": "eus", "gl": "glg", "id": "ind",
    "ms": "msa", "vi": "vie", "he": "heb", "ar": "ara", "fa": "fas", "hi": "hin", "th": "tha",
    "ja": "jpn", "ko": "kor", "yue": "chi_tra",
}
CHINESE = {"Hans": "chi_sim", "Hant": "chi_tra"}
LATIN = {"en", "de", "fr", "es", "it", "pt", "nl", "sv", "da", "no", "nb", "nn", "fi", "is", "pl", "cs", "sk",
         "sl", "hr", "bs", "sr", "hu", "ro", "tr", "et", "lv", "lt", "ca", "eu", "gl", "id", "ms", "vi"}

LINE_HEIGHT = 36  # pixels from the top of a line's letters to the bottom of its descenders
LINES = True  # each line read on its own (--psm 7); a block of lines (--psm 6) drops parts of Chinese lines
MIN_CONFIDENCE = 80.0  # a track's mean confidence under this is not attached
DOUBTFUL = 50.0  # a cue under this is counted as doubtful
ITALIC_DEGREES = 7.0  # a line leaning this much or more is italic
PAGES_A_PROCESS = 40  # a Tesseract process loads its model first: no fewer pages a process than this
BATCH = 600  # pictures prepared and read at a time
SECOND_LOOK = 60.0  # a part read less surely than this is read again in grey, the surer reading kept


def model_for(lang: Lang | str, script: str | None = None) -> str | None:
    """Tesseract's model for a language (and a Chinese script: Simplified unless it says Traditional)."""
    code = lang.code if isinstance(lang, Lang) else lang
    if code == "zh":
        return CHINESE.get(script or (lang.script if isinstance(lang, Lang) else None) or "Hans")
    return MODELS.get(code)


# --- the pictures, made ready to read -----------------------------------------

def _resize(img: np.ndarray, f: float) -> np.ndarray:
    """Bilinear scaling by f (numpy only)."""
    if abs(f - 1.0) < 1e-3:
        return img.astype(np.float32)
    h, w = img.shape
    H, W = max(1, round(h * f)), max(1, round(w * f))
    ys = (np.arange(H) + 0.5) / f - 0.5
    xs = (np.arange(W) + 0.5) / f - 0.5
    y0 = np.clip(np.floor(ys).astype(np.int64), 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(np.int64), 0, w - 1)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy = np.clip(ys - y0, 0, 1).astype(np.float32)[:, None]
    wx = np.clip(xs - x0, 0, 1).astype(np.float32)[None, :]
    a = img.astype(np.float32)
    top = a[y0][:, x0] * (1 - wx) + a[y0][:, x1] * wx
    bottom = a[y1][:, x0] * (1 - wx) + a[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def otsu(values: np.ndarray) -> float:
    """The threshold that best splits values (0-255) in two (Otsu)."""
    hist = np.bincount(np.clip(values, 0, 255).astype(np.uint8).ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128.0
    w0 = hist.cumsum()
    mu = (hist * np.arange(256)).cumsum()
    w1 = total - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu[-1] * w0 - mu * total) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = 0
    return float(np.argmax(between)) + 0.5


def _runs(rows: np.ndarray) -> list[list[int]]:
    """The runs [first, end) of true rows."""
    out, start = [], None
    for y, r in enumerate(rows):
        if r and start is None:
            start = y
        elif not r and start is not None:
            out.append([start, y])
            start = None
    if start is not None:
        out.append([start, len(rows)])
    return out


def bands(ink: np.ndarray) -> list[tuple[int, int]]:
    """A part's lines, as the rows [top, bottom) each takes: the runs of rows with ink in them; a
    thin one (accents, a dot, a lone mark) goes with the nearer of its neighbours, and one as tall
    as two lines is cut where it has the least ink."""
    counts = ink.sum(axis=1)
    runs = _runs(counts > 0)
    if not runs:
        return []
    tallest = max(b - a for a, b in runs)
    text = [b - a for a, b in runs if b - a >= 0.4 * tallest]
    line = float(np.median(text))
    merged: list[list[int]] = []
    for r in runs:  # thin runs onto a line
        if merged and (r[1] - r[0] < 0.4 * line or merged[-1][1] - merged[-1][0] < 0.4 * line) \
                and r[0] - merged[-1][1] < 0.5 * line:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    out: list[tuple[int, int]] = []
    for a, b in merged:  # two lines that touch, cut apart
        while b - a > 1.6 * line:
            lo, hi = a + int(0.6 * line), min(b - int(0.6 * line), a + int(1.4 * line))
            if hi <= lo:
                break
            cut = lo + int(np.argmin(counts[lo:hi]))
            out.append((a, cut))
            a = cut
        out.append((a, b))
    return out


def _line_heights(ink: np.ndarray) -> list[int]:
    """The heights of a part's lines (bands)."""
    return [b - a for a, b in bands(ink)]


@dataclass
class Part:
    """A part of a picture as Tesseract gets it: dark letters on white (uint8), and its scale."""

    image: np.ndarray
    scale: float
    line_height: float


def groups(objects: list) -> list[list]:
    """A picture's objects in the parts read one by one: objects that touch or lie a line apart at
    most are one part; parts from the top down."""
    out: list[list] = []
    for o in sorted(objects, key=lambda o: (o.y, o.x)):
        if out:
            g = out[-1]
            bottom = max(p.y + p.height for p in g)
            if o.y <= bottom + min(o.height, max(p.height for p in g)):
                g.append(o)
                continue
        out.append([o])
    return out


def compose(group: list) -> tuple[np.ndarray, np.ndarray]:
    """The objects of a part laid onto black by their alpha: (luma, alpha), float32 0-255."""
    x0, y0 = min(o.x for o in group), min(o.y for o in group)
    x1, y1 = max(o.x + o.width for o in group), max(o.y + o.height for o in group)
    luma = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    alpha = np.zeros_like(luma)
    for o in group:
        a = o.alpha.astype(np.float32) / 255.0
        ys, xs = slice(o.y - y0, o.y - y0 + o.height), slice(o.x - x0, o.x - x0 + o.width)
        luma[ys, xs] = luma[ys, xs] * (1 - a) + o.luma.astype(np.float32) * a
        alpha[ys, xs] = np.maximum(alpha[ys, xs], o.alpha)
    return luma, alpha


def _part(luma: np.ndarray, t: float, lh: float, line_height: float, grey: bool) -> Part | None:
    scale = float(np.clip(line_height / lh, 1.0, 4.0))
    scaled = _resize(luma, scale)
    ink = scaled > t
    if not ink.any():
        return None
    pad = max(8, int(round(lh * scale * 0.5)))
    image = np.full((ink.shape[0] + 2 * pad, ink.shape[1] + 2 * pad), 255, dtype=np.uint8)
    if grey:
        image[pad:pad + ink.shape[0], pad:pad + ink.shape[1]] = 255 - np.clip(scaled, 0, 255).astype(np.uint8)
    else:
        image[pad:pad + ink.shape[0], pad:pad + ink.shape[1]][ink] = 0
    return Part(image, scale, lh * scale)


def prepare(objects: list, line_height: float = LINE_HEIGHT, grey: bool = False, lines: bool = True) -> list[Part]:
    """A picture's objects as the parts Tesseract reads, each line on its own (lines; else a part
    of lines): laid onto black, inverted to dark on light, scaled to `line_height`, thresholded
    between fill and outline (grey: not, Tesseract thresholds it), a margin around."""
    parts = []
    for g in groups(objects):
        luma, alpha = compose(g)
        drawn = luma[alpha > 0]
        if not drawn.size:
            continue
        t = otsu(drawn)
        rows = bands(luma > t)
        if not rows:
            continue
        lh = float(np.median([b - a for a, b in rows]))
        if not lines:
            rows = [(0, luma.shape[0])]
        for k, (a, b) in enumerate(rows):
            # The band, and its rows of anti-aliasing up to the neighbours' bands.
            top = rows[k - 1][1] if k else 0
            bottom = rows[k + 1][0] if k + 1 < len(rows) else luma.shape[0]
            a, b = max(top, a - 2), min(bottom, b + 2)
            cols = np.nonzero((luma[a:b] > t).any(axis=0))[0]
            if not cols.size:
                continue
            x0, x1 = max(0, cols[0] - 2), min(luma.shape[1], cols[-1] + 3)
            p = _part(luma[a:b, x0:x1], t, lh, line_height, grey)
            if p is not None:
                parts.append(p)
    return parts


def slant(ink: np.ndarray) -> float:
    """How far a line's strokes lean to the right, in degrees: about 0 upright, 10-15 italic. The
    shear that makes its columns of ink the sharpest (the largest sum of squares of the column
    counts) is its slant."""
    ys, xs = np.nonzero(ink)
    if xs.size < 40:
        return 0.0
    h = ink.shape[0]
    rise = (h - 1 - ys).astype(np.float64)

    def score(deg: float) -> float:
        shifted = np.round(xs - rise * math.tan(math.radians(deg))).astype(np.int64)
        counts = np.bincount(shifted - shifted.min())
        return float((counts.astype(np.float64) ** 2).sum())

    best = max(range(-8, 27, 3), key=score)
    return float(max((best + d for d in (-2, -1, 0, 1, 2)), key=score))


# --- Tesseract -----------------------------------------------------------------

@dataclass
class Word:
    line: tuple  # (block, paragraph, line)
    left: int
    top: int
    width: int
    height: int
    conf: float
    text: str


def _pgm(path: Path, image: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (image.shape[1], image.shape[0]))
        f.write(np.ascontiguousarray(image, dtype=np.uint8).tobytes())


def parse_tsv(text: str) -> dict[int, list[Word]]:
    """Tesseract's TSV: the words of each page (1-based), in reading order."""
    pages: dict[int, list[Word]] = {}
    for row in text.splitlines()[1:]:
        f = row.split("\t")
        if len(f) < 12 or f[0] != "5" or not f[11].strip():
            continue
        pages.setdefault(int(f[1]), []).append(Word((int(f[2]), int(f[3]), int(f[4])), int(f[6]), int(f[7]),
                                                    int(f[8]), int(f[9]), float(f[10]), f[11].strip()))
    return pages


class Tesseract:
    """Tesseract's command line over many images: a list of files a process, `jobs` processes at
    once, one thread each, TSV out."""

    def __init__(self, model: str, tessdata: str | Path | None = None, jobs: int = 0, binary: str = "tesseract",
                 psm: int = 6):
        self.model = model
        self.tessdata = str(tessdata) if tessdata else None
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 2
        self.jobs = jobs or max(1, cpus - 1)
        self.binary = binary
        self.psm = psm
        self.cpu_seconds = 0.0

    def _cmd(self, inp: str) -> list[str]:
        cmd = [self.binary, inp, "stdout", "-l", self.model, "--oem", "1", "--psm", str(self.psm)]
        if self.tessdata:
            cmd += ["--tessdata-dir", self.tessdata]
        return cmd + ["-c", "tessedit_create_txt=0", "-c", "tessedit_create_tsv=1", "-c", "tessedit_do_invert=0"]

    def _env(self) -> dict:
        return {**os.environ, "OMP_THREAD_LIMIT": "1"}

    def version(self) -> str:
        try:
            out = subprocess.run([self.binary, "--version"], capture_output=True, text=True, timeout=30)
            m = re.search(r"tesseract\s+v?([\d.]+)", out.stdout + out.stderr)
            return m.group(1) if m else "unknown"
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    def models(self) -> set[str]:
        cmd = [self.binary, "--list-langs"] + (["--tessdata-dir", self.tessdata] if self.tessdata else [])
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return set()
        return {x.strip() for x in out.stdout.splitlines()[1:] if x.strip()}

    def _run(self, files: list[Path], listing: Path) -> dict[int, list[Word]]:
        listing.write_text("".join(f"{f}\n" for f in files))
        proc = subprocess.run(self._cmd(str(listing)), capture_output=True, text=True, env=self._env(),
                              timeout=120 + 10 * len(files))
        if proc.returncode != 0:
            raise RuntimeError(f"tesseract exited {proc.returncode}: {proc.stderr.strip()[-500:]}")
        return parse_tsv(proc.stdout)

    def read(self, images: list[np.ndarray]) -> list[list[Word]]:
        """The words Tesseract reads in each image, in order."""
        if not images:
            return []
        tmp = Path(tempfile.mkdtemp(prefix="subtitler-ocr-"))
        try:
            files = []
            for i, im in enumerate(images):
                f = tmp / f"{i:06d}.pgm"
                _pgm(f, im)
                files.append(f)
            n = len(files)
            k = max(1, min(self.jobs, -(-n // PAGES_A_PROCESS)))
            chunks = [list(range(j * n // k, (j + 1) * n // k)) for j in range(k)]

            def work(j: int) -> dict[int, list[Word]]:
                idx = chunks[j]
                try:
                    pages = self._run([files[i] for i in idx], tmp / f"list-{j}.txt")
                    return {idx[p - 1]: words for p, words in pages.items() if 0 < p <= len(idx)}
                except (RuntimeError, subprocess.SubprocessError):
                    out = {}  # page by page: one image that fails costs that image alone
                    for i in idx:
                        try:
                            out[i] = self._run([files[i]], tmp / f"one-{i}.txt").get(1, [])
                        except (RuntimeError, subprocess.SubprocessError):
                            out[i] = []
                    return out

            found: dict[int, list[Word]] = {}
            r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
            with ThreadPoolExecutor(k) as pool:
                for part in pool.map(work, range(k)):
                    found.update(part)
            r1 = resource.getrusage(resource.RUSAGE_CHILDREN)  # every process of this read has ended
            self.cpu_seconds += (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
            return [found.get(i, []) for i in range(n)]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata_{}/raw/4.1.0/{}.traineddata"
# Chinese reads better with the larger models (tessdata_best: a quarter fewer errors, 2.3 times the CPU
# time); the Latin scripts read as well with the fast ones.
BEST = {"chi_sim", "chi_tra"}


def tessdata_url(model: str) -> str:
    return TESSDATA_URL.format("best" if model in BEST else "fast", model)


def tessdata_for(model: str, tessdata: str | Path | None = None, cache: str | Path | None = None,
                 binary: str = "tesseract") -> str | None:
    """The models folder that has every model of `model` ("chi_sim+eng"): Tesseract's own (or
    `tessdata`) where it has them; else `cache`, the models it lacks fetched into it (tessdata_fast
    4.1.0, Apache-2.0; BEST's from tessdata_best) and the others copied from Tesseract's own.
    LookupError without a cache."""
    wanted = model.split("+")
    have = Tesseract(model, tessdata, binary=binary).models()
    if all(m in have for m in wanted):
        return str(tessdata) if tessdata else None
    if not cache:
        raise LookupError(f"Tesseract has no model {', '.join(m for m in wanted if m not in have)}; "
                          f"a models folder (--models) would fetch it")
    folder = Path(cache)
    folder.mkdir(parents=True, exist_ok=True)
    own = Path(tessdata or os.environ.get("TESSDATA_PREFIX") or "")
    for m in wanted:
        target = folder / f"{m}.traineddata"
        if target.exists():
            continue
        part = target.with_suffix(".part")
        if m in have and (own / f"{m}.traineddata").is_file():
            shutil.copyfile(own / f"{m}.traineddata", part)
        else:
            with urllib.request.urlopen(tessdata_url(m), timeout=120) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f)
        part.replace(target)
    return str(folder)


# --- words to lines, lines to cues ----------------------------------------------

_CJK = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿豈-﫿＀-￯]")


def fix_latin(text: str, code: str) -> str:
    """What Tesseract confuses in subtitles of a Latin script: a pipe for a capital I, typographic
    apostrophes; in English a lone lower-case l for I; in French, which puts a space before an
    exclamation mark, a lone I, pipe or l there for the mark, and a ç within a sentence read as
    capital."""
    if code == "fr":
        text = re.sub(r"(?<= )[I|l]!?(?=\s|$)", "!", text)
        text = re.sub(r"(?<=[a-zà-ÿ,;] )Ç(?=[a-zà-ÿ])", "ç", text)
    text = text.replace("|", "I").replace("’", "'").replace("‘", "'")
    if code == "en":
        text = re.sub(r"(?<![\w'])l(?=(?:'(?:m|ll|ve|d)\b)|\s|$|[,.!?])", "I", text)
    return text


@dataclass
class Line:
    text: str
    conf: float
    chars: int
    italic: bool
    slant: float


def wide_gaps(ink: np.ndarray, height: float) -> list[tuple[int, int]]:
    """The columns [from, to) of a line with no ink over half its height wide or more: where
    Chinese and Japanese subtitles put a space between phrases."""
    cols = ink.any(axis=0)
    xs = np.nonzero(cols)[0]
    if not xs.size:
        return []
    return [(a + xs[0], b + xs[0]) for a, b in _runs(~cols[xs[0]:xs[-1] + 1]) if b - a >= 0.45 * height]


def lines_of(words: list[Word], part: Part, code: str) -> list[Line]:
    """A part's words as its lines: joined by spaces (Chinese and Japanese: by nothing, but where
    the picture has a wide gap before a word: Tesseract's boxes of their characters are too loose
    to tell), each with its confidence and whether it leans as italics."""
    by: dict[tuple, list[Word]] = {}
    for w in words:
        by.setdefault(w.line, []).append(w)
    out = []
    ink = part.image < 128
    for ws in by.values():
        x0, y0 = min(w.left for w in ws), min(w.top for w in ws)
        x1, y1 = max(w.left + w.width for w in ws), max(w.top + w.height for w in ws)
        if code in UNSPACED:
            gaps = wide_gaps(ink[y0:y1], part.line_height)
            text = ws[0].text
            for a, b in zip(ws, ws[1:]):
                wide = any(a.left < end <= b.left + 0.25 * part.line_height for _, end in gaps)
                text += (" " if wide else "") + b.text
        else:
            text = " ".join(w.text for w in ws)
        chars = sum(len(w.text) for w in ws)
        conf = sum(max(0.0, w.conf) * len(w.text) for w in ws) / max(1, chars)
        lean = slant(ink[y0:y1, x0:x1]) if code not in UNSPACED else 0.0
        if code in LATIN:
            text = fix_latin(text, code)
        text = text.replace("\ufffd", "")  # the catalog takes no text that holds it
        out.append(Line(text, conf, chars, lean >= ITALIC_DEGREES, lean))
    return out


@dataclass
class Recognized:
    """A track's text copy and what the reading of it found."""

    cues: list[vtt.Cue]
    model: str
    engine: str
    confidence: float  # mean over its characters, 0-100
    pictures: int
    empty: int  # pictures nothing was read in
    doubtful: int  # cues under DOUBTFUL
    italic: int  # cues with a line in italics
    seconds: float
    cpu_seconds: float
    subtitle_seconds: float  # how long its pictures are shown, in all
    rejected: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.rejected is None

    def summary(self) -> dict:
        """What the reading found, without the text: the cues it doubts by their time alone."""
        doubted = [{"at": vtt.timestamp(c.start), "conf": c.meta["conf"]} for c in self.cues
                   if c.meta["conf"] < DOUBTFUL][:50]
        return {"model": self.model, "engine": self.engine, "confidence": round(self.confidence, 1),
                "cues": len(self.cues), "pictures": self.pictures, "empty": self.empty, "doubtful": self.doubtful,
                "doubtful_cues": doubted, "italic": self.italic, "shown_seconds": round(self.subtitle_seconds, 1),
                "seconds": round(self.seconds, 1), "cpu_seconds": round(self.cpu_seconds, 1),
                "rejected": self.rejected, **self.meta}


def _mean(lines: list[Line]) -> float:
    chars = sum(x.chars for x in lines)
    return sum(x.conf * x.chars for x in lines) / chars if chars else 0.0


def _escape(text: str) -> str:
    """Text as WebVTT cue text: &, < and > as character references (as FFmpeg writes them)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def cue_text(lines: list[Line]) -> str:
    """A cue's lines; in italics one span over all of them where every line is, else line by line."""
    lines = [x for x in lines if x.text]
    if lines and all(x.italic for x in lines):
        return "<i>" + "\n".join(_escape(x.text) for x in lines) + "</i>"
    return "\n".join(f"<i>{_escape(x.text)}</i>" if x.italic else _escape(x.text) for x in lines)


def recognize(pictures: list, lang: Lang | str, model: str | None = None, tessdata: str | Path | None = None,
              jobs: int = 0, min_confidence: float = MIN_CONFIDENCE, engine: Tesseract | None = None,
              line_height: float = LINE_HEIGHT, default_duration: float = 4.0, psm: int | None = None,
              second_look: float = SECOND_LOOK, by_line: bool = LINES) -> Recognized:
    """A bitmap track's pictures (pgs.Picture, vobsub's the same) read into cues with their times:
    each line on its own (by_line, `--psm 7`), or each part as a block of lines (`--psm 6`)."""
    code = lang.code if isinstance(lang, Lang) else lang
    model = model or model_for(lang)
    t0 = time.perf_counter()
    engine = engine or Tesseract(model, tessdata=tessdata, jobs=jobs, psm=psm or (7 if by_line else 6))
    lines: dict[int, list[Line]] = {}
    for b in range(0, len(pictures), BATCH):  # a batch's images at a time in memory, not a track's
        parts, owner = [], []  # owner: (the picture, the part's place among the picture's)
        for k in range(b, min(b + BATCH, len(pictures))):
            for j, p in enumerate(prepare(pictures[k].objects, line_height, lines=by_line)):
                parts.append(p)
                owner.append((k, j))
        read = [lines_of(ws, p, code) for p, ws in zip(parts, engine.read([p.image for p in parts]))]
        if second_look:  # an unsure part read again in grey (the same picture's parts, in the same order)
            again = [i for i, ls in enumerate(read) if ls and _mean(ls) < second_look]
            greys = {k: prepare(pictures[k].objects, line_height, grey=True, lines=by_line)
                     for k in {owner[i][0] for i in again}}
            redo = [(i, greys[owner[i][0]][owner[i][1]]) for i in again if owner[i][1] < len(greys[owner[i][0]])]
            for (i, p), ws in zip(redo, engine.read([p.image for _, p in redo])):
                ls = lines_of(ws, p, code)
                if ls and _mean(ls) > _mean(read[i]) + 5:
                    read[i] = ls
        for (k, _), ls in zip(owner, read):
            lines.setdefault(k, []).extend(ls)
    cues: list[vtt.Cue] = []
    empty = 0
    shown = 0.0
    for k, pic in enumerate(pictures):
        ls = [x for x in lines.get(k, []) if x.text.strip()]
        end = pic.end if pic.end is not None else pic.start + default_duration
        shown += max(0.0, end - pic.start)
        if not ls:
            empty += 1
            continue
        chars = sum(x.chars for x in ls)
        conf = sum(x.conf * x.chars for x in ls) / max(1, chars)
        text = cue_text(ls)
        prev = cues[-1] if cues else None
        if prev is not None and prev.text == text and pic.start - prev.end < 0.1:
            prev.end = end  # the same lines again: one cue
            continue
        cues.append(vtt.Cue(pic.start, end, text, meta={"conf": round(conf, 1), "chars": chars,
                                                         "italic": any(x.italic for x in ls), "forced": pic.forced,
                                                         "slant": round(max(x.slant for x in ls), 1)}))
    cues = [c for c in cues if c.end - c.start >= 0.08]
    chars = sum(c.meta["chars"] for c in cues)
    confidence = sum(c.meta["conf"] * c.meta["chars"] for c in cues) / chars if chars else 0.0
    r = Recognized(cues, model, f"Tesseract {engine.version()}", confidence, len(pictures), empty,
                   sum(c.meta["conf"] < DOUBTFUL for c in cues), sum(c.meta["italic"] for c in cues),
                   time.perf_counter() - t0, engine.cpu_seconds, shown)
    if not cues:
        r.rejected = "nothing was read in its pictures"
    elif confidence < min_confidence:
        r.rejected = f"its mean confidence {confidence:.1f} is under {min_confidence:g}"
    return r


# --- a track ---------------------------------------------------------------------

def pictures_of(path: str | Path, fmt: str) -> list:
    """An image subtitle's pictures: PGS (.sup) or VobSub (the .idx, its .sub beside it)."""
    from . import pgs, vobsub

    if fmt.lower() in ("vobsub", "dvd_subtitle", "idx"):
        return vobsub.read(path)
    return pgs.read(path)


def probe_script(pictures: list, tessdata: str | Path | None = None, jobs: int = 0, sample: int = 40,
                 engine_factory=None) -> tuple[str, dict]:
    """Which script a Chinese image subtitle is in, when its title does not say: a sample of its
    pictures read with both models, the one that reads them more surely (and whose text OpenCC
    finds of its script) wins."""
    from . import script

    step = max(1, len(pictures) // sample)
    picked = pictures[::step][:sample]
    found = {}
    for sc, model in CHINESE.items():
        engine = engine_factory(model) if engine_factory else Tesseract(model, tessdata=tessdata, jobs=jobs)
        r = recognize(picked, Lang("zh", sc), model=model, engine=engine, min_confidence=0)
        text = "".join(vtt.plain(c.text) for c in r.cues)
        found[sc] = {"confidence": round(r.confidence, 1), "by_characters": script.detect_script(text)[0]}
    best = max(found, key=lambda sc: (found[sc]["by_characters"] == sc, found[sc]["confidence"]))
    return best, found


_FORMAT_WORDS = re.compile(r"[\s/\-–(\[]*\b(?:pgs|sup|vobsub|idx|image|bitmap)\b[)\]]?", re.I)
# A name that says nothing but the kind or the language's code: the language's name says more.
_NO_NAME = re.compile(r"[\[(]?\s*(?:forced|sdh|cc|hi|full|subs?|subtitles?|[a-z]{2,3})\s*[\])]?", re.I)


def label(title: str, lang: Lang, kind: str) -> str:
    """The label of a text copy: the track's own name, else its language's (with SDH or forced, as
    extract names the source's tracks), then a plain "OCR" mark: "English (Forced) · OCR",
    "English · OCR". Words of the image format ("PGS") are left out: the copy is text."""
    name = _FORMAT_WORDS.sub(" ", title or "").strip(" -–/·|,")
    name = re.sub(r"\s{2,}", " ", name)
    if not name or _NO_NAME.fullmatch(name):
        name = lang.name + {"sdh": " · SDH", "forced": " · forced"}.get(kind, "")
    return f"{name} · OCR"


def read_pictures(pictures: list, lang: Lang, tessdata: str | Path | None = None, jobs: int = 0,
                  min_confidence: float = MIN_CONFIDENCE, bilingual: bool = False, models: str | Path | None = None,
                  **kw) -> tuple[Lang, Recognized]:
    """An image subtitle's pictures read into its text copy: the language it is in (a Chinese one's
    script found by reading it where it is not known) and what was read. A bilingual track (Chinese
    above, English below) is read with English's model too. A model Tesseract lacks is fetched into
    `models`/tessdata where that is given (tessdata_for)."""
    meta = {}
    cache = Path(models) / "tessdata" if models else None
    if lang.code == "zh" and lang.script is None:
        script, meta["script_probe"] = probe_script(pictures, tessdata_for("chi_sim+chi_tra", tessdata, cache), jobs)
        lang = Lang("zh", script)
    model = model_for(lang)
    if model is None:
        raise LookupError(f"no OCR model for the language {lang.tag}")
    if bilingual and lang.code != "en":
        model += "+eng"
    tessdata = tessdata_for(model, tessdata, cache)
    r = recognize(pictures, lang, model=model, tessdata=tessdata, jobs=jobs, min_confidence=min_confidence, **kw)
    r.meta.update(meta)
    return lang, r


def read_track(path: str | Path, fmt: str, lang: Lang, tessdata: str | Path | None = None, jobs: int = 0,
               min_confidence: float = MIN_CONFIDENCE, bilingual: bool = False,
               models: str | Path | None = None) -> tuple[Lang, Recognized]:
    """An image subtitle file read into its text copy (read_pictures)."""
    return read_pictures(pictures_of(path, fmt), lang, tessdata, jobs, min_confidence, bilingual, models)
