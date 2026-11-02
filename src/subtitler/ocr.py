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
   the same are one cue; a cue of one or two characters read less surely
   than DOUBTFUL is left out (a mark, a stray shape: what a music note or a
   graphic reads as), and a "#" at a line's edge is the music note it
   stands for (♪).

A track is not attached, its copy written aside and the report saying why,
where its confidence (Tesseract's, per character, over the track) is under
MIN_CONFIDENCE; where more than GARBAGE_SHARE of its pictures read as
nothing, as such a stray mark or under DOUBTFUL (graphics, a font Tesseract
does not read); and where it reads as another language than its track's
(wrong_language: Chinese whose letters are mostly not Chinese characters, a
Latin-script language whose commonest words another's outnumber), as a
track tagged with the wrong language reads almost as surely with the
model of its tag.
"""

from __future__ import annotations

import hashlib
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
SHORT = 2  # a cue of this many characters or fewer read under DOUBTFUL is left out
# A track with more than this share of its pictures read as nothing, as a short cue left out or under
# DOUBTFUL (and at least GARBAGE_MIN of them) is not attached; the 45 tracks measured had 2 % at most.
GARBAGE_SHARE = 0.10
GARBAGE_MIN = 3
HAN_SHARE = 0.5  # a Chinese copy with fewer of its letters Chinese characters is not attached
HAN_SHARE_BILINGUAL = 0.05  # a bilingual one's English line is most of its letters
MIN_LETTERS = 50  # the script of a copy with fewer letters is not judged
MIN_WORDS = 40  # the language of a copy with fewer words is not judged
LABEL_WIDTH = 120  # the catalog keeps this many characters of a label (katalog-manager's FitLabel)
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


CGROUP = Path("/sys/fs/cgroup")


def cpus(cgroup: Path = CGROUP) -> int:
    """The CPUs this process may use: the ones it may run on, and no more than its container's CPU
    limit (cgroup v2 cpu.max, v1 cpu.cfs_quota_us; rounded up), which a pod's node does not show."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 2
    for quota_file, period_file in (("cpu.max", None), ("cpu/cpu.cfs_quota_us", "cpu/cpu.cfs_period_us")):
        try:
            if period_file is None:
                quota, period = (cgroup / quota_file).read_text().split()[:2]
            else:
                quota, period = (cgroup / quota_file).read_text(), (cgroup / period_file).read_text()
            if quota.strip() in ("max", "-1"):
                break
            q, p = int(quota), int(period)
        except (OSError, ValueError):
            continue
        if q > 0 and p > 0:
            n = min(n, max(1, math.ceil(q / p)))
        break
    return n


class Tesseract:
    """Tesseract's command line over many images: a list of files a process, `jobs` processes at
    once, one thread each, TSV out; by default one process fewer than the CPUs it may use (cpus)."""

    def __init__(self, model: str, tessdata: str | Path | None = None, jobs: int = 0, binary: str = "tesseract",
                 psm: int = 6):
        self.model = model
        self.tessdata = str(tessdata) if tessdata else None
        self.jobs = jobs or max(1, cpus() - 1)
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
# The SHA-256 of each model that may be fetched (tessdata_fast 4.1.0; BEST's from tessdata_best 4.1.0):
# a file that differs is not used.
TESSDATA_SHA256 = {
    "ara": "e3206d3dc87fd50c24a0fb9f01838615911d25168f4e64415244b67d2bb3e729",
    "bos": "6cc8cc87cf1afbfa6a41febb725dbadb14bed96a46685906440a6eb8a7892f04",
    "bul": "aebc9b0fcc8cfaf8a9f38a02bb7b85052bd850744696a2c11cf0081820e5b21e",
    "cat": "250db73cd5b380d2798581295dc12f20d0828cdb335a65d833d12dfdbf57117d",
    "ces": "934bcaf97ef3348413263331131c9fa7f55f30db333c711929c124fb635f7e1b",
    "chi_sim": "4fef2d1306c8e87616d4d3e4c6c67faf5d44be3342290cf8f2f0f6e3aa7e735b",
    "chi_tra": "1aa60488574cafa69486d919284f079ca9b68fcc7f6ad8dc1ff1b318dfd97028",
    "dan": "acb1fd074487a31d1294fcdfd7d7c673467ffd8aeacb2ccd61ebcbf04eb4e2fa",
    "deu": "19d219bbb6672c869d20a9636c6816a81eb9a71796cb93ebe0cb1530e2cdb22d",
    "ell": "4fba8a0b461038d51f1c20d043d4f2ac38c4e778f1b90830847f7bd8fa3ba726",
    "eng": "7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2",
    "est": "515d4a773682b286369511e83fe412bcff16a92a886f99c761c1d760a7e30456",
    "eus": "40e7418296c355d9fd9ca843d115c51e740089e576e887443e61704823cd6624",
    "fas": "db1c0a91208aff00d3cf1ed2c1d23f76419afd5f024688b4f71adc3f2ce4a505",
    "fin": "61a04cd62b507c3d9ae0e1cda399e6715ebf49dea9df47897c8acdcd3bd3e13c",
    "fra": "ced037562e8c80c13122dece28dd477d399af80911a28791a66a63ac1e3445ca",
    "glg": "7947619c5544d86849f563bd737ad90dbea9f5319fbd838c4747a7a1cd40b260",
    "heb": "11f9e43ab227f786352a50f75c94c2e9906f1baba86d93276da19da7ce0904db",
    "hin": "4c73ffc59d497c186b19d1e90f5d721d678ea6b2e277b719bee4e2af12271825",
    "hrv": "9e515d9832ce259dbab550b1cc6b998f8b929faf2edacaaca981b05adb130571",
    "hun": "35067e7cfe102dcdc953f9a758fdfaa6296b17a1ee6d874ee780fa306430b9fb",
    "ind": "69786901da87ab8766c1ea7fbb10b28f2110c14da3f6c8f2735df131fba95d88",
    "isl": "5ec828c363f3f0062c3caf08216677cf5f3c0c981b05fab445123a16b98f10a9",
    "ita": "b8f89e1e785118dac4d51ae042c029a64edb5c3ee42ef73027a6d412748d8827",
    "jpn": "1f5de9236d2e85f5fdf4b3c500f2d4926f8d9449f28f5394472d9e8d83b91b4d",
    "kor": "6b85e11d9bbf07863b97b3523b1b112844c43e713df8b66418a081fd1060b3b2",
    "lav": "7d9eaf22254b381f18b806cd6cb647862a397a7221adbce3e22cc421793a8847",
    "lit": "1e383df5b055583bc01cb5764ecdf74c540753f2cb3f8205e7105361da4bc989",
    "msa": "e41a3e5febfec50c90371eb1cbb17a48b10cad387900e3420b1f134c1b766cba",
    "nld": "ced0e5e046a84c908a6aa7accbef9a232c4a5d9a8276691b81c6ee64d02963f6",
    "nor": "0451eb4f8049ae78196806bf878a389a2f40f1386fe038568cf4441226ba6ef2",
    "pol": "c4476cdbc0e33d898d32345122b7be1cbf85ace15f920f06c7714756e1ef79b2",
    "por": "c4932b937207a9514b7514d518b931a99938c02a28a5a5a553f8599ed58b7deb",
    "ron": "9adfde6b51ba4b97efd10ea37c3070fd3fc2bad7815e81f5c3c198cd96216cc9",
    "rus": "e16e5e036cce1d9ec2b00063cf8b54472625b9e14d893a169e2b0dedeb4df225",
    "slk": "fbcc400a9c74c6a13d922fcb1211b655d1b165387b675ed75cd2dbd756b974a5",
    "slv": "b937632c17ce5fdf20535d25feec044da10a0c4c8234f4302d5606fd671cf60d",
    "spa": "6f2e04d02774a18f01bed44b1111f2cd7f3ba7ac9dc4373cd3f898a40ea6b464",
    "srp_latn": "54b2e001bd8a336bdffc249c1487b8b7cb712f0a031f2551930ba050ebefa2a0",
    "swe": "f7304988d41f833efebcc2d529df54b1903ecebbc3da1faabd19a0fddd4fe586",
    "tha": "294227cc2d1292b0acb28d61d4115c88252b96d466ca90b417cf4cf0c67bf07c",
    "tur": "7393381111e1152420fc4092cb44eef4237580d21b92bf30d7d221aad192c6b7",
    "ukr": "d59e53e2bded32f4445f124b4b00240fcac7e8044c003ab822ccb94f0b3db59b",
    "vie": "79df64caf7bcfb2a27df5042ecb6121e196eada34da774956995747636d5bfa1",
}


def tessdata_url(model: str) -> str:
    return TESSDATA_URL.format("best" if model in BEST else "fast", model)


def _fetch(model: str, part: Path) -> None:
    """A model fetched into part, its SHA-256 checked against TESSDATA_SHA256 (LookupError where it
    has none or differs, nothing left behind)."""
    want = TESSDATA_SHA256.get(model)
    if want is None:
        raise LookupError(f"no checksum is known for the model {model}: it is not fetched (put it into the models "
                          f"folder by hand)")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(tessdata_url(model), timeout=120) as r, open(part, "wb") as f:
            while chunk := r.read(1 << 20):
                digest.update(chunk)
                f.write(chunk)
        if digest.hexdigest() != want:
            raise LookupError(f"the model {model} fetched from {tessdata_url(model)} is not the one expected "
                              f"(SHA-256 {digest.hexdigest()}, not {want}): it is not used")
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def tessdata_for(model: str, tessdata: str | Path | None = None, cache: str | Path | None = None,
                 binary: str = "tesseract") -> str | None:
    """The models folder that has every model of `model` ("chi_sim+eng"): Tesseract's own (or
    `tessdata`) where it has them; else `cache`, the models it lacks fetched into it (tessdata_fast
    4.1.0, Apache-2.0; BEST's from tessdata_best), each checked against its SHA-256, and the others
    copied from Tesseract's own. LookupError without a cache, or for a model that differs."""
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
            _fetch(m, part)
        part.replace(target)
    return str(folder)


# --- words to lines, lines to cues ----------------------------------------------

_CJK = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿豈-﫿＀-￯]")


def fix_latin(text: str, code: str) -> str:
    """What Tesseract confuses in subtitles of a Latin script: a pipe for a capital I, typographic
    apostrophes, a lone double quote before a question or exclamation mark; in English a lone
    lower-case l for I; in French, which puts a space before an exclamation mark, a lone I, pipe or
    l there for the mark, and a ç within a sentence read as capital."""
    if code == "fr":
        text = re.sub(r"(?<= )[I|l]!?(?=\s|$)", "!", text)
        text = re.sub(r"(?<=[a-zà-ÿ,;] )Ç(?=[a-zà-ÿ])", "ç", text)
    if sum(text.count(q) for q in '"“”„') == 1:  # a lone quote right before ? or ! is the mark's top, read twice
        text = re.sub(r'(?<=\w)["“”](?=[?!])', "", text)
    text = text.replace("|", "I").replace("’", "'").replace("‘", "'")
    if code == "en":
        text = re.sub(r"(?<![\w'])l(?=(?:'(?:m|ll|ve|d)\b)|\s|$|[,.!?])", "I", text)
    return text


def music_notes(text: str, code: str) -> str:
    """A music note at a line's edge, which Tesseract reads as "#" (125 of 127 times in the tracks
    measured), put back: "# La la la #" is "♪ La la la ♪". A "#" within a line ("#1") stays; in
    Chinese and Japanese, which leave no space before it, one at either edge is a note."""
    note = lambda m: "♪" * len(m.group())
    if code in UNSPACED:
        return re.sub(r"#+$", note, re.sub(r"^#+", note, text))
    return re.sub(r"(?<=\s)#+$", note, re.sub(r"^#+(?=\s|$)", note, text))


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


def dots(ink: np.ndarray) -> tuple[int, int]:
    """How many dots a line begins and ends with, by its pixels: marks small and low in the line,
    one after the other (Tesseract leaves an ellipsis at a line's edge out, or reads it short)."""
    rows = np.nonzero(ink.any(axis=1))[0]
    if not rows.size:
        return 0, 0
    top, lh = rows[0], rows[-1] + 1 - rows[0]
    segs = _runs(ink.any(axis=0))

    def is_dot(a: int, b: int) -> bool:
        ys = np.nonzero(ink[:, a:b].any(axis=1))[0]
        return b - a <= 0.25 * lh and ys[-1] + 1 - ys[0] <= 0.25 * lh and ys[0] >= top + 0.5 * lh

    lead = next((k for k, (a, b) in enumerate(segs) if not is_dot(a, b)), len(segs))
    if lead == len(segs):  # dots alone
        return 0, 0
    trail = next((k for k, (a, b) in enumerate(reversed(segs)) if not is_dot(a, b)), 0)
    return lead, trail


def ellipses(text: str, lead: int, trail: int) -> str:
    """A line's ellipses as its pixels have them: two or three dots at its start or end are '...'."""
    if lead >= 2 and not text.startswith(("...", "…")):
        text = "..." + text.lstrip(".…· ")
    if trail >= 2 and not text.endswith(("...", "…")):
        text = text.rstrip(".…· ") + "..."
    return text


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
            text = ellipses(fix_latin(text, code), *dots(ink[y0:y1] if len(by) > 1 else ink))
        text = music_notes(text.replace("\ufffd", ""), code)  # the catalog takes no text that holds U+FFFD
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
    dropped: int = 0  # cues of SHORT characters or fewer read under DOUBTFUL, left out

    @property
    def accepted(self) -> bool:
        return self.rejected is None

    def summary(self) -> dict:
        """What the reading found, without the text: the cues it doubts by their time alone."""
        doubted = [{"at": vtt.timestamp(c.start), "conf": c.meta["conf"]} for c in self.cues
                   if c.meta["conf"] < DOUBTFUL][:50]
        return {"model": self.model, "engine": self.engine, "confidence": round(self.confidence, 1),
                "cues": len(self.cues), "pictures": self.pictures, "empty": self.empty, "dropped": self.dropped,
                "doubtful": self.doubtful, "doubtful_cues": doubted, "italic": self.italic, "shown_seconds": round(self.subtitle_seconds, 1),
                "seconds": round(self.seconds, 1), "cpu_seconds": round(self.cpu_seconds, 1),
                "rejected": self.rejected, **self.meta}


def _mean(lines: list[Line]) -> float:
    chars = sum(x.chars for x in lines)
    return sum(x.conf * x.chars for x in lines) / chars if chars else 0.0


def _escape(text: str) -> str:
    """Text as WebVTT cue text: &, < and > as character references (as FFmpeg writes them)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --- which language a copy reads as ------------------------------------------------

# The commonest words of the Latin-script languages a copy's language is checked in: a track tagged with
# a language it is not in reads almost as surely with that language's model (German read with the
# English one at a confidence of 90), and its words tell. Per copy of the tracks measured, its own
# language's are 21-50 % of its words, any other's 25 % at most; Danish and Norwegian, which share
# most of theirs (35-39 %), are not told apart.
COMMON_WORDS = {
    "en": "the a an and or but to of in on at for with is are was were be been it it's that this what you your i "
          "i'm me my we he she they him her them not no don't do have has can will just so all there here know like "
          "get go oh yes yeah okay if about up out how why who now",
    "de": "der die das den dem des ein eine einen und oder aber zu von mit auf für ist sind war bin bist es ich du er "
          "sie wir ihr mich dich mir dir uns nicht kein keine was wie wo warum wer ja nein doch schon noch nur auch "
          "hier da jetzt mal so dass wenn kann habe hast hat haben wird sein mein dein",
    "fr": "le la les un une des du de et ou mais à au en dans pour avec sur est suis es sont était ai as a je tu il "
          "elle nous vous ils me te moi toi ne pas non oui ce c'est ça qui que quoi comment où pourquoi on y bien "
          "très tout plus si mon ma ton ta",
    "es": "el la los las un una unos y o pero de del a al en con por para es son está estoy eres fue yo tú él ella "
          "nosotros me te lo le se mi tu su no sí qué que quién cómo dónde esto eso aquí ahora bien muy todo hay ya "
          "vamos",
    "it": "il lo la i gli le un uno una e o ma di da a in con su per è sono sei era ho hai ha io tu lui lei noi voi "
          "mi ti ci si non sì che chi cosa come dove perché questo quello qui qua bene molto tutto già anche",
    "nl": "de het een en of maar van in op te met voor naar is zijn was ben bent heb hebt heeft ik je jij hij zij "
          "ze we wij jullie mij me niet geen nee ja wat wie waar hoe waarom dat dit die er hier nu dan ook nog al "
          "wel zo mijn",
    "sv": "och att det den en ett i på med för till av om är var har hade jag du han hon vi ni de mig dig oss inte "
          "ja nej vad vem hur varför som här där nu så men kan ska vill",
    "da": "og at det den en et i på med for til af om er var har havde jeg du han hun vi de mig dig os ikke ja nej "
          "hvad hvem hvor hvordan hvorfor som her der nu så men kan skal vil",
    "no": "og å det den en et i på med for til av om er var har hadde jeg du han hun vi dere de meg deg oss ikke ja "
          "nei hva hvem hvor hvordan hvorfor som her der nå så men kan skal vil",
    "fi": "ja tai mutta on ole oli olen olet ei en et se ne tämä tuo minä mä sinä sä hän me te he minua sinua mitä "
          "mikä kuka missä miten miksi että kun jos niin nyt vain jo kyllä täällä siellä",
    "pl": "i a w z na do od o że to jest są był była jestem jesteś nie tak co kto gdzie jak dlaczego ja ty on ona "
          "my wy oni mnie cię mi ci się ale czy już tu tam teraz dobrze wszystko",
    "pt": "o a os as um uma e ou mas de do da em no na com por para é são está estou foi eu tu você ele ela nós me "
          "te se meu minha não sim que quem como onde isso isto aqui agora bem muito tudo",
}
_COMMON = {k: frozenset(v.split()) for k, v in COMMON_WORDS.items()}
_ALIKE = {"da": {"no"}, "no": {"da"}}
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
_LETTER = re.compile(r"[A-Za-zÀ-ɏ]")


def word_rates(text: str) -> tuple[int, dict[str, float]]:
    """How many words a text has, and per language of COMMON_WORDS the share of them that are its."""
    words = [w.lower() for w in _WORD.findall(text.replace("’", "'"))]
    if not words:
        return 0, {}
    return len(words), {k: sum(w in s for w in words) / len(words) for k, s in _COMMON.items()}


def wrong_language(cues: list[vtt.Cue], code: str, bilingual: bool = False) -> str | None:
    """Why a copy reads as another language than its track's, or None: Chinese whose letters are
    under HAN_SHARE Chinese characters (HAN_SHARE_BILINGUAL for a track with an English line); a
    language of COMMON_WORDS whose commonest words another language's outnumber in it. A copy of
    fewer letters (MIN_LETTERS) or words (MIN_WORDS) than that, or of another language, is not
    judged."""
    from . import script

    text = "\n".join(vtt.plain(c.text) for c in cues)
    if code in ("zh", "yue"):
        han, latin = script.han_count(text), len(_LETTER.findall(text))
        if han + latin < MIN_LETTERS:
            return None
        share, need = han / (han + latin), HAN_SHARE_BILINGUAL if bilingual else HAN_SHARE
        if share < need:
            return f"it reads as another script: {share:.0%} of its letters are Chinese characters (under {need:.0%})"
        return None
    key = "no" if code in ("nb", "nn") else code
    if key not in _COMMON:
        return None
    n, rates = word_rates(text)
    if n < MIN_WORDS:
        return None
    other = max((k for k in rates if k != key and k not in _ALIKE.get(key, ())), key=rates.get)
    if rates[other] > rates[key]:
        return (f"it reads as {Lang(other).name}, not {Lang(key).name}: {rates[other]:.0%} of its {n} words are "
                f"{Lang(other).name}'s commonest, {rates[key]:.0%} {Lang(key).name}'s")
    return None


def cue_text(lines: list[Line]) -> str:
    """A cue's lines; in italics one span over all of them where every line is, else line by line."""
    lines = [x for x in lines if x.text]
    if lines and all(x.italic for x in lines):
        return "<i>" + "\n".join(_escape(x.text) for x in lines) + "</i>"
    return "\n".join(f"<i>{_escape(x.text)}</i>" if x.italic else _escape(x.text) for x in lines)


def recognize(pictures: list, lang: Lang | str, model: str | None = None, tessdata: str | Path | None = None,
              jobs: int = 0, min_confidence: float = MIN_CONFIDENCE, engine: Tesseract | None = None,
              line_height: float = LINE_HEIGHT, default_duration: float = 4.0, psm: int | None = None,
              second_look: float = SECOND_LOOK, by_line: bool = LINES, bilingual: bool = False) -> Recognized:
    """A bitmap track's pictures (pgs.Picture, vobsub's the same) read into cues with their times:
    each line on its own (by_line, `--psm 7`), or each part as a block of lines (`--psm 6`); a cue of
    SHORT characters or fewer read under DOUBTFUL left out. Rejected (not to be attached, and why):
    nothing read; a confidence under min_confidence; more than GARBAGE_SHARE of its pictures read as
    nothing, as such a cue or under DOUBTFUL; another language than `lang` (wrong_language; a
    bilingual track's second language aside)."""
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
    empty = dropped = 0
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
        if chars <= SHORT and conf < DOUBTFUL:  # a mark or a stray shape, not a word
            dropped += 1
            continue
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
                   time.perf_counter() - t0, engine.cpu_seconds, shown, dropped=dropped)
    garbage = empty + dropped + r.doubtful
    if not cues:
        r.rejected = "nothing was read in its pictures"
    elif confidence < min_confidence:
        r.rejected = f"its mean confidence {confidence:.1f} is under {min_confidence:g}"
    elif garbage >= GARBAGE_MIN and garbage > GARBAGE_SHARE * len(pictures):
        r.rejected = (f"{garbage} of its {len(pictures)} pictures read as nothing, as a stray mark or under "
                      f"{DOUBTFUL:g} ({garbage / len(pictures):.0%}, over {GARBAGE_SHARE:.0%})")
    else:
        r.rejected = wrong_language(cues, code, bilingual)
    return r


# --- a track ---------------------------------------------------------------------

def pictures_of(path: str | Path, fmt: str) -> list:
    """An image subtitle's pictures: PGS (.sup) or VobSub (the .idx, its .sub beside it)."""
    from . import pgs, vobsub

    if fmt.lower() in ("vobsub", "dvd_subtitle", "idx"):
        return vobsub.read(path)
    return pgs.read(path)


PROBE = "chi_sim+chi_tra"  # both Chinese models at once: per character the one that reads it better


def probe_script(pictures: list, tessdata: str | Path | None = None, jobs: int = 0, sample: int = 40,
                 engine_factory=None) -> tuple[str, dict]:
    """Which script a Chinese image subtitle is in, when its title does not say: a sample of its
    pictures read once with both models together (PROBE), the script of the characters read deciding
    (script.detect_script; Simplified where they tell neither). The two models' confidence alone
    picked the wrong one for a sample in eight."""
    from . import script

    step = max(1, len(pictures) // sample)
    picked = pictures[::step][:sample]
    engine = engine_factory(PROBE) if engine_factory else Tesseract(PROBE, tessdata=tessdata, jobs=jobs, psm=7)
    r = recognize(picked, Lang("zh"), model=PROBE, engine=engine, min_confidence=0)
    found, share = script.detect_script("".join(vtt.plain(c.text) for c in r.cues))
    return found or "Hans", {"model": PROBE, "pictures": len(picked), "by_characters": found,
                             "share": round(share, 3), "confidence": round(r.confidence, 1)}


_FORMAT_WORDS = re.compile(r"[\s/\-–(\[]*\b(?:pgs|sup|vobsub|idx|image|bitmap)\b[)\]]?", re.I)
# A name that says nothing but the kind or the language's code: the language's name says more.
_NO_NAME = re.compile(r"[\[(]?\s*(?:forced|sdh|cc|hi|full|subs?|subtitles?|[a-z]{2,3})\s*[\])]?", re.I)


_SIGNS = re.compile(r"\bsigns?\b", re.I)
MARK = " · OCR"


def label(title: str, lang: Lang, kind: str) -> str:
    """The label of a text copy: the track's own name, else its language's (with SDH or forced, as
    extract names the source's tracks), then a plain "OCR" mark: "English (Forced) · OCR",
    "English · OCR". Words of the image format ("PGS") are left out: the copy is text. A copy of a
    track of signs and songs says Signs ("English · Signs · OCR"): it is attached as forced, and a
    viewer and the catalog tell it from a forced track by its label. A name too long for the
    catalog (LABEL_WIDTH) is cut, and the mark kept."""
    name = _FORMAT_WORDS.sub(" ", title or "").strip(" -–/·|,")
    name = re.sub(r"\s{2,}", " ", name)
    if not name or _NO_NAME.fullmatch(name):
        name = lang.name + {"sdh": " · SDH", "forced": " · forced", "signs": " · Signs"}.get(kind, "")
    elif kind == "signs" and not _SIGNS.search(name):
        name += " · Signs"
    if len(name) + len(MARK) > LABEL_WIDTH:
        name = name[:LABEL_WIDTH - len(MARK) - 1].rstrip() + "…"
    return name + MARK


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
    r = recognize(pictures, lang, model=model, tessdata=tessdata, jobs=jobs, min_confidence=min_confidence,
                  bilingual=bilingual, **kw)
    r.meta.update(meta)
    return lang, r


def read_track(path: str | Path, fmt: str, lang: Lang, tessdata: str | Path | None = None, jobs: int = 0,
               min_confidence: float = MIN_CONFIDENCE, bilingual: bool = False,
               models: str | Path | None = None) -> tuple[Lang, Recognized]:
    """An image subtitle file read into its text copy (read_pictures)."""
    return read_pictures(pictures_of(path, fmt), lang, tessdata, jobs, min_confidence, bilingual, models)
