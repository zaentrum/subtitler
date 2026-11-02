"""OCR against a text track of the same subtitle: character error rate per track and per language.

    python bench/ocr_score.py --pairs pairs.json --out results-dir [--tessdata DIR] [--jobs N] [--sample N]
    python bench/ocr_score.py --pairs pairs.json --out results-dir --rescore      (the copies read before)

pairs.json lists image subtitles beside a text track of the same language
that a title carries too (often the same subtitle, made from the image one):

    [{"n": 0, "lang": "en", "image": "0.sup", "format": "pgs", "text": "0.vtt"}, ...]

Each image track is read by the OCR step as the run reads it; its cues are
paired with the text track's by time (a pair overlaps by half the longer of
the two at least), and per pair the edit distance of their characters is
counted:

- `cer`: tags out, line breaks and runs of spaces as one space, typographic
  forms as plain ones (… as ..., ’ as ', “ ” as ", – — as -); Chinese and
  Japanese without spaces, full-width forms as half-width (NFKC);
- `cer_exact`: tags out, nothing else changed (line breaks count);
- `cer_dialogue`: as `cer`, the annotations of an SDH track ([door slams],
  (laughs), speakers' names) and the dialogue dashes out of both: a text
  track of the dialogue beside an image track for the deaf and hard of
  hearing (labelled alike) shows in `cer`, not here;
- `cer_loose`: as `cer_dialogue`, without case and spaces: what is left are
  letters, digits and punctuation read for others (a text track that writes
  names in lower case shows in the others, not here);
- `same`: the paired cues that read exactly as the text track's (`cer`'s way);
- `tight`: the paired cues whose start and end lie within 50 ms of the text
  track's, the track's own shift aside (the median of its pairs'): a text
  track made from the same subtitle keeps its times, another version of the
  subtitle (a translation of its own, cues cut otherwise) does not;
  `cer_tight` is `cer_dialogue` over them alone, `cer_tight_loose`
  `cer_loose`.
- `lines`: pairs whose lines are as many as the text track's;
- `italic`: of the text track's cues in italics, how many the copy has in
  italics (recall), and of the copy's, how many the text track has (precision).

Pairs are counted only where the two cues overlap: a text track of other
cues (another translation, an SDH track for a dialogue one) shows as few
pairs. Results: <out>/<n>.json per track and <out>/<n>.vtt, the copy (its
cues' confidence and slant in <n>.meta.json), so `--rescore` scores again
without reading; summary.json, and a table on stdout per language and per
confidence band. No subtitle text is printed.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subtitler import ocr, sdh, vtt  # noqa: E402
from subtitler.langs import UNSPACED, Lang  # noqa: E402


def distance(a: str, b: str) -> int:
    """Levenshtein distance of two strings (characters): rapidfuzz's (MIT) where it is installed."""
    if a == b:
        return 0
    if _rapidfuzz is not None:
        return _rapidfuzz.distance(a, b)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


try:
    from rapidfuzz.distance import Levenshtein as _rapidfuzz
except ImportError:  # the pure Python one below
    _rapidfuzz = None

_TYPO = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "„": '"', "–": "-", "—": "-", "‐": "-"})


def norm(text: str, code: str) -> str:
    t = vtt.plain(text).replace("…", "...").translate(_TYPO)
    if code in UNSPACED:
        t = unicodedata.normalize("NFKC", t)
        return re.sub(r"\s+", "", t)
    return re.sub(r"\s+", " ", t).strip()


def exact(text: str) -> str:
    return vtt.plain(text).strip()


def dialogue(text: str, code: str) -> str:
    t = sdh.clean_text(text, drop_lyrics=False)
    t = re.sub(r"(^|\s)[-‐–—]\s*(?=\S)", r"\1", t)  # dialogue dashes out: kept, spaced or not, as either track has them
    return norm(t, code)


def loose(text: str, code: str) -> str:
    return re.sub(r"\s+", "", dialogue(text, code)).lower()


def italic(text: str) -> bool:
    return "<i>" in text


def pairs_of(ocr_cues: list[vtt.Cue], ref: list[vtt.Cue]) -> list[tuple[vtt.Cue, vtt.Cue]]:
    """Each text cue with the copy's cue that overlaps it most, where that is half the longer one."""
    out, j = [], 0
    for r in ref:
        best, most = None, 0.0
        while j < len(ocr_cues) and ocr_cues[j].end < r.start - 5:
            j += 1
        for c in ocr_cues[j:j + 12]:
            ov = min(c.end, r.end) - max(c.start, r.start)
            if ov > most:
                best, most = c, ov
        if best is not None and most >= 0.5 * max(best.end - best.start, r.end - r.start):
            out.append((best, r))
    return out


def score(cues: list[vtt.Cue], ref: list[vtt.Cue], code: str) -> dict:
    ps = pairs_of(cues, ref)
    shift = statistics.median(c.start - r.start for c, r in ps) if ps else 0.0
    shift_end = statistics.median(c.end - r.end for c, r in ps) if ps else 0.0
    e = n = ex = nx = le = ln = de = dn = te = tn = tle = tln = tight = same = same_lines = 0
    it_ref = it_ocr = it_both = 0
    bands: dict[str, list[int]] = {}
    slants: list[tuple[float, bool]] = []
    for c, r in ps:
        a, b = norm(c.text, code), norm(r.text, code)
        d = distance(a, b)
        e, n = e + d, n + len(b)
        dx = distance(exact(c.text), exact(r.text))
        ex, nx = ex + dx, nx + len(exact(r.text))
        da, db = dialogue(c.text, code), dialogue(r.text, code)
        dd = distance(da, db)
        de, dn = de + dd, dn + len(db)
        la, lb = loose(c.text, code), loose(r.text, code)
        if abs(c.start - r.start - shift) <= 0.05 and abs(c.end - r.end - shift_end) <= 0.05:
            tight, te, tn = tight + 1, te + dd, tn + len(db)
            tle, tln = tle + distance(la, lb), tln + len(lb)
        le, ln = le + distance(la, lb), ln + len(lb)
        same += d == 0
        same_lines += len(exact(c.text).split("\n")) == len(exact(r.text).split("\n"))
        ir, io = italic(r.text), italic(c.text)
        it_ref, it_ocr, it_both = it_ref + ir, it_ocr + io, it_both + (ir and io)
        slants.append((c.meta.get("slant", 0.0), ir))
        conf = c.meta.get("conf", 0)
        band = "<50" if conf < 50 else "50-70" if conf < 70 else "70-85" if conf < 85 else "85-90" if conf < 90 else ">=90"
        acc = bands.setdefault(band, [0, 0, 0])
        acc[0], acc[1], acc[2] = acc[0] + d, acc[1] + len(b), acc[2] + 1
    return {"ref_cues": len(ref), "ocr_cues": len(cues), "pairs": len(ps), "paired_share": round(len(ps) / max(1, len(ref)), 3),
            "edits": e, "chars": n, "cer": round(e / max(1, n), 4), "cer_exact": round(ex / max(1, nx), 4),
            "dialogue_edits": de, "dialogue_chars": dn, "cer_dialogue": round(de / max(1, dn), 4),
            "tight": round(tight / max(1, len(ps)), 3), "tight_pairs": tight, "tight_edits": te, "tight_chars": tn,
            "shift": round(shift, 3), "cer_tight": round(te / max(1, tn), 4), "tight_loose_edits": tle,
            "tight_loose_chars": tln,
            "cer_tight_loose": round(tle / max(1, tln), 4),
            "loose_edits": le, "loose_chars": ln, "cer_loose": round(le / max(1, ln), 4),
            "same": round(same / max(1, len(ps)), 3),
            "lines_same": round(same_lines / max(1, len(ps)), 3),
            "italic": {"ref": it_ref, "ocr": it_ocr, "both": it_both,
                       "recall": round(it_both / it_ref, 3) if it_ref else None,
                       "precision": round(it_both / it_ocr, 3) if it_ocr else None},
            "bands": bands, "slants": slants}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tessdata", default=None)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0, help="read every k-th picture only, k so that about N are read")
    ap.add_argument("--line-height", type=float, default=ocr.LINE_HEIGHT)
    ap.add_argument("--psm", type=int, default=0, help="Tesseract's page mode (default: the OCR step's)")
    ap.add_argument("--only", default="", help="pair numbers, comma-separated")
    ap.add_argument("--rescore", action="store_true", help="score the copies a run before wrote, reading nothing")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(a.pairs).parent
    pairs = json.loads(Path(a.pairs).read_text())
    if a.only:
        keep = {int(x) for x in a.only.split(",")}
        pairs = [p for p in pairs if p["n"] in keep]
    results = []
    for p in pairs:
        lang = Lang.parse(p["lang"])
        t0 = time.perf_counter()
        if a.rescore:
            before = json.loads((out / f"{p['n']}.json").read_text())
            cues = vtt.read(out / f"{p['n']}.vtt")
            for c, m in zip(cues, json.loads((out / f"{p['n']}.meta.json").read_text())):
                c.meta = m
            r = ocr.Recognized(cues, before["model"], "", before["confidence"], before["pictures"], before["empty"], 0,
                               before["italic_cues"], 0.0, before["cpu_seconds"], 0.0)
            got, pictures = Lang.parse(before["lang"]), None
        else:
            pictures = ocr.pictures_of(base / p["image"], p.get("format", "pgs"))
            if a.sample and len(pictures) > a.sample:
                pictures = pictures[::len(pictures) // a.sample]
            got, r = ocr.read_pictures(pictures, lang, a.tessdata, a.jobs, min_confidence=0,
                                       bilingual=bool(p.get("bilingual")), line_height=a.line_height,
                                       **({"psm": a.psm} if a.psm else {}))
            vtt.write(out / f"{p['n']}.vtt", r.cues)
            (out / f"{p['n']}.meta.json").write_text(json.dumps([c.meta for c in r.cues]))
        ref = vtt.read(base / p["text"])
        if a.sample and pictures:
            lo, hi = pictures[0].start - 1, (pictures[-1].end or pictures[-1].start) + 1
            ref = [c for c in ref if lo <= c.start <= hi]
        s = score(r.cues, ref, lang.code)
        if pictures is None:  # rescored: what the run before measured
            keep = {k: before[k] for k in ("pictures", "empty", "italic_cues", "cpu_seconds", "wall_seconds",
                                           "shown_hours", "runtime_hours", "bytes") if k in before}
        else:
            keep = {"pictures": len(pictures), "empty": r.empty, "italic_cues": r.italic,
                    "cpu_seconds": round(r.cpu_seconds, 1), "wall_seconds": round(time.perf_counter() - t0, 1),
                    "shown_hours": round(sum(max(0.0, (x.end or x.start + 4) - x.start) for x in pictures) / 3600, 3),
                    "runtime_hours": round(((pictures[-1].end or 0) - pictures[0].start) / 3600, 3) if pictures else 0,
                    "bytes": (base / p["image"]).stat().st_size}
        res = {"n": p["n"], "lang": got.tag, "model": r.model, "confidence": round(r.confidence, 1), **keep,
               "forced": bool(p.get("forced")), **s, **r.meta}
        (out / f"{p['n']}.json").write_text(json.dumps({k: v for k, v in res.items() if k != "slants"}, indent=1,
                                                       ensure_ascii=False))
        results.append(res)
        print(f"{p['n']:>3} {got.tag:8s} conf {r.confidence:5.1f}  pairs {s['pairs']:>5}/{s['ref_cues']:<5} "
              f"cer {100 * s['cer']:5.2f} %  dialogue {100 * s['cer_dialogue']:5.2f} %  loose {100 * s['cer_loose']:5.2f} %  "
              f"tight {100 * s['cer_tight']:5.2f} % of {100 * s['tight']:3.0f} %  "
              f"exact {100 * s['cer_exact']:5.2f} %  "
              f"same {100 * s['same']:5.1f} %  lines {100 * s['lines_same']:5.1f} %  "
              f"italic r/p {s['italic']['recall']}/{s['italic']['precision']}  cpu {r.cpu_seconds:6.1f} s", flush=True)
    by: dict[str, list[dict]] = {}
    for r in results:
        by.setdefault(r["lang"] + (" forced" if r.get("forced") else ""), []).append(r)
    summary = {}
    for lang, rs in sorted(by.items()):
        # The same subtitle: at least half of its cues pair with the text track's at the same times.
        ok = [r for r in rs if r["tight"] >= 0.5]
        te, tn = sum(r["tight_edits"] for r in ok), sum(r["tight_chars"] for r in ok)
        le, ln = sum(r["tight_loose_edits"] for r in ok), sum(r["tight_loose_chars"] for r in ok)
        hours = sum(r["runtime_hours"] for r in rs)
        summary[lang] = {"tracks": len(rs), "same_subtitle": len(ok), "pairs": sum(r["tight_pairs"] for r in ok),
                         "cer": round(te / tn, 4) if tn else None, "cer_loose": round(le / ln, 4) if ln else None,
                         "cer_median": round(statistics.median(r["cer_tight"] for r in ok), 4) if ok else None,
                         "cer_all_pairs": round(sum(r["dialogue_edits"] for r in rs) / max(1, sum(r["dialogue_chars"] for r in rs)), 4),
                         "confidence": round(statistics.mean(r["confidence"] for r in rs), 1),
                         "cpu_seconds_per_runtime_hour": round(sum(r["cpu_seconds"] for r in rs) / max(1e-9, hours), 1),
                         "mb_per_runtime_hour": round(sum(r.get("bytes", 0) for r in rs) / 1e6 / max(1e-9, hours), 1)}
        x = summary[lang]
        print(f"{lang:14s} tracks {len(rs):>2} same subtitle {len(ok):>2} pairs {x['pairs']:>6}  CER {x['cer']}  "
              f"loose {x['cer_loose']}  median {x['cer_median']}  all pairs {x['cer_all_pairs']}  "
              f"confidence {x['confidence']}  CPU s/h {x['cpu_seconds_per_runtime_hour']}  MB/h {x['mb_per_runtime_hour']}")
    bands: dict[str, list[int]] = {}
    for r in results:
        for k, v in r["bands"].items():
            acc = bands.setdefault(k, [0, 0, 0])
            for i in range(3):
                acc[i] += v[i]
    for k in ("<50", "50-70", "70-85", "85-90", ">=90"):
        if k in bands:
            e, n, c = bands[k]
            print(f"confidence {k:6s} cues {c:>6}  CER {100 * e / max(1, n):5.2f} %")
    # Italics: which slant tells them best, over the tracks whose text track marks any.
    sl = [x for r in results if r["italic"]["ref"] for x in r["slants"]]
    for deg in (5, 6, 7, 8, 9, 10):
        tp = sum(1 for a, ir in sl if a >= deg and ir)
        fp = sum(1 for a, ir in sl if a >= deg and not ir)
        fn = sum(1 for a, ir in sl if a < deg and ir)
        print(f"italic at {deg:>2} degrees: precision {tp / max(1, tp + fp):.3f} recall {tp / max(1, tp + fn):.3f} "
              f"({tp + fn} cues in italics of {len(sl)})")
    (out / "summary.json").write_text(json.dumps({"languages": summary, "bands": bands, "tracks": results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
