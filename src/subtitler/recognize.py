"""Text copies of the image subtitles of packaged titles (OCR), title by title, on the CPU.

    subtitler ocr --titles titles.json --out DIR [--languages en,de,zh-Hans] [options]

titles.json lists the titles in the order they are to be done, each by its
version folder (the package: package.json and its subs/):

    [{"id": "<item id>", "name": "<what the log calls it>", "package": "<version folder>"}, ...]

Per title the copies plan.copies names (an image subtitle in one of the
languages, of a kind the title has no text subtitle of), each read by
ocr.read_track and written to <DIR>/<id>/<tag>[.sdh|.forced|.signs].vtt with
a NOTE that says how it was made; ocr.json lists each file with what
attaching it takes (origin `recognized`, its language, label, forced and
SDH) and the copies not attached, with why (confidence too low, no model for
the language). Nothing is attached here, and nothing is written into the
package.

Paced for a shared store: titles are read only within `--hours` (UTC, by
default 22-17, so never in the evening's 17-22), each file at most
`--read-mbps`, a pause between titles. A title whose ocr.json is in <DIR>
from a run before is not read again (`--again` reads it again), so a run cut
short by its deadline goes on where it stopped. `--dry-run` reads
package.json alone and says what would be read, how much, and in which
language.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import tarfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import __version__, ocr, vtt
from .langs import Lang, parse_list
from .package import load
from .plan import Copy, copies


def say(line: str) -> None:
    print(time.strftime("%H:%M:%S ", time.gmtime()) + line, flush=True)


SUFFIX = {"sdh": ".sdh", "forced": ".forced", "signs": ".signs"}


def file_name(lang: Lang, kind: str) -> str:
    return f"{lang.tag}{SUFFIX.get(kind, '')}.vtt"


def entry(c: Copy, lang: Lang, r: ocr.Recognized, version: str = __version__) -> dict:
    """A copy's line in the manifest: its file, label, how it was read, the NOTE of its file, and
    what attaching it takes."""
    lbl = ocr.label(c.sub.title, lang, c.kind)
    what = f"{c.sub.id} ({c.sub.format}" + (f", '{c.sub.title}'" if c.sub.title else "") + ")"
    note = (f"{lbl}. Made by subtitler {version}: the text of the title's own image subtitle {what}, "
            f"recognized (OCR) by {r.engine} with its {r.model} model; mean confidence {r.confidence:.1f}.")
    forced, sdh = c.kind == "forced", c.kind == "sdh"
    return {"file": file_name(lang, c.kind), "language": lang.tag, "label": lbl, "origin": "recognized",
            "source": c.sub.id, "source_path": c.sub.path, "kind": c.kind, "forced": forced, "sdh": sdh,
            "model": f"{r.engine} ({r.model})", "cues": len(r.cues), "confidence": round(r.confidence, 1),
            "ocr": r.summary(), "note": note,
            "attach": {"language": lang.tag, "label": lbl, "origin": "recognized", "forced": forced, "sdh": sdh,
                       "format": "webvtt"}}


# --- pacing --------------------------------------------------------------------

def parse_hours(spec: str) -> tuple[int, int]:
    """"22-17": from 22:00 to 17:00 UTC, over midnight; "0-24": always."""
    a, b = (int(x) for x in spec.split("-"))
    if not (0 <= a <= 24 and 0 <= b <= 24):
        raise ValueError(f"hours {spec!r}: two hours of the day, 0-24")
    return a, b


def within(hours: tuple[int, int], now: datetime) -> bool:
    a, b = hours
    h = now.hour + now.minute / 60
    if a == b or (a, b) == (0, 24):
        return True
    return a <= h < b if a < b else (h >= a or h < b)


def until_open(hours: tuple[int, int], now: datetime) -> float:
    """Seconds until the hours begin again (0 within them)."""
    if within(hours, now):
        return 0.0
    start = now.replace(hour=hours[0] % 24, minute=0, second=0, microsecond=0)
    if start <= now:
        start += timedelta(days=1)
    return (start - now).total_seconds()


def paced_read(path: Path, mbps: float | None) -> bytes:
    """A file read at most mbps megabytes a second (all at once without a limit)."""
    if not mbps:
        return path.read_bytes()
    out = bytearray()
    chunk = 1 << 20
    t0 = time.monotonic()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            out += b
            ahead = len(out) / (mbps * 1e6) - (time.monotonic() - t0)
            if ahead > 0:
                time.sleep(ahead)
    return bytes(out)


def _read_pictures(c: Copy, root: Path, read) -> list:
    from . import pgs, vobsub

    path = root / c.sub.path
    if c.sub.format.lower() in ("vobsub", "dvd_subtitle", "idx"):
        return vobsub.pictures(read(path).decode("latin-1"), read(path.with_suffix(".sub")))
    return pgs.pictures(read(path))


def _size(c: Copy, root: Path) -> int:
    path = root / c.sub.path
    try:
        n = path.stat().st_size
        if c.sub.format.lower() in ("vobsub", "dvd_subtitle", "idx"):
            n += path.with_suffix(".sub").stat().st_size
        return n
    except OSError:
        return 0


# --- a title -----------------------------------------------------------------

def bundle(item: str, folder: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in sorted(folder.iterdir()):
            if f.suffix in (".vtt", ".json"):
                tar.add(f, arcname=f"{item}/{f.name}")
    data = base64.b64encode(buf.getvalue()).decode()
    print(f"-----BEGIN SUBTITLER OUTPUT {item}-----", flush=True)
    for i in range(0, len(data), 76):
        print(data[i:i + 76])
    print(f"-----END SUBTITLER OUTPUT {item}-----", flush=True)


def run_title(t: dict, langs: list[Lang], a, folder: Path) -> dict:
    root = Path(t["package"])
    kept: dict[Path, bytes] = {}

    def read(path: Path) -> bytes:  # each file once, paced; what counting read the OCR step reads
        if path not in kept:
            kept[path] = paced_read(path, a.read_mbps)
        return kept[path]

    # The pictures of image subtitles in the languages counted (a read of each), the others' not read.
    pkg = load(root, {l.code for l in langs}, read, count=not a.dry_run)
    todo = copies(pkg, langs)
    r = {"id": t["id"], "name": t.get("name") or t["id"], "made_by": f"subtitler {__version__}", "made": [],
         "not_attached": [], "copies": [{"source": c.sub.id, "format": c.sub.format, "language": c.lang.tag
                                         if c.lang.script else c.lang.code, "kind": c.kind, "pictures": c.sub.cues,
                                         "bytes": _size(c, root)} for c in todo]}
    if a.dry_run:
        r["dry_run"] = True
        r["model"] = {c.sub.id: (ocr.model_for(c.lang) or None) if c.lang.code != "zh" or c.lang.script
                      else "chi_sim or chi_tra (read to tell)" for c in todo}
        return r
    for c in todo:
        t0 = time.perf_counter()
        try:
            lang, res = ocr.read_pictures(_read_pictures(c, root, read), c.lang, a.tessdata, a.jobs,
                                          a.min_confidence, c.sub.bilingual, a.models)
        except (LookupError, ValueError, OSError, RuntimeError) as e:
            r["not_attached"].append({"source": c.sub.id, "language": c.lang.tag, "kind": c.kind, "why": str(e)})
            say(f"{r['name']}: {c.sub.id}: no text copy: {e}")
            continue
        e = entry(c, lang, res)
        note = e.pop("note")
        e["seconds"] = round(time.perf_counter() - t0, 1)
        if res.accepted:
            vtt.write(folder / e["file"], res.cues, notes=[note])
            r["made"].append(e)
        else:
            e["why"] = res.rejected
            e["file"] = "not-attached-" + e["file"]  # written aside, for a person to look at
            vtt.write(folder / e["file"], res.cues, notes=[note])
            r["not_attached"].append(e)
        say(f"{r['name']}: {c.sub.id} {lang.tag} {c.kind}: {len(res.cues)} cues, confidence {res.confidence:.1f}, "
            f"{res.cpu_seconds:.0f} CPU s" + (f"; not attached: {res.rejected}" if not res.accepted else ""))
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="subtitler ocr", description=__doc__.split("\n\n")[0])
    ap.add_argument("--titles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--languages", default="en,de,zh-Hans")
    ap.add_argument("--tessdata", default=None, help="Tesseract's models (default: its own)")
    ap.add_argument("--models", default=None, help="a folder a model Tesseract lacks is fetched into (<DIR>/tessdata)")
    ap.add_argument("--jobs", type=int, default=0, help="Tesseract processes at once (default: CPUs - 1)")
    ap.add_argument("--min-confidence", type=float, default=ocr.MIN_CONFIDENCE)
    ap.add_argument("--hours", default="22-17", help="UTC hours the store is read in (default: not 17-22)")
    ap.add_argument("--read-mbps", type=float, default=20.0, help="read a file at most this many MB/s")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between titles")
    ap.add_argument("--dry-run", action="store_true", help="package.json alone: what would be read")
    ap.add_argument("--again", action="store_true", help="read titles again that a run before did")
    ap.add_argument("--bundle", action="store_true", help="write each title's files to the log as well")
    a = ap.parse_args(argv)
    langs = parse_list(a.languages)
    hours = parse_hours(a.hours)
    titles = json.loads(Path(a.titles).read_text())
    say(f"subtitler {__version__} ocr: {len(titles)} titles, languages {', '.join(x.tag for x in langs)}, "
        f"hours {a.hours} UTC" + (", dry run" if a.dry_run else ""))
    results, rc = [], 0
    for t in titles:
        done = Path(a.out) / t["id"] / "ocr.json"
        if not a.dry_run and not a.again and done.exists():
            before = json.loads(done.read_text())
            if "error" not in before and not before.get("dry_run"):
                results.append(before)
                continue
        wait = 0.0 if a.dry_run else until_open(hours, datetime.now(timezone.utc))
        if wait:
            say(f"outside the hours {a.hours} UTC: waiting {wait / 3600:.1f} h")
            time.sleep(wait)
        folder = Path(a.out) / t["id"]
        folder.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        try:
            r = run_title(t, langs, a, folder)
        except Exception as e:  # noqa: BLE001 - one title's failure is not the run's
            r = {"id": t["id"], "name": t.get("name") or t["id"], "error": repr(e)}
            rc = 1
        r["seconds"] = round(time.perf_counter() - t0, 1)
        (folder / "ocr.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        results.append(r)
        if "error" in r:
            say(f"{r['name']}: failed: {r['error']}")
        elif a.dry_run:
            what = ", ".join(f"{c['source']} ({c['format']}, {c['language']}, {c['kind']}, {c['bytes'] / 1e6:.1f} MB)"
                             for c in r["copies"])
            say(f"{r['name']}: would read {what or 'nothing'}")
        if a.bundle and not a.dry_run:
            bundle(t["id"], folder)
        if not a.dry_run and a.pause:
            time.sleep(a.pause)
    summary = {"made_by": f"subtitler {__version__}", "mode": "ocr" + (" dry-run" if a.dry_run else ""),
               "titles": results}
    (Path(a.out) / "run.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    made = sum(len(r.get("made", [])) for r in results)
    left = sum(len(r.get("not_attached", [])) for r in results)
    say(f"run: {len(results)} titles, {made} text copies, {left} not attached, "
        f"{sum('error' in r for r in results)} failed")
    return rc


if __name__ == "__main__":
    sys.exit(main())
