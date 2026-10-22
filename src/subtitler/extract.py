"""The source's own subtitles first (ADR-0014): a title's embedded subtitle tracks, as WebVTT files.

    subtitler extract --titles titles.json --out DIR [--languages en,de,zh-Hans] [--bundle]

A title that has no package yet plays from its original file, which often
carries subtitles of its own, made by people. Before anything is made with a
model, those are the subtitles a title has: this mode takes the original's
subtitle streams in the instance's languages out of the file as WebVTT, one
file a stream, and makes nothing. On the CPU, one read of the file for all
of its streams (at most `--readrate` times real time), the file opened
read-only.

Which streams: every text stream (SubRip, ASS/SSA, WebVTT, MP4 text) in one of
the languages, dialogue, SDH and forced alike — each its own file, so SDH and
forced tracks keep what they are. Commentary is left out. Chinese is told
apart by script: the track title where it says Simplified or Traditional
(script.hints), else the characters themselves (script.detect_script);
written Cantonese is another language and never stands for zh-Hans. Image
streams (PGS, VobSub) become no text without OCR: they are listed as left out.

Out, per title, in <DIR>/<id>/: <tag>[.sdh][.forced]-<stream>.vtt (a NOTE says
which stream it came from), and extracted.json: per file its language, SDH,
forced, the stream's index, codec and title, cues, and what attaching it
takes (origin "source": made by people, taken from the title's own file).
Nothing is attached here.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import __version__, script, vtt
from .langs import Lang, iso1, parse_list
from .original import probe

TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}
_COMMENTARY = ("commentary", "comment", "kommentar")


def say(line: str) -> None:
    print(time.strftime("%H:%M:%S ", time.gmtime()) + line, flush=True)


@dataclass
class SubStream:
    index: int
    codec: str
    language: str | None  # ISO 639-1 where there is one
    title: str
    forced: bool
    sdh: bool
    commentary: bool
    default: bool

    @property
    def text(self) -> bool:
        return self.codec in TEXT_CODECS


def subtitle_streams(info: dict) -> list[SubStream]:
    out = []
    for s in info.get("streams", []):
        if s.get("codec_type") != "subtitle":
            continue
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
        disp = s.get("disposition") or {}
        title = (tags.get("title") or "").strip()
        low = title.lower()
        out.append(SubStream(index=int(s["index"]), codec=(s.get("codec_name") or "").lower(),
                             language=iso1(tags.get("language")), title=title,
                             forced=bool(disp.get("forced")) or "forced" in low,
                             sdh=bool(disp.get("hearing_impaired")) or "sdh" in low or "hard of hearing" in low
                             or "captions" in low,
                             commentary=bool(disp.get("comment")) or any(w in low for w in _COMMENTARY),
                             default=bool(disp.get("default"))))
    return out


@dataclass
class Pick:
    stream: SubStream
    lang: Lang  # the instance's language it serves (zh-Hans for a Simplified track)
    script_from: str  # "title", "text" or "" (no script to tell)

    @property
    def name(self) -> str:
        kind = (".sdh" if self.stream.sdh else "") + (".forced" if self.stream.forced else "")
        return f"{self.lang.tag}{kind}-{self.stream.index}.vtt"


def choose(streams: list[SubStream], langs: list[Lang]) -> tuple[list[Pick], list[dict]]:
    """The streams to take out for the languages, and the ones of those languages left out, with why.

    A Chinese stream whose title names no script is picked for the Chinese language wanted and its
    script checked against the text after extraction (finish)."""
    picks, left = [], []
    for s in streams:
        want = [l for l in langs if l.code == s.language]
        if not want:
            continue
        if s.commentary:
            left.append({"stream": s.index, "why": "commentary"})
            continue
        if not s.text:
            left.append({"stream": s.index, "why": f"an image subtitle ({s.codec}): no text without OCR"})
            continue
        if s.language == "zh":
            h = script.hints(s.title)
            if h["cantonese"]:
                left.append({"stream": s.index, "why": "written Cantonese"})
                continue
            if h["script"]:
                match = [l for l in want if (l.script or "Hans") == h["script"]]
                if not match:
                    left.append({"stream": s.index, "why": f"Chinese in the other script ({h['script']})"})
                    continue
                picks.append(Pick(s, match[0], "title"))
                continue
            picks.append(Pick(s, want[0], ""))
            continue
        picks.append(Pick(s, want[0], ""))
    return picks, left


def extract(path: str | Path, picks: list[Pick], folder: Path, readrate: float | None = None,
            ffmpeg: str = "ffmpeg") -> None:
    """All picked streams in one read of the file, each to its WebVTT file."""
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-y"]
    if readrate:
        cmd += ["-readrate", f"{readrate:g}"]
    cmd += ["-i", str(path)]
    for p in picks:
        cmd += ["-map", f"0:{p.stream.index}", "-c:s", "webvtt", "-f", "webvtt", str(folder / (p.name + ".part"))]
    subprocess.run(cmd, check=True)


def finish(picks: list[Pick], folder: Path, title_id: str) -> tuple[list[dict], list[dict]]:
    """Each extracted file checked (cues, Chinese script by its characters), given its NOTE, and listed."""
    made, left = [], []
    for p in picks:
        part = folder / (p.name + ".part")
        cues = vtt.read(part) if part.exists() else []
        if not cues:
            left.append({"stream": p.stream.index, "why": "no cue in the stream"})
            part.unlink(missing_ok=True)
            continue
        if p.stream.language == "zh":
            text = "\n".join(vtt.plain(c.text) for c in cues)
            if script.looks_cantonese(text):
                left.append({"stream": p.stream.index, "why": "written Cantonese, by its characters"})
                part.unlink()
                continue
            found, conf = script.detect_script(text)
            if p.script_from == "title" and found and found != (p.lang.script or "Hans") and conf > 0.02:
                left.append({"stream": p.stream.index, "why": f"its title says {p.lang.script}, its characters {found}"})
                part.unlink()
                continue
            if p.script_from == "":
                if found and found != (p.lang.script or "Hans"):
                    left.append({"stream": p.stream.index, "why": f"Chinese in the other script ({found}, by its characters)"})
                    part.unlink()
                    continue
                p.script_from = "text" if found else ""
        kind = " · ".join(k for k, on in (("SDH", p.stream.sdh), ("forced", p.stream.forced)) if on)
        label = f"{p.lang.name}" + (f" · {kind}" if kind else "")
        what = f"stream {p.stream.index} ({p.stream.codec}" + (f", '{p.stream.title}'" if p.stream.title else "") + ")"
        out = folder / p.name
        vtt.write(out, cues, notes=[f"{label}. Extracted by subtitler {__version__} from the title's original file, "
                                    f"its subtitle {what}: the source's own, made by people."])
        part.unlink()
        made.append({"file": out.name, "language": p.lang.tag, "sdh": p.stream.sdh, "forced": p.stream.forced,
                     "stream": p.stream.index, "codec": p.stream.codec, "title": p.stream.title or None,
                     "scriptFrom": p.script_from or None, "cues": len(cues),
                     "first": vtt.timestamp(cues[0].start), "last": vtt.timestamp(cues[-1].end),
                     "attach": {"language": p.lang.tag, "origin": "source", "sdh": p.stream.sdh,
                                "forced": p.stream.forced, "format": "webvtt"}})
    return made, left


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="subtitler extract", description=__doc__.split("\n\n")[0])
    ap.add_argument("--titles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--languages", default="en,de,zh-Hans")
    ap.add_argument("--readrate", type=float, default=30.0)
    ap.add_argument("--bundle", action="store_true")
    a = ap.parse_args(argv)
    langs = parse_list(a.languages)
    titles = json.loads(Path(a.titles).read_text())
    say(f"subtitler {__version__} extract: {len(titles)} titles, languages {', '.join(x.tag for x in langs)}")
    results, rc = [], 0
    for t in titles:
        folder = Path(a.out) / t["id"]
        folder.mkdir(parents=True, exist_ok=True)
        name = t.get("name") or t["id"]
        t0 = time.perf_counter()
        try:
            streams = subtitle_streams(probe(t["original"]))
            picks, left = choose(streams, langs)
            say(f"{name}: {len(streams)} subtitle streams; taking out " +
                (", ".join(f"{p.stream.index} {p.lang.tag}{' sdh' if p.stream.sdh else ''}{' forced' if p.stream.forced else ''}"
                           for p in picks) or "none"))
            if picks:
                extract(t["original"], picks, folder, a.readrate)
            made, more = finish(picks, folder, t["id"])
            left += more
            missing = [l.tag for l in langs if not any(m["language"] == l.tag and not m["forced"] for m in made)]
            r = {"id": t["id"], "name": name, "made_by": f"subtitler {__version__}", "streams": len(streams),
                 "made": made, "left_out": left, "languages_without_a_track": missing,
                 "seconds": round(time.perf_counter() - t0, 1)}
        except Exception as e:  # noqa: BLE001 - one title's failure is not the run's
            r = {"id": t["id"], "name": name, "error": repr(e)}
            rc = 1
        (folder / "extracted.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        results.append(r)
        say(f"{name}: " + (f"failed: {r['error']}" if "error" in r else
                           f"{len(r['made'])} files in {r['seconds']:.0f} s: " +
                           ", ".join(f"{m['file']} ({m['cues']} cues)" for m in r["made"]) +
                           (f"; no track for {', '.join(r['languages_without_a_track'])}" if r["languages_without_a_track"] else "")))
        if a.bundle:
            bundle(t["id"], folder)
    (Path(a.out) / "run.json").write_text(json.dumps({"made_by": f"subtitler {__version__}", "mode": "extract",
                                                      "titles": results}, indent=1, ensure_ascii=False))
    say("run: " + "; ".join(f"{r['name']} {'ok' if 'error' not in r else 'ERROR'}" for r in results))
    return rc


if __name__ == "__main__":
    sys.exit(main())
