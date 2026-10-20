"""subtitler: subtitles in the instance's languages for one package.

    subtitler plan PACKAGE --languages en,de,zh-Hans [--original-language it] [--exclude SOURCE ...]
    subtitler run  PACKAGE --out DIR --languages en,de,zh-Hans [options]

`plan` prints what would be done per language. `run` does it and writes
<DIR>/<language>.vtt and <DIR>/subtitles.json (label, origin, source,
models). It never writes into the package: DIR inside it is refused.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

from . import __version__, script, vtt
from .langs import Lang, made_label, parse_list
from .package import Package, load
from .plan import TRANSCRIPT, LanguagePlan, plan


def _print_plan(pkg: Package, plans: list[LanguagePlan]) -> None:
    a = pkg.main_audio
    print(f"package {pkg.root}")
    print(f"main audio: {a.id if a else '-'} ({a.language if a and a.language else 'unknown'})")
    for p in pkg.subtitles:
        flags = ",".join(f for f in [p.format, p.purpose, "forced" if p.forced else "", p.script or "",
                                      p.region or "", "partial" if p.partial else "",
                                      "missing" if not p.exists else ""] if f)
        print(f"  {p.id:7s} {p.language or 'und':4s} {flags:28s} {p.title!r}")
    for lp in plans:
        c = lp.choice
        rest = "; then ".join(x.describe() for x in lp.candidates[1:4])
        print(f"{lp.language.tag:8s} {c.describe() if c else 'nothing to do it from'}" + (f"   [then {rest}]" if rest else ""))


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


class Runner:
    def __init__(self, args):
        self.a = args
        self._asr = None
        self._mt = None
        self.transcript: list[vtt.Cue] | None = None
        self.transcript_lang: str | None = None
        self.asr_info: dict = {}

    # --- speech ---
    def transcribe(self, pkg: Package) -> None:
        if self.transcript is not None:
            return
        from . import asr, cues

        audio = pkg.main_audio
        if audio is None:
            raise SystemExit("no audio rendition to transcribe")
        kw = {}
        if self.a.asr_backend == "faster-whisper":
            kw = dict(device=self.a.asr_device, compute_type=self.a.compute_type, batch_size=self.a.batch_size)
        model = asr.backend(self.a.asr_backend, self.a.asr_model, **kw)
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "audio.wav"
            offset = asr.extract(pkg.root / audio.dir, wav)
            lang = audio.language if audio.language not in (None, "zh") else None  # Chinese: Mandarin or Cantonese
            r = model.transcribe(str(wav), language=lang)
        r.offset = offset
        r = r.shifted()
        detected = r.language
        self.transcript = cues.build(r.words, detected)
        self.transcript_lang = detected
        self.asr_info = {"model": Path(self.a.asr_model).name, "backend": r.backend, "language": detected,
                         "language_probability": round(r.language_probability, 3),
                         "audio_seconds": round(r.audio_seconds, 1), "seconds": round(r.seconds, 1),
                         "offset": round(offset, 3)}
        print(f"transcribed {r.audio_seconds:.0f}s of audio ({detected}) in {r.seconds:.0f}s: "
              f"{len(self.transcript)} cues", file=sys.stderr)

    # --- translation ---
    def translator(self):
        if self._mt is None:
            if self.a.mt == "llm":
                from .translate.llm import Client, LlmTranslator

                client = Client(self.a.llm_url)
                self._mt = LlmTranslator(client, batch=self.a.batch, length_budget=self.a.length_budget)
                self.mt_model = Path(client.model_id()).name
            else:
                from .translate.madlad import MadladTranslator

                self._mt = MadladTranslator(self.a.madlad_model, merge=not self.a.sentences)
                self.mt_model = self._mt.name
        return self._mt

    def run(self, pkg: Package, plans: list[LanguagePlan], out: Path) -> dict:
        from .translate import translate_cues

        made, kept = [], []
        for lp in plans:
            c = lp.choice
            if c is None:
                print(f"{lp.language.tag}: no candidate", file=sys.stderr)
                continue
            if c.action == "keep":
                kept.append({"language": lp.language.tag, "subtitle": c.source})
                continue
            t0 = time.perf_counter()
            entry = {"language": lp.language.tag, "origin": c.origin, "source": c.source,
                     "source_language": c.source_language}
            if c.action == "convert":
                sub = next(s for s in pkg.subtitles if s.id == c.source)
                res = [cue.with_text(script.convert(cue.text, c.detail["config"])) for cue in vtt.read(pkg.root / sub.path)]
                entry.update(model=f"opencc {c.detail['config']}")
            elif c.action == "transcribe":
                self.transcribe(pkg)
                res = self.transcript
                if lp.language.code == "zh":  # Whisper writes either script
                    res = [cue.with_text(script.convert(cue.text, "s2t" if lp.language.script == "Hant" else "t2s"))
                           for cue in res]
                entry.update(model=self.asr_info["model"], asr=self.asr_info)
            else:  # translate
                if c.source == TRANSCRIPT:
                    self.transcribe(pkg)
                    src_lang, cues_in, is_sdh = self.transcript_lang, self.transcript, False
                    entry.update(source_language=src_lang, asr=self.asr_info)
                    if src_lang == lp.language.code:  # detection found the target language itself
                        res = self.transcript
                        if lp.language.code == "zh":
                            res = [cue.with_text(script.convert(cue.text, "s2t" if lp.language.script == "Hant" else "t2s"))
                                   for cue in res]
                        entry.update(origin="transcribed", model=self.asr_info["model"])
                        self._write(out, lp, entry, res, t0, made)
                        continue
                else:
                    sub = next(s for s in pkg.subtitles if s.id == c.source)
                    src_lang, cues_in, is_sdh = sub.language, vtt.read(pkg.root / sub.path), sub.is_sdh
                tr = self.translator()
                res, usage, failures = translate_cues(cues_in, Lang.parse(src_lang or "en"), lp.language, tr,
                                                      is_sdh=is_sdh, sentences=self.a.sentences)
                entry.update(model=self.mt_model, usage=usage.as_dict(), failures=failures)
            self._write(out, lp, entry, res, t0, made)
        manifest = {"package": str(pkg.root), "made_by": f"subtitler {__version__}",
                    "languages": [lp.language.tag for lp in plans], "kept": kept, "made": made,
                    "plan": {lp.language.tag: [x.describe() for x in lp.candidates] for lp in plans}}
        (out / "subtitles.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
        return manifest

    def _write(self, out: Path, lp: LanguagePlan, entry: dict, res: list[vtt.Cue], t0: float, made: list) -> None:
        label = made_label(lp.language, entry["origin"])
        path = out / f"{lp.language.tag}.vtt"
        note = (f"{label}. Made by subtitler {__version__}: {entry['origin']} from {entry['source']}"
                f" ({entry.get('source_language') or '?'}) by {entry.get('model')}.")
        vtt.write(path, res, notes=[note])
        entry.update(label=label, file=path.name, cues=len(res), seconds=round(time.perf_counter() - t0, 1))
        made.append(entry)
        print(f"{lp.language.tag}: {label} <- {entry['origin']} from {entry['source']} ({len(res)} cues)", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="subtitler", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("package")
        p.add_argument("--languages", default="en,de,zh-Hans")
        p.add_argument("--original-language", default=None,
                       help="the title's original language when the catalog knows it (default: the main audio's)")
        p.add_argument("--exclude", nargs="*", default=[], help="sources a viewer reported (next candidate)")
        p.add_argument("--source-order", choices=["original", "english"], default="original",
                       help="original: ADR-0014's order; english: a complete English track first")
        if name == "run":
            p.add_argument("--out", required=True)
            p.add_argument("--mt", choices=["llm", "madlad"], default="llm")
            p.add_argument("--llm-url", default="http://127.0.0.1:8091")
            p.add_argument("--batch", type=int, default=30)
            p.add_argument("--length-budget", action=argparse.BooleanOptionalAction, default=False,
                           help="ask for at most N characters where a cue is short for its text")
            p.add_argument("--sentences", action=argparse.BooleanOptionalAction, default=True,
                           help="translate whole sentences and place them back over their cues")
            p.add_argument("--madlad-model", default=None)
            p.add_argument("--asr-backend", choices=["faster-whisper", "mlx-whisper"], default="faster-whisper")
            p.add_argument("--asr-model", default="large-v3-turbo")
            p.add_argument("--asr-device", default="auto")
            p.add_argument("--compute-type", default="default")
            p.add_argument("--batch-size", type=int, default=0, help="faster-whisper batched inference (GPU: 8-16)")
            p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    pkg = load(a.package)
    plans = plan(pkg, parse_list(a.languages), a.original_language, tuple(a.exclude), a.source_order)
    _print_plan(pkg, plans)
    if a.cmd == "plan":
        return 0
    out = Path(a.out)
    if _inside(out, pkg.root):
        print(f"refusing to write into the package: {out}", file=sys.stderr)
        return 2
    if a.dry_run:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    Runner(a).run(pkg, plans, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
