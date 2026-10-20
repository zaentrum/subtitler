"""Run speech recognition over clips and keep the raw result per clip.

    python bench/asr_run.py --backend faster-whisper --model PATH --out DIR \
        [--compute-type int8] [--threads 4] [--batch-size 0] [--seconds 120] \
        clip.wav[:lang[:offset]] ...

Each clip's result (words, segments, language, timings) goes to
DIR/<clip>.json; a line per clip goes to stdout. --seconds cuts the clip to
its first N seconds (speed runs). Scoring happens afterwards (asr_score.py).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subtitler import asr  # noqa: E402


def cut(wav: str, seconds: float, tmp: Path) -> str:
    out = tmp / (Path(wav).stem + f".first{int(seconds)}.wav")
    with wave.open(wav, "rb") as r, wave.open(str(out), "wb") as w:
        w.setparams(r.getparams())
        w.writeframes(r.readframes(int(seconds * r.getframerate())))
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="faster-whisper")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compute-type", default="default")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("clips", nargs="+", help="wav[:lang[:offset]]; lang 'auto' detects")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    kw = {}
    if a.backend == "faster-whisper":
        kw = dict(device=a.device, compute_type=a.compute_type, cpu_threads=a.threads, batch_size=a.batch_size)
    model = asr.backend(a.backend, a.model, **kw)
    print(f"# {a.backend} {Path(a.model).name} load {model.load_seconds:.1f}s {a.tag}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        for spec in a.clips:
            parts = spec.split(":")
            wav = parts[0]
            lang = parts[1] if len(parts) > 1 and parts[1] not in ("", "auto") else None
            offset = float(parts[2]) if len(parts) > 2 else 0.0
            src = cut(wav, a.seconds, Path(tmp)) if a.seconds else wav
            r = model.transcribe(src, language=lang, beam_size=a.beam_size)
            r.offset = offset
            r.settings["tag"] = a.tag
            name = Path(wav).stem
            (out / f"{name}.json").write_text(r.to_json(), encoding="utf-8")
            print(f"{name}\tlang={r.language}({r.language_probability:.2f})\taudio={r.audio_seconds:.0f}s"
                  f"\twall={r.seconds:.1f}s\tspeed={r.speed:.2f}x\twords={len(r.words)}", flush=True)


if __name__ == "__main__":
    main()
