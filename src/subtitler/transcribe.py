"""One title's speech recognition, as a process of its own.

`subtitler transcribe` runs in its own process so that the GPU memory it
takes is given back the moment it ends: a run alternates speech
recognition and translation on one GPU and never holds both models at
once.

It writes one JSON file:
- `result`: the recognised words and segments (asr.Result, times on the WAV's
  timeline), with the model, the settings and the seconds it took;
- `detected`: the language the model hears in the audio and how sure it is,
  even when the language was given;
- `speech`: the stretches Silero VAD finds speech in, for the timing check;
- `check`: windows of the audio transcribed a second time by another model
  (`--check-model`, decoded one window at a time), for the agreement check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import asr

SR = asr.SAMPLE_RATE


def speech_stretches(audio) -> list[tuple[float, float]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    ts = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500, speech_pad_ms=100))
    return [(t["start"] / SR, t["end"] / SR) for t in ts]


def check_windows(speech: list[tuple[float, float]], n: int, length: float = 30.0) -> list[tuple[float, float]]:
    """n windows of `length` seconds spread evenly over the speech, each starting where a stretch starts."""
    if not speech or n <= 0:
        return []
    total = sum(e - s for s, e in speech)
    out: list[tuple[float, float]] = []
    acc, k = 0.0, 0
    targets = [total * (i + 0.5) / n for i in range(n)]
    for s, e in speech:
        while k < len(targets) and acc + (e - s) >= targets[k]:
            start = max(0.0, s - 0.5)
            if not out or start >= out[-1][1]:
                out.append((start, start + length))
            k += 1
        acc += e - s
    return out


def detect(model, audio) -> tuple[str, float]:
    try:
        lang, prob, _ = model.model.detect_language(audio, vad_filter=True, language_detection_segments=4)
        return lang, float(prob)
    except Exception:  # an older faster-whisper: the first 30 s of speech
        _, info = model.model.transcribe(audio[: 30 * SR], vad_filter=True)
        return info.language, float(info.language_probability)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="subtitler transcribe")
    ap.add_argument("wav")
    ap.add_argument("--model", required=True, help="the speech model's folder (faster-whisper)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--language", default=None, help="the audio's language when known (ISO 639-1)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--check-model", default=None, help="a second speech model for the agreement check")
    ap.add_argument("--check-windows", type=int, default=10)
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    audio = asr.load_wav(a.wav)
    speech = speech_stretches(audio)
    model = asr.FasterWhisper(a.model, device=a.device, compute_type=a.compute_type, batch_size=a.batch_size)
    detected = detect(model, audio)
    r = model.transcribe(a.wav, language=a.language)
    out = {"result": json.loads(r.to_json()), "detected": {"language": detected[0], "probability": round(detected[1], 3)},
           "speech": [[round(s, 3), round(e, 3)] for s, e in speech], "check": None}
    del model
    if a.check_model:
        c0 = time.perf_counter()
        check = asr.FasterWhisper(a.check_model, device=a.device, compute_type=a.compute_type, batch_size=0)
        windows = []
        for s, e in check_windows(speech, a.check_windows):
            seg = audio[int(s * SR): int(e * SR)]
            segs, _ = check.model.transcribe(seg, language=a.language or r.language, beam_size=5, vad_filter=True,
                                             condition_on_previous_text=False)
            windows.append({"start": round(s, 3), "end": round(e, 3), "text": " ".join(x.text.strip() for x in segs)})
        out["check"] = {"model": Path(a.check_model).name, "windows": windows,
                        "seconds": round(time.perf_counter() - c0, 1)}
        del check
    out["seconds"] = round(time.perf_counter() - t0, 1)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False))
    print(f"transcribed {r.audio_seconds:.0f}s of audio ({r.language}, detected {detected[0]} "
          f"p={detected[1]:.2f}) in {r.seconds:.0f}s: {len(r.words)} words", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
