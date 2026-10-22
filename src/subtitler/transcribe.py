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
  (`--check-model`, decoded one window at a time), for the agreement check;
- `filled`: stretches of speech the batched pass gave (almost) no words
  for, decoded again one window at a time by the same model, whose words
  took their place. Batched decoding drops a whole window now and then
  (a scene with music under quiet dialogue): this pass gets it back.
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


def uncovered(speech: list[tuple[float, float]], words: list, min_len: float = 1.0, covered: float = 0.2,
              join: float = 2.0, longest: float = 28.0, pad: float = 0.3) -> list[tuple[float, float]]:
    """Windows of speech the words leave (almost) uncovered: stretches of `min_len` s and more with under
    `covered` of their time under words, the ones less than `join` s apart in one window of at most
    `longest` s, padded by `pad` s."""
    spans = sorted((w.start, w.end) for w in words)
    out: list[list[float]] = []
    for s, e in speech:
        if e - s < min_len:
            continue
        under = sum(max(0.0, min(e, b) - max(s, a)) for a, b in spans if a < e and b > s)
        if under >= covered * (e - s):
            continue
        if out and s - out[-1][1] < join and e - out[-1][0] <= longest:
            out[-1][1] = e
        else:
            out.append([s, min(e, s + longest)])
    return [(max(0.0, s - pad), e + pad) for s, e in out]


def fill(model, audio, result: asr.Result, speech: list[tuple[float, float]], language: str | None) -> list[dict]:
    """Decode the uncovered windows one at a time and put their words in place of what was there."""
    from .cues import Word

    filled = []
    for s, e in uncovered(speech, result.words):
        seg = audio[int(s * SR): int(e * SR)]
        segs, _ = model.model.transcribe(seg, language=language or result.language, beam_size=5, word_timestamps=True,
                                         vad_filter=True, condition_on_previous_text=False, no_speech_threshold=None)
        words = [Word(w.start + s, w.end + s, w.word, w.probability) for x in segs for w in (x.words or [])]
        words = [w for w in words if w.probability >= 0.3 or len(w.text.strip()) > 2]
        text = "".join(w.text for w in words).strip()
        filled.append({"start": round(s, 3), "end": round(e, 3), "words": len(words), "text": text[:300]})
        if not words:
            continue
        result.words = [w for w in result.words if not (s <= (w.start + w.end) / 2 < e)] + words
        result.words.sort(key=lambda w: w.start)
    return filled


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
    f0 = time.perf_counter()
    filled = fill(model, audio, r, speech, a.language)
    out = {"result": json.loads(r.to_json()), "detected": {"language": detected[0], "probability": round(detected[1], 3)},
           "speech": [[round(s, 3), round(e, 3)] for s, e in speech], "check": None,
           "filled": {"windows": filled, "seconds": round(time.perf_counter() - f0, 1)}}
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
    got = sum(1 for x in filled if x["words"])
    print(f"transcribed {r.audio_seconds:.0f}s of audio ({r.language}, detected {detected[0]} "
          f"p={detected[1]:.2f}) in {r.seconds:.0f}s: {len(r.words)} words; {len(filled)} windows of speech "
          f"without words decoded again, {got} gave words", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
