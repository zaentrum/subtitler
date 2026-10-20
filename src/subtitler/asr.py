"""Speech recognition of a package's main audio.

The audio comes straight from the package's stereo rendition (hls/aN:
init.mp4 plus fMP4 segments), decoded by ffmpeg to 16 kHz mono. Its times
are the title's: the first packet's timestamp (the segment's tfdt) is the
offset of everything recognised. A partial copy of a package (the segments
of a window) works the same way: the longest run of segments present is
used.

Two backends with the same output:
- faster-whisper (CTranslate2): CPU here, CUDA on the GPU node; the target.
- mlx-whisper: Apple's Metal GPU, to see what this Mac's GPU does. It has no
  VAD of its own, so it gets faster-whisper's Silero VAD as clip ranges.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .cues import Word

SAMPLE_RATE = 16000


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Result:
    language: str
    language_probability: float
    words: list[Word]
    segments: list[Segment]
    audio_seconds: float
    seconds: float
    load_seconds: float
    model: str
    backend: str
    settings: dict = field(default_factory=dict)
    offset: float = 0.0

    @property
    def speed(self) -> float:
        """× real time: audio seconds per wall-clock second of recognition."""
        return self.audio_seconds / self.seconds if self.seconds else 0.0

    def shifted(self) -> "Result":
        o = self.offset
        return Result(self.language, self.language_probability,
                      [Word(w.start + o, w.end + o, w.text, w.probability) for w in self.words],
                      [Segment(s.start + o, s.end + o, s.text) for s in self.segments],
                      self.audio_seconds, self.seconds, self.load_seconds, self.model, self.backend,
                      dict(self.settings), 0.0)

    def to_json(self) -> str:
        d = asdict(self)
        d["speed"] = round(self.speed, 2)
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Result":
        d = json.loads(s)
        d.pop("speed", None)
        d["words"] = [Word(**w) for w in d["words"]]
        d["segments"] = [Segment(**x) for x in d["segments"]]
        return cls(**d)


# --- audio -------------------------------------------------------------------

def _playlist(path: Path) -> tuple[str | None, list[tuple[str, float]]]:
    init, segs, dur = None, [], None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#EXT-X-MAP:"):
            init = line.split('URI="', 1)[1].split('"', 1)[0]
        elif line.startswith("#EXTINF:"):
            dur = float(line[8:].split(",")[0])
        elif line and not line.startswith("#"):
            segs.append((line, dur or 0.0))
            dur = None
    return init, segs


def present_run(rendition_dir: Path) -> tuple[str, list[str]]:
    """The init segment and the longest run of consecutive media segments on disk."""
    init, segs = _playlist(rendition_dir / "playlist.m3u8")
    if not init or not (rendition_dir / init).exists():
        raise FileNotFoundError(f"{rendition_dir}: no init segment")
    best: list[str] = []
    run: list[str] = []
    for uri, _ in segs:
        if (rendition_dir / uri).exists():
            run.append(uri)
            if len(run) > len(best):
                best = list(run)
        else:
            run = []
    if not best:
        raise FileNotFoundError(f"{rendition_dir}: no media segments")
    return init, best


def first_pts(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "packet=pts_time",
         "-read_intervals", "%+#1", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True).stdout
    return float(out.strip().splitlines()[0].strip(","))


def extract(rendition_dir: str | Path, wav: str | Path, ffmpeg: str = "ffmpeg") -> float:
    """Decode a package audio rendition (what is on disk of it) to 16 kHz mono WAV.

    Returns the offset in seconds of the WAV's first sample on the title's timeline.
    """
    d = Path(rendition_dir)
    init, segs = present_run(d)
    with tempfile.NamedTemporaryFile(suffix=".mp4") as head:
        head.write((d / init).read_bytes() + (d / segs[0]).read_bytes())
        head.flush()
        offset = first_pts(Path(head.name))
    proc = subprocess.Popen(
        [ffmpeg, "-nostdin", "-v", "error", "-y", "-f", "mp4", "-i", "pipe:0",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav)],
        stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for name in [init, *segs]:
            proc.stdin.write((d / name).read_bytes())
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed decoding {d}")
    return offset


def wav_seconds(path: str | Path) -> float:
    return (Path(path).stat().st_size - 44) / (2 * SAMPLE_RATE)


def load_wav(path: str | Path):
    """16 kHz mono 16-bit WAV as float32 samples (what extract() writes).

    Read here rather than through faster-whisper's PyAV decoder, whose
    1.2.1 release does not run with PyAV 15 and later.
    """
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: want 16 kHz mono 16-bit PCM")
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


# --- backends ----------------------------------------------------------------

DEFAULTS = dict(beam_size=5, vad_min_silence_ms=500, vad_speech_pad_ms=200,
                condition_on_previous_text=False, hallucination_silence_threshold=2.0)


class FasterWhisper:
    backend = "faster-whisper"

    def __init__(self, model: str, device: str = "auto", compute_type: str = "default",
                 cpu_threads: int = 0, batch_size: int = 0):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        t0 = time.perf_counter()
        self.model_name = model
        self.model = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
        self.batch_size = batch_size
        self.pipe = BatchedInferencePipeline(self.model) if batch_size > 0 else None
        self.load_seconds = time.perf_counter() - t0
        self.settings = {"device": device, "compute_type": compute_type, "cpu_threads": cpu_threads,
                         "batch_size": batch_size}

    def transcribe(self, wav: str, language: str | None = None, **opts) -> Result:
        o = {**DEFAULTS, **opts}
        kw = dict(language=language, beam_size=o["beam_size"], word_timestamps=True, vad_filter=True,
                  vad_parameters=dict(min_silence_duration_ms=o["vad_min_silence_ms"],
                                      speech_pad_ms=o["vad_speech_pad_ms"]),
                  condition_on_previous_text=o["condition_on_previous_text"],
                  language_detection_segments=4)
        audio = load_wav(wav)
        t0 = time.perf_counter()
        if self.pipe is not None:
            segs, info = self.pipe.transcribe(audio, batch_size=self.batch_size, **kw)
        else:
            segs, info = self.model.transcribe(
                audio, hallucination_silence_threshold=o["hallucination_silence_threshold"], **kw)
        segments, words = [], []
        for s in segs:  # a generator: recognition happens here
            segments.append(Segment(s.start, s.end, s.text))
            for w in s.words or []:
                words.append(Word(w.start, w.end, w.word, w.probability))
        dt = time.perf_counter() - t0
        return Result(info.language, info.language_probability, words, segments, info.duration, dt,
                      self.load_seconds, self.model_name, self.backend, {**self.settings, **o})


class MlxWhisper:
    backend = "mlx-whisper"

    def __init__(self, model: str):
        self.model_name = model
        self.load_seconds = 0.0
        self.settings = {"device": "metal"}

    def transcribe(self, wav: str, language: str | None = None, **opts) -> Result:
        import mlx_whisper
        import numpy as np
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        o = {**DEFAULTS, **opts}
        audio = load_wav(wav)
        t0 = time.perf_counter()
        speech = get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=o["vad_min_silence_ms"],
                                                         speech_pad_ms=o["vad_speech_pad_ms"]))
        clips: list[float] = []
        for s in speech:
            clips += [s["start"] / SAMPLE_RATE, s["end"] / SAMPLE_RATE]
        r = mlx_whisper.transcribe(
            audio.astype(np.float32), path_or_hf_repo=self.model_name, language=language, verbose=None,
            word_timestamps=True, condition_on_previous_text=o["condition_on_previous_text"],
            clip_timestamps=clips or "0", hallucination_silence_threshold=o["hallucination_silence_threshold"])
        dt = time.perf_counter() - t0
        segments, words = [], []
        for s in r["segments"]:
            segments.append(Segment(s["start"], s["end"], s["text"]))
            for w in s.get("words", []):
                words.append(Word(w["start"], w["end"], w["word"], w.get("probability", 1.0)))
        return Result(r.get("language") or language or "", 1.0 if language else 0.0, words, segments,
                      len(audio) / SAMPLE_RATE, dt, 0.0, self.model_name, self.backend,
                      {**self.settings, **o, "beam_size": 1})


def backend(name: str, model: str, **kw):
    if name == "faster-whisper":
        return FasterWhisper(model, **kw)
    if name == "mlx-whisper":
        return MlxWhisper(model)
    raise ValueError(f"unknown speech backend {name}")


def default_threads() -> int:
    return int(os.environ.get("ASR_THREADS", "0"))
