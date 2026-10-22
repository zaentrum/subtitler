"""The main audio of an original file: a title that has no package yet.

A title taken into the library without a package plays from its original
file (Matroska, MP4, ...). Its subtitles are made from that file's main
audio: the audio stream in the title's language that a viewer hears (not
commentary, not audio description), the default one first, then the one
with the most channels.

ffmpeg reads the file once from start to end and decodes that one stream
to 16 kHz mono; the file is opened read-only and nothing is written next to
it. `readrate` caps the read at that many times real time, so a shared
store is not read flat out while others play from it. A surround track is
mixed down with the centre channel first, where dialogue sits, which keeps
music and effects in the side channels quieter for speech recognition.

Times are the file's: the stream's start time is the offset of everything
recognised.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .asr import SAMPLE_RATE
from .langs import iso1

# Words in a stream's title that say it is not the dialogue a viewer hears.
_NOT_MAIN = ("commentary", "comment", "description", "descriptive", "described", "narration for the blind",
             "kommentar", "audiodeskription", "hörfilm")


@dataclass(frozen=True)
class AudioStream:
    index: int  # the stream's index in the file (ffmpeg's 0:<index>)
    language: str | None  # ISO 639-1 where there is one
    codec: str
    channels: int
    layout: str
    default: bool
    title: str
    start: float  # seconds: the stream's first timestamp on the file's timeline
    other: bool = False  # flagged as commentary or audio description

    @property
    def main(self) -> bool:
        t = self.title.lower()
        return not self.other and not any(w in t for w in _NOT_MAIN)

    def describe(self) -> str:
        return (f"0:{self.index} {self.language or 'und'} {self.codec} {self.layout or self.channels} ch"
                + (" default" if self.default else "") + (" (not main)" if not self.main else "")
                + (f" '{self.title}'" if self.title else ""))


def probe(path: str | Path, ffprobe: str = "ffprobe") -> dict:
    out = subprocess.run([ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
                          str(path)], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def audio_streams(info: dict) -> list[AudioStream]:
    out = []
    for s in info.get("streams", []):
        if s.get("codec_type") != "audio":
            continue
        tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
        disp = s.get("disposition") or {}
        try:
            start = float(s.get("start_time") or 0.0)
        except ValueError:
            start = 0.0
        out.append(AudioStream(index=int(s["index"]), language=iso1(tags.get("language")),
                               codec=s.get("codec_name") or "", channels=int(s.get("channels") or 0),
                               layout=s.get("channel_layout") or "", default=bool(disp.get("default")),
                               title=(tags.get("title") or "").strip(), start=start,
                               other=bool(disp.get("comment") or disp.get("visual_impaired"))))
    return out


def main_audio(streams: list[AudioStream], language: str | None = "en") -> AudioStream | None:
    """The stream to transcribe: the title's language when it has a stream in it, else the default one."""
    main = [s for s in streams if s.main] or list(streams)
    if not main:
        return None
    if language:
        want = iso1(language)
        same = [s for s in main if s.language == want]
        if same:
            main = same
    return sorted(main, key=lambda s: (not s.default, -s.channels, s.index))[0]


def downmix(stream: AudioStream) -> list[str]:
    """ffmpeg arguments that make one channel of it: centre first where there is one."""
    layout = stream.layout.lower()
    if stream.channels > 2 and any(layout.startswith(x) for x in ("5.1", "6.1", "7.1")):
        # "<" renormalises the gains: FC 0.625, FL and FR 0.1875 each; surrounds and LFE left out.
        return ["-af", "pan=mono|c0<FC+0.3*FL+0.3*FR"]
    return ["-ac", "1"]


def extract(path: str | Path, stream: AudioStream, wav: str | Path, readrate: float | None = None,
            ffmpeg: str = "ffmpeg") -> float:
    """Decode the stream to 16 kHz mono WAV; returns its offset on the file's timeline."""
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-y"]
    if readrate:
        cmd += ["-readrate", f"{readrate:g}"]
    cmd += ["-i", str(path), "-map", f"0:{stream.index}", "-vn", "-sn", "-dn", *downmix(stream),
            "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav)]
    subprocess.run(cmd, check=True)
    return stream.start
