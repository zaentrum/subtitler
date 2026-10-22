"""A run over titles that have no subtitle at all: transcribe, translate, check.

    subtitler job --titles titles.json --out DIR --models DIR [options]

titles.json lists the titles in the order they are to be done:

    [{"id": "<item id>", "name": "<what the log calls it>", "original": "<file>", "language": "en"}, ...]

For each title, on one GPU, one model at a time:

1. its original's main audio to 16 kHz mono (original.py), read ahead of
   the GPU work, one title in advance, at most `--readrate` times real time;
2. speech recognition by Whisper large-v3 in a process of its own
   (transcribe.py), which gives its GPU memory back when it ends; the
   transcript becomes the subtitle in the audio's language (origin
   `transcribed`);
3. translation of the transcript into the other languages by an instruction
   model behind llama.cpp's server, `--chains` parts of the title at once,
   whole sentences placed back over their cues (origin `translated`);
4. the checks (qc.py), and <DIR>/<id>/: one WebVTT file a language
   (<tag>.vtt, with a NOTE saying how it was made), subtitles.json (label,
   origin, models, GPU seconds a step) and qc.json.

Nothing is attached anywhere: the files are the run's output. With
`--bundle` each title's files are also written to the log, as a base64
tar.gz between BEGIN and END lines, for whoever collects them.

Live playback first (ADR-0013): before every GPU step and every
translation request the run looks at the live worker's queue on the stream
service's metrics (`--live-metrics`) and waits while segments a viewer is
waiting for are being made, at most `--live-wait` seconds at a time; and it
starts a model only when the GPU has its memory free plus `--vram-reserve`
MiB left for the live worker.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__, cues, models, original, qc, script, vtt
from .langs import Lang, made_label, parse_list


def say(line: str = "") -> None:
    print(time.strftime("%H:%M:%S ", time.gmtime()) + line, flush=True)


# --- the GPU and live playback -------------------------------------------------

def gpu_memory() -> tuple[int, int]:
    """(used, free) MiB of the GPU; (-1, -1) when nvidia-smi cannot say."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        used, free = (int(x.strip()) for x in out.splitlines()[0].split(","))
        return used, free
    except Exception:
        return -1, -1


_INFLIGHT = re.compile(r'^katalog_stream_live_worker_inflight\{[^}]*priority="playhead"[^}]*\}\s+([0-9.eE+-]+)', re.M)


class LiveFirst:
    """Waits while the live worker makes segments a viewer is waiting for."""

    def __init__(self, url: str | None, most: float = 20.0):
        self.url = url
        self.most = most
        self.off = not url
        self.waited = 0.0
        self.waits = 0
        self._lock = threading.Lock()
        self._seen = (0.0, 0.0)  # (when, value)

    def playhead(self) -> float:
        if self.off:
            return 0.0
        now = time.monotonic()
        with self._lock:
            if now - self._seen[0] < 1.0:
                return self._seen[1]
            try:
                with urllib.request.urlopen(self.url, timeout=3) as r:
                    m = _INFLIGHT.search(r.read().decode("utf-8", "replace"))
                v = float(m.group(1)) if m else 0.0
            except Exception as e:
                say(f"live first: the live worker's metrics at {self.url} cannot be read ({e}); going on without")
                self.off = True
                return 0.0
            self._seen = (now, v)
            return v

    def wait(self) -> None:
        t0 = time.monotonic()
        while self.playhead() > 0 and time.monotonic() - t0 < self.most:
            time.sleep(0.5)
        dt = time.monotonic() - t0
        if dt > 0.4:
            with self._lock:
                self.waited += dt
                self.waits += 1


def wait_for_memory(need: int, reserve: int, most: float, live: LiveFirst) -> bool:
    """Whether the GPU has `need` + `reserve` MiB free, waiting up to `most` seconds for it."""
    t0 = time.monotonic()
    said = False
    while True:
        used, free = gpu_memory()
        if free < 0 or free >= need + reserve:
            live.wait()
            return True
        if time.monotonic() - t0 > most:
            say(f"gpu: {free} MiB free, {need} + {reserve} needed, for {most:.0f} s: this step is skipped")
            return False
        if not said:
            say(f"gpu: {free} MiB free ({used} used), waiting for {need} + {reserve} MiB")
            said = True
        time.sleep(15)


# --- llama.cpp's server --------------------------------------------------------

class LlamaServer:
    def __init__(self, gguf: Path, slots: int, ctx_per_slot: int = 8192, port: int = 8091):
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.cmd = ["llama-server", "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port), "-ngl", "99",
                    "-fa", "on", "--reasoning", "off", "-np", str(slots), "-c", str(ctx_per_slot * slots),
                    "--no-webui", "--metrics"]
        self.proc: subprocess.Popen | None = None

    def __enter__(self):
        self.log = tempfile.NamedTemporaryFile("w+", prefix="llama-server-", suffix=".log", delete=False)
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(600):
            if self.proc.poll() is not None:
                self.log.seek(0)
                raise RuntimeError("llama-server exited: " + self.log.read()[-2000:])
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if b"ok" in r.read():
                        return self
            except Exception:
                time.sleep(0.5)
        raise RuntimeError("llama-server did not come up")

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


def llama_version(text: str | None = None) -> str:
    """llama.cpp's release as its tags name it ("b11430") and its commit, from `llama-server --version`:
    "version: 0.6.0-dev (build 11430, commit 8345f3339)", or the older "version: 11430 (8345f3339)"."""
    try:
        if text is None:
            out = subprocess.run(["llama-server", "--version"], capture_output=True, text=True, timeout=30)
            text = out.stdout + out.stderr
        m = re.search(r"build (\d+), commit ([0-9a-f]+)", text) or re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", text)
        return f"b{m.group(1)} ({m.group(2)})" if m else "unknown"
    except Exception:
        return "unknown"


def asr_versions() -> str:
    try:
        import ctranslate2
        import faster_whisper

        return f"faster-whisper {faster_whisper.__version__}, CTranslate2 {ctranslate2.__version__}"
    except Exception:
        return "faster-whisper"


# --- one title -----------------------------------------------------------------

@dataclass
class Title:
    id: str
    name: str
    original: Path
    language: str | None = "en"
    stream: original.AudioStream | None = None
    wav: Path | None = None
    offset: float = 0.0
    duration: float = 0.0
    error: str | None = None
    read_seconds: float = 0.0
    gpu: dict = field(default_factory=dict)


def read_ahead(titles: list[Title], tmp: Path, readrate: float | None, ready: dict[str, threading.Event]) -> None:
    """The main audio of each title, in order, ahead of the GPU work."""
    for t in titles:
        t0 = time.perf_counter()
        try:
            info = original.probe(t.original)
            t.duration = float((info.get("format") or {}).get("duration") or 0.0)
            streams = original.audio_streams(info)
            t.stream = original.main_audio(streams, t.language)
            if t.stream is None:
                raise RuntimeError("the original has no audio stream")
            say(f"{t.name}: audio streams " + "; ".join(s.describe() for s in streams) + f" -> {t.stream.describe()}")
            t.wav = tmp / f"{t.id}.wav"
            t.offset = original.extract(t.original, t.stream, t.wav, readrate=readrate)
            t.read_seconds = round(time.perf_counter() - t0, 1)
            say(f"{t.name}: audio read in {t.read_seconds:.0f} s ({t.duration / 60:.1f} min, offset {t.offset:.3f} s)")
        except Exception as e:
            t.error = f"the audio could not be read: {e}"
            say(f"{t.name}: {t.error}")
        finally:
            ready[t.id].set()


class GatedClient:
    """A llama-server client that lets live playback go first before every request."""

    def __init__(self, url: str, live: LiveFirst):
        from .translate.llm import Client

        self.inner = Client(url)
        self.live = live
        self.model = self.inner.model

    def chat(self, *args, **kw):
        self.live.wait()
        return self.inner.chat(*args, **kw)

    def model_id(self) -> str:
        return self.inner.model_id()


def translate(cues_in: list[vtt.Cue], src: Lang, tgt: Lang, url: str, chains: int, live: LiveFirst,
              length_budget: bool) -> tuple[list[vtt.Cue], dict]:
    """`chains` contiguous parts of the track translated at once, each a sequential chain."""
    from .translate import translate_cues
    from .translate.llm import LlmTranslator

    n = len(cues_in)
    chains = max(1, min(chains, n // 40 or 1))
    parts = [cues_in[i * n // chains:(i + 1) * n // chains] for i in range(chains)]
    results: list = [None] * chains
    errors: list = [None] * chains

    def work(k: int) -> None:
        try:
            # No lines after the batch as context: a model that translated them shifted the batch's
            # answers by a line or two and kept the count; the batch before stays as context.
            # Spans: a model that wants to translate two lines as one says so ("12-13|") instead of
            # folding one into the other and moving the rest up.
            tr = LlmTranslator(GatedClient(url, live), batch=20, after=0, length_budget=length_budget, spans=True)
            results[k] = translate_cues(parts[k], src, tgt, tr, sentences=True)
        except Exception as e:  # noqa: BLE001 - reported with the part
            errors[k] = repr(e)

    threads = [threading.Thread(target=work, args=(k,)) for k in range(chains)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if any(errors):
        raise RuntimeError("; ".join(e for e in errors if e))
    out: list[vtt.Cue] = []
    usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0, "fallbacks": 0,
             "budgeted": 0, "merged": 0, "rejected": 0, "failures": 0, "chains": chains}
    for res, u, failures in results:
        out += res
        d = u.as_dict()
        for k in ("requests", "prompt_tokens", "completion_tokens", "retries", "fallbacks", "budgeted", "merged"):
            usage[k] += d.get(k, 0)
        usage["rejected"] += len(u.rejected)
        usage["failures"] += failures
    # Each part had its reading time stretched on its own: the last cue of a part may reach into the
    # first of the next one. Never two cues at once.
    out.sort(key=lambda c: c.start)
    for a, b in zip(out, out[1:]):
        if a.end > b.start - 0.083:
            a.end = max(a.start + 0.3, b.start - 0.083)
    return out, usage


def bundle(t: Title, folder: Path) -> None:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(folder.iterdir()):
            if p.suffix in (".vtt", ".json"):
                tar.add(p, arcname=f"{t.id}/{p.name}")
    data = base64.b64encode(buf.getvalue()).decode()
    print(f"-----BEGIN SUBTITLER OUTPUT {t.id}-----", flush=True)
    for i in range(0, len(data), 76):
        print(data[i:i + 76])
    print(f"-----END SUBTITLER OUTPUT {t.id}-----", flush=True)


def run_title(t: Title, a, paths: dict[str, Path], langs: list[Lang], live: LiveFirst, versions: dict) -> dict:
    folder = Path(a.out) / t.id
    folder.mkdir(parents=True, exist_ok=True)
    summary = {"id": t.id, "name": t.name, "made_by": f"subtitler {__version__}", "made": [], "errors": []}
    audio_lang = Lang.parse(t.language or "en")

    # speech recognition, a process of its own
    if not wait_for_memory(a.vram_asr, a.vram_reserve, a.vram_wait, live):
        summary["errors"].append("no GPU memory for speech recognition")
        return summary
    words_json = folder / "transcript.json"
    cmd = [sys.executable, "-m", "subtitler", "transcribe", str(t.wav), "--model", str(paths["asr"]),
           "--out", str(words_json), "--batch-size", str(a.asr_batch)]
    if t.language:
        cmd += ["--language", audio_lang.code]
    if paths.get("check"):
        cmd += ["--check-model", str(paths["check"]), "--check-windows", str(a.check_windows)]
    g0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    t.gpu["transcribe"] = round(time.perf_counter() - g0, 1)
    if proc.returncode != 0:
        summary["errors"].append("speech recognition failed: " + proc.stderr[-1500:])
        say(f"{t.name}: speech recognition failed:\n{proc.stderr[-1500:]}")
        return summary
    lines = [x for x in proc.stderr.strip().splitlines() if x.startswith("transcribed ")]
    say(f"{t.name}: {lines[-1] if lines else 'transcribed'} ({t.gpu['transcribe']:.0f} s on the GPU)")
    tr = json.loads(words_json.read_text())
    res = tr["result"]
    words = [cues.Word(w["start"] + t.offset, w["end"] + t.offset, w["text"], w["probability"]) for w in res["words"]]
    heard = res["language"]
    speech = [(x[0] + t.offset, x[1] + t.offset) for x in tr.get("speech", [])]
    transcript = cues.build(words, heard, speech=speech or None)
    if audio_lang.code == "zh":
        transcript = [c.with_text(script.convert(c.text, "s2t" if audio_lang.script == "Hant" else "t2s")) for c in transcript]
    tracks: dict[str, list[vtt.Cue]] = {}
    asr_model = f"Whisper {Path(paths['asr']).name.replace('faster-whisper-', '')} ({versions['asr']})"
    s = t.stream
    stream = f"0:{s.index} {s.language or 'und'} {s.codec} {s.layout or s.channels}" if s else "?"

    def write(lang: Lang, origin: str, cs: list[vtt.Cue], model: str, how: str, extra: dict) -> None:
        label = made_label(lang, origin)
        path = folder / f"{lang.tag}.vtt"
        vtt.write(path, cs, notes=[f"{label}. Made by subtitler {__version__}: {how} by {model}."])
        tracks[lang.tag] = cs
        entry = {"language": lang.tag, "label": label, "origin": origin, "model": model, "file": path.name,
                 "cues": len(cs), **extra}
        summary["made"].append(entry)

    for lang in langs:
        if lang.code == audio_lang.code and (lang.script or None) in (None, audio_lang.script):
            write(lang, "transcribed", transcript, asr_model, f"transcribed from the title's audio ({heard}, stream {stream})",
                  {"source": "audio", "gpu_seconds": t.gpu["transcribe"]})

    # translations, one server for every language
    targets = [x for x in langs if x.tag not in tracks]
    if targets:
        if not wait_for_memory(a.vram_mt, a.vram_reserve, a.vram_wait, live):
            summary["errors"].append("no GPU memory for translation")
        else:
            s0 = time.perf_counter()
            with LlamaServer(paths["llm"], slots=a.chains) as srv:
                start = round(time.perf_counter() - s0, 1)
                for lang in targets:
                    l0 = time.perf_counter()
                    try:
                        out, usage = translate(transcript, Lang.parse(heard), lang, srv.url, a.chains, live,
                                               length_budget=lang.code in a.budget_languages)
                    except Exception as e:  # noqa: BLE001
                        summary["errors"].append(f"{lang.tag}: translation failed: {e}")
                        say(f"{t.name}: {lang.tag} translation failed: {e}")
                        continue
                    secs = round(time.perf_counter() - l0 + start, 1)
                    start = 0.0
                    t.gpu[lang.tag] = secs
                    llm = f"{versions['llm_name']} (llama.cpp {versions['llm']})"
                    write(lang, "translated", out, llm,
                          f"translated from the {Lang.parse(heard).name} transcript of the title's audio",
                          {"source": f"transcript ({heard})", "gpu_seconds": secs, "usage": usage})
                    say(f"{t.name}: {lang.tag} {len(out)} cues in {secs:.0f} s, {usage['completion_tokens']} tokens, "
                        f"{usage['failures']} empty, {usage['retries']} retries")

    # the checks
    report = {"id": t.id, "name": t.name, "duration_s": round(t.duration, 1),
              "audio": {"stream": stream, "detected": tr.get("detected"), "language": heard,
                        "read_seconds": t.read_seconds},
              "timing": qc.timing(transcript, [tuple(x) for x in tr.get("speech", [])], t.offset),
              "agreement": qc.agreement(res["words"], (tr.get("check") or {}).get("windows", []), heard, t.offset)
              if tr.get("check") else None,
              "filled": tr.get("filled"),
              "alignment": {tag: qc.alignment(cs, transcript) for tag, cs in tracks.items() if tag != audio_lang.tag},
              "languages": {tag: qc.language(cs, Lang.parse(tag).code, transcript if tag != audio_lang.tag else None)
                            for tag, cs in tracks.items()},
              "samples": qc.samples(tracks, n=a.samples, first=audio_lang.tag),
              "live_first": {"waits": live.waits, "waited_s": round(live.waited, 1)}}
    (folder / "qc.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    summary["gpu_seconds"] = t.gpu
    summary["audio"] = report["audio"]
    (folder / "subtitles.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    tim, agr = report["timing"], report["agreement"] or {}
    say(f"{t.name}: checks: speech covered {tim['speech_covered_pct']} %, {tim['cues_without_speech']} cues without "
        f"speech, {tim['speech_without_cues']} stretches of speech without a cue, agreement WER {agr.get('mean_wer')}; "
        + "; ".join(f"{k} {v['cues']} cues, {v['over_cps_pct']} % fast, {v['empty']} empty, "
                    f"{v['left_in_source_language']} left untranslated" for k, v in report["languages"].items()))
    for row in report["samples"]:
        say(f"{t.name}: sample {row['at']} | " + " | ".join(f"{k}: {row.get(k, '')}" for k in tracks))
    if a.bundle:
        bundle(t, folder)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="subtitler job", description=__doc__.split("\n\n")[0])
    ap.add_argument("--titles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", required=True, help="the model cache folder")
    ap.add_argument("--languages", default=os.environ.get("SUBTITLE_LANGUAGES", "en,de,zh-Hans"))
    ap.add_argument("--asr-model", default="large-v3")
    ap.add_argument("--check-model", default="large-v3-turbo", help="'' for no agreement check")
    ap.add_argument("--llm", default="qwen3.5-9b-q4km")
    ap.add_argument("--asr-batch", type=int, default=16)
    ap.add_argument("--check-windows", type=int, default=10)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--budget-languages", default="de", help="languages asked to condense lines short of time")
    ap.add_argument("--readrate", type=float, default=30.0, help="read originals at most this many times real time")
    ap.add_argument("--live-metrics", default=os.environ.get("LIVE_METRICS_URL"))
    ap.add_argument("--live-wait", type=float, default=20.0)
    ap.add_argument("--vram-asr", type=int, default=5500)
    ap.add_argument("--vram-mt", type=int, default=8000)
    ap.add_argument("--vram-reserve", type=int, default=4000, help="MiB left free for the live worker")
    ap.add_argument("--vram-wait", type=float, default=1800.0)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--bundle", action="store_true", help="write each title's files to the log as well")
    a = ap.parse_args(argv)
    a.budget_languages = {Lang.parse(x).code for x in a.budget_languages.split(",") if x.strip()}

    titles = [Title(id=x["id"], name=x.get("name") or x["id"], original=Path(x["original"]),
                    language=x.get("language", "en")) for x in json.loads(Path(a.titles).read_text())]
    langs = parse_list(a.languages)
    say(f"subtitler {__version__}: {len(titles)} titles, languages {', '.join(x.tag for x in langs)}")
    used, free = gpu_memory()
    say(f"gpu: {used} MiB used, {free} MiB free before anything")
    live = LiveFirst(a.live_metrics, a.live_wait)
    say(f"live first: {'playhead segments in flight now: %g' % live.playhead() if not live.off else 'off'}")

    f0 = time.perf_counter()
    paths = {"asr": models.fetch(a.models, a.asr_model), "llm": models.fetch(a.models, a.llm)}
    if a.check_model:
        paths["check"] = models.fetch(a.models, a.check_model)
    say(f"models ready in {time.perf_counter() - f0:.0f} s: " + ", ".join(models.describe(m) for m in
        [a.asr_model, a.llm] + ([a.check_model] if a.check_model else [])))
    versions = {"asr": asr_versions(), "llm": llama_version(), "llm_name": "Qwen3.5-9B Q4_K_M"
                if a.llm == "qwen3.5-9b-q4km" else a.llm}

    tmp = Path(tempfile.mkdtemp(prefix="subtitler-audio-"))
    ready = {t.id: threading.Event() for t in titles}
    reader = threading.Thread(target=read_ahead, args=(titles, tmp, a.readrate, ready), daemon=True)
    reader.start()
    done = []
    for t in titles:
        ready[t.id].wait()
        if t.error:
            done.append({"id": t.id, "name": t.name, "errors": [t.error]})
            continue
        r0 = time.perf_counter()
        try:
            s = run_title(t, a, paths, langs, live, versions)
        except Exception as e:  # noqa: BLE001 - one title's failure is not the run's
            s = {"id": t.id, "name": t.name, "errors": [repr(e)]}
            say(f"{t.name}: failed: {e!r}")
        s["wall_seconds"] = round(time.perf_counter() - r0, 1)
        done.append(s)
        if t.wav and t.wav.exists():
            t.wav.unlink()
        say(f"{t.name}: done in {s['wall_seconds']:.0f} s; GPU seconds {s.get('gpu_seconds')}; errors {s.get('errors')}")
    (Path(a.out) / "run.json").write_text(json.dumps({"made_by": f"subtitler {__version__}", "titles": done},
                                                     indent=1, ensure_ascii=False))
    say("run: " + "; ".join(f"{d['name']} {'ok' if not d.get('errors') else 'ERRORS'}" for d in done))
    return 0 if all(not d.get("errors") for d in done) else 1


if __name__ == "__main__":
    sys.exit(main())
