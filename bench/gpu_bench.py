"""The GPU run: speech recognition × real time and translation tokens/s, alone and next to a busy GPU.

    python bench/gpu_bench.py --sources /etc/bench/sources.json --models /models --out /tmp/report

sources.json lists packages (folders under the read-only media mount):
    [{"name": "movie-1", "package": "/media/movies/.../versions/<id>"}, ...]

Phases, each optional (--phases):
  asr    every source's main audio (its first --asr-minutes, 0 = all) through
         faster-whisper: large-v3-turbo and large-v3, sequential (beam 5) and
         batched; × real time, GPU memory, and WER etc. against the package's
         own subtitle in the audio's language where it has one;
  mt     the best text subtitle of each source (English first) translated into
         German and Simplified Chinese by each GGUF model behind a CUDA
         llama-server, with 1, 4 and 8 chains in flight; tokens/s per chain
         and in total; then MADLAD-400 on CTranslate2 (CUDA);
  busy   the same, shortened, while live-worker-like load runs on the GPU:
         --busy-streams ffmpeg processes decoding a package's video with
         NVDEC, scaling with scale_cuda and encoding with NVENC, in a loop.
The estimate at the end: GPU minutes for a two-hour film with no subtitle
(transcription plus two translations), alone and busy.

Everything goes to stdout (the Job's log) and to --out as report.md and results.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

from subtitler import asr, cues, package, script, vtt  # noqa: E402
from subtitler.langs import Lang  # noqa: E402

MODELS = {
    # name: (Hugging Face repo, file or None for the whole repo)
    "large-v3-turbo": ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", None),
    "large-v3": ("Systran/faster-whisper-large-v3", None),
    "qwen3.5-9b-q4km": ("bartowski/Qwen_Qwen3.5-9B-GGUF", "Qwen_Qwen3.5-9B-Q4_K_M.gguf"),
    "qwen2.5-7b-q4km": ("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    "madlad400-3b": ("Nextcloud-AI/madlad400-3b-mt-ct2-int8", None),
}

REPORT: list[str] = []
# cuda on the GPU node; cpu for a smoke run of this script anywhere else.
DEVICE = os.environ.get("BENCH_DEVICE", "cuda")
COMPUTE = {"cuda": {"asr": "float16", "mt": "int8_float16"}, "cpu": {"asr": "int8", "mt": "int8"}}
RESULTS: dict = {"asr": [], "mt": [], "gpu": {}, "estimate": {}}


def say(line: str = "") -> None:
    print(line, flush=True)
    REPORT.append(line)


# --- GPU memory --------------------------------------------------------------

class GpuSampler:
    """nvidia-smi every 0.5 s: peak memory used and mean utilisation while running."""

    def __init__(self):
        self.peak_mib = 0
        self.util: list[int] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
                mem, util = (int(x.strip()) for x in out.splitlines()[0].split(","))
                self.peak_mib = max(self.peak_mib, mem)
                self.util.append(util)
            except Exception:
                pass
            self._stop.wait(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=5)

    @property
    def mean_util(self) -> float:
        return sum(self.util) / len(self.util) if self.util else 0.0


def gpu_used_mib() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        return int(out.splitlines()[0].strip())
    except Exception:
        return -1


# --- models ------------------------------------------------------------------

def fetch(models_dir: Path, name: str) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    repo, file = MODELS[name]
    target = models_dir / name
    if file:
        path = target / file
        if not path.exists():
            hf_hub_download(repo, file, local_dir=target)
        return path
    if not (target / "model.bin").exists():
        snapshot_download(repo, local_dir=target)
    return target


# --- busy GPU ----------------------------------------------------------------

class BusyGpu:
    """Live-worker-like load: NVDEC decode, scale_cuda, NVENC encode, looping."""

    def __init__(self, video_playlist: str, streams: int, height: int = 720):
        self.cmd = ["ffmpeg", "-nostdin", "-v", "error", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                    "-stream_loop", "-1", "-i", video_playlist, "-vf", f"scale_cuda=-2:{height}",
                    "-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "3M", "-an", "-f", "null", "-"]
        self.streams = streams
        self.procs: list[subprocess.Popen] = []

    def __enter__(self):
        for _ in range(self.streams):
            self.procs.append(subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE))
        time.sleep(8)  # let the encoders reach steady state
        alive = sum(p.poll() is None for p in self.procs)
        say(f"busy GPU: {alive}/{self.streams} NVDEC->scale_cuda->NVENC streams running, "
            f"GPU memory {gpu_used_mib()} MiB")
        if alive == 0:
            err = self.procs[0].stderr.read().decode(errors="replace")[-500:] if self.procs else ""
            raise RuntimeError(f"busy load did not start: {err}")
        return self

    def __exit__(self, *exc):
        for p in self.procs:
            p.send_signal(signal.SIGINT)
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


# --- speech recognition ------------------------------------------------------

def audio_of(pkg: package.Package, minutes: float, tmp: Path, name: str) -> tuple[Path, float, float]:
    a = pkg.main_audio
    wav = tmp / f"{name}.wav"
    if not wav.exists():
        offset = asr.extract(pkg.root / a.dir, wav)
        if minutes:
            cut = tmp / f"{name}.cut.wav"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(wav), "-t", str(minutes * 60),
                            "-c", "copy", str(cut)], check=True)
            cut.replace(wav)
        (tmp / f"{name}.offset").write_text(str(offset))
    offset = float((tmp / f"{name}.offset").read_text())
    return wav, offset, asr.wav_seconds(wav)


def own_subtitle(pkg: package.Package, lang: str | None):
    if not lang:
        return None
    subs = [s for s in pkg.subtitles if s.exists and s.is_text and s.is_full and s.language == lang]
    subs.sort(key=lambda s: (s.is_sdh, -(s.cues or 0)))
    return subs[0] if subs else None


def run_asr(sources, models_dir: Path, tmp: Path, minutes: float, modes, tag: str) -> None:
    import metrics

    for model_name in [m for m in os.environ.get("ASR_MODELS", "large-v3-turbo,large-v3").split(",") if m]:
        path = fetch(models_dir, model_name)
        for mode, batch in modes:
            before = gpu_used_mib()
            m = asr.FasterWhisper(str(path), device=DEVICE, compute_type=COMPUTE[DEVICE]["asr"], batch_size=batch)
            loaded = gpu_used_mib()
            for src in sources:
                pkg = package.load(src["package"])
                try:
                    wav, offset, secs = audio_of(pkg, minutes, tmp, src["name"])
                    lang = pkg.main_audio.language
                    with GpuSampler() as g:
                        r = m.transcribe(str(wav), language=lang if lang != "zh" else None)
                    r.offset = offset
                    r = r.shifted()
                    row = {"tag": tag, "model": model_name, "mode": mode, "source": src["name"],
                           "language": r.language, "audio_s": round(secs, 1), "wall_s": round(r.seconds, 2),
                           "x_realtime": round(r.speed, 1), "gpu_peak_mib": g.peak_mib,
                           "gpu_model_mib": loaded - before, "gpu_util": round(g.mean_util, 1)}
                    sub = own_subtitle(pkg, r.language if r.language != "yue" else None)
                    if sub and tag == "alone":
                        words = r.words
                        if r.language == "zh":
                            words = [cues.Word(w.start, w.end, script.convert(w.text, "t2s"), w.probability) for w in words]
                        built = cues.build(words, r.language)
                        ref = vtt.read(pkg.root / sub.path)
                        s = metrics.score_asr(ref, words, built, r.language, offset + 2, offset + secs - 2)
                        row.update(wer=round(s.wer, 4), sd_rate=round(s.sd_rate, 4), local_recall=round(s.local_recall, 4),
                                   coverage=round(s.coverage, 3), ref=sub.id)
                    RESULTS["asr"].append(row)
                    say(f"asr {tag:5s} {model_name:14s} {mode:10s} {src['name']:14s} {r.language:3s} "
                        f"{secs:6.0f}s audio {r.seconds:7.1f}s {r.speed:6.1f}x  GPU {g.peak_mib} MiB "
                        f"(model {loaded - before})" + (f"  WER {row['wer']:.3f} SD {row['sd_rate']:.3f} "
                                                        f"recall {row['local_recall']:.3f}" if "wer" in row else ""))
                except Exception as e:  # one source's failure is that source's only
                    say(f"asr {tag} {model_name} {mode} {src['name']}: FAILED {e!r}")
            del m
            gc.collect()


# --- translation -------------------------------------------------------------

class LlamaServer:
    def __init__(self, gguf: Path, slots: int, ctx_per_slot: int = 8192, port: int = int(os.environ.get("BENCH_PORT", "8091"))):
        self.port = port
        self.cmd = ["llama-server", "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port), "-ngl", "99",
                    "-fa", "on", "--reasoning", "off", "-np", str(slots), "-c", str(ctx_per_slot * slots), "--no-webui", "--metrics"]
        self.proc: subprocess.Popen | None = None

    def __enter__(self):
        self.log = open(f"/tmp/llama-server-{self.port}.log", "w")
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        url = f"http://127.0.0.1:{self.port}/health"
        for _ in range(300):
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited: see {self.log.name}")
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if b"ok" in r.read():
                        return self
            except Exception:
                time.sleep(1)
        raise RuntimeError("llama-server did not come up")

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()


def source_track(pkg: package.Package):
    """The subtitle a translation would start from: English first, then the original language."""
    from subtitler.plan import _source_rank

    subs = [s for s in pkg.subtitles if s.exists and s.is_text and s.is_full and not s.bilingual]
    if not subs:
        return None
    a = pkg.main_audio
    subs.sort(key=lambda s: (s.language != "en",) + _source_rank(s, a.language if a else None, []))
    return subs[0]


def translate_chains(cues_in, src: Lang, tgt: Lang, url: str, chains: int, batch: int, is_sdh: bool):
    """`chains` contiguous parts of the track translated at once, each a sequential chain."""
    from subtitler.translate import translate_cues
    from subtitler.translate.llm import Client, LlmTranslator

    n = len(cues_in)
    parts = [cues_in[i * n // chains:(i + 1) * n // chains] for i in range(chains)]
    results = [None] * chains

    def work(k):
        tr = LlmTranslator(Client(url), batch=batch)
        results[k] = translate_cues(parts[k], src, tgt, tr, is_sdh=is_sdh,
                                    sentences=os.environ.get("MT_SENTENCES", "1") == "1")

    t0 = time.perf_counter()
    threads = [threading.Thread(target=work, args=(k,)) for k in range(chains)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    done = [r for r in results if r]
    tokens = sum(r[1].completion_tokens for r in done)
    prompt = sum(r[1].prompt_tokens for r in done)
    per_chain = [r[1].completion_tokens / (r[1].predicted_ms / 1000) for r in done if r[1].predicted_ms]
    return {"cues": sum(len(p) for p in parts), "wall_s": round(wall, 2), "completion_tokens": tokens,
            "prompt_tokens": prompt, "tok_s_total": round(tokens / wall, 1) if wall else 0,
            "tok_s_per_chain": round(sum(per_chain) / len(per_chain), 1) if per_chain else 0,
            "failures": sum(r[2] for r in done), "retries": sum(r[1].retries for r in done)}


def run_mt(sources, models_dir: Path, llms, chains_list, cue_limit: int, tag: str, madlad: bool) -> None:
    jobs = []
    density = []
    for src in sources:
        pkg = package.load(src["package"])
        sub = source_track(pkg)
        if sub is None:
            continue
        cs = vtt.read(pkg.root / sub.path)
        minutes = json.loads((pkg.root / "package.json").read_text()).get("durationMs", 0) / 60000
        if minutes > 20:
            density.append(len(cs) / minutes)
        if cue_limit:
            mid = max(0, len(cs) // 2 - cue_limit // 2)
            cs = cs[mid:mid + cue_limit]
        jobs.append((src["name"], sub, cs))
    if not jobs:
        say("mt: no source has a text subtitle")
        return
    if density and tag == "alone":
        RESULTS["cues_per_2h"] = round(120 * sum(density) / len(density))
        say(f"mt: the sources' subtitles run at {sum(density) / len(density):.1f} cues a minute "
            f"({RESULTS['cues_per_2h']} for two hours)")
    targets = [Lang.parse("de"), Lang.parse("zh-Hans")]
    for llm in llms:
        gguf = fetch(models_dir, llm)
        if tag == "alone":
            # llama.cpp's own measure: sequences in parallel, each a translation batch's size.
            npl = ",".join(str(c) for c in sorted(set([1, *chains_list])))
            out = subprocess.run(["llama-batched-bench", "-m", str(gguf), "-c", "32768", "-b", "2048", "-ub", "512",
                                  "-npp", "1300", "-ntg", "450", "-npl", npl, "-fa", "on", "-ngl", "99"],
                                 capture_output=True, text=True)
            say(f"llama-batched-bench {llm}:")
            for line in out.stdout.splitlines():
                if line.startswith("|"):
                    say("  " + line)
        for chains in chains_list:
            before = gpu_used_mib()
            with LlamaServer(gguf, slots=chains) as srv, GpuSampler() as g:
                url = f"http://127.0.0.1:{srv.port}"
                for name, sub, cs in jobs:
                    for tgt in targets:
                        if tgt.code == sub.language:
                            continue
                        try:
                            r = translate_chains(cs, Lang(sub.language or "en"), tgt, url, chains, 30, sub.is_sdh)
                        except Exception as e:
                            say(f"mt {tag} {llm} chains={chains} {name} ->{tgt.tag}: FAILED {e!r}")
                            continue
                        row = {"tag": tag, "model": llm, "chains": chains, "source": name, "src": sub.language,
                               "tgt": tgt.tag, **r}
                        RESULTS["mt"].append(row)
                        say(f"mt  {tag:5s} {llm:16s} chains={chains} {name:14s} {sub.language}->{tgt.tag:7s} "
                            f"{r['cues']:4d} cues {r['wall_s']:6.1f}s {r['tok_s_total']:7.1f} tok/s total "
                            f"{r['tok_s_per_chain']:6.1f}/chain  {r['completion_tokens'] / max(1, r['cues']):.1f} tok/cue")
            say(f"mt  {tag} {llm} chains={chains}: GPU peak {g.peak_mib} MiB (before {before} MiB), util {g.mean_util:.0f} %")
    if madlad:
        from subtitler.translate import translate_cues
        from subtitler.translate.madlad import MadladTranslator

        tr = MadladTranslator(str(fetch(models_dir, "madlad400-3b")), device=DEVICE, compute_type=COMPUTE[DEVICE]["mt"],
                              beam_size=4, batch=32, merge=False)
        for name, sub, cs in jobs:
            for tgt in targets:
                if tgt.code == sub.language:
                    continue
                with GpuSampler() as g:
                    t0 = time.perf_counter()
                    res, usage, failures = translate_cues(cs, Lang(sub.language or "en"), tgt, tr, is_sdh=sub.is_sdh,
                                                          sentences=True)
                    wall = time.perf_counter() - t0
                u = usage.as_dict()
                row = {"tag": tag, "model": "madlad400-3b", "chains": 1, "source": name, "src": sub.language,
                       "tgt": tgt.tag, "cues": len(cs), "wall_s": round(wall, 2), "completion_tokens": u["output_tokens"],
                       "tok_s_total": round(u["output_tokens"] / wall, 1), "gpu_peak_mib": g.peak_mib}
                RESULTS["mt"].append(row)
                say(f"mt  {tag:5s} madlad400-3b     batch      {name:14s} {sub.language}->{tgt.tag:7s} "
                    f"{len(cs):4d} cues {wall:6.1f}s {row['tok_s_total']:7.1f} tok/s  GPU {g.peak_mib} MiB")
        del tr
        gc.collect()


# --- estimate ----------------------------------------------------------------

def estimate() -> None:
    """GPU minutes for a two-hour film without subtitles: transcription + two translations."""
    cues_per_film = RESULTS.get("cues_per_2h", 1500)
    for tag in ("alone", "busy"):
        asr_rows = [r for r in RESULTS["asr"] if r["tag"] == tag and r["mode"] == "batched"]
        mt_rows = [r for r in RESULTS["mt"] if r["tag"] == tag and r["model"] != "madlad400-3b"]
        if not asr_rows or not mt_rows:
            continue
        for model in sorted({r["model"] for r in asr_rows}):
            xs = [r["x_realtime"] for r in asr_rows if r["model"] == model]
            asr_min = 120 / (sum(xs) / len(xs))
            for llm in sorted({r["model"] for r in mt_rows}):
                best = max((r for r in mt_rows if r["model"] == llm), key=lambda r: r["tok_s_total"])
                tok_per_cue = best["completion_tokens"] / max(1, best["cues"])
                mt_min = 2 * cues_per_film * tok_per_cue / best["tok_s_total"] / 60
                key = f"{tag}/{model}+{llm}"
                RESULTS["estimate"][key] = {"asr_min": round(asr_min, 1), "mt_min_two_languages": round(mt_min, 1),
                                            "total_min": round(asr_min + mt_min, 1), "chains": best["chains"]}
                say(f"estimate {tag:5s} {model} + {llm} (chains {best['chains']}): transcription {asr_min:.1f} min "
                    f"+ two translations {mt_min:.1f} min = {asr_min + mt_min:.1f} min of GPU for a 2-hour film")


def preflight() -> None:
    say("===== preflight")
    for cmd in (["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv"],
                ["ffmpeg", "-hide_banner", "-version"], ["llama-server", "--version"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            say("$ " + " ".join(cmd[:2]))
            say((out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else "(no output)")
            if cmd[0] == "nvidia-smi":
                say(out.stdout.strip())
        except Exception as e:
            say(f"{cmd[0]}: {e!r}")
    try:
        import ctranslate2

        say(f"ctranslate2 {ctranslate2.__version__}: CUDA devices {ctranslate2.get_cuda_device_count()}, "
            f"compute types {sorted(ctranslate2.get_supported_compute_types('cuda'))}")
    except Exception as e:
        say(f"ctranslate2 CUDA: {e!r}")
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    flt = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    say(f"ffmpeg h264_nvenc: {'yes' if ' h264_nvenc ' in enc else 'MISSING'}, scale_cuda: {'yes' if ' scale_cuda ' in flt else 'MISSING'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=os.environ.get("SOURCES", "/etc/bench/sources.json"))
    ap.add_argument("--models", default=os.environ.get("MODELS_DIR", "/models"))
    ap.add_argument("--out", default=os.environ.get("OUT", "/tmp/report"))
    ap.add_argument("--phases", default=os.environ.get("PHASES", "asr,mt,busy"))
    ap.add_argument("--asr-minutes", type=float, default=float(os.environ.get("ASR_MINUTES", "0")))
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("ASR_BATCH", "16")))
    ap.add_argument("--llms", default=os.environ.get("LLMS", "qwen3.5-9b-q4km,qwen2.5-7b-q4km"))
    ap.add_argument("--chains", default=os.environ.get("CHAINS", "1,4,8"))
    ap.add_argument("--cue-limit", type=int, default=int(os.environ.get("CUE_LIMIT", "300")))
    ap.add_argument("--mt-sources", type=int, default=int(os.environ.get("MT_SOURCES", "4")))
    ap.add_argument("--busy-streams", type=int, default=int(os.environ.get("BUSY_STREAMS", "4")))
    ap.add_argument("--busy-video", default=os.environ.get("BUSY_VIDEO", ""))
    ap.add_argument("--busy-sources", type=int, default=int(os.environ.get("BUSY_SOURCES", "3")))
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = json.loads(Path(a.sources).read_text())
    models_dir = Path(a.models)
    models_dir.mkdir(parents=True, exist_ok=True)
    phases = set(a.phases.split(","))
    llms = [x for x in a.llms.split(",") if x]
    chains = [int(x) for x in a.chains.split(",")]
    modes = [("sequential", 0), ("batched", a.batch_size)]
    preflight()
    say(f"GPU memory before anything: {gpu_used_mib()} MiB")
    tmp = Path(tempfile.mkdtemp(prefix="subtitler-", dir="/tmp"))
    try:
        if "asr" in phases:
            say("===== speech recognition, alone")
            run_asr(sources, models_dir, tmp, a.asr_minutes, modes, "alone")
        if "mt" in phases:
            say("===== translation, alone")
            with_subs = [s for s in sources if source_track(package.load(s["package"])) is not None]
            run_mt(with_subs[: a.mt_sources], models_dir, llms, chains, a.cue_limit, "alone", madlad=True)
        if "busy" in phases:
            video = a.busy_video or str(Path(sources[0]["package"]) / "hls" / "v0" / "playlist.m3u8")
            say(f"===== next to a busy GPU ({a.busy_streams} streams of {video})")
            try:
                with BusyGpu(video, a.busy_streams):
                    run_asr(sources[: a.busy_sources], models_dir, tmp, a.asr_minutes, [("batched", a.batch_size)], "busy")
                    with_subs = [s for s in sources if source_track(package.load(s["package"])) is not None]
                    run_mt(with_subs[: a.busy_sources], models_dir, llms[:1], [max(chains)], a.cue_limit, "busy",
                           madlad=False)
            except Exception as e:
                say(f"busy phase FAILED: {e!r}")
        say("===== estimate")
        estimate()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        (out / "results.json").write_text(json.dumps(RESULTS, indent=1))
        (out / "report.md").write_text("\n".join(REPORT) + "\n")
        say(f"===== results.json and report.md in {out}")
        print("===== results.json")
        print(json.dumps(RESULTS))


if __name__ == "__main__":
    main()
