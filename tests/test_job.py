"""The run over titles without subtitles: audio choice, checks, live first, one title end to end (fakes for the GPU)."""

from __future__ import annotations

import base64
import io
import json
import re
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from subtitler import job, original, qc, transcribe, vtt
from subtitler.langs import Lang

PROBE = {"format": {"duration": "2246.9"}, "streams": [
    {"index": 0, "codec_type": "video", "codec_name": "hevc"},
    {"index": 1, "codec_type": "audio", "codec_name": "eac3", "channels": 6, "channel_layout": "5.1(side)",
     "start_time": "0.021", "disposition": {"default": 1}, "tags": {"language": "eng"}},
    {"index": 2, "codec_type": "audio", "codec_name": "aac", "channels": 2, "channel_layout": "stereo",
     "start_time": "0.000", "disposition": {"default": 0, "comment": 1}, "tags": {"language": "eng", "title": "Commentary"}},
    {"index": 3, "codec_type": "audio", "codec_name": "ac3", "channels": 6, "channel_layout": "5.1",
     "disposition": {"default": 0}, "tags": {"language": "ger"}},
    {"index": 4, "codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "eng"}},
]}


def test_the_main_audio_is_the_dialogue_in_the_titles_language():
    streams = original.audio_streams(PROBE)
    assert [s.index for s in streams] == [1, 2, 3]
    main = original.main_audio(streams, "en")
    assert main.index == 1 and main.language == "en" and main.start == pytest.approx(0.021)
    assert not streams[1].main  # commentary
    assert original.main_audio(streams, "de").index == 3
    assert original.main_audio(streams, "ja").index == 1  # no Japanese: the default one
    assert original.downmix(main) == ["-af", "pan=mono|c0<FC+0.3*FL+0.3*FR"]
    assert original.downmix(streams[1]) == ["-ac", "1"]


def test_check_windows_spread_over_the_speech():
    speech = [(i * 60.0, i * 60.0 + 40.0) for i in range(30)]  # 30 stretches of 40 s
    w = transcribe.check_windows(speech, 10)
    assert len(w) == 10
    assert all(b - a == pytest.approx(30.0) for a, b in w)
    assert w[0][0] < 200 and w[-1][0] > 1500
    assert all(x[1] <= y[0] for x, y in zip(w, w[1:]))  # no two overlap


def test_wer_and_agreement():
    assert qc.wer("a b c d".split(), "a b c d".split()) == 0
    assert qc.wer("a b c d".split(), "a x c".split()) == pytest.approx(0.5)
    words = [{"start": 10.0 + i, "end": 10.5 + i, "text": t, "probability": 0.9}
             for i, t in enumerate("the old ferry carried far too many people".split())]
    a = qc.agreement(words, [{"start": 9.0, "end": 39.0, "text": "The old ferry carried far too many people."}], "en")
    assert a["mean_wer"] == 0.0 and a["rows"][0]["words"] == 8


def test_timing_finds_cues_without_speech_and_speech_without_cues():
    cues = [vtt.Cue(1.0, 3.0, "Hello there."), vtt.Cue(20.0, 22.0, "Thanks for watching!")]
    speech = [(0.9, 3.2), (40.0, 46.0)]
    t = qc.timing(cues, speech)
    assert t["cues_without_speech"] == 1 and t["cues_without_speech_examples"][0]["at"] == "00:00:20.000"
    assert t["speech_without_cues"] == 1 and t["speech_without_cues_examples"][0]["at"] == "00:00:40.000"
    assert 0 < t["speech_covered_pct"] < 50


def test_language_counts_and_samples():
    en = [vtt.Cue(i * 3.0, i * 3.0 + 2.0, f"Line number {i} of the title.") for i in range(30)]
    zh = [vtt.Cue(c.start, c.end, "这是第一行" if i % 10 else "Line number 0 of the title.") for i, c in enumerate(en)]
    st = qc.language(zh, "zh", en)
    assert st["cues"] == 30 and st["left_in_source_language"] == 3 and st["overlaps"] == 0
    s = qc.samples({"en": en, "zh-Hans": zh}, n=5)
    assert len(s) == 5 and all(r["en"].startswith("Line number") for r in s)


class _Metrics(BaseHTTPRequestHandler):
    value = "2"

    def do_GET(self):
        body = (f'katalog_stream_live_worker_inflight{{priority="playhead"}} {self.value}\n'
                f'katalog_stream_live_worker_inflight{{priority="prefetch"}} 3\n').encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def serve(handler):
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_live_first_reads_the_playhead_queue_and_waits_at_most_its_limit():
    srv = serve(_Metrics)
    try:
        live = job.LiveFirst(f"http://127.0.0.1:{srv.server_port}/metrics", most=1.0)
        assert live.playhead() == 2.0
        live.wait()
        assert live.waits == 1 and 0.9 < live.waited < 3.0
        _Metrics.value = "0"
        live._seen = (0.0, 0.0)
        live.wait()
        assert live.waits == 1
    finally:
        srv.shutdown()
    dead = job.LiveFirst("http://127.0.0.1:9/metrics", most=1.0)
    assert dead.playhead() == 0.0 and dead.off  # unreachable: the run goes on without


class _Chat(BaseHTTPRequestHandler):
    """llama-server's chat endpoint: every numbered line answered as 'N| <tgt>: <text>'."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = body["messages"][-1]["content"].split("for context only")[0]
        lines = re.findall(r"^(\d+)(?: \[max \d+\])?\| (.*)$", user, re.M)
        tgt = "de" if "into German" in body["messages"][0]["content"] else "zh"
        text = "\n".join(f"{n}| {'这是一句话' if tgt == 'zh' else 'Das ist ein Satz.'}" for n, _ in lines)
        out = json.dumps({"choices": [{"message": {"content": text}}], "usage": {"completion_tokens": 5 * len(lines)},
                          "timings": {"prompt_n": 50, "predicted_n": 5 * len(lines), "prompt_ms": 5, "predicted_ms": 50}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_one_title_end_to_end_with_fakes_for_the_gpu(tmp_path, monkeypatch, capsys):
    chat = serve(_Chat)
    words = []
    for k in range(120):  # 120 sentences of 4 words, 3 s apart
        for i, w in enumerate(["We", "wait", "at", "home."]):
            words.append({"start": k * 3.0 + i * 0.4, "end": k * 3.0 + i * 0.4 + 0.3, "text": " " + w, "probability": 0.9})
    speech = [[k * 3.0, k * 3.0 + 1.6] for k in range(120)]

    def fake_run(cmd, capture_output=True, text=True):
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text(json.dumps({"result": {"language": "en", "words": words, "segments": []},
                                   "detected": {"language": "en", "probability": 0.99}, "speech": speech,
                                   "check": {"model": "turbo", "windows": [{"start": 0.0, "end": 30.0,
                                             "text": " ".join("We wait at home." for _ in range(10))}]}}))
        return SimpleNamespace(returncode=0, stderr="transcribed 360s of audio (en) in 3s: 480 words\n", stdout="")

    class FakeServer:
        def __init__(self, gguf, slots, **kw):
            self.url = f"http://127.0.0.1:{chat.server_port}"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr(job.subprocess, "run", fake_run)
    monkeypatch.setattr(job, "LlamaServer", FakeServer)
    monkeypatch.setattr(job, "gpu_memory", lambda: (1000, 20000))
    t = job.Title(id="item-1", name="E06", original=tmp_path / "original.mkv", language="en",
                  stream=original.audio_streams(PROBE)[0], wav=tmp_path / "a.wav", offset=0.021, duration=360.0)
    a = SimpleNamespace(out=str(tmp_path / "out"), vram_asr=5500, vram_mt=8000, vram_reserve=4000, vram_wait=1,
                        asr_batch=16, check_windows=10, chains=2, budget_languages={"de"}, samples=6, bundle=True)
    paths = {"asr": Path("/models/faster-whisper-large-v3"), "llm": Path("/models/q.gguf"), "check": Path("/models/turbo")}
    versions = {"asr": "faster-whisper 1.2.1, CTranslate2 4.8.3", "llm": "b11430 (abc)", "llm_name": "Qwen3.5-9B Q4_K_M"}
    try:
        s = job.run_title(t, a, paths, [Lang.parse(x) for x in ("en", "de", "zh-Hans")], job.LiveFirst(None), versions)
    finally:
        chat.shutdown()
    assert not s["errors"], s["errors"]
    folder = tmp_path / "out" / "item-1"
    made = {m["language"]: m for m in s["made"]}
    assert {k: m["label"] for k, m in made.items()} == {"en": "English · AI", "de": "German · AI",
                                                       "zh-Hans": "Chinese (Simplified) · AI"}
    assert made["en"]["origin"] == "transcribed" and made["de"]["origin"] == "translated"
    en = vtt.read(folder / "en.vtt")
    assert en and en[0].start == pytest.approx(0.021) and en[0].text.startswith("We wait at home.")
    head = (folder / "de.vtt").read_text().split("\n\n")[1]
    assert head.startswith("NOTE German · AI. Made by subtitler") and "Qwen3.5-9B Q4_K_M (llama.cpp b11430" in head
    assert all("这是一句话" in c.text for c in vtt.read(folder / "zh-Hans.vtt"))
    report = json.loads((folder / "qc.json").read_text())
    assert report["agreement"]["mean_wer"] == 0.0
    assert report["timing"]["cues_without_speech"] == 0
    assert set(report["languages"]) == {"en", "de", "zh-Hans"}
    # the bundle in the log carries the files, not the transcript
    log = capsys.readouterr().out
    data = log.split("-----BEGIN SUBTITLER OUTPUT item-1-----\n")[1].split("-----END")[0]
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode("".join(data.split()))), mode="r:gz") as tar:
        names = sorted(tar.getnames())
    assert names == ["item-1/de.vtt", "item-1/en.vtt", "item-1/qc.json", "item-1/subtitles.json", "item-1/zh-Hans.vtt"]
