"""Synthetic packages: a package.json and subtitle files with invented text."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LINES = {
    "en": ["Where are we going tonight?", "Nobody knows yet.", "Then we stay home."],
    "de": ["Wohin gehen wir heute Abend?", "Das weiß noch keiner.", "Dann bleiben wir zu Hause."],
    "it": ["Dove andiamo stasera?", "Nessuno lo sa ancora.", "Allora restiamo a casa."],
    "fr": ["On va où ce soir ?", "Personne ne le sait encore.", "Alors on reste à la maison."],
    "ko": ["오늘 밤에 어디 가?", "아직 아무도 몰라.", "그럼 집에 있자."],
    "Hans": ["我们今天晚上去哪里吃饭？", "这个问题很难回答，因为时间已经太晚了。", "那我们就在家里等着吧。"],
    "Hant": ["我們今天晚上去哪裡吃飯？", "這個問題很難回答，因為時間已經太晚了。", "那我們就在家裡等著吧。"],
    "yue": ["我哋今晚去邊度食飯呀？", "佢話唔知啊，你喺邊度？", "冇問題，嚟啦，我哋等你。"],
}


def vtt_text(kind: str, n: int = 40) -> str:
    out = ["WEBVTT", ""]
    for i in range(n):
        s = 2.0 + i * 3.0
        out += [f"00:{int(s // 60):02d}:{s % 60:06.3f} --> 00:{int((s + 2) // 60):02d}:{(s + 2) % 60:06.3f}",
                LINES[kind][i % 3], ""]
    return "\n".join(out)


@pytest.fixture
def make_package(tmp_path):
    """make_package(audio=[...], subs=[...]) -> package folder.

    audio: (language, default[, purpose]); subs: dicts with language, kind
    (key of LINES; default the language), title, forced, purpose, format,
    cues, exists.
    """

    def make(audio=(("eng", True),), subs=(), duration_ms=600_000):
        root = tmp_path / "pkg"
        (root / "subs").mkdir(parents=True, exist_ok=True)
        entries = []
        for i, s in enumerate(subs):
            fmt = s.get("format", "webvtt")
            path = f"subs/{i}.{'vtt' if fmt == 'webvtt' else 'sup'}"
            if s.get("exists", True):
                if fmt == "webvtt":
                    from subtitler.langs import iso1

                    code = iso1(s["language"])
                    kind = s.get("kind") or ("Hans" if code == "zh" else code)
                    (root / path).write_text(vtt_text(kind, s.get("cues", 40)), encoding="utf-8")
                else:
                    (root / path).write_bytes(b"PG\x00\x00")
            entries.append({"id": f"sub{i}", "path": path, "language": s["language"], "title": s.get("title", ""),
                            "forced": s.get("forced", False), "purpose": s.get("purpose", "dialogue"),
                            "format": fmt, "visible": s.get("visible", True)})
        manifest = {"durationMs": duration_ms, "subtitles": entries, "renditions": {"audio": [
            {"id": f"a{k}", "dir": f"hls/a{k}", "language": a[0], "default": a[1],
             "purpose": a[2] if len(a) > 2 else "main"} for k, a in enumerate(audio)]}}
        (root / "package.json").write_text(json.dumps(manifest), encoding="utf-8")
        return root

    return make
