"""subtitler run on a synthetic package: the convert path end to end, and the package guard."""

import json

from subtitler import vtt
from subtitler.cli import main


def test_run_converts_traditional_and_writes_the_manifest(make_package, tmp_path):
    root = make_package(subs=[{"language": "eng"}, {"language": "ger"}, {"language": "chi", "kind": "Hant"}])
    out = tmp_path / "out"
    assert main(["run", str(root), "--out", str(out), "--languages", "en,de,zh-Hans"]) == 0
    manifest = json.loads((out / "subtitles.json").read_text())
    assert [k["language"] for k in manifest["kept"]] == ["en", "de"]
    made = manifest["made"][0]
    assert made["origin"] == "converted" and made["label"] == "Chinese (Simplified) · Converted"
    cues = vtt.read(out / "zh-Hans.vtt")
    assert cues[0].text == "我们今天晚上去哪里吃饭？"
    assert "Converted" in (out / "zh-Hans.vtt").read_text().split("\n")[2]  # the NOTE says how it was made


def test_run_refuses_to_write_into_the_package(make_package):
    root = make_package(subs=[{"language": "chi", "kind": "Hant"}])
    assert main(["run", str(root), "--out", str(root / "subs" / "made")]) == 2
    assert not (root / "subs" / "made").exists()


def test_plan_prints_and_writes_nothing(make_package, capsys):
    root = make_package(audio=[("kor", True)], subs=[])
    assert main(["plan", str(root)]) == 0
    text = capsys.readouterr().out
    assert "translate the transcript (ko)" in text
    assert sorted(p.name for p in root.iterdir()) == ["package.json", "subs"]
