"""The source's own subtitles: the choice of streams, and a real extraction where ffmpeg is there."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from subtitler import extract, vtt
from subtitler.langs import parse_list

PROBE = {"streams": [
    {"index": 0, "codec_type": "video", "codec_name": "hevc"},
    {"index": 1, "codec_type": "audio", "codec_name": "eac3", "tags": {"language": "eng"}},
    {"index": 2, "codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "eng"}, "disposition": {}},
    {"index": 3, "codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "eng", "title": "SDH"},
     "disposition": {"hearing_impaired": 1}},
    {"index": 4, "codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "chi", "title": "Simplified"}},
    {"index": 5, "codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "chi", "title": "Traditional"}},
    {"index": 6, "codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "ger"}},
    {"index": 7, "codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "eng", "title": "Commentary"}},
    {"index": 8, "codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle", "tags": {"language": "ger"}},
    {"index": 9, "codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "ger", "title": "Forced"},
     "disposition": {"forced": 1}},
    {"index": 10, "codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "fre"}},
]}


def test_the_streams_in_the_instance_languages_are_chosen_each_as_it_is():
    picks, left = extract.choose(extract.subtitle_streams(PROBE), parse_list("en,de,zh-Hans"))
    assert [(p.stream.index, p.lang.tag, p.stream.sdh, p.stream.forced) for p in picks] == [
        (2, "en", False, False), (3, "en", True, False), (4, "zh-Hans", False, False),
        (6, "de", False, False), (9, "de", False, True)]
    assert [p.name for p in picks] == ["en-2.vtt", "en.sdh-3.vtt", "zh-Hans-4.vtt", "de-6.vtt", "de.forced-9.vtt"]
    why = {x["stream"]: x["why"] for x in left}
    assert why[5].startswith("Chinese in the other script") and why[7] == "commentary" and "image subtitle" in why[8]
    assert 10 not in why  # French is no language of the instance


SRT = {"en": ["Where are we going tonight?", "Nobody knows yet."],
       "zh": ["我们今天晚上去哪里吃饭？这个问题很难回答，因为时间已经太晚了。那我们就在家里等着吧。" * 2,
              "这个问题很难回答，因为时间已经太晚了。我们今天晚上去哪里吃饭？" * 2],
       "zht": ["我們今天晚上去哪裡吃飯？這個問題很難回答，因為時間已經太晚了。那我們就在家裡等著吧。" * 2,
               "這個問題很難回答，因為時間已經太晚了。我們今天晚上去哪裡吃飯？" * 2]}


def srt(lines):
    return "".join(f"{i + 1}\n00:00:0{1 + 2 * i},000 --> 00:00:0{2 + 2 * i},500\n{t}\n\n" for i, t in enumerate(lines))


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")
def test_a_files_own_tracks_come_out_as_webvtt(tmp_path):
    for k, lines in SRT.items():
        (tmp_path / f"{k}.srt").write_text(srt(lines), encoding="utf-8")
    mkv = tmp_path / "original.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=6",
                    "-i", str(tmp_path / "en.srt"), "-i", str(tmp_path / "en.srt"), "-i", str(tmp_path / "zh.srt"),
                    "-i", str(tmp_path / "zht.srt"),
                    "-map", "0:v", "-map", "1", "-map", "2", "-map", "3", "-map", "4", "-c:v", "mpeg4",
                    "-c:s", "srt",
                    "-metadata:s:s:0", "language=eng", "-metadata:s:s:1", "language=eng", "-metadata:s:s:1", "title=SDH",
                    "-disposition:s:1", "hearing_impaired",
                    "-metadata:s:s:2", "language=chi", "-metadata:s:s:3", "language=chi", str(mkv)], check=True)
    titles = tmp_path / "titles.json"
    titles.write_text(json.dumps([{"id": "item-1", "name": "E1", "original": str(mkv)}]))
    out = tmp_path / "out"
    assert extract.main(["--titles", str(titles), "--out", str(out), "--languages", "en,de,zh-Hans"]) == 0
    r = json.loads((out / "item-1" / "extracted.json").read_text())
    files = {m["file"]: m for m in r["made"]}
    assert set(files) == {"en-1.vtt", "en.sdh-2.vtt", "zh-Hans-3.vtt"}
    assert files["zh-Hans-3.vtt"]["scriptFrom"] == "text"  # no title: told by its characters
    assert any("other script" in x["why"] for x in r["left_out"])  # the Traditional one
    assert r["languages_without_a_track"] == ["de"]
    assert files["en.sdh-2.vtt"]["attach"] == {"language": "en", "origin": "source", "sdh": True, "forced": False,
                                              "format": "webvtt"}
    en = vtt.read(out / "item-1" / "en-1.vtt")
    assert [c.text for c in en] == SRT["en"] and en[0].start == pytest.approx(1.0)
    note = (out / "item-1" / "en.sdh-2.vtt").read_text().split("\n\n")[1]
    assert note.startswith("NOTE English · SDH. Extracted by subtitler") and "stream 2 (subrip, 'SDH')" in note
