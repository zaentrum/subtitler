"""subtitler ocr: the text copies of packaged titles, title by title; the hours it reads in; its dry run."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from subtitler import ocr, recognize, vtt
from subtitler.langs import Lang


def at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 11, 2, hour, minute, tzinfo=timezone.utc)


def test_the_hours_leave_the_evening_out():
    h = recognize.parse_hours("22-17")
    assert recognize.within(h, at(23)) and recognize.within(h, at(3)) and recognize.within(h, at(16, 59))
    assert not recognize.within(h, at(17)) and not recognize.within(h, at(21, 59))
    assert recognize.until_open(h, at(18)) == 4 * 3600 and recognize.until_open(h, at(2)) == 0
    assert recognize.within(recognize.parse_hours("0-24"), at(19))
    assert not recognize.within(recognize.parse_hours("1-5"), at(5)) and recognize.within(recognize.parse_hours("1-5"), at(1))
    with pytest.raises(ValueError):
        recognize.parse_hours("22-30")


def test_a_paced_read_is_the_file(tmp_path):
    f = tmp_path / "x.sup"
    f.write_bytes(bytes(range(256)) * 9000)
    assert recognize.paced_read(f, 1000.0) == f.read_bytes() == recognize.paced_read(f, None)


def titles(tmp_path, root) -> str:
    t = tmp_path / "titles.json"
    t.write_text(json.dumps([{"id": "item-9", "name": "E9", "package": str(root)}]))
    return str(t)


def test_a_dry_run_says_what_it_would_read_and_reads_no_picture(make_package, tmp_path, monkeypatch):
    root = make_package(subs=[{"language": "eng", "format": "pgs"}, {"language": "chi", "format": "pgs", "title": ""},
                              {"language": "fre", "format": "pgs"}])
    monkeypatch.setattr(recognize, "_read_pictures", lambda *a: pytest.fail("a dry run reads no picture"))
    out = tmp_path / "out"
    assert recognize.main(["--titles", titles(tmp_path, root), "--out", str(out), "--dry-run"]) == 0
    r = json.loads((out / "item-9" / "ocr.json").read_text())
    assert [(c["source"], c["language"], c["kind"], c["pictures"]) for c in r["copies"]] == [
        ("sub0", "en", "full", None), ("sub1", "zh", "full", None)]  # package.json alone: nothing counted
    assert r["copies"][0]["bytes"] == (root / "subs" / "0.sup").stat().st_size
    assert r["model"] == {"sub0": "eng", "sub1": "chi_sim or chi_tra (read to tell)"}
    assert not list((out / "item-9").glob("*.vtt"))


def test_a_run_writes_the_copies_and_sets_aside_the_unsure_ones(make_package, tmp_path, monkeypatch):
    root = make_package(subs=[{"language": "eng", "format": "pgs", "title": "English PGS"},
                              {"language": "ger", "format": "pgs", "forced": True, "purpose": "forced", "cues": 4},
                              {"language": "ger"}])

    def read(pictures, lang, tessdata=None, jobs=0, min_confidence=80.0, bilingual=False, models=None, **kw):
        conf = 92.0 if lang.code == "en" else 55.0
        cues = [vtt.Cue(p.start, p.end, "Then we stay home.", meta={"conf": conf, "chars": 18, "italic": False})
                for p in pictures]
        r = ocr.Recognized(cues, ocr.model_for(lang), "Tesseract 5.0.0", conf, len(pictures), 0, 0, 0, 1.0, 2.0, 60.0)
        if conf < min_confidence:
            r.rejected = f"its mean confidence {conf} is under {min_confidence:g}"
        return Lang(lang.code), r

    monkeypatch.setattr(ocr, "read_pictures", read)
    out = tmp_path / "out"
    assert recognize.main(["--titles", titles(tmp_path, root), "--out", str(out), "--hours", "0-24", "--pause", "0",
                           "--read-mbps", "0"]) == 0
    r = json.loads((out / "item-9" / "ocr.json").read_text())
    (made,) = r["made"]
    assert made["file"] == "en.vtt" and made["attach"] == {"language": "en", "label": "English · OCR",
                                                           "origin": "recognized", "forced": False, "sdh": False,
                                                           "format": "webvtt"}
    assert len(vtt.read(out / "item-9" / "en.vtt")) == 40
    (left,) = r["not_attached"]
    assert left["kind"] == "forced" and "under 80" in left["why"] and left["file"] == "not-attached-de.forced.vtt"
    assert (out / "item-9" / "not-attached-de.forced.vtt").exists()  # aside, for a person to look at
    run = json.loads((out / "run.json").read_text())
    assert run["mode"] == "ocr" and run["titles"][0]["id"] == "item-9"
    # A run after it takes the titles it has not done: this one it does not read again.
    monkeypatch.setattr(ocr, "read_pictures", lambda *a, **k: pytest.fail("read again"))
    assert recognize.main(["--titles", titles(tmp_path, root), "--out", str(out), "--hours", "0-24", "--pause", "0"]) == 0
