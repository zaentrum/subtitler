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
    assert left["kind"] == "forced" and "under 80" in left["why"] and left["file"] == "not-attached-sub1.de.forced.vtt"
    assert (out / "item-9" / "not-attached-sub1.de.forced.vtt").exists()  # aside, for a person to look at
    run = json.loads((out / "run.json").read_text())
    assert run["mode"] == "ocr" and run["titles"][0]["id"] == "item-9"
    # A run after it takes the titles it has not done: this one it does not read again.
    monkeypatch.setattr(ocr, "read_pictures", lambda *a, **k: pytest.fail("read again"))
    assert recognize.main(["--titles", titles(tmp_path, root), "--out", str(out), "--hours", "0-24", "--pause", "0"]) == 0


def fake_read(conf=92.0, script="Hans", fail=None):
    """ocr.read_pictures as a test reads it: every picture one cue; a Chinese track of no script read
    as `script`; a track of `fail` pictures raises."""

    def read(pictures, lang, tessdata=None, jobs=0, min_confidence=80.0, bilingual=False, models=None, **kw):
        if fail is not None and len(pictures) == fail:
            raise KeyError("a reader that broke")
        got = lang if lang.code != "zh" or lang.script else Lang("zh", script)
        cues = [vtt.Cue(p.start, p.end, "Then we stay home.", meta={"conf": conf, "chars": 18, "italic": False})
                for p in pictures]
        return got, ocr.Recognized(cues, ocr.model_for(got), "Tesseract 5.0.0", conf, len(pictures), 0, 0, 0, 1.0,
                                   2.0, 60.0)

    return read


RUN = ["--hours", "0-24", "--pause", "0", "--read-mbps", "0"]


def test_two_chinese_tracks_read_as_one_script_make_one_copy_and_signs_attach_as_forced(make_package, tmp_path,
                                                                                         monkeypatch):
    root = make_package(subs=[{"language": "chi", "format": "pgs", "title": "简体", "cues": 300},
                              {"language": "chi", "format": "pgs", "title": "", "cues": 250},
                              {"language": "eng", "format": "pgs", "cues": 320},
                              {"language": "eng", "format": "pgs", "cues": 12}], duration_ms=1_200_000)
    monkeypatch.setattr(ocr, "read_pictures", fake_read())
    out = tmp_path / "out"
    assert recognize.main(["--titles", titles(tmp_path, root), "--out", str(out), *RUN]) == 0
    r = json.loads((out / "item-9" / "ocr.json").read_text())
    assert r["version"] == root.name and r["package"] == str(root)  # what the copies were read from
    made = {m["source"]: m for m in r["made"]}
    assert sorted(made) == ["sub0", "sub2", "sub3"]
    (twice,) = r["not_attached"]  # the untitled track, read as Simplified too
    assert twice["source"] == "sub1" and "made already, from sub0" in twice["why"]
    assert twice["file"] == "not-attached-sub1.zh-Hans.vtt" and (out / "item-9" / twice["file"]).exists()
    assert len(vtt.read(out / "item-9" / "zh-Hans.vtt")) == 300  # the first one's, not overwritten
    signs = made["sub3"]  # 12 pictures in 20 minutes: signs and songs
    assert signs["kind"] == "signs" and signs["file"] == "en.signs.vtt"
    assert signs["attach"] == {"language": "en", "label": "English · Signs · OCR", "origin": "recognized",
                               "forced": True, "sdh": False, "format": "webvtt"}
    assert not list((out / "item-9").glob(".*.tmp")) and not (out / "item-9" / "started").exists()


def test_a_track_whose_reading_breaks_costs_its_title_nothing_else(make_package, tmp_path, monkeypatch):
    root = make_package(subs=[{"language": "eng", "format": "pgs", "cues": 40},
                              {"language": "ger", "format": "pgs", "cues": 30}])
    monkeypatch.setattr(ocr, "read_pictures", fake_read(fail=30))
    out = tmp_path / "out"
    assert recognize.main(["--titles", titles(tmp_path, root), "--out", str(out), *RUN]) == 0
    r = json.loads((out / "item-9" / "ocr.json").read_text())
    assert [m["source"] for m in r["made"]] == ["sub0"] and "error" not in r
    assert r["not_attached"] == [{"source": "sub1", "language": "de", "kind": "full",
                                  "why": "'a reader that broke'"}]


def test_a_title_that_killed_a_run_is_left_out_and_one_stopped_by_a_signal_is_read_again(make_package, tmp_path,
                                                                                          monkeypatch):
    root = make_package(subs=[{"language": "eng", "format": "pgs"}])
    out = tmp_path / "out"
    (out / "item-9").mkdir(parents=True)
    (out / "item-9" / "started").write_text("{}")  # what a run killed in the title leaves
    monkeypatch.setattr(ocr, "read_pictures", lambda *a, **k: pytest.fail("read again"))
    assert recognize.main(["--titles", titles(tmp_path, root), "--out", str(out), *RUN]) == 1
    r = json.loads((out / "item-9" / "ocr.json").read_text())
    assert r["stopped"] and "killed" in r["error"] and not (out / "item-9" / "started").exists()
    assert recognize.main(["--titles", titles(tmp_path, root), "--out", str(out), *RUN]) == 0  # left out again
    # --again reads it; a SIGTERM while it does takes the mark away, so the next run reads it.
    import signal

    def stopped(*a, **k):
        assert (out / "item-9" / "started").exists()
        signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(ocr, "read_pictures", stopped)
    with pytest.raises(SystemExit):
        recognize.main(["--titles", titles(tmp_path, root), "--out", str(out), "--again", *RUN])
    assert not (out / "item-9" / "started").exists()
    assert signal.getsignal(signal.SIGTERM) is not recognize._stopped  # the run's handler is its own


def test_titles_that_are_no_list_of_titles_are_refused_with_why(tmp_path, capsys):
    t = tmp_path / "titles.json"
    t.write_text('SET\n[{"id": "item-1", "package": "/x"}]\n')  # a query's output with its SET line
    assert recognize.main(["--titles", str(t), "--out", str(tmp_path / "out")]) == 2
    assert "no JSON list of titles" in capsys.readouterr().out
    t.write_text(json.dumps([{"id": "../item-1", "package": "/x"}]))
    assert recognize.main(["--titles", str(t), "--out", str(tmp_path / "out")]) == 2
    assert "no plain folder name" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_a_stop_at_any_point_of_a_title_leaves_no_mark_and_a_stale_one_is_no_kill(make_package, tmp_path,
                                                                                  monkeypatch):
    import signal

    root = make_package(subs=[{"language": "eng", "format": "pgs"}])
    out = tmp_path / "out"
    real = recognize.write_json

    def write(path, data):  # a SIGTERM right after the mark is written, before anything else happens
        real(path, data)
        if path.name == "started":
            signal.raise_signal(signal.SIGTERM)

    monkeypatch.setattr(recognize, "write_json", write)
    with pytest.raises(SystemExit):
        recognize.main(["--titles", titles(tmp_path, root), "--out", str(out), *RUN])
    assert not (out / "item-9" / "started").exists()
    monkeypatch.setattr(recognize, "write_json", real)
    # A Ctrl-C while the title is read: the same.
    monkeypatch.setattr(ocr, "read_pictures", lambda *a, **k: signal.raise_signal(signal.SIGINT))
    with pytest.raises(SystemExit):
        recognize.main(["--titles", titles(tmp_path, root), "--out", str(out), *RUN])
    assert not (out / "item-9" / "started").exists()
    # A mark left beside a title's result (a stop after it was written) is stale: the result stands.
    monkeypatch.setattr(ocr, "read_pictures", fake_read())
    assert recognize.main(["--titles", titles(tmp_path, root), "--out", str(out), *RUN]) == 0
    (out / "item-9" / "started").write_text("{}")
    monkeypatch.setattr(ocr, "read_pictures", lambda *a, **k: pytest.fail("read again"))
    assert recognize.main(["--titles", titles(tmp_path, root), "--out", str(out), *RUN]) == 0
    r = json.loads((out / "item-9" / "ocr.json").read_text())
    assert "error" not in r and r["made"] and not (out / "item-9" / "started").exists()
