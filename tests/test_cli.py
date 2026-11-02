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


class EchoTranslator:
    """Translates by marking each line with the target language."""

    def translate(self, texts, src, tgt):
        from subtitler.translate.madlad import Usage

        return [f"[{tgt.tag}] {t}" for t in texts], Usage(segments=len(texts))


def fake_reading(confident: dict):
    """ocr.read_track as a test wants it: per track path a confidence, the cues invented."""
    from subtitler import ocr
    from subtitler.langs import Lang

    def read_track(path, fmt, lang, tessdata=None, jobs=0, min_confidence=80.0, bilingual=False, models=None):
        conf = confident[path.name]
        got = lang if lang.code != "zh" or lang.script else Lang("zh", "Hant")
        text = {"en": "Where are we going tonight?", "zh": "我們今天晚上去哪裡吃飯？"}[got.code]
        cues = [vtt.Cue(2.0 + 3 * i, 4.0 + 3 * i, text, meta={"conf": conf, "chars": len(text), "italic": False})
                for i in range(40)]
        r = ocr.Recognized(cues, ocr.model_for(got), "Tesseract 5.0.0", conf, 40, 0, 0, 0, 1.0, 1.0, 80.0)
        if conf < min_confidence:
            r.rejected = f"its mean confidence {conf} is under {min_confidence:g}"
        return got, r

    return read_track


def test_run_makes_the_text_copy_first_and_translates_from_it(make_package, tmp_path, monkeypatch):
    from subtitler import cli, ocr

    root = make_package(audio=[("eng", True)], subs=[{"language": "eng", "format": "pgs", "title": "English PGS"},
                                                     {"language": "chi", "format": "pgs", "title": ""}])
    monkeypatch.setattr(ocr, "read_track", fake_reading({"0.sup": 93.0, "1.sup": 91.0}))
    monkeypatch.setattr(cli.Runner, "translator", lambda self: setattr(self, "mt_model", "echo") or EchoTranslator())
    out = tmp_path / "out"
    assert main(["run", str(root), "--out", str(out), "--languages", "en,de,zh-Hans"]) == 0
    m = json.loads((out / "subtitles.json").read_text())
    made = {x["language"]: x for x in m["made"]}
    assert made["en"]["origin"] == "recognized" and made["en"]["label"] == "English · OCR"
    assert made["en"]["attach"] == {"language": "en", "label": "English · OCR", "origin": "recognized",
                                    "forced": False, "sdh": False, "format": "webvtt"}
    # The Chinese track read as Traditional: its copy is zh-Hant, and zh-Hans its conversion.
    assert made["zh-Hant"]["origin"] == "recognized" and made["zh-Hans"]["origin"] == "converted"
    assert made["zh-Hans"]["source"] == "ocr:sub1"
    assert vtt.read(out / "zh-Hans.vtt")[0].text == "我们今天晚上去哪里吃饭？"
    assert made["de"]["origin"] == "translated" and made["de"]["source"] == "ocr:sub0"
    assert vtt.read(out / "de.vtt")[0].text.startswith("[de] Where")
    assert "recognized (OCR) by Tesseract 5.0.0" in (out / "en.vtt").read_text().split("\n")[2]


def test_a_copy_read_too_unsurely_leaves_its_language_to_the_next_candidate(make_package, tmp_path, monkeypatch):
    from subtitler import cli, ocr

    root = make_package(audio=[("chi", True)], subs=[{"language": "eng", "format": "pgs"},
                                                     {"language": "chi", "kind": "Hant"}])
    monkeypatch.setattr(ocr, "read_track", fake_reading({"0.sup": 61.0}))
    monkeypatch.setattr(cli.Runner, "translator", lambda self: setattr(self, "mt_model", "echo") or EchoTranslator())
    out = tmp_path / "out"
    assert main(["run", str(root), "--out", str(out), "--languages", "en"]) == 0
    m = json.loads((out / "subtitles.json").read_text())
    assert m["not_attached"][0]["source"] == "sub0" and "under 80" in m["not_attached"][0]["why"]
    assert [(x["language"], x["origin"], x["source"]) for x in m["made"]] == [("en", "translated", "sub1")]
    assert not (out / "en.forced.vtt").exists()


def test_plan_names_the_copies(make_package, capsys):
    root = make_package(subs=[{"language": "eng", "format": "pgs", "forced": True, "purpose": "forced", "cues": 5},
                              {"language": "eng"}])
    assert main(["plan", str(root), "--languages", "en"]) == 0
    text = capsys.readouterr().out
    assert "ocr      sub0 (pgs, en, forced): a text copy" in text and "en       keep sub1" in text


def test_without_ocr_the_plan_is_the_one_before_it(make_package, capsys):
    # A sparse image track (signs and songs, by its pictures) is not counted without OCR: it is a
    # subtitle a client draws, as it was.
    root = make_package(subs=[{"language": "ger", "format": "pgs", "cues": 3}], duration_ms=6_000_000)
    assert main(["plan", str(root), "--languages", "de", "--no-ocr"]) == 0
    text = capsys.readouterr().out
    assert "de       keep sub0" in text and "partial" not in text
    assert main(["plan", str(root), "--languages", "de"]) == 0
    text = capsys.readouterr().out
    assert "ocr      sub0 (pgs, de, signs): a text copy" in text and "partial" in text
