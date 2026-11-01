"""The OCR step: pictures made ready to read, words to lines and cues, the confidence gate, labels;
end to end with Tesseract where it is installed (text drawn with Pillow's own font)."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from synth import clear, outlined, show, text_mask

from subtitler import ocr, pgs, vtt
from subtitler.langs import Lang
from subtitler.pgs import Picture, Placed


def placed(mask: np.ndarray, x: int = 0, y: int = 0) -> Placed:
    b = outlined(mask)
    luma = np.where(b == 1, 255, 0).astype(np.uint8)
    alpha = np.where(b == 0, 0, 255).astype(np.uint8)
    return Placed(x, y, luma, alpha)


def test_a_part_is_dark_letters_on_white_scaled_to_the_line_height():
    m = np.zeros((18, 60), dtype=bool)
    m[2:16, 5:9] = m[2:16, 20:24] = True  # two strokes 14 px high
    parts = ocr.prepare([placed(m)], line_height=36)
    assert len(parts) == 1
    p = parts[0]
    assert p.scale == pytest.approx(36 / 14, rel=0.05)
    assert p.image.dtype == np.uint8 and set(np.unique(p.image)) == {0, 255}
    assert (p.image[:5] == 255).all() and (p.image[:, :5] == 255).all()  # a white margin
    fill = 2 * 14 * 4 * p.scale ** 2  # the letters' fill, scaled; with its outline it would be 2.5 times that
    assert 0.7 * fill < (p.image == 0).sum() < 1.4 * fill


def test_objects_far_apart_are_read_one_by_one_from_the_top():
    m = np.ones((20, 100), dtype=bool)
    near = [placed(m, 0, 900), placed(m, 0, 935)]
    far = [placed(m, 0, 950), placed(m, 0, 80)]
    assert [len(g) for g in ocr.groups(near)] == [2]
    assert [[o.y for o in g] for g in ocr.groups(far)] == [[80], [950]]


def test_slant_tells_italics():
    upright = np.zeros((40, 200), dtype=bool)
    for x in range(10, 190, 15):
        upright[4:36, x:x + 4] = True
    leaning = np.zeros_like(upright)
    for y in range(40):
        s = int(round((39 - y) * np.tan(np.radians(13))))
        leaning[y, s:] = upright[y, :200 - s]
    assert abs(ocr.slant(upright)) <= 2
    assert 10 <= ocr.slant(leaning) <= 16


def test_models_and_labels():
    assert ocr.model_for("en") == "eng" and ocr.model_for(Lang("de")) == "deu"
    assert ocr.model_for(Lang("zh", "Hans")) == "chi_sim" and ocr.model_for(Lang("zh", "Hant")) == "chi_tra"
    assert ocr.model_for(Lang("zh")) == "chi_sim" and ocr.model_for("xx") is None
    en = Lang("en")
    assert ocr.label("", en, "full") == "English · OCR"
    assert ocr.label("", en, "sdh") == "English · SDH · OCR"
    assert ocr.label("English (Forced)", en, "forced") == "English (Forced) · OCR"
    assert ocr.label("English PGS SDH", en, "sdh") == "English SDH · OCR"
    assert ocr.label("PGS", en, "full") == "English · OCR"
    assert ocr.label("Deutsch (PGS)", Lang("de"), "full") == "Deutsch · OCR"
    assert ocr.label("forced PGS", en, "forced") == "English · forced · OCR"  # a name that is only its kind
    assert ocr.label("ENG", en, "full") == "English · OCR" and ocr.label("简体中字", Lang("zh", "Hans"), "full") == "简体中字 · OCR"


def test_what_tesseract_confuses_is_put_right():
    assert ocr.fix_latin("| know. l'm here, l said.", "en") == "I know. I'm here, I said."
    assert ocr.fix_latin("- Viens vite I - Attends-moi I", "fr") == "- Viens vite ! - Attends-moi !"
    assert ocr.fix_latin("Voilà |", "fr") == "Voilà !" and ocr.fix_latin("Tu aimes pas Ça.", "fr") == "Tu aimes pas ça."
    assert ocr.fix_latin("Ça va, toll!", "de") == "Ça va, toll!"
    assert ocr.fix_latin('Kom je morgen"?', "nl") == "Kom je morgen?" == ocr.fix_latin("Kom je morgen”?", "nl")
    assert ocr.fix_latin('Er sagte: "Nein"?', "de") == 'Er sagte: "Nein"?'  # a pair of quotes stays


def test_an_ellipsis_at_a_lines_edge_is_told_by_its_pixels():
    ink = np.zeros((40, 300), dtype=bool)
    for x in (2, 10, 18):  # three dots low in the line
        ink[30:35, x:x + 5] = True
    ink[2:38, 30:240:12] = True  # letters
    for x in (250, 258):  # two dots at its end
        ink[30:35, x:x + 5] = True
    assert ocr.dots(ink) == (3, 2)
    assert ocr.ellipses("en dan gaan we naar huis .", 3, 2) == "...en dan gaan we naar huis..."
    assert ocr.ellipses("..We will see.", 3, 0) == "...We will see."
    assert ocr.ellipses("...en toen?", 3, 0) == "...en toen?" and ocr.ellipses("Gut.", 0, 1) == "Gut."


def test_tsv_words_become_lines():
    tsv = ("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
           "4\t1\t1\t1\t1\t0\t10\t10\t300\t40\t-1\t\n"
           "5\t1\t1\t1\t1\t1\t10\t10\t80\t40\t96.5\t|\n"
           "5\t1\t1\t1\t1\t2\t100\t10\t90\t40\t91.0\tl'm\n"
           "5\t1\t1\t1\t2\t1\t10\t60\t80\t40\t50.0\there.\n"
           "5\t2\t1\t1\t1\t1\t10\t10\t40\t40\t90.0\t我们\n"
           "5\t2\t1\t1\t1\t2\t52\t10\t40\t40\t80.0\t走吧\n"
           "5\t2\t1\t1\t1\t3\t140\t10\t40\t40\t70.0\t快\n")
    pages = ocr.parse_tsv(tsv)
    part = ocr.Part(np.full((120, 400), 255, dtype=np.uint8), 1.0, 40)
    lines = ocr.lines_of(pages[1], part, "en")
    assert [x.text for x in lines] == ["I I'm", "here."]
    assert lines[0].conf == pytest.approx((96.5 * 1 + 91.0 * 3) / 4)
    img = np.full((120, 400), 255, dtype=np.uint8)
    for x0, x1 in ((10, 50), (52, 92), (140, 180)):  # the characters' ink: a narrow gap, then a wide one
        img[12:48, x0:x1] = 0
    zh = ocr.lines_of(pages[2], ocr.Part(img, 1.0, 40), "zh")
    assert [x.text for x in zh] == ["我们走吧 快"]  # a wide gap in the picture is a space, a narrow one none
    both = [ocr.Line("Tom & <Jerry>", 90, 13, True, 12.0), ocr.Line("x", 90, 1, False, 0.0)]
    assert ocr.cue_text(both) == "<i>Tom &amp; &lt;Jerry&gt;</i>\nx" and vtt.plain(ocr.cue_text(both)) == "Tom & <Jerry>\nx"


class FakeEngine:
    """Reads what the test says each image holds."""

    def __init__(self, answers):
        self.answers = answers
        self.cpu_seconds = 0.0

    def version(self):
        return "5.0.0"

    def read(self, images):
        return [[ocr.Word((1, 1, 1), 10, 10, 50, 30, conf, t) for t, conf in self.answers.get(i, [])]
                for i, _ in enumerate(images)]


def picture(start, end, y=900):
    m = np.zeros((30, 120), dtype=bool)
    m[5:25, 10:110:9] = True
    return Picture(start, end, [placed(m, 400, y)], (1920, 1080))


def test_cues_with_the_pictures_times_and_the_confidence_gate():
    pics = [picture(1.0, 3.0), picture(3.0, 4.0), picture(5.0, 6.0), picture(7.0, None)]
    eng = FakeEngine({0: [("Hello", 95.0)], 1: [("Hello", 93.0)], 3: [("again.", 60.0)]})
    r = ocr.recognize(pics, "en", engine=eng, min_confidence=70)
    assert [(c.start, c.end, c.text) for c in r.cues] == [(1.0, 4.0, "Hello"), (7.0, 11.0, "again.")]
    assert r.empty == 1 and r.pictures == 4 and r.accepted  # the same lines in a row are one cue
    assert r.confidence == pytest.approx((95 * 5 + 60 * 6) / 11)
    low = ocr.recognize(pics, "en", engine=FakeEngine({0: [("Hm", 40.0)]}), min_confidence=80)
    assert not low.accepted and "under 80" in low.rejected
    none = ocr.recognize(pics, "en", engine=FakeEngine({}))
    assert not none.accepted and none.cues == []


needs_tesseract = pytest.mark.skipif(not shutil.which("tesseract") or text_mask("x") is None,
                                     reason="tesseract or Pillow is not installed")


@needs_tesseract
def test_tesseract_reads_a_synthetic_pgs_track(tmp_path):
    lines = ["Where are we going tonight?", "Nobody knows yet.\nThen we stay home."]
    data = b""
    for k, text in enumerate(lines):
        b = outlined(text_mask(text, 38))
        data += show(10.0 + 4 * k, [(0, 300, 860, b)]) + clear(12.0 + 4 * k)
    it = outlined(text_mask("Wait for me!", 38, italic=True))
    data += show(20.0, [(0, 300, 860, it)]) + clear(22.0)
    f = tmp_path / "2.sup"
    f.write_bytes(data)
    if "eng" not in ocr.Tesseract("eng").models():
        pytest.skip("Tesseract has no English model")
    lang, r = ocr.read_track(f, "pgs", Lang("en"), jobs=2)
    assert lang.tag == "en" and r.accepted and r.confidence > 80
    assert [(c.start, c.end) for c in r.cues] == [(10.0, 12.0), (14.0, 16.0), (20.0, 22.0)]
    assert vtt.plain(r.cues[0].text) == lines[0]
    assert r.cues[1].text.split("\n") == lines[1].split("\n")  # its line break kept
    assert r.cues[2].text == "<i>Wait for me!</i>" and r.italic == 1
    assert r.cpu_seconds > 0


def test_pictures_of_a_sup_file(tmp_path):
    m = np.zeros((30, 120), dtype=bool)
    m[5:25, 10:110:9] = True
    f = tmp_path / "1.sup"
    f.write_bytes(show(1.0, [(0, 10, 10, outlined(m))]) + clear(2.0))
    assert [(p.start, p.end) for p in ocr.pictures_of(f, "pgs")] == [(1.0, 2.0)] == [(p.start, p.end) for p in pgs.read(f)]


def test_a_model_tesseract_lacks_is_fetched_into_the_models_folder(tmp_path, monkeypatch):
    own = tmp_path / "own"
    own.mkdir()
    (own / "eng.traineddata").write_bytes(b"eng model")
    monkeypatch.setattr(ocr.Tesseract, "models", lambda self: {"eng"} if self.tessdata in (None, str(own)) else set())
    fetched = []

    class Answer:
        def __init__(self, url):
            fetched.append(url)
            self.data = b"a model fetched"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            out, self.data = self.data, b""
            return out

    monkeypatch.setattr(ocr.urllib.request, "urlopen", lambda url, timeout=0: Answer(url))
    assert ocr.tessdata_for("eng", own) == str(own)  # it has the model: nothing fetched
    with pytest.raises(LookupError, match="fra"):
        ocr.tessdata_for("fra", own)
    cache = tmp_path / "models" / "tessdata"
    assert ocr.tessdata_for("chi_sim+eng", own, cache) == str(cache)
    assert fetched == ["https://github.com/tesseract-ocr/tessdata_best/raw/4.1.0/chi_sim.traineddata"]
    assert ocr.tessdata_url("deu") == "https://github.com/tesseract-ocr/tessdata_fast/raw/4.1.0/deu.traineddata"
    assert (cache / "eng.traineddata").read_bytes() == b"eng model"  # its own, copied beside the fetched one
    assert (cache / "chi_sim.traineddata").read_bytes() == b"a model fetched"


def test_an_unsure_part_is_read_again_in_grey_and_the_surer_reading_kept():
    class TwoLooks(FakeEngine):
        def __init__(self):
            super().__init__({})
            self.calls = []

        def read(self, images):
            self.calls.append(images)
            if len(self.calls) == 1:  # thresholded: one sure, one unsure
                answers = [[("Hello", 95.0)], [("Hcllo", 40.0)]]
            else:  # grey: the unsure one again
                answers = [[("Hello", 88.0)]]
            return [[ocr.Word((1, 1, 1), 10, 10, 50, 30, c, t) for t, c in a] for a in answers[:len(images)]]

    eng = TwoLooks()
    r = ocr.recognize([picture(1.0, 2.0), picture(3.0, 4.0)], "en", engine=eng, min_confidence=0)
    assert [c.text for c in r.cues] == ["Hello", "Hello"] and len(eng.calls[1]) == 1
    grey = eng.calls[1][0]
    assert len(np.unique(grey)) > 2  # the second look is not thresholded
    off = TwoLooks()
    r = ocr.recognize([picture(1.0, 2.0), picture(3.0, 4.0)], "en", engine=off, min_confidence=0, second_look=0)
    assert [c.text for c in r.cues] == ["Hello", "Hcllo"] and len(off.calls) == 1


@needs_tesseract
def test_tesseract_reads_a_synthetic_vobsub_track(tmp_path):
    from synth import spu, vobsub

    units = [(5.0, spu(outlined(text_mask("Nobody knows yet.", 22)), 200, 400, 0.0, 2.0)),
             (9.0, spu(outlined(text_mask("Then we stay home.", 22)), 190, 400, 0.0, 2.5))]
    idx, sub = vobsub(units)
    (tmp_path / "4.idx").write_text(idx, encoding="latin-1")
    (tmp_path / "4.sub").write_bytes(sub)
    lang, r = ocr.read_track(tmp_path / "4.idx", "vobsub", Lang("en"), jobs=1)
    assert [c.text for c in r.cues] == ["Nobody knows yet.", "Then we stay home."]
    assert r.cues[0].start == pytest.approx(5.0) and r.cues[1].end == pytest.approx(11.5, abs=0.012)
