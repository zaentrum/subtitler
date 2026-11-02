"""The OCR step: pictures made ready to read, words to lines and cues, the confidence gate, labels;
end to end with Tesseract where it is installed (text drawn with Pillow's own font)."""

from __future__ import annotations

import hashlib
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
    # A copy of signs and songs says so: it is attached as forced.
    assert ocr.label("", en, "signs") == "English · Signs · OCR" == ocr.label("English", en, "signs")
    assert ocr.label("Signs & Songs", en, "signs") == "Signs & Songs · OCR"
    # The catalog keeps 120 characters of a label: a longer name is cut, the mark kept.
    long = ocr.label("English, " + "the release group's notes on this track " * 5, en, "full")
    assert len(long) == ocr.LABEL_WIDTH and long.endswith("… · OCR")


def test_what_tesseract_confuses_is_put_right():
    assert ocr.fix_latin("| know. l'm here, l said.", "en") == "I know. I'm here, I said."
    assert ocr.fix_latin("- Viens vite I - Attends-moi I", "fr") == "- Viens vite ! - Attends-moi !"
    assert ocr.fix_latin("Voilà |", "fr") == "Voilà !" and ocr.fix_latin("Tu aimes pas Ça.", "fr") == "Tu aimes pas ça."
    assert ocr.fix_latin("Ça va, toll!", "de") == "Ça va, toll!"
    assert ocr.fix_latin('Kom je morgen"?', "nl") == "Kom je morgen?" == ocr.fix_latin("Kom je morgen”?", "nl")
    assert ocr.fix_latin('Er sagte: "Nein"?', "de") == 'Er sagte: "Nein"?'  # a pair of quotes stays


def test_a_hash_at_a_lines_edge_is_the_music_note_it_stands_for():
    assert ocr.music_notes("# La la la #", "en") == "♪ La la la ♪" and ocr.music_notes("#", "de") == "♪"
    assert ocr.music_notes("## Somewhere far ##", "en") == "♪♪ Somewhere far ♪♪"
    assert ocr.music_notes("We're #1", "en") == "We're #1" and ocr.music_notes("Room # 4", "en") == "Room # 4"
    assert ocr.music_notes("#我们走吧#", "zh") == "♪我们走吧♪"


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
    both = [ocr.Line("Salt & <pepper>", 90, 13, True, 12.0), ocr.Line("x", 90, 1, False, 0.0)]
    assert ocr.cue_text(both) == "<i>Salt &amp; &lt;pepper&gt;</i>\nx" and vtt.plain(ocr.cue_text(both)) == "Salt & <pepper>\nx"


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
    low = ocr.recognize(pics, "en", engine=FakeEngine({0: [("Hmmm", 40.0)]}), min_confidence=80)
    assert not low.accepted and "under 80" in low.rejected
    none = ocr.recognize(pics, "en", engine=FakeEngine({}))
    assert not none.accepted and none.cues == []


def test_a_short_unsure_cue_is_left_out_and_a_track_of_them_is_not_attached():
    pics = [picture(1.0 + 2 * k, 2.0 + 2 * k) for k in range(20)]
    words = {k: [("Nobody knows yet.", 95.0)] for k in range(20)}
    words[3] = [("DS", 44.0)]  # a mark read as letters, unsure
    words[4] = [("Oh", 96.0)]  # a short word read surely stays
    r = ocr.recognize(pics, "en", engine=FakeEngine(words), second_look=0)
    assert r.accepted and r.dropped == 1 and len(r.cues) == 19 and "DS" not in [c.text for c in r.cues]
    assert r.summary()["dropped"] == 1
    # A third of the pictures graphics: read as nothing, or as marks. The gate of confidence alone
    # would pass it (the words read are sure); their share does not.
    for k in range(13, 20):
        words[k] = [] if k % 2 else [("a", 30.0)]
    r = ocr.recognize(pics, "en", engine=FakeEngine(words), second_look=0)
    assert r.confidence > 90 and not r.accepted and "read as nothing, as a stray mark" in r.rejected
    # A few of them in a long track: attached.
    assert ocr.recognize(pics * 5, "en", engine=FakeEngine({k: [("Fine.", 95.0)] if k % 30 else []
                                                            for k in range(100)}), second_look=0).accepted


def test_a_copy_that_reads_as_another_language_is_not_attached():
    def cues(lines):
        return [vtt.Cue(2.0 * k, 2.0 * k + 1, line) for k, line in enumerate(lines)]

    german = ["Ich weiß nicht, wo er ist.", "Das ist doch nicht dein Ernst!", "Wir haben noch Zeit, oder?",
              "Er hat mir das nicht gesagt.", "Kannst du mir bitte helfen?"] * 4
    english = ["I don't know where he is.", "You can't be serious!", "We still have time, don't we?",
               "He didn't tell me that.", "Can you help me with this?"] * 4
    assert "reads as German, not English" in ocr.wrong_language(cues(german), "en")
    assert "reads as English, not German" in ocr.wrong_language(cues(english), "de")
    assert ocr.wrong_language(cues(english), "en") is None and ocr.wrong_language(cues(german), "de") is None
    assert ocr.wrong_language(cues(german[:3]), "en") is None  # too few words to tell
    assert ocr.wrong_language(cues(english), "cs") is None  # a language it has no words of: not judged
    chinese = ["我不知道他在哪里。", "你不是认真的吧！", "我们还有时间，对吧？", "他没有告诉我这件事。"] * 4
    assert ocr.wrong_language(cues(chinese), "zh") is None
    assert "another script" in ocr.wrong_language(cues(english), "zh")  # Latin read with a Chinese model
    both = [f"{a}\n{b}" for a, b in zip(chinese, english)]
    assert ocr.wrong_language(cues(both), "zh", bilingual=True) is None and ocr.wrong_language(cues(both), "zh")


def test_the_cpus_are_the_containers_limit(tmp_path):
    assert ocr.cpus(tmp_path) >= 1  # no cgroup files: the CPUs it may run on
    (tmp_path / "cpu.max").write_text("400000 100000\n")
    assert ocr.cpus(tmp_path) == min(4, ocr.cpus(tmp_path / "none"))
    (tmp_path / "cpu.max").write_text("150000 100000\n")
    assert ocr.cpus(tmp_path) == min(2, ocr.cpus(tmp_path / "none"))  # rounded up
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert ocr.cpus(tmp_path) == ocr.cpus(tmp_path / "none")
    (tmp_path / "cpu.max").unlink()
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("100000\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert ocr.cpus(tmp_path) == 1


def test_the_script_probe_reads_once_with_both_models_and_the_characters_decide():
    pics = [picture(1.0 + 2 * k, 2.0 + 2 * k) for k in range(100)]
    made = []

    def factory(model):
        made.append(model)
        return FakeEngine({k: [("我們今天晚上去哪裡吃飯", 70.0)] for k in range(40)})

    script, found = ocr.probe_script(pics, engine_factory=factory)
    assert script == "Hant" and made == ["chi_sim+chi_tra"] and found["by_characters"] == "Hant"
    assert found["pictures"] == 40
    script, found = ocr.probe_script(pics, engine_factory=lambda m: FakeEngine({0: [("好", 90.0)]}))
    assert script == "Hans" and found["by_characters"] is None  # too few characters to tell: Simplified


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
    # A file that is not the model it should be is not used, nor left behind.
    with pytest.raises(LookupError, match="chi_sim fetched .* is not the one expected"):
        ocr.tessdata_for("chi_sim+eng", own, cache)
    assert not list(cache.glob("chi_sim.*"))
    monkeypatch.setitem(ocr.TESSDATA_SHA256, "chi_sim", hashlib.sha256(b"a model fetched").hexdigest())
    assert ocr.tessdata_for("chi_sim+eng", own, cache) == str(cache)
    assert fetched == ["https://github.com/tesseract-ocr/tessdata_best/raw/4.1.0/chi_sim.traineddata"] * 2
    assert ocr.tessdata_url("deu") == "https://github.com/tesseract-ocr/tessdata_fast/raw/4.1.0/deu.traineddata"
    assert (cache / "eng.traineddata").read_bytes() == b"eng model"  # its own, copied beside the fetched one
    assert (cache / "chi_sim.traineddata").read_bytes() == b"a model fetched"
    # Every model a language is read with has its checksum.
    assert set(ocr.MODELS.values()) | set(ocr.CHINESE.values()) <= set(ocr.TESSDATA_SHA256)


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
