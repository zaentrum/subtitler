"""WebVTT, wrapping, SDH clean-up, cue building, and the translators' text handling."""

from subtitler import cues, langs, sdh, vtt, wrap
from subtitler.translate import finish, prepare
from subtitler.translate.llm import parse_answer
from subtitler.translate.madlad import split_by_share

SAMPLE = """﻿WEBVTT - a header

NOTE a note

STYLE
::cue { color: yellow }

1
00:01.000 --> 00:02.500 align:center
<i>Hello there.</i>

01:02:03.004 --> 01:02:05.000
-Yes.
-No &amp; never.
"""


def test_vtt_parse_and_round_trip():
    cs = vtt.parse(SAMPLE)
    assert [(c.start, c.end) for c in cs] == [(1.0, 2.5), (3723.004, 3725.0)]
    assert cs[0].settings == "align:center"
    assert vtt.plain(cs[1].text) == "-Yes.\n-No & never."
    again = vtt.parse(vtt.dumps(cs, notes=["made --> by test"]))
    assert [(c.start, c.end, c.text) for c in again] == [(c.start, c.end, c.text) for c in cs]


def test_window_by_midpoint():
    cs = [vtt.Cue(0, 2, "a"), vtt.Cue(9, 12, "b"), vtt.Cue(19, 22, "c")]
    assert [c.text for c in vtt.window(cs, 10, 20)] == ["b"]


def test_language_tags():
    assert langs.iso1("ger") == langs.iso1("deu") == langs.iso1("de") == "de"
    assert langs.iso1("chi") == "zh" and langs.iso1("und") is None
    assert langs.Lang.parse("zh-Hans").tag == "zh-Hans"
    assert langs.Lang.parse("zh-TW").script == "Hant"
    assert langs.made_label(langs.Lang.parse("de"), "translated") == "German · AI"
    assert langs.made_label(langs.Lang.parse("zh-Hans"), "converted") == "Chinese (Simplified) · Converted"


def test_wrap_spaced_two_balanced_lines():
    text = "I had no idea which of the two trains you took, so I waited."
    lines = wrap.rewrap(text, "en").split("\n")
    # The comma break would leave 47 characters on the first line: balance wins.
    assert len(lines) == 2 and all(len(l) <= 42 for l in lines)
    lines = wrap.rewrap("When we get there, nobody will be waiting for us at all.", "en").split("\n")
    assert lines == ["When we get there,", "nobody will be waiting for us at all."]
    assert wrap.rewrap("Short line.", "en") == "Short line."


def test_wrap_keeps_dialogue_lines():
    assert wrap.rewrap("-Yes.\n-No, never.", "en") == "-Yes.\n-No, never."


def test_zh_style_and_wrap():
    assert wrap.zh_style("你好，我们走吧。") == "你好 我们走吧"
    assert wrap.zh_style("价格是3.5元，一共1,000元。") == "价格是3.5元 一共1,000元"
    lines = wrap.rewrap("这个问题很难回答，因为时间已经太晚了，我们明天再说吧。", "zh").split("\n")
    assert len(lines) == 2 and all(wrap.width(l) <= 16 for l in lines)


def test_sdh_clean():
    assert sdh.clean_text("-[shouting, grunting]\n-That's a story.") == "That's a story."
    assert sdh.clean_text("JOHN: Come here.") == "Come here."
    assert sdh.clean_text("[door slams]") == ""
    assert sdh.clean_text("♪ la la la ♪") == ""
    assert sdh.clean_text("-MAN: Hi.\n-WOMAN: Hello.") == "-Hi.\n-Hello."


def _words(spec):
    """'word@start' items, each word 0.3 s long."""
    out = []
    for item in spec.split():
        w, t = item.rsplit("@", 1)
        out.append(cues.Word(float(t), float(t) + 0.3, " " + w))
    return out


def test_cues_sentences_pauses_and_lead_out():
    ws = _words("Hello@0.0 there.@0.4 How@0.9 are@1.3 you?@1.7 Fine.@5.0")
    cs = cues.build(ws, "en")
    assert [c.text for c in cs] == ["Hello there.\nHow are you?", "Fine."]
    assert cs[0].end <= cs[1].start - 0.08 and cs[0].end > 2.0


def test_cues_long_sentence_is_cut_at_a_comma():
    words = ("Every winter the old ferry carried far too many people, which nobody in the harbour "
             "office noticed.").split()
    ws = [cues.Word(i * 0.25, i * 0.25 + 0.2, " " + w) for i, w in enumerate(words)]
    cs = cues.build(ws, "en")
    assert len(cs) == 2
    assert all(len(l) <= 42 for c in cs for l in c.text.split("\n"))
    assert cs[0].text.replace("\n", " ").endswith("people,")


def test_cues_very_long_sentence_fits_the_limits():
    words = ("In the spring, the village council approved a remarkably dull and entirely unopposed plan "
             "to repair the bridge by summer, which nobody noticed at all.").split()
    ws = [cues.Word(i * 0.25, i * 0.25 + 0.2, " " + w) for i, w in enumerate(words)]
    cs = cues.build(ws, "en")
    assert len(cs) >= 2
    assert all(len(l) <= 42 for c in cs for l in c.text.split("\n"))
    assert all(len(c.text.split("\n")) <= 2 and c.duration <= 7.5 for c in cs)


def test_cues_orphan_word_is_moved_to_its_phrase():
    ws = [cues.Word(0.0, 0.4, " Thank"), cues.Word(20.0, 20.3, " you.")]
    cs = cues.build(ws, "en")
    assert len(cs) == 1 and cs[0].text == "Thank you." and cs[0].start > 19.0


def test_cues_drop_loops_and_hallucinations():
    ws = _words("Thanks@0 for@0.3 watching!@0.6") + _words("Go.@3 Go.@5 Go.@7 Go.@9")
    texts = [c.text for c in cues.build(ws, "en")]
    assert "Thanks for watching!" not in texts
    assert texts.count("Go.") == 2


def test_abbreviations_end_no_sentence_and_fillers_go():
    ws = _words("From@0 Mr.@0.3 Brown.@0.6") + _words("Uh,@3.0 who@4.2 is@4.5 he?@4.8 Mm@6.5 -hmm.@6.7")
    texts = [c.text.replace("\n", " ") for c in cues.build(ws, "en")]
    assert texts == ["From Mr. Brown.", "Who is he?"]


def test_collapse_repeats_inside_a_cue():
    assert cues.collapse_repeats("现在不试试试试试试试试试试试", "zh") == "现在不试试"
    assert cues.collapse_repeats("no no no no no no no no stop", "en") == "no no stop"
    assert cues.collapse_repeats("no, no, no, stop", "en") == "no, no, no, stop"


def test_prepare_and_finish_markers():
    cs = [vtt.Cue(0, 1, "-Yes.\n-No."), vtt.Cue(1, 2, "<i>Far away\nfrom here.</i>"), vtt.Cue(2, 3, "[music]")]
    p = prepare(cs, is_sdh=True)
    assert p.texts == ["-Yes. <br> -No.", "Far away from here."] and p.italic == [False, True]
    assert finish("-Ja. <br> -Nein.", langs.Lang.parse("de"), False) == "-Ja.\n-Nein."
    assert finish("遠方。", langs.Lang.parse("zh-Hans"), True) == "<i>远方</i>"


def test_parse_answer():
    text = "Here you go:\n3| Eins\n4｜Zwei\n5: Drei\n3| doppelt"
    assert parse_answer(text, [3, 4, 5]) == {3: "Eins", 4: "Zwei", 5: "Drei"}
    assert parse_answer("7 [max 20]| Kurz.", [7]) == {7: "Kurz."}


def test_length_budget_only_where_the_cue_is_short():
    from subtitler.translate.llm import budget

    en, de, zh = langs.Lang.parse("en"), langs.Lang.parse("de"), langs.Lang.parse("zh-Hans")
    long_text = "I had no idea which of the two trains you took, so I waited at both of the stations."
    assert budget(long_text, 2.0, en, de) == 34  # 2 s at 17 characters a second
    assert budget(long_text, 6.0, en, de) == 84  # time enough, but more than two lines
    assert budget("I had no idea which train you took.", 6.0, en, de) is None
    assert budget("Yes.", 0.5, en, de) is None
    assert budget(long_text, 2.0, en, zh) == 18  # 9 a second, at least 6


def test_split_by_share():
    assert split_by_share("eins zwei drei vier", [1, 1], False) == ["eins zwei", "drei vier"]
    pieces = split_by_share("我们走吧，明天再说", [4, 4], True)
    assert "".join(pieces) == "我们走吧，明天再说" and pieces[0].endswith("，")
    # The source's 1:2 share would leave 98 characters for the second cue: the limit wins.
    de = ("Im letzten Frühling hat der Gemeinderat einen ziemlich langweiligen, einstimmigen Plan "
          "zur Reparatur der alten Brücke über den Fluss beschlossen.")
    a, b = split_by_share(de, [26, 51], False, max_len=84)
    assert len(a) <= 84 and len(b) <= 84
    assert f"{a} {b}" == de
