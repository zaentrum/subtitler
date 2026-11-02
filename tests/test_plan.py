"""The decision per language, in ADR-0014's order."""

from subtitler.langs import parse_list
from subtitler.package import load
from subtitler.plan import COPY, TRANSCRIPT, copies, plan

LANGS = parse_list("en,de,zh-Hans")


def choices(root, langs=LANGS, **kw):
    return {p.language.tag: p.choice for p in plan(load(root), langs, **kw)}


def test_everything_present_keeps_all(make_package):
    root = make_package(subs=[{"language": "eng"}, {"language": "ger"}, {"language": "chi", "title": "简体"}])
    c = choices(root)
    assert {k: v.action for k, v in c.items()} == {"en": "keep", "de": "keep", "zh-Hans": "keep"}


def test_traditional_only_is_converted_not_translated(make_package):
    root = make_package(subs=[{"language": "eng"}, {"language": "chi", "kind": "Hant", "title": ""}])
    c = choices(root)
    assert c["zh-Hans"].action == "convert"
    assert c["zh-Hans"].detail["config"] == "t2s"
    assert c["de"].action == "translate" and c["de"].source_language == "en"


def test_script_comes_from_the_text_not_the_title(make_package):
    # Labelled "Simplified" but written in Traditional characters.
    root = make_package(subs=[{"language": "chi", "kind": "Hant", "title": "Simplified"}])
    pkg = load(root)
    assert pkg.subtitles[0].script == "Hant" and pkg.subtitles[0].script_from == "text"
    assert choices(root)["zh-Hans"].action == "convert"


def test_hong_kong_traditional_converts_with_hk_tables(make_package):
    root = make_package(subs=[{"language": "chi", "kind": "Hant", "title": "Hong Kong"}])
    assert choices(root)["zh-Hans"].detail["config"] == "hk2s"


def test_cantonese_is_another_language_not_another_script(make_package):
    root = make_package(subs=[{"language": "eng"}, {"language": "chi", "kind": "yue", "title": "Traditional"}])
    pkg = load(root)
    assert pkg.subtitles[1].language == "yue"
    c = choices(root)
    assert c["zh-Hans"].action == "translate" and c["zh-Hans"].source_language == "en"


def test_cantonese_by_title(make_package):
    root = make_package(subs=[{"language": "chi", "kind": "Hant", "title": "Cantonese (Traditional)"}])
    assert load(root).subtitles[0].language == "yue"


def test_original_language_first_then_english(make_package):
    root = make_package(audio=[("ita", True)], subs=[{"language": "eng"}, {"language": "ita"}])
    c = choices(root)
    assert c["de"].action == "translate" and c["de"].source_language == "it"
    assert c["zh-Hans"].source_language == "it"


def test_source_order_english_takes_a_complete_english_track_first(make_package):
    root = make_package(audio=[("ita", True)], subs=[{"language": "ita", "cues": 40}, {"language": "eng", "cues": 40}])
    assert choices(root, source_order="english")["de"].source_language == "en"
    assert choices(root)["de"].source_language == "it"  # ADR-0014's order stays the default


def test_an_english_track_that_leaves_a_scene_out_is_not_complete(make_package):
    # The last ten Italian cues have no English: a scene spoken in English.
    root = make_package(audio=[("ita", True)], subs=[{"language": "ita", "cues": 40}, {"language": "eng", "cues": 30}])
    assert choices(root, source_order="english")["de"].source_language == "it"


def test_english_before_other_languages(make_package):
    root = make_package(audio=[("kor", True)], subs=[{"language": "fre"}, {"language": "eng"}])
    c = choices(root)
    assert c["de"].source_language == "en"
    order = [x.source_language for x in plan(load(root), LANGS)[1].candidates]
    assert order == ["en", "fr", "ko"]  # then the transcript of the Korean audio


def test_original_language_override_for_a_dub(make_package):
    # German dub as the default audio, English original second.
    root = make_package(audio=[("deu", True), ("eng", False)],
                        subs=[{"language": "deu"}, {"language": "eng"}])
    assert choices(root, parse_list("zh-Hans"))["zh-Hans"].source_language == "de"
    assert choices(root, parse_list("zh-Hans"), original_language="en")["zh-Hans"].source_language == "en"


def test_no_subtitle_at_all_transcribes(make_package):
    root = make_package(audio=[("eng", True)], subs=[])
    c = choices(root)
    assert c["en"].action == "transcribe" and c["en"].source == "audio:a0"
    assert c["de"].action == "translate" and c["de"].source == TRANSCRIPT
    assert c["zh-Hans"].source == TRANSCRIPT


def test_chinese_audio_transcribes_into_simplified(make_package):
    root = make_package(audio=[("chi", True)], subs=[])
    c = choices(root)
    assert c["zh-Hans"].action == "transcribe" and c["zh-Hans"].detail == {"script": "Hans"}
    assert c["en"].source == TRANSCRIPT


def test_unknown_audio_language_translates_the_transcript(make_package):
    root = make_package(audio=[("und", True)], subs=[])
    c = choices(root)
    assert all(v.source == TRANSCRIPT and v.source_language is None for v in c.values())


def test_forced_only_is_no_subtitle(make_package):
    root = make_package(subs=[{"language": "eng", "forced": True, "purpose": "forced"}])
    c = choices(root)
    assert c["en"].action == "transcribe"
    assert c["de"].source == TRANSCRIPT


def test_unflagged_signs_track_is_partial(make_package):
    # 6 cues in 10 minutes: signs and songs, not the dialogue.
    root = make_package(audio=[("deu", True)], subs=[{"language": "deu", "cues": 6}, {"language": "eng"}])
    pkg = load(root)
    assert pkg.subtitles[0].partial
    c = choices(root)
    assert c["de"].action == "translate" and c["de"].source_language == "en"


def test_a_sparse_but_only_track_still_counts(make_package):
    # 15 cues in 10 minutes and nothing fuller beside it: a quiet film, not signs.
    root = make_package(subs=[{"language": "eng", "cues": 15}])
    assert not load(root).subtitles[0].partial
    assert choices(root)["en"].action == "keep"


def test_commentary_is_no_subtitle(make_package):
    root = make_package(subs=[{"language": "eng", "purpose": "commentary"}])
    assert choices(root)["en"].action == "transcribe"


def test_without_ocr_an_image_subtitle_counts_as_present_but_is_no_source(make_package):
    root = make_package(subs=[{"language": "ger", "format": "pgs"}, {"language": "eng", "format": "pgs"}])
    c = choices(root, ocr=False)
    assert c["de"].action == "keep" and c["en"].action == "keep"
    # Nothing readable to translate from: Chinese comes from the audio.
    assert c["zh-Hans"].source == TRANSCRIPT


def test_sdh_counts_and_dialogue_is_preferred_as_source(make_package):
    root = make_package(subs=[{"language": "eng", "purpose": "sdh", "title": "SDH"}, {"language": "eng"}])
    c = choices(root)
    assert c["en"].action == "keep"
    assert c["de"].source == "sub1"  # the dialogue track, not the SDH one
    sdh_only = make_package(subs=[{"language": "eng", "purpose": "sdh"}])
    d = choices(sdh_only)["de"]
    assert d.source == "sub0" and d.detail["sdh"] is True


def test_missing_file_does_not_count(make_package):
    root = make_package(subs=[{"language": "ger", "exists": False}, {"language": "eng"}])
    c = choices(root)
    assert c["de"].action == "translate" and c["de"].source == "sub1"


def test_bilingual_track_is_kept_but_no_source(make_package):
    root = make_package(audio=[("fre", True)],
                        subs=[{"language": "chi", "title": "Simplified Chinese&English"}, {"language": "fre"}])
    c = choices(root)
    assert c["zh-Hans"].action == "keep"
    assert c["de"].source_language == "fr"


def test_a_report_moves_to_the_next_candidate(make_package):
    root = make_package(audio=[("ita", True)], subs=[{"language": "ita"}, {"language": "eng"}])
    first = choices(root)["de"]
    assert first.source == "sub0"
    assert choices(root, exclude=["sub0"])["de"].source == "sub1"
    last = choices(root, exclude=["sub0", "sub1"])["de"]
    assert last.source == TRANSCRIPT


def test_target_without_script_takes_any_chinese(make_package):
    root = make_package(subs=[{"language": "chi", "kind": "Hant"}])
    assert choices(root, parse_list("zh"))["zh"].action == "keep"


def test_traditional_target_converts_from_simplified(make_package):
    root = make_package(subs=[{"language": "chi", "kind": "Hans"}])
    c = choices(root, parse_list("zh-Hant"))["zh-Hant"]
    assert c.action == "convert" and c.detail["config"] == "s2t"


def test_an_image_subtitle_gets_a_text_copy_that_is_present_and_a_source(make_package):
    root = make_package(subs=[{"language": "ger", "format": "pgs"}, {"language": "eng", "format": "pgs"}])
    c = choices(root)
    assert (c["de"].action, c["de"].source) == ("recognize", COPY + "sub0")
    assert (c["en"].action, c["en"].source) == ("recognize", COPY + "sub1")
    # Chinese is translated from the English copy (English before German), the audio last.
    order = [x.source for x in plan(load(root), LANGS)[2].candidates]
    assert order == [COPY + "sub1", COPY + "sub0", TRANSCRIPT]
    assert plan(load(root), LANGS)[2].candidates[0].action == "translate"


def test_a_text_subtitle_of_the_language_and_kind_needs_no_copy(make_package):
    root = make_package(subs=[{"language": "eng"}, {"language": "eng", "format": "pgs"},
                              {"language": "eng", "format": "pgs", "purpose": "sdh", "title": "SDH"},
                              {"language": "eng", "format": "pgs", "forced": True, "purpose": "forced", "cues": 6}])
    made = {(x.sub.id, x.kind) for x in copies(load(root), LANGS)}
    # The full one has a text track beside it; SDH and forced are kinds of their own.
    assert made == {("sub2", "sdh"), ("sub3", "forced")}
    assert choices(root)["en"].action == "keep"


def test_one_copy_a_language_and_kind_the_fullest_and_no_commentary(make_package):
    root = make_package(subs=[{"language": "eng", "format": "pgs", "cues": 30}, {"language": "eng", "format": "pgs"},
                              {"language": "eng", "format": "pgs", "purpose": "commentary"},
                              {"language": "fre", "format": "pgs"}])
    made = copies(load(root), LANGS)
    assert [(x.sub.id, x.kind) for x in made] == [("sub1", "full")]  # French is no language of the instance


def test_a_sparse_image_track_is_copied_as_signs_and_is_no_source(make_package):
    root = make_package(audio=[("jpn", True)], subs=[{"language": "eng", "format": "pgs", "cues": 5},
                                                     {"language": "eng", "format": "pgs", "cues": 40}])
    made = {x.sub.id: x.kind for x in copies(load(root), LANGS)}
    assert made == {"sub0": "signs", "sub1": "full"}
    assert choices(root)["de"].source == COPY + "sub1"


def test_a_vobsub_track_is_copied_too(make_package):
    root = make_package(subs=[{"language": "ger", "format": "vobsub"}])
    assert [(x.sub.format, x.lang.tag) for x in copies(load(root), LANGS)] == [("vobsub", "de")]
    assert choices(root)["de"].action == "recognize"


def test_a_traditional_image_track_is_copied_then_converted(make_package):
    root = make_package(subs=[{"language": "chi", "format": "pgs", "title": "繁體中文"}])
    made = copies(load(root), LANGS)
    assert [x.lang.tag for x in made] == ["zh-Hant"]
    c = choices(root)["zh-Hans"]
    assert (c.action, c.source, c.detail["config"]) == ("convert", COPY + "sub0", "t2s")


def test_a_chinese_image_tracks_title_may_name_the_wrong_script(make_package):
    # Reading tells its script: a copy titled Traditional is converted for Simplified where it reads
    # Traditional, and is the Simplified one where it reads Simplified; the other way round for Traditional.
    root = make_package(subs=[{"language": "chi", "format": "pgs", "title": "繁體中文"}])
    plans = {p.language.tag: p for p in plan(load(root), parse_list("zh-Hans,zh-Hant"))}
    assert [(x.action, x.source) for x in plans["zh-Hans"].candidates[:2]] == [("convert", COPY + "sub0"),
                                                                             ("recognize", COPY + "sub0")]
    hant = plans["zh-Hant"].candidates
    assert [(x.action, x.source) for x in hant[:2]] == [("recognize", COPY + "sub0"), ("convert", COPY + "sub0")]
    assert hant[1].detail["config"] == "s2t"


def test_a_chinese_image_track_of_no_script_said_is_read_then_kept_or_converted(make_package):
    root = make_package(subs=[{"language": "chi", "format": "pgs", "title": ""}])
    cands = plan(load(root), LANGS)[2].candidates
    assert [(x.action, x.source) for x in cands[:2]] == [("recognize", COPY + "sub0"), ("convert", COPY + "sub0")]


def test_a_report_on_a_copy_moves_to_the_next_candidate(make_package):
    root = make_package(audio=[("eng", True)], subs=[{"language": "eng", "format": "pgs"}])
    assert choices(root)["en"].action == "recognize"
    c = choices(root, exclude=[COPY + "sub0"])
    assert c["en"].action == "transcribe" and c["de"].source == TRANSCRIPT


def test_a_text_subtitle_goes_before_a_copy_of_the_same_rank(make_package):
    root = make_package(audio=[("kor", True)], subs=[{"language": "eng", "format": "pgs", "purpose": "sdh"},
                                                     {"language": "eng"}, {"language": "ger", "format": "pgs"}])
    assert [x.sub.id for x in copies(load(root), LANGS)] == ["sub0", "sub2"]
    zh = [x.source for x in plan(load(root), LANGS)[2].candidates]
    assert zh == ["sub1", COPY + "sub0", COPY + "sub2", TRANSCRIPT]  # English text, its SDH copy, the German copy


def test_only_the_image_tracks_of_the_languages_are_read_to_count_them(make_package):
    root = make_package(subs=[{"language": "eng", "format": "pgs", "cues": 12}, {"language": "fre", "format": "pgs"}])
    read = []

    def reader(path):
        read.append(path.name)
        return path.read_bytes()

    pkg = load(root, {"en", "de", "zh"}, reader)
    assert read == ["0.sup"] and [s.cues for s in pkg.subtitles] == [12, None]
    assert [s.cues for s in load(root, count=False).subtitles] == [None, None]


def test_signs_are_not_copied_where_the_language_has_a_forced_subtitle(make_package):
    # A forced text track of the language, or a forced image track of it (copied), covers its signs track.
    root = make_package(subs=[{"language": "eng", "format": "pgs", "cues": 400},
                              {"language": "eng", "format": "pgs", "cues": 6},
                              {"language": "eng", "forced": True, "purpose": "forced", "cues": 5}],
                        duration_ms=6_000_000)
    assert [(c.sub.id, c.kind) for c in copies(load(root), LANGS)] == [("sub0", "full")]
    root = make_package(subs=[{"language": "ger", "format": "pgs", "cues": 400},
                              {"language": "ger", "format": "pgs", "cues": 6},
                              {"language": "ger", "format": "pgs", "forced": True, "purpose": "forced", "cues": 4}],
                        duration_ms=6_000_000)
    assert [(c.sub.id, c.kind) for c in copies(load(root), LANGS)] == [("sub0", "full"), ("sub2", "forced")]
