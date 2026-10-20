"""The decision per language, in ADR-0014's order."""

from subtitler.langs import parse_list
from subtitler.package import load
from subtitler.plan import TRANSCRIPT, plan

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


def test_commentary_is_no_subtitle(make_package):
    root = make_package(subs=[{"language": "eng", "purpose": "commentary"}])
    assert choices(root)["en"].action == "transcribe"


def test_image_subtitle_counts_as_present_but_is_no_source(make_package):
    root = make_package(subs=[{"language": "ger", "format": "pgs"}, {"language": "eng", "format": "pgs"}])
    c = choices(root)
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
