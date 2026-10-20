"""The instruction-model translator against a fake chat endpoint."""

import re

from subtitler import vtt
from subtitler.langs import Lang
from subtitler.translate import translate_cues
from subtitler.translate.llm import LlmTranslator, covers_all, parse_spans


class FakeClient:
    """Answers each request with what `answer(numbers, attempt)` returns."""

    def __init__(self, answer):
        self.answer = answer
        self.requests = []

    def chat(self, messages, max_tokens, temperature=0.0, extra=None):
        user = messages[-1]["content"].split("for context only")[0]
        numbers = [int(m) for m in re.findall(r"^(\d+)(?: \[max \d+\])?\|", user, re.M)]
        self.requests.append(numbers)
        text = self.answer(numbers, len(self.requests))
        return text, {"prompt_n": 100, "predicted_n": 20, "prompt_ms": 10.0, "predicted_ms": 100.0}, {}, 0.1


def cues(*texts, dur=2.0):
    return [vtt.Cue(i * 3.0, i * 3.0 + dur, t) for i, t in enumerate(texts)]


EN, DE = Lang.parse("en"), Lang.parse("de")


def test_parse_spans_and_coverage():
    got = parse_spans("1| Eins\n2-3| Zwei und drei\n4| Vier", [1, 2, 3, 4])
    assert got == {1: (1, "Eins"), 2: (3, "Zwei und drei"), 4: (4, "Vier")}
    assert covers_all(got, [1, 2, 3, 4])
    assert not covers_all({1: (2, "x"), 2: (2, "y")}, [1, 2])  # line 2 twice
    assert not covers_all({1: (1, "x")}, [1, 2])  # line 2 missing


def test_a_span_that_fits_becomes_one_cue_over_both_times():
    def answer(numbers, n):
        return "1| Hallo.\n2-3| Im Frühling hat der Gemeinderat einen Plan beschlossen."

    tr = LlmTranslator(FakeClient(answer), spans=True)
    out, usage, failures = translate_cues(cues("Hello.", "In the spring, the council", "approved a plan."), EN, DE, tr)
    assert [c.text.replace("\n", " ") for c in out] == ["Hallo.", "Im Frühling hat der Gemeinderat einen Plan beschlossen."]
    assert (out[1].start, out[1].end) == (3.0, 8.0)
    assert usage.merged == 1 and failures == 0


def test_a_span_too_long_for_one_cue_is_split_back():
    long = ("Im letzten Frühling hat der Gemeinderat einen ziemlich langweiligen, einstimmigen Plan "
            "zur Reparatur der alten Brücke über den Fluss beschlossen.")

    def answer(numbers, n):
        return f"1-2| {long}"

    tr = LlmTranslator(FakeClient(answer), spans=True)
    out, _, _ = translate_cues(cues("Last spring, the council approved a rather dull,", "unanimous plan to fix the bridge."),
                               EN, DE, tr)
    assert len(out) == 2 and (out[0].start, out[1].start) == (0.0, 3.0)
    assert " ".join(c.text.replace("\n", " ") for c in out) == long


def test_without_spans_a_range_answer_is_asked_again():
    def answer(numbers, n):
        if n == 1:
            return "1-2| Beides zusammen."
        return "1| Eins.\n2| Zwei."

    fake = FakeClient(answer)
    tr = LlmTranslator(fake, spans=False)
    out, usage, _ = translate_cues(cues("One.", "Two."), EN, DE, tr)
    assert [c.text for c in out] == ["Eins.", "Zwei."]
    assert usage.retries == 1 and len(fake.requests) == 2


def test_length_budget_marks_short_cues_in_the_prompt():
    seen = []

    def answer(numbers, n):
        return "1| Kurz."

    fake = FakeClient(answer)
    orig = fake.chat

    def chat(messages, **kw):
        seen.append(messages[-1]["content"])
        return orig(messages, **kw)

    fake.chat = chat
    tr = LlmTranslator(fake, length_budget=True)
    translate_cues(cues("I really had no idea which of the two trains you would take today.", dur=1.5), EN, DE, tr)
    assert "1 [max 25]|" in seen[0]
