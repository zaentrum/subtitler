"""Subtitle translation by an instruction model behind a llama.cpp server.

The server speaks the OpenAI chat API (llama-server); the same GGUF file
runs on Metal here and on CUDA on the GPU node, so what is measured here is
the model the GPU will run.

A film is translated in batches of numbered cues. Each request carries:
- the previous batch and its translation as an earlier turn of the chat, so
  names, terms and forms of address (du/Sie) stay as they were;
- the batch to translate, one numbered line per cue;
- the next few cues as context the model must not translate.
The answer must carry every number once. A batch whose answer does not is
asked again once, then split in halves, down to single cues.

Cue timing never changes: translation N takes cue N's times.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass, field

from ..langs import Lang

_LINE = re.compile(r"^\s*(\d+)(?:\s*[-–]\s*(\d+))?\s*(?:\[[^\]]*\]\s*)?[|｜:：.)]\s?(.*)$")
MAX_SPAN = 3  # lines one answer may cover

_TARGET_RULES = {
    "de": ("Use 'du' between friends, family and children and 'Sie' with strangers, superiors and in "
           "formal situations, as the scene suggests, and keep each relationship's form once chosen."),
    "zh-Hans": ("Write Simplified Chinese as used in mainland China. Use full-width punctuation. Do not end "
                "a line with a full stop (。); use a space instead of a comma inside a line."),
    "zh-Hant": "Write Traditional Chinese. Use full-width punctuation. Do not end a line with a full stop.",
    "en": "Write natural, idiomatic English.",
}


# Characters a language needs for the same content, English = 1 (subtitle
# text; rough, for deciding which lines need a length budget).
_CHAR_RATIO = {"en": 1.0, "de": 1.15, "fr": 1.1, "it": 1.05, "es": 1.05, "pt": 1.05, "nl": 1.1,
               "zh": 0.33, "yue": 0.35, "ja": 0.45, "ko": 0.5, "ru": 1.05}

BUDGET_RULE = ("- A line marked [max N] has little time on screen: translate it in at most N characters. "
               "Condense it as a subtitler would, keep what matters. Do not repeat the marker.\n")


def budget(text: str, duration: float, src: Lang, tgt: Lang) -> int | None:
    """The characters a cue's translation may have at the target's reading speed, if it is likely to need fewer."""
    from ..wrap import limits

    lim = limits(tgt.code)
    allowed = max(6 if tgt.code in ("zh", "yue", "ja") else 14, int(duration * lim.cps))
    allowed = min(allowed, lim.chars_per_cue)
    expected = len(text.replace(" <br> ", " ")) * _CHAR_RATIO.get(tgt.code, 1.0) / _CHAR_RATIO.get(src.code, 1.0)
    return allowed if expected > allowed * 1.1 else None


SPAN_RULE = ("- Keep each line's content in that line. When a sentence runs over several lines and reads "
             "better translated as a whole, translate those lines together as one answer numbered with the "
             "range, like \"12-13| translation\" (at most three lines). Never move words from one line into "
             "another line's answer.\n")
NO_SPAN_RULE = ("- Keep each line's content in that line. Do not merge lines or move words between them, even "
                "when a sentence runs over several lines.\n")


def system_prompt(src: Lang, tgt: Lang, with_budget: bool = False, spans: bool = False) -> str:
    rules = _TARGET_RULES.get(tgt.tag) or _TARGET_RULES.get(tgt.code, "")
    return (
        f"You are a professional subtitle translator. You translate the subtitles of a film or TV episode "
        f"from {src.name} into {tgt.name}.\n"
        "Rules:\n"
        "- Translate every numbered line. Answer with the same numbers, one line each, as \"N| translation\". "
        "Write nothing else: no notes, no explanations, no source text.\n"
        + (SPAN_RULE if spans else NO_SPAN_RULE) +
        "- Write natural spoken language as a professional subtitler would: concise and quick to read. "
        "Condense slightly rather than add words.\n"
        "- Keep names as they are. Keep the marker <br> where it appears: it separates two speakers.\n"
        + (BUDGET_RULE if with_budget else "") +
        f"- {rules}\n"
    )


def user_prompt(batch: list[tuple[int, str]], after: list[tuple[int, str]],
                budgets: dict[int, int] | None = None) -> str:
    budgets = budgets or {}
    lines = ["Translate these lines:"]
    lines += [f"{n} [max {budgets[n]}]| {t}" if n in budgets else f"{n}| {t}" for n, t in batch]
    if after:
        lines.append("")
        lines.append("The lines that follow, for context only (do not translate them):")
        lines += [f"{n}| {t}" for n, t in after]
    return "\n".join(lines)


def parse_spans(text: str, wanted: list[int]) -> dict[int, tuple[int, str]]:
    """{first line: (last line, translation)}; a range covers consecutive lines."""
    want = set(wanted)
    out: dict[int, tuple[int, str]] = {}
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if a in want and b in want and a <= b <= a + MAX_SPAN - 1 and a not in out:
            out[a] = (b, m.group(3).strip())
    return out


def covers_all(spans: dict[int, tuple[int, str]], wanted: list[int]) -> bool:
    """Every wanted line in exactly one answer, every answer non-empty."""
    seen: list[int] = []
    for a, (b, t) in spans.items():
        if not t:
            return False
        seen += range(a, b + 1)
    return sorted(seen) == sorted(wanted)


def parse_answer(text: str, wanted: list[int]) -> dict[int, str]:
    """One translation per line number (answers covering a range are left out)."""
    return {a: t for a, (b, t) in parse_spans(text, wanted).items() if a == b}


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n < 3:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a) ** 0.5
    vb = sum((y - mb) ** 2 for y in b) ** 0.5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) if va and vb else 0.0


def misaligned(sources: list[str], answers: list[str], unspaced: bool = False) -> bool:
    """Whether a batch's answers look shifted against its lines: every number answered, but the
    translations' lengths follow the lines one or two places away far better than their own lines
    (a model that dropped a line and moved the rest up keeps the count and loses the alignment)."""
    n = len(sources)
    if n < 10 or len(answers) != n:
        return False
    a = [float(len(t.replace(" <br> ", " "))) for t in sources]
    b = [float(len(t.replace(" ", "")) if unspaced else len(t)) for t in answers]
    own = _pearson(a, b)
    moved = max(_pearson(a[:n - k], b[k:]) if k > 0 else _pearson(a[-k:], b[:n + k]) for k in (-2, -1, 1, 2))
    return moved > 0.6 and moved > own + 0.25


@dataclass
class Usage:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_ms: float = 0.0
    predicted_ms: float = 0.0
    wall_s: float = 0.0
    retries: int = 0
    fallbacks: int = 0
    budgeted: int = 0  # lines given a length budget
    merged: int = 0  # answers that covered several lines
    calls: list = field(default_factory=list)
    rejected: list = field(default_factory=list)  # answers that did not cover the batch

    def add(self, timings: dict, usage: dict, wall: float) -> None:
        self.requests += 1
        pn = timings.get("prompt_n", usage.get("prompt_tokens", 0))
        cn = timings.get("predicted_n", usage.get("completion_tokens", 0))
        self.prompt_tokens += pn
        self.completion_tokens += cn
        self.prompt_ms += timings.get("prompt_ms", 0.0)
        self.predicted_ms += timings.get("predicted_ms", 0.0)
        self.wall_s += wall
        self.calls.append({"prompt_n": pn, "predicted_n": cn, "prompt_ms": timings.get("prompt_ms"),
                           "predicted_ms": timings.get("predicted_ms"), "wall_s": round(wall, 3)})

    def as_dict(self) -> dict:
        gen = self.completion_tokens / (self.predicted_ms / 1000) if self.predicted_ms else 0.0
        pp = self.prompt_tokens / (self.prompt_ms / 1000) if self.prompt_ms else 0.0
        return {"requests": self.requests, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "gen_tok_s": round(gen, 1),
                "prompt_tok_s": round(pp, 1), "wall_s": round(self.wall_s, 2),
                "retries": self.retries, "fallbacks": self.fallbacks, "budgeted": self.budgeted,
                "merged": self.merged}


class Client:
    """A llama-server's OpenAI-compatible chat endpoint."""

    def __init__(self, url: str = "http://127.0.0.1:8091", model: str = "local", timeout: float = 600):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[dict], max_tokens: int, temperature: float = 0.0,
             extra: dict | None = None) -> tuple[str, dict, dict, float]:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "cache_prompt": True, **(extra or {})}
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        wall = time.perf_counter() - t0
        text = data["choices"][0]["message"]["content"] or ""
        return text, data.get("timings", {}), data.get("usage", {}), wall

    def model_id(self) -> str:
        try:
            with urllib.request.urlopen(self.url + "/v1/models", timeout=10) as r:
                data = json.loads(r.read())
            return data["data"][0]["id"]
        except Exception:
            return self.model


class LlmTranslator:
    """translate() returns, per text, (translation, span): span 1 for a line of its own, n for the
    first of n lines translated together, 0 for the lines a span took in."""

    uses_durations = True

    def __init__(self, client: Client, batch: int = 30, after: int = 4, temperature: float = 0.0,
                 extra: dict | None = None, length_budget: bool = False, spans: bool = False):
        self.client = client
        self.batch = batch
        self.after = after
        self.temperature = temperature
        self.extra = extra or {}
        self.length_budget = length_budget
        self.spans = spans
        self.unspaced = False

    def translate(self, texts: list[str], src: Lang, tgt: Lang,
                  durations: list[float] | None = None) -> tuple[list[tuple[str, int]], Usage]:
        usage = Usage()
        self.unspaced = tgt.code in ("zh", "yue", "ja")
        sys_msg = {"role": "system", "content": system_prompt(src, tgt, self.length_budget, self.spans)}
        budgets: dict[int, int] = {}
        if self.length_budget and durations:
            for i, (t, d) in enumerate(zip(texts, durations)):
                b = budget(t, d, src, tgt)
                if b is not None:
                    budgets[i + 1] = b
        usage.budgeted = len(budgets)
        out: list[tuple[str, int]] = [("", 1)] * len(texts)
        prev: list[dict] = []
        for b0 in range(0, len(texts), self.batch):
            idx = list(range(b0, min(len(texts), b0 + self.batch)))
            after = list(range(idx[-1] + 1, min(len(texts), idx[-1] + 1 + self.after)))
            got = self._batch(texts, idx, after, sys_msg, prev, usage, budgets)
            answer = []
            for i in idx:
                if i in got:
                    last, t = got[i]
                    out[i] = (t, last - i + 1)
                    for k in range(i + 1, last + 1):
                        out[k] = ("", 0)
                    answer.append(f"{i + 1}-{last + 1}| {t}" if last > i else f"{i + 1}| {t}")
            # The batch just done becomes the earlier turn of the next request.
            prev = [{"role": "user", "content": user_prompt([(i + 1, texts[i]) for i in idx], [], budgets)},
                    {"role": "assistant", "content": "\n".join(answer)}]
        return out, usage

    def _ask(self, msgs, batch, max_tokens, usage) -> dict[int, tuple[int, str]] | None:
        text, timings, use, wall = self.client.chat(msgs, max_tokens=max_tokens, temperature=self.temperature,
                                                   extra=self.extra)
        usage.add(timings, use, wall)
        numbers = [n for n, _ in batch]
        if self.spans:
            got = parse_spans(text, numbers)
            if covers_all(got, numbers):
                by = dict(batch)
                units = sorted(got.items())
                sources = [" ".join(by[k] for k in range(a, b + 1)) for a, (b, _) in units]
                if not misaligned(sources, [t for _, (_, t) in units], self.unspaced):
                    return {a - 1: (b - 1, t) for a, (b, t) in got.items()}
                usage.rejected.append({"lines": f"{numbers[0]}-{numbers[-1]}", "shifted": True, "answer": text[:8000]})
                return None
        else:
            single = parse_answer(text, numbers)
            if len(single) == len(batch) and all(single.values()):
                answers = [single[n] for n, _ in batch]
                if not misaligned([t for _, t in batch], answers, self.unspaced):
                    return {n - 1: (n - 1, t) for n, t in single.items()}
                usage.rejected.append({"lines": f"{numbers[0]}-{numbers[-1]}", "shifted": True, "answer": text[:8000]})
                return None
        answered = [int(m) for m in re.findall(r"^\s*(\d+)", text, re.M)]
        spans = parse_spans(text, numbers)
        usage.rejected.append({
            "lines": f"{numbers[0]}-{numbers[-1]}",
            "missing": sorted(set(numbers) - {k for a, (b, _) in spans.items() for k in range(a, b + 1)}),
            "doubled": sorted({n for n in answered if answered.count(n) > 1}),
            "empty": sorted(a for a, (_, t) in spans.items() if not t),
            "answer": text[:8000]})
        return None

    def _batch(self, texts, idx, after, sys_msg, prev, usage, budgets) -> dict[int, tuple[int, str]]:
        batch = [(i + 1, texts[i]) for i in idx]
        ctx = [(i + 1, texts[i]) for i in after]
        msgs = [sys_msg, *prev, {"role": "user", "content": user_prompt(batch, ctx, budgets)}]
        max_tokens = 64 + sum(len(t) for _, t in batch) * 3
        for attempt in range(2):
            got = self._ask(msgs, batch, max_tokens, usage)
            if got is not None:
                return got
            usage.retries += 1
            msgs = msgs[:-1] + [{"role": "user", "content": user_prompt(batch, ctx, budgets) +
                                 f"\n\nAnswer every line from {batch[0][0]} to {batch[-1][0]} exactly once."}]
        usage.fallbacks += 1
        if len(idx) == 1:
            return {}
        half = len(idx) // 2
        left = self._batch(texts, idx[:half], idx[half:half + len(after)], sys_msg, prev, usage, budgets)
        right = self._batch(texts, idx[half:], after, sys_msg, prev, usage, budgets)
        return {**left, **right}
