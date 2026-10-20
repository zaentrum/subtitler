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

_LINE = re.compile(r"^\s*(\d+)\s*[|｜:：.)]\s?(.*)$")

_TARGET_RULES = {
    "de": ("Use 'du' between friends, family and children and 'Sie' with strangers, superiors and in "
           "formal situations, as the scene suggests, and keep each relationship's form once chosen."),
    "zh-Hans": ("Write Simplified Chinese as used in mainland China. Use full-width punctuation. Do not end "
                "a line with a full stop (。); use a space instead of a comma inside a line."),
    "zh-Hant": "Write Traditional Chinese. Use full-width punctuation. Do not end a line with a full stop.",
    "en": "Write natural, idiomatic English.",
}


def system_prompt(src: Lang, tgt: Lang) -> str:
    rules = _TARGET_RULES.get(tgt.tag) or _TARGET_RULES.get(tgt.code, "")
    return (
        f"You are a professional subtitle translator. You translate the subtitles of a film or TV episode "
        f"from {src.name} into {tgt.name}.\n"
        "Rules:\n"
        "- Translate every numbered line. Answer with the same numbers, one line each, as \"N| translation\". "
        "Write nothing else: no notes, no explanations, no source text.\n"
        "- Keep each line's content in that line. Do not merge lines or move words between them, even when a "
        "sentence runs over several lines.\n"
        "- Write natural spoken language as a professional subtitler would: concise and quick to read. "
        "Condense slightly rather than add words.\n"
        "- Keep names as they are. Keep the marker <br> where it appears: it separates two speakers.\n"
        f"- {rules}\n"
    )


def user_prompt(batch: list[tuple[int, str]], after: list[tuple[int, str]]) -> str:
    lines = ["Translate these lines:"]
    lines += [f"{n}| {t}" for n, t in batch]
    if after:
        lines.append("")
        lines.append("The lines that follow, for context only (do not translate them):")
        lines += [f"{n}| {t}" for n, t in after]
    return "\n".join(lines)


def parse_answer(text: str, wanted: list[int]) -> dict[int, str]:
    want = set(wanted)
    out: dict[int, str] = {}
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        n = int(m.group(1))
        if n in want and n not in out:
            out[n] = m.group(2).strip()
    return out


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
    calls: list = field(default_factory=list)

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
                "retries": self.retries, "fallbacks": self.fallbacks}


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
    def __init__(self, client: Client, batch: int = 30, after: int = 4, temperature: float = 0.0,
                 extra: dict | None = None):
        self.client = client
        self.batch = batch
        self.after = after
        self.temperature = temperature
        self.extra = extra or {}

    def translate(self, texts: list[str], src: Lang, tgt: Lang) -> tuple[list[str], Usage]:
        """One translation per text, in order. `texts` are single-line cue texts."""
        usage = Usage()
        sys_msg = {"role": "system", "content": system_prompt(src, tgt)}
        out: list[str | None] = [None] * len(texts)
        prev: list[dict] = []
        for b0 in range(0, len(texts), self.batch):
            idx = list(range(b0, min(len(texts), b0 + self.batch)))
            after = list(range(idx[-1] + 1, min(len(texts), idx[-1] + 1 + self.after)))
            got = self._batch(texts, idx, after, sys_msg, prev, usage)
            for i in idx:
                out[i] = got.get(i)
            # The batch just done becomes the earlier turn of the next request.
            prev = [{"role": "user", "content": user_prompt([(i + 1, texts[i]) for i in idx], [])},
                    {"role": "assistant", "content": "\n".join(f"{i + 1}| {out[i] or ''}" for i in idx)}]
        return [o if o is not None else "" for o in out], usage

    def _batch(self, texts, idx, after, sys_msg, prev, usage, depth=0) -> dict[int, str]:
        batch = [(i + 1, texts[i]) for i in idx]
        ctx = [(i + 1, texts[i]) for i in after]
        msgs = [sys_msg, *prev, {"role": "user", "content": user_prompt(batch, ctx)}]
        budget = 64 + sum(len(t) for _, t in batch) * 3
        for attempt in range(2):
            text, timings, use, wall = self.client.chat(msgs, max_tokens=budget, temperature=self.temperature,
                                                       extra=self.extra)
            usage.add(timings, use, wall)
            got = parse_answer(text, [n for n, _ in batch])
            if len(got) == len(batch) and all(v for v in got.values()):
                return {n - 1: v for n, v in got.items()}
            usage.retries += 1
            msgs = msgs[:-1] + [{"role": "user", "content": user_prompt(batch, ctx) +
                                 f"\n\nAnswer with exactly {len(batch)} lines, numbered "
                                 f"{batch[0][0]} to {batch[-1][0]}."}]
        if len(idx) == 1:
            usage.fallbacks += 1
            return {idx[0]: got.get(idx[0] + 1, "")} if got else {}
        half = len(idx) // 2
        usage.fallbacks += 1
        left = self._batch(texts, idx[:half], idx[half:half + len(after)], sys_msg, prev, usage, depth + 1)
        right = self._batch(texts, idx[half:], after, sys_msg, prev, usage, depth + 1)
        return {**left, **right}
