"""Subtitle translation by MADLAD-400 (a T5 machine translation model) on CTranslate2.

MADLAD translates one segment at a time and sees no context. Two modes:
- per cue: every cue is a segment;
- merged: cues that continue a sentence (no closing punctuation) are joined
  with the next ones into one segment (at most four cues), translated, and
  the translation is split back over the cues by their share of the source
  length, at word boundaries (characters in Chinese and Japanese).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from ..langs import UNSPACED, Lang

_END = re.compile(r"[.?!…。？！\"”»)\]]\s*$")
_PUNCT = set(",.;:!?…，。；：！？、 ")


@dataclass
class Usage:
    segments: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_s: float = 0.0

    def as_dict(self) -> dict:
        return {"segments": self.segments, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "wall_s": round(self.wall_s, 2),
                "gen_tok_s": round(self.output_tokens / self.wall_s, 1) if self.wall_s else 0.0}


def _target_tag(tgt: Lang) -> str:
    return "<2zh>" if tgt.code == "zh" else f"<2{tgt.code}>"


def split_by_share(text: str, shares: list[int], unspaced: bool) -> list[str]:
    """Cut `text` into len(shares) pieces whose lengths follow `shares`."""
    k = len(shares)
    if k == 1:
        return [text]
    units = list(text) if unspaced else text.split()
    joiner = "" if unspaced else " "
    n, total = len(units), sum(shares) or 1
    cuts, prev, acc = [], 0, 0
    for j, s in enumerate(shares[:-1]):
        acc += s
        remaining = k - j - 1
        lo, hi = prev + (1 if n - prev > remaining else 0), max(prev, n - remaining)
        c = min(max(round(n * acc / total), lo), hi)
        # Move the cut to a punctuation mark or space close by, if there is one.
        reach = 3 if unspaced else 1
        for d in sorted(range(-reach, reach + 1), key=abs):
            m = c + d
            if lo <= m <= hi and 0 < m <= n and units[m - 1][-1:] in _PUNCT:
                c = m
                break
        cuts.append(c)
        prev = c
    pieces, p = [], 0
    for c in cuts + [n]:
        pieces.append(joiner.join(units[p:c]).strip())
        p = c
    return pieces


class MadladTranslator:
    def __init__(self, model_dir: str, device: str = "cpu", compute_type: str = "int8", threads: int = 4,
                 beam_size: int = 4, batch: int = 16, merge: bool = True):
        import ctranslate2
        import sentencepiece

        self.translator = ctranslate2.Translator(model_dir, device=device, compute_type=compute_type,
                                                 intra_threads=threads)
        self.sp = sentencepiece.SentencePieceProcessor(model_file=str(Path(model_dir) / "spiece.model"))
        self.beam_size = beam_size
        self.batch = batch
        self.merge = merge
        self.name = Path(model_dir).name

    def _run(self, segments: list[str], tgt: Lang, usage: Usage) -> list[str]:
        tag = _target_tag(tgt)
        toks = [self.sp.encode(f"{tag} {s}", out_type=str) for s in segments]
        t0 = time.perf_counter()
        res = self.translator.translate_batch(toks, beam_size=self.beam_size, max_batch_size=self.batch,
                                              max_decoding_length=256, repetition_penalty=1.1)
        usage.wall_s += time.perf_counter() - t0
        outs = []
        for t, r in zip(toks, res):
            hyp = r.hypotheses[0]
            usage.segments += 1
            usage.input_tokens += len(t)
            usage.output_tokens += len(hyp)
            outs.append(self.sp.decode(hyp))
        return outs

    def translate(self, texts: list[str], src: Lang, tgt: Lang) -> tuple[list[str], Usage]:
        usage = Usage()
        unspaced = tgt.code in UNSPACED
        if not self.merge:
            return self._translate_lines(texts, tgt, usage), usage
        groups: list[list[int]] = []
        cur: list[int] = []
        for i, t in enumerate(texts):
            if "<br>" in t:  # two speakers: a group of its own
                if cur:
                    groups.append(cur)
                    cur = []
                groups.append([i])
                continue
            cur.append(i)
            if _END.search(t) or len(cur) >= 4:
                groups.append(cur)
                cur = []
        if cur:
            groups.append(cur)
        segs = [" ".join(texts[i] for i in g) for g in groups]
        outs = self._translate_lines(segs, tgt, usage)
        result = [""] * len(texts)
        for g, o in zip(groups, outs):
            pieces = split_by_share(o, [max(1, len(texts[i])) for i in g], unspaced)
            for i, p in zip(g, pieces):
                result[i] = p
        return result, usage

    def _translate_lines(self, texts: list[str], tgt: Lang, usage: Usage) -> list[str]:
        # Two speakers in one cue are two segments.
        flat, owner = [], []
        for i, t in enumerate(texts):
            for part in t.split("<br>"):
                flat.append(part.strip().lstrip("-").strip())
                owner.append(i)
        outs = self._run(flat, tgt, usage)
        joined: list[list[str]] = [[] for _ in texts]
        for i, o in zip(owner, outs):
            joined[i].append(o)
        return [" <br> ".join(("-" + p) if len(parts) > 1 else p for p in parts) for parts in joined]
