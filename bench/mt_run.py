"""Translate a title's subtitle (its clip window) and keep the result and its cost.

    python bench/mt_run.py --engine llm --tag qwen7 ep-multi:en:de ep-multi:en:zh-Hans ...
    python bench/mt_run.py --engine madlad --tag madlad-merged ep-multi:en:de ...
    python bench/mt_run.py --engine llm --tag qwen7-asr --from-asr results/asr-cues/fw-turbo ep-multi:en:de

Each pair is TITLE:SOURCE:TARGET with tags from local.toml. --from-asr takes
the source from TITLE.vtt in that folder (a transcription) instead of the
title's own subtitle. Output: results/mt/<tag>/<title>.<src>-<tgt>.vtt and
.json (usage: tokens, tokens/s, wall time, cues).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import common
from subtitler import vtt
from subtitler.langs import Lang
from subtitler.translate import translate_cues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["llm", "madlad"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--llm-url", default="http://127.0.0.1:8091")
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--after", type=int, default=4)
    ap.add_argument("--budget", action="store_true", help="mark lines that need condensing with [max N]")
    ap.add_argument("--spans", action="store_true", help="let the model translate a sentence over lines as one")
    ap.add_argument("--madlad-merge", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--from-asr", default=None)
    ap.add_argument("--full", action="store_true", help="the whole track, not the clip window")
    ap.add_argument("pairs", nargs="+")
    a = ap.parse_args()

    cfg = common.load()
    out = cfg.out("mt", a.tag)
    if a.engine == "llm":
        from subtitler.translate.llm import Client, LlmTranslator

        client = Client(a.llm_url)
        tr = LlmTranslator(client, batch=a.batch, after=a.after, length_budget=a.budget, spans=a.spans)
        model = client.model_id()
    else:
        from subtitler.translate.madlad import MadladTranslator

        tr = MadladTranslator(str(cfg.models["madlad"]), threads=a.threads, beam_size=a.beam, merge=a.madlad_merge)
        model = tr.name
    print(f"# {a.engine} {model} tag={a.tag}", flush=True)

    for spec in a.pairs:
        name, src_tag, tgt_tag = spec.split(":")
        t = cfg.titles[name]
        if a.from_asr:
            src_path = Path(a.from_asr) / f"{name}.vtt"
            is_sdh = False
        else:
            src_path = t.subs[src_tag]
            is_sdh = src_path in t.sdh
        cues = vtt.read(src_path)
        if not a.full and t.window:
            cues = vtt.window(cues, *t.window)
        t0 = time.perf_counter()
        res, usage, failures = translate_cues(cues, Lang.parse(src_tag), Lang.parse(tgt_tag), tr, is_sdh=is_sdh)
        wall = time.perf_counter() - t0
        stem = f"{name}.{src_tag}-{tgt_tag}"
        vtt.write(out / f"{stem}.vtt", res, notes=[f"made by subtitler spike: translated {src_tag}->{tgt_tag} by {model}"])
        u = usage.as_dict()
        info = {"title": name, "src": src_tag, "tgt": tgt_tag, "engine": a.engine, "model": model,
                "source": str(src_path.name), "from_asr": bool(a.from_asr),
                "asr_dir": str(Path(a.from_asr).resolve()) if a.from_asr else None, "cues_in": len(cues),
                "cues_out": len(res), "failures": failures, "wall_s": round(wall, 2), "usage": u,
                "full": a.full}
        (out / f"{stem}.json").write_text(json.dumps(info, indent=1))
        print(f"{stem}\tcues={len(cues)}\tout={len(res)}\tfail={failures}\twall={wall:.1f}s\t"
              f"gen={u.get('gen_tok_s')}tok/s\tcompl={u.get('completion_tokens', u.get('output_tokens'))}"
              f"\tretries={u.get('retries', '-')}", flush=True)


if __name__ == "__main__":
    main()
