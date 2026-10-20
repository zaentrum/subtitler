"""Translation scores against the human subtitle in the target language.

    python bench/mt_score.py [TAG ...]      (default: every folder in results/mt/)

For each results/mt/<tag>/<title>.<src>-<tgt>.vtt: chrF and BLEU over the
time-aligned units (metrics.score_mt), the length ratio, the share of cues
over the reading-speed and line limits, and the units as (source,
translation, reference) triplets in results/mt-units/<tag>/<stem>.jsonl for
COMET. A table goes to stdout and results/mt-scores.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import common
import metrics
from subtitler import vtt, wrap
from subtitler.langs import Lang
from subtitler.translate import prepare


def reference(t: common.Title, tgt: Lang) -> Path | None:
    return t.subs.get(tgt.tag) or (t.subs.get(tgt.code) if tgt.code != "zh" else None)


def main() -> None:
    cfg = common.load()
    mt_root = cfg.scratch / "results" / "mt"
    tags = sys.argv[1:] or sorted(p.name for p in mt_root.iterdir() if p.is_dir())
    out_json = cfg.scratch / "results" / "mt-scores.json"
    table = json.loads(out_json.read_text()) if out_json.exists() else {}
    print("tag\tpair\tunits\tchrF\tBLEU\tlen\tover_cps%\tover_line%\tref_over_cps%\tlate%\tearly%\tanchors r/p\tgen_tok/s\twall_s")
    for tag in tags:
        for f in sorted((mt_root / tag).glob("*.vtt")):
            stem = f.stem
            name, pair = stem.split(".", 1)
            src_tag, tgt_tag = common.split_pair(pair)
            t = cfg.titles[name]
            tgt = Lang.parse(tgt_tag)
            ref_path = reference(t, tgt)
            info_path = f.with_suffix(".json")
            info = json.loads(info_path.read_text()) if info_path.exists() else {}
            if ref_path is None or not t.window:
                continue
            hyp = vtt.read(f)
            ref = vtt.window(vtt.read(ref_path), *t.window)
            score, units = metrics.score_mt(hyp, ref, tgt.code)
            # Source text per hypothesis cue: the prepared source cue with the same times.
            if info.get("from_asr"):
                src_file, is_sdh = Path(info["asr_dir"]) / f"{name}.vtt", False
            else:
                src_file = t.subs[src_tag]
                is_sdh = src_file in t.sdh
            prep = prepare(vtt.window(vtt.read(src_file), *t.window), is_sdh)
            by_time = {(round(c.start, 3), round(c.end, 3)): s for c, s in zip(prep.cues, prep.texts)}
            anch = metrics.anchor_agreement([c.with_text(s) for c, s in zip(prep.cues, prep.texts)], hyp)
            udir = cfg.out("mt-units", tag)
            with open(udir / f"{stem}.jsonl", "w", encoding="utf-8") as fh:
                for h_text, r_text, u in units:
                    srcs = [by_time.get((round(hyp[i].start, 3), round(hyp[i].end, 3)), "") for i in u.hyp]
                    fh.write(json.dumps({"src": " ".join(s.replace(" <br> ", " ") for s in srcs if s),
                                         "mt": h_text, "ref": r_text}, ensure_ascii=False) + "\n")
            st = wrap.stats(hyp, tgt.code).as_dict()
            rst = wrap.stats(ref, tgt.code).as_dict()
            # Runs made before translate_cues applied wrap.reading_time: the same rule here.
            retimed = wrap.reading_time([vtt.Cue(c.start, c.end, c.text) for c in hyp], tgt.code)
            st_rt = wrap.stats(retimed, tgt.code).as_dict()
            disp = metrics.displacement(hyp, ref, tgt.code)
            u = info.get("usage", {})
            row = {"tag": tag, "title": name, "src": src_tag, "tgt": tgt_tag, **score.as_dict(), **disp, **anch,
                   "over_cps_pct": st["over_cps_pct"], "over_cps_retimed_pct": st_rt["over_cps_pct"],
                   "over_line_pct": st["over_line_pct"],
                   "over_lines_pct": st["over_lines_pct"], "ref_over_cps_pct": rst["over_cps_pct"],
                   "cues": len(hyp), "failures": info.get("failures"), "gen_tok_s": u.get("gen_tok_s"),
                   "completion_tokens": u.get("completion_tokens", u.get("output_tokens")),
                   "prompt_tokens": u.get("prompt_tokens", u.get("input_tokens")), "wall_s": info.get("wall_s"),
                   "from_asr": info.get("from_asr", False), "model": info.get("model")}
            table[f"{tag}/{stem}"] = row
            print(f"{tag}\t{stem}\t{score.units}\t{score.chrf:.1f}\t{score.bleu:.1f}\t{score.len_ratio:.2f}\t"
                  f"{st['over_cps_pct']}\t{st['over_line_pct']}\t{rst['over_cps_pct']}\t{disp['late_pct']}\t"
                  f"{disp['early_pct']}\t{anch['anchor_recall_pct']}/{anch['anchor_precision_pct']}\t"
                  f"{u.get('gen_tok_s')}\t{info.get('wall_s')}")
    out_json.write_text(json.dumps(table, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
