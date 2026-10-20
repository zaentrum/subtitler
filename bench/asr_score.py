"""Cues from recognised words, and their scores against the title's own subtitle.

    python bench/asr_score.py RUN_DIR [RUN_DIR ...]

RUN_DIR holds asr_run.py's <clip>.json. For each clip: cues built from the
words on the title's timeline (results/asr-cues/<run>/<clip>.vtt), and, where
the title has a subtitle in the audio's language, the scores of
metrics.score_asr. A table goes to stdout and results/asr-scores.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import common
import metrics
from subtitler import asr, cues, script, vtt, wrap
from subtitler.langs import Lang


def main() -> None:
    cfg = common.load()
    table = {}
    out_json = cfg.scratch / "results" / "asr-scores.json"
    if out_json.exists():
        table = json.loads(out_json.read_text())
    print("run\tclip\tlang\tspeed\tWER\tSD\tlocal_recall\tcoverage\tt_prec\toffset_s\tcues\tover_cps%")
    for run_dir in map(Path, sys.argv[1:]):
        run = run_dir.name
        cue_dir = cfg.out("asr-cues", run)
        for f in sorted(run_dir.glob("*.json")):
            r = asr.Result.from_json(f.read_text()).shifted()
            t = cfg.titles.get(f.stem)
            if t is None:
                continue
            lang = Lang.parse(t.audio if t.audio != "und" else r.language)
            words = r.words
            if lang.code == "zh":
                words = [cues.Word(w.start, w.end, script.convert(w.text, "t2s"), w.probability) for w in words]
            built = cues.build(words, lang.code)
            vtt.write(cue_dir / f"{f.stem}.vtt", built,
                      notes=[f"made by subtitler spike: transcribed ({r.language}) by {Path(r.model).name}"])
            st = wrap.stats(built, lang.code).as_dict()
            row = {"run": run, "clip": f.stem, "detected": r.language, "detected_p": round(r.language_probability, 3),
                   "speed": round(r.speed, 2), "audio_s": round(r.audio_seconds), "wall_s": round(r.seconds, 1),
                   "words": len(r.words), "cues": len(built), **{f"wrap_{k}": v for k, v in st.items()}}
            ref_path = t.subs.get(t.audio) or t.subs.get(f"{t.audio}-Hans")
            if ref_path and t.window:
                ref = vtt.read(ref_path)
                s = metrics.score_asr(ref, words, built, lang.code, t.window[0] + 2, t.window[1] - 2)
                row.update(s.as_dict())
                ref_win = vtt.window(ref, *t.window)
                row["ref_cues"] = len(ref_win)
                row["ref_over_cps_pct"] = wrap.stats(ref_win, lang.code).as_dict()["over_cps_pct"]
            table[f"{run}/{f.stem}"] = row
            print(f"{run}\t{f.stem}\t{r.language}\t{row['speed']}x\t{row.get('wer', '-'):.3}\t"
                  f"{row.get('sd_rate', float('nan')):.3}\t{row.get('local_recall', float('nan')):.3}\t"
                  f"{row.get('coverage', float('nan')):.3}\t{row.get('precision_time', float('nan')):.3}\t"
                  f"{row.get('median_start_offset', float('nan')):+.2f}\t{len(built)}\t{st['over_cps_pct']}"
                  if "wer" in row else
                  f"{run}\t{f.stem}\t{r.language}({r.language_probability:.2f})\t{row['speed']}x\t-\t-\t-\t-\t-\t-\t{len(built)}\t{st['over_cps_pct']}")
    out_json.write_text(json.dumps(table, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
