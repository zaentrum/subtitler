"""Traditional -> Simplified Chinese by OpenCC on the titles' real tracks.

    python bench/opencc_demo.py

For every title with a Traditional track (zh-Hant, zh-HK): each OpenCC
configuration (t2s; tw2s and tw2sp for Taiwan, with tw2sp also rewriting
Taiwanese vocabulary; hk2s for Hong Kong) converts the whole track. Where
the title also has a human Simplified track, the conversion is compared
with it over time-aligned units: chrF, and the share of units that match
character for character. Speed is characters per second. Side-by-side
samples go to results/samples/opencc/<title>.md.
"""

from __future__ import annotations

import json
import time

import common
import metrics
from subtitler import script, vtt

CONFIGS = ["t2s", "tw2s", "tw2sp", "hk2s"]


def main() -> None:
    cfg = common.load()
    out = cfg.out("samples", "opencc")
    rows = []
    for name, t in cfg.titles.items():
        for trad_tag in ("zh-Hant", "zh-HK"):
            if trad_tag not in t.subs:
                continue
            trad = vtt.read(t.subs[trad_tag])
            simp = vtt.read(t.subs["zh-Hans"]) if "zh-Hans" in t.subs else None
            text = "\n".join(c.text for c in trad)
            detected, conf = script.detect_script(text)
            yue = script.cantonese_ratio(text)
            converted = {}
            for config in CONFIGS:
                t0 = time.perf_counter()
                conv = [c.with_text(script.convert(c.text, config)) for c in trad]
                dt = time.perf_counter() - t0
                converted[config] = conv
                row = {"title": name, "track": trad_tag, "config": config, "cues": len(trad),
                       "detected": detected, "detect_conf": round(conf, 3), "cantonese_ratio": round(yue, 4),
                       "chars": len(text), "chars_per_s": round(len(text) / dt) if dt else None,
                       "changed_chars": sum(a != b for a, b in zip(text, "\n".join(c.text for c in conv)))}
                if simp:
                    score, pairs = metrics.score_mt(conv, simp, "zh")
                    exact = sum(1 for h, r, _ in pairs if h.replace(" ", "") == r.replace(" ", ""))
                    row.update(units=score.units, chrf=round(score.chrf, 1), exact_units_pct=round(100 * exact / max(1, len(pairs)), 1))
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))
            # Samples: 20 cues from the middle of the track.
            mid = len(trad) // 2
            lines = [f"# {name}: {trad_tag} -> Simplified (OpenCC)", "",
                     f"Detected script: {detected} (confidence {conf:.3f}); written-Cantonese share {yue:.4f}.", ""]
            head = "| time | Traditional (source) | t2s | tw2sp | hk2s |" + (" human Simplified |" if simp else "")
            lines += [head, "|" + "---|" * (head.count("|") - 1)]
            for k in range(mid, min(len(trad), mid + 20)):
                c = trad[k]
                ref = ""
                if simp:
                    ov = [s for s in simp if min(s.end, c.end) - max(s.start, c.start) > 0.2]
                    ref = " / ".join(vtt.plain(s.text).replace("\n", " ") for s in ov)
                cells = [vtt.timestamp(c.start), vtt.plain(c.text).replace("\n", " ")]
                cells += [vtt.plain(converted[x][k].text).replace("\n", " ") for x in ("t2s", "tw2sp", "hk2s")]
                if simp:
                    cells.append(ref)
                lines.append("| " + " | ".join(x.replace("|", "/") for x in cells) + " |")
            (out / f"{name}.{trad_tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            vtt.write(out / f"{name}.{trad_tag}.t2s.vtt", converted["hk2s" if trad_tag == "zh-HK" else "t2s"],
                      notes=["Chinese (Simplified) · Converted: OpenCC from the Traditional track"])
    (cfg.scratch / "results" / "opencc.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
