"""The results as Markdown tables: speech recognition, translation, script conversion.

    python bench/summary.py > results/summary.md

Reads results/asr-scores.json, mt-scores.json, comet-scores.json and
opencc.json. Titles appear under their local names (no library names).
"""

from __future__ import annotations

import json

import common


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    cfg = common.load()
    r = cfg.scratch / "results"
    asr = load(r / "asr-scores.json")
    mt = load(r / "mt-scores.json")
    comet = load(r / "comet-scores.json")
    opencc = json.loads((r / "opencc.json").read_text()) if (r / "opencc.json").exists() else []

    print("## Speech recognition (10-minute windows)\n")
    print("| run | clip | language | WER | SD rate | local recall | time coverage | start offset s | cues | over reading speed % (human) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k, v in sorted(asr.items()):
        if "wer" not in v:
            continue
        print(f"| {v['run']} | {v['clip']} | {v['detected']} | {v['wer']:.3f} | {v['sd_rate']:.3f} | "
              f"{v['local_recall']:.3f} | {v['coverage']:.3f} | {v['median_start_offset']:+.2f} | {v['cues']} | "
              f"{v['wrap_over_cps_pct']} ({v.get('ref_over_cps_pct')}) |")
    print()
    print("| run | clip | detected (p) | cues |")
    print("|---|---|---|---|")
    for k, v in sorted(asr.items()):
        if "wer" in v:
            continue
        print(f"| {v['run']} | {v['clip']} | {v['detected']} ({v['detected_p']}) | {v['cues']} |")

    print("\n## Translation (the same windows, against the title's human subtitle)\n")
    print("| system | title | pair | units | chrF | BLEU | COMET | length vs human | over reading speed %: made / re-timed (human) | over line % |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k, v in sorted(mt.items(), key=lambda kv: (kv[1]["title"], kv[1]["src"], kv[1]["tgt"], kv[1]["tag"])):
        c = comet.get(k, {}).get("comet")
        print(f"| {v['tag']} | {v['title']} | {v['src']}→{v['tgt']} | {v['units']} | {v['chrf']:.1f} | {v['bleu']:.1f} | "
              f"{c if c is not None else '-'} | {v['len_ratio']:.2f} | {v['over_cps_pct']} / "
              f"{v.get('over_cps_retimed_pct', '-')} ({v['ref_over_cps_pct']}) | {v['over_line_pct']} |")

    # Means per system over the pairs every compared system has.
    tags = sorted({v["tag"] for v in mt.values()})
    pairs = {}
    for k, v in mt.items():
        pairs.setdefault((v["title"], v["src"], v["tgt"]), {})[v["tag"]] = (v, comet.get(k, {}).get("comet"))
    print("\n### Means per system over the pairs all of them translated\n")
    for group in (tags,):
        common_pairs = [p for p, d in pairs.items() if all(t in d for t in group)]
        if not common_pairs:
            continue
        print(f"Pairs: {len(common_pairs)}\n")
        print("| system | chrF | COMET | over reading speed % |")
        print("|---|---|---|---|")
        for t in group:
            ch = [pairs[p][t][0]["chrf"] for p in common_pairs]
            co = [pairs[p][t][1] for p in common_pairs if pairs[p][t][1] is not None]
            cp = [pairs[p][t][0]["over_cps_pct"] for p in common_pairs]
            print(f"| {t} | {sum(ch) / len(ch):.1f} | {(sum(co) / len(co)) if co else float('nan'):.4f} | "
                  f"{sum(cp) / len(cp):.1f} |")

    if opencc:
        print("\n## Script conversion (OpenCC, whole tracks)\n")
        print("| title | track | config | cues | detected | Cantonese share | chrF vs human Simplified | identical units % | chars/s |")
        print("|---|---|---|---|---|---|---|---|---|")
        for o in opencc:
            print(f"| {o['title']} | {o['track']} | {o['config']} | {o['cues']} | {o['detected']} | {o['cantonese_ratio']} | "
                  f"{o.get('chrf', '-')} | {o.get('exact_units_pct', '-')} | {o['chars_per_s']} |")


if __name__ == "__main__":
    main()
