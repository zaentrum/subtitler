"""Side-by-side samples for a person to read: 20 cues per language pair.

    python bench/samples.py TAG [TAG ...]

For each pair translated under the first TAG (results/mt/<tag>/<title>.<src>-<tgt>.vtt):
20 consecutive source cues from the window, each with the translation of
every TAG and the human subtitle over the same time. Markdown in
results/samples/mt/<title>.<src>-<tgt>.md. Transcriptions get the same
(results/samples/asr/<title>.md) for every results/asr-cues/<run>/.
"""

from __future__ import annotations

import sys

import common
from subtitler import vtt
from subtitler.langs import Lang
from subtitler.translate import prepare


def cell(text: str) -> str:
    return vtt.plain(text).replace("\n", " / ").replace("|", "∣").strip()


def overlapping(cues: list[vtt.Cue], start: float, end: float, share: float = 0.3) -> str:
    hits = [c for c in cues if min(c.end, end) - max(c.start, start) > share * min(c.duration, end - start)]
    return " ‖ ".join(cell(c.text) for c in hits)


def main() -> None:
    cfg = common.load()
    tags = sys.argv[1:]
    root = cfg.scratch / "results"
    out = cfg.out("samples", "mt")
    for f in sorted((root / "mt" / tags[0]).glob("*.vtt")):
        name, pair = f.stem.split(".", 1)
        src_tag, tgt_tag = common.split_pair(pair)
        t = cfg.titles[name]
        tgt = Lang.parse(tgt_tag)
        src_path = t.subs[src_tag]
        prep = prepare(vtt.window(vtt.read(src_path), *t.window), src_path in t.sdh)
        systems = {}
        for tag in tags:
            p = root / "mt" / tag / f.name
            if p.exists():
                systems[tag] = vtt.read(p)
        ref_path = t.subs.get(tgt.tag)
        ref = vtt.window(vtt.read(ref_path), *t.window) if ref_path else []
        k0 = len(prep.cues) // 3
        lines = [f"# {name}: {src_tag} → {tgt_tag}", "",
                 f"Source: the title's own {src_tag} subtitle{' (SDH, cleaned)' if src_path in t.sdh else ''}; "
                 f"20 cues from {vtt.timestamp(prep.cues[k0].start)}. The last column is the title's human "
                 f"{tgt_tag} subtitle over the same time (‖ between its cues).", ""]
        head = ["time", f"source ({src_tag})"] + list(systems) + [f"human {tgt_tag}"]
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        shown: dict[str, set] = {tag: set() for tag in systems}
        for c, s in list(zip(prep.cues, prep.texts))[k0:k0 + 20]:
            row = [vtt.timestamp(c.start), s.replace(" <br> ", " / ").replace("|", "∣")]
            for tag, hyp in systems.items():
                # The translated cue over this source cue's time; one cue may cover a
                # whole sentence of several source cues (shown once, then ↑).
                over = [(min(h.end, c.end) - max(h.start, c.start), k) for k, h in enumerate(hyp)]
                best = max(over, default=(0, None))
                if best[1] is None or best[0] <= 0:
                    row.append("")
                elif best[1] in shown[tag]:
                    row.append("↑")
                else:
                    shown[tag].add(best[1])
                    row.append(cell(hyp[best[1]].text))
            row.append(overlapping(ref, c.start, c.end))
            lines.append("| " + " | ".join(row) + " |")
        (out / f"{f.stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(out / f"{f.stem}.md")

    asr_out = cfg.out("samples", "asr")
    runs = sorted(p for p in (root / "asr-cues").iterdir() if p.is_dir()) if (root / "asr-cues").exists() else []
    for name, t in cfg.titles.items():
        if not t.window:
            continue
        hyps = {r.name: vtt.read(r / f"{name}.vtt") for r in runs if (r / f"{name}.vtt").exists()}
        if not hyps:
            continue
        ref_path = t.subs.get(t.audio) or t.subs.get(f"{t.audio}-Hans")
        ref = vtt.window(vtt.read(ref_path), *t.window) if ref_path else []
        first = next(iter(hyps.values()))
        k0 = len(first) // 3
        lines = [f"# {name}: transcription ({t.audio})", "",
                 f"20 cues of the first run from {vtt.timestamp(first[k0].start) if first else '-'}; the other "
                 f"runs and the title's own subtitle over the same time.", ""]
        head = ["time"] + list(hyps) + (["title's subtitle"] if ref else [])
        lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for c in first[k0:k0 + 20]:
            row = [vtt.timestamp(c.start), cell(c.text)]
            row += [overlapping(h, c.start, c.end) for h in list(hyps.values())[1:]]
            if ref:
                row.append(overlapping(ref, c.start, c.end))
            lines.append("| " + " | ".join(row) + " |")
        (asr_out / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(asr_out / f"{name}.md")


if __name__ == "__main__":
    main()
