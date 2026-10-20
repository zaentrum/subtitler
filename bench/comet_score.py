"""COMET (wmt22-comet-da, Apache-2.0) over the units mt_score.py wrote.

    python bench/comet_score.py RESULTS_DIR [TAG ...]

Runs in its own environment (unbabel-comet pins older numpy and
huggingface-hub than the speech stack): it reads only
RESULTS_DIR/mt-units/<tag>/<stem>.jsonl and writes RESULTS_DIR/comet-scores.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    from comet import download_model, load_from_checkpoint

    import os

    root = Path(sys.argv[1])
    tags = sys.argv[2:] or sorted(p.name for p in (root / "mt-units").iterdir() if p.is_dir())
    accelerator = os.environ.get("COMET_ACCELERATOR", "cpu")  # mps on a Mac, cuda on a GPU
    out_path = root / "comet-scores.json"
    scores = json.loads(out_path.read_text()) if out_path.exists() else {}
    model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    for tag in tags:
        for f in sorted((root / "mt-units" / tag).glob("*.jsonl")):
            key = f"{tag}/{f.stem}"
            data = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            data = [d for d in data if d["src"] and d["mt"] and d["ref"]]
            if not data:
                continue
            if accelerator == "cpu":
                res = model.predict(data, batch_size=32, gpus=0, progress_bar=False)
            else:
                res = model.predict(data, batch_size=32, gpus=1, accelerator=accelerator, progress_bar=False)
            scores[key] = {"comet": round(float(res.system_score), 4), "units": len(data)}
            print(f"{key}\tCOMET={res.system_score:.4f}\tunits={len(data)}", flush=True)
            out_path.write_text(json.dumps(scores, indent=1))


if __name__ == "__main__":
    main()
