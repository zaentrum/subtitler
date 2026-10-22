"""The models a run uses, fetched from Hugging Face into a cache folder.

None of them is in the image: a run downloads what its cache folder lacks
(a volume kept between runs saves the download). Every one is open-weight
under a permissive licence:

| name | Hugging Face repository | licence |
|---|---|---|
| large-v3 | Systran/faster-whisper-large-v3 | Apache-2.0 (weights), MIT (conversion) |
| large-v3-turbo | mobiuslabsgmbh/faster-whisper-large-v3-turbo | MIT |
| qwen3.5-9b-q4km | bartowski/Qwen_Qwen3.5-9B-GGUF (Q4_K_M) | Apache-2.0 |
"""

from __future__ import annotations

from pathlib import Path

MODELS = {
    # name: (Hugging Face repository, file or None for the whole repository)
    "large-v3": ("Systran/faster-whisper-large-v3", None),
    "large-v3-turbo": ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", None),
    "qwen3.5-9b-q4km": ("bartowski/Qwen_Qwen3.5-9B-GGUF", "Qwen_Qwen3.5-9B-Q4_K_M.gguf"),
}


def describe(name: str) -> str:
    repo, file = MODELS[name]
    return f"{repo}/{file}" if file else repo


def fetch(models_dir: str | Path, name: str, revision: str | None = None) -> Path:
    """The model's local path, downloaded first when the cache folder lacks it."""
    from huggingface_hub import hf_hub_download, snapshot_download

    repo, file = MODELS[name]
    target = Path(models_dir) / name
    if file:
        path = target / file
        if not path.exists():
            hf_hub_download(repo, file, local_dir=target, revision=revision)
        return path
    if not (target / "model.bin").exists():
        snapshot_download(repo, local_dir=target, revision=revision)
    return target
