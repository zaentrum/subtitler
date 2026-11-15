# subtitler

The subtitler is a Python command line tool that makes subtitles in each of an instance's
languages for titles that have none ([ADR-0014](https://github.com/zaentrum/zaentrum/blob/main/docs/adr/0014-subtitles-in-every-language.md)). Per language it keeps the source's own,
gives an image subtitle (PGS, VobSub) a text copy with Tesseract, converts Chinese script
with OpenCC, translates the best subtitle with an instruction model behind llama.cpp's
server, or transcribes the audio with faster-whisper and translates the transcript. It
writes into a folder of its own and never into a package; `subtitler attach` hands the text
copies to katalog-manager's attach endpoint. It runs as a Job from its image, not as a
service, and the operator's chart does not start it. The [README](README.md) has the
decision, the commands and what was measured.

How a change is proposed and what a pull request has to pass is in the
[guide to contributing](https://github.com/zaentrum/.github/blob/main/CONTRIBUTING.md);
what may be built is in [building from the specification](https://github.com/zaentrum/zaentrum/blob/main/AGENTS.md).

## Build

```sh
python3.12 -m venv .venv && . .venv/bin/activate
pip install -c requirements-image.txt -e '.[zh,dev]'
```

Python 3.10 or later (`requires-python` in `pyproject.toml`; `ci.yml` uses 3.12).
`requirements-image.txt` pins what the image installs. The extras: `asr` (faster-whisper),
`mlx` (mlx-whisper, macOS), `mt` (MADLAD-400 on CTranslate2), `zh` (jieba), `eval` and
`comet` (the bench's scores), `dev` (pytest, Pillow). The image:

```sh
docker build -t subtitler .
```

It is CUDA 12.8 on Ubuntu 24.04 with llama.cpp's release build (`LLAMA_CPP_TAG`), BtbN's
FFmpeg 7.1 and Tesseract 5.5.4 on Leptonica, built from source in a stage of their own, with
the `eng`, `deu`, `chi_sim` and `chi_tra` models. No speech or translation model is in it: a
run fetches them into its model folder. `ci.yml` builds and pushes it after the tests, on
`main` and on a tag `vX.Y.Z`. `deploy/Dockerfile` is the spike's GPU bench; no workflow builds it.

## Test

`ci.yml` installs `tesseract-ocr` and `ffmpeg` from apt, then runs:

```sh
pip install -c requirements-image.txt -e '.[zh,dev]'
python -m pytest -q
```

- `tests/test_ocr.py`: two of its 20 cases read a synthetic PGS track with Tesseract end to
  end and skip without `tesseract` on the PATH or Pillow, or when Tesseract has no English
  model; the other 18 run with nothing installed.
- `tests/test_extract.py`: two of its five cases extract from a file ffmpeg makes and skip
  without `ffmpeg`, the OCR one also without `tesseract` or Pillow; the other three run with
  nothing installed.
- The rest needs nothing installed: packages and image subtitles are built in the tests
  (`tests/conftest.py`, `tests/synth.py`), the translator talks to a fake chat endpoint, the
  GPU steps are fakes, and `subtitler attach` talks to a seam of the test's own.

## Run it

`subtitler plan` reads a package folder (its `package.json` and subtitle files) and prints
the decision per language; it runs no model. The version folders of the schemas
repository's example library serve as recorded input, from a checkout beside this one:

```sh
subtitler plan ../schemas/library/v2/examples/movies/b6/b68e2fff-ad52-5f70-97ef-ff618b37c4bb/versions/c4d1ec81-7696-5e12-a2ee-00924217ce0e
```

`subtitler run PACKAGE --out DIR` makes the subtitles. A text copy needs Tesseract 5, a
translation a llama.cpp server with the model at `--llm-url` (`http://127.0.0.1:8091`), and
speech recognition the `asr` extra and a Whisper model. `subtitler extract`, `ocr` and `job`
take a list of titles and read their originals or packages on the library share, `job` on a
GPU; `deploy/job.yaml` and `deploy/ocr-job.yaml` are Jobs for `job` and `ocr`, with
placeholders. `subtitler attach` takes katalog-manager's address (`--url` or `KATALOG_URL`)
and a token (`--token-file` or `KATALOG_TOKEN`). `run`, `ocr` and `attach` have `--dry-run`.
`subtitler <command> --help` lists every flag; the README's "Running it locally" has example
commands.

## What the tests leave unchecked

- No model runs in a test: speech recognition (faster-whisper, mlx-whisper), translation by
  llama.cpp's server or MADLAD-400. The GPU, its memory checks (`nvidia-smi`) and the
  transcription process are fakes; no test needs a GPU.
- `subtitler attach` is never run against katalog-manager, only against the test's seam.
- The OCR tests read text drawn with Pillow's own font, with the Tesseract on the PATH
  (Ubuntu's in `ci.yml`), not the image's own build. The README's error rates come from
  `bench/ocr_score.py` on library titles, not from a test.
- `bench/` and `deploy/` (the bench image, the Jobs, `deploy/ocr-titles.sql`) have no tests.
- A pull request runs the tests only: the image is built on `main` and on tags.
