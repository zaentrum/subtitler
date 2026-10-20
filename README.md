# subtitler spike (ADR-0014)

ADR-0014 proposes that every title gets a subtitle in each of the
instance's languages, made on the platform where the source brings none:
keep the source's, convert the script, translate the best subtitle the
title has, or transcribe the audio first. This spike measures whether that
holds before anything is built: speech recognition and translation with
open-weight models, their speed and quality on library titles, the decision
per language as a prototype command, and a GPU run prepared for the
platform's RTX 3090.

Results are in [Results](#results). Nothing here touches a package: the
command writes into a folder of its own and refuses one inside the package.

## Layout

| Path | What |
|---|---|
| `src/subtitler/plan.py` | the decision per language, in the ADR's order, with the ranked candidates a report falls back to |
| `src/subtitler/package.py` | what a package holds: `package.json`'s audio and subtitles; Chinese script and Cantonese from the text itself; signs-only tracks |
| `src/subtitler/asr.py` | the package's main audio (its HLS rendition) to 16 kHz, faster-whisper (CPU, CUDA) or mlx-whisper (Metal) |
| `src/subtitler/cues.py` | timed cues from recognised words: sentences, pauses, line and reading limits, Whisper's loops and stock phrases out |
| `src/subtitler/translate/` | translation by an instruction model behind llama.cpp's server, or by MADLAD-400 on CTranslate2; cue times kept, lines re-wrapped |
| `src/subtitler/script.py` | Traditional ↔ Simplified Chinese by OpenCC, script and Cantonese detection |
| `src/subtitler/wrap.py` | line limits per language, re-wrapping, Chinese punctuation house style, reading-speed statistics |
| `src/subtitler/cli.py` | `subtitler plan` and `subtitler run` |
| `bench/` | the local measurements (`asr_run`, `asr_score`, `mt_run`, `mt_score`, `comet_score`, `opencc_demo`, `samples`, `summary`) and the GPU run (`gpu_bench`) |
| `deploy/` | the GPU run: `Dockerfile`, `bench-job.yaml` (placeholders), `run-gpu-bench.sh` |
| `tests/` | the decision logic and the text handling (`pytest`) |

## The decision per language

For each language of `--languages` (the instance's `subtitles.languages`,
BCP 47 with a script where it matters, default `en,de,zh-Hans`):

1. **keep**: the package has a complete subtitle in it (and in its script)
   that a viewer can turn on: WebVTT, or PGS, which the web client draws;
2. **convert**: the same language in the other script as text: OpenCC,
   `t2s` (`hk2s` for Hong Kong tracks), no model;
3. **translate**: the best complete *text* subtitle in another language:
   the original language first, then English, then the instance's
   languages, then widely translated ones; dialogue before SDH, the fuller
   track first;
4. **transcribe**: speech recognition of the main audio: the subtitle when
   the audio is in the language, else the source of 3. It is the last
   candidate of every language, so it is chosen only when the title has no
   text subtitle at all, or a viewer reported every other one.

What counts as a subtitle: not forced, not commentary, not a track of signs
(under 30 % of the cues of the title's fullest text track, or under one cue
a minute: packages carry such tracks unflagged); its file on disk. Chinese tracks carry `chi` whatever they are: the script comes from
the text (OpenCC changes many characters of the other script and almost
none of its own), the title only where the text says nothing; written
Cantonese (粵語 characters, or the title) is another language and never
converted into Mandarin. A bilingual track counts as present but is no
source.

`--original-language` gives the title's original language; without it, the
main audio's. The catalog has no original language today, and a dub as the
default audio makes the dub's language "original" (see Findings).
`--exclude` drops reported sources: the next candidate makes it again.

```
$ subtitler plan PACKAGE --languages en,de,zh-Hans
main audio: a0 (it)
  sub0    en   webvtt,dialogue              ''
  sub2    de   webvtt,dialogue              ''
  sub3    it   webvtt,forced,forced         'Forced / OCR'
  sub5    it   webvtt,dialogue              ''
en       keep sub0
de       keep sub2
zh-Hans  translate sub5 (it)   [then translate sub0 (en); then translate sub2 (de); then translate the transcript (it)]
```

`subtitler run PACKAGE --out DIR` does it and writes `DIR/<language>.vtt`
with a NOTE saying how it was made, and `DIR/subtitles.json`: per language
the label ("German · AI", "Chinese (Simplified) · Converted"), the origin
(`transcribed`, `translated`, `converted`), the source track and language,
the model, timings and tokens.

## Running it locally

```sh
uv venv -p 3.12 .venv && . .venv/bin/activate
uv pip install -e '.[asr,mt,eval,dev]'          # mlx-whisper too on a Mac: '.[mlx]'
pytest

# Translation needs a llama.cpp server with the model (release b11430 here):
llama-server -m Qwen_Qwen3.5-9B-Q4_K_M.gguf --port 8091 -ngl 99 -fa on --reasoning off -c 16384

subtitler plan /path/to/package
subtitler run  /path/to/package --out /path/to/scratch --asr-model large-v3-turbo --spans
```

The bench reads `local.toml` (untracked; `local.example.toml` shows it):
local copies of a few packages, ten minutes of audio each. Heavy runs go
under `taskpolicy -b` on a shared Mac; speeds are measured in short runs at
normal priority, as background QoS runs on the efficiency cores.

```sh
python bench/asr_run.py --model DIR --compute-type int8 --threads 6 --out results-dir clip.wav:en:OFFSET ...
python bench/asr_score.py results-dir
python bench/mt_run.py --engine llm --tag NAME --spans TITLE:en:de TITLE:en:zh-Hans ...
python bench/mt_run.py --engine madlad --tag NAME TITLE:en:de ...
python bench/mt_score.py
python bench/comet_score.py RESULTS                    # its own environment: unbabel-comet pins old numpy
python bench/opencc_demo.py
python bench/samples.py TAG TAG ...
python bench/summary.py
```

## The GPU run

Prepared, not run: it needs the platform owner's go.

```sh
docker build -f deploy/Dockerfile -t IMAGE .      # llama.cpp b11430 for sm_86, FFmpeg n7.1.5 (BtbN), CUDA 12.3.2
# fill NAMESPACE, IMAGE, MEDIA_CLAIM and the package paths in deploy/bench-job.yaml
kubectl apply -f deploy/bench-job.yaml
kubectl -n NAMESPACE logs -f job/subtitler-bench > subtitler-gpu-report.txt
```

The Job requests one GPU, mounts the media volume read-only at `/media`,
fetches the models (about 19 GB) into `/models`, and measures:

- **speech recognition** of every source's main audio (`ASR_MINUTES` of
  it): large-v3-turbo and large-v3 in fp16, sequential (beam 5) and
  batched (`ASR_BATCH`); × real time, GPU memory, and WER, SD rate and
  local recall against the package's own subtitle where it has one in the
  audio's language;
- **translation** of the first `MT_SOURCES` sources' best subtitle
  (`CUE_LIMIT` cues) into German and Simplified Chinese by each `LLMS`
  GGUF model behind a CUDA llama-server with 1, 4 and 8 chains in flight
  (tokens/s per chain and in total), then MADLAD-400 on CTranslate2;
- **next to a busy GPU**: the same, shortened, while `BUSY_STREAMS` FFmpeg
  processes decode a package's video with NVDEC, scale with `scale_cuda` and
  encode with NVENC in a loop, as live rungs do;
- **the estimate**: GPU minutes for a two-hour film without subtitles,
  transcription plus two translations, alone and busy.

## Models and licences

| Use | Model | Licence | Runtime |
|---|---|---|---|
| speech | Whisper large-v3-turbo (CTranslate2: mobiuslabsgmbh/faster-whisper-large-v3-turbo) | MIT | faster-whisper 1.2 / CTranslate2 4.8 (MIT) |
| speech | Whisper large-v3 (Systran/faster-whisper-large-v3) | Apache-2.0 (OpenAI), MIT (conversion) | same |
| speech, VAD | Silero VAD (bundled with faster-whisper) | MIT | ONNX Runtime |
| translation | Qwen3.5-9B (bartowski GGUF Q4_K_M) | Apache-2.0 | llama.cpp (MIT) |
| translation | Qwen2.5-7B-Instruct (bartowski GGUF Q4_K_M) | Apache-2.0 | llama.cpp |
| translation | MADLAD-400 3B MT (Nextcloud-AI CTranslate2 int8) | Apache-2.0 | CTranslate2 |
| script | OpenCC | Apache-2.0 | opencc (Python) |
| scoring only | COMET wmt22-comet-da | Apache-2.0 | unbabel-comet |

Left out for their licences: NLLB-200 and SeamlessM4T (CC-BY-NC-4.0),
Tower (CC-BY-NC-SA-4.0), Hunyuan-MT (Tencent's community licence "does not
apply in the European Union, United Kingdom and South Korea"). Gemma-based
models (TranslateGemma) come under Google's Gemma terms, a custom licence
with use restrictions: possible, not chosen. Seed-X (OpenMDW) and
Qwen3-ASR (Apache-2.0, June 2026) are worth a later look.
