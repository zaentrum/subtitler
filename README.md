# subtitler spike (ADR-0014)

ADR-0014 proposes that every title gets a subtitle in each of the
instance's languages, made on the platform where the source brings none:
keep the source's, convert the script, translate the best subtitle the
title has, or transcribe the audio first. This spike measures whether that
holds before anything is built: speech recognition and translation with
open-weight models, their speed and quality on library titles, the decision
per language as a prototype command, and a GPU run prepared for the
platform's RTX 3090.

Results are in [Results](#results), what they mean for the ADR in
[Findings](#findings). Nothing here touches a package: the command writes
into a folder of its own and refuses one inside the package.

## The picks

- **Speech: Whisper large-v3** (Apache-2.0; the CTranslate2 conversion
  MIT) on faster-whisper (MIT), **batched**, fp16 on the GPU, with Silero
  VAD. Equal to large-v3-turbo on English, 5–10 points better on Italian
  and Mandarin; batched decoding does not skip speech the way long-form
  decoding did. turbo (MIT) stays the fast fallback.
- **Translation: Qwen3.5-9B** (Apache-2.0) as a Q4_K_M GGUF on llama.cpp
  (MIT), **whole sentences** placed back over their cues, a length budget
  for German. Better than Qwen2.5-7B (COMET 0.758 against 0.738 over ten
  pairs) and MADLAD-400 3B (0.785 against 0.747), which repeats short lines.
  The same file runs on Metal and CUDA; about 9 GB of GPU memory with eight
  chains.
- **Script: OpenCC** (Apache-2.0), `t2s`, `hk2s` for Hong Kong tracks.

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
a minute: packages carry such tracks unflagged); its file on disk. Chinese
tracks carry `chi` whatever they are: the script comes from the text (OpenCC changes many characters of the other script and almost
none of its own), the title only where the text says nothing; written
Cantonese (粵語 characters, or the title) is another language and never
converted into Mandarin. A bilingual track counts as present but is no
source.

`--original-language` gives the title's original language; without it, the
main audio's. The catalog has no original language today, and a dub as the
default audio makes the dub's language "original" (see Findings).
`--exclude` drops reported sources: the next candidate makes it again.
`--source-order english` puts a complete English track before the original
language (see Findings); the ADR's order is the default.

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
uv pip install -e '.[asr,mt,eval,zh,dev]'       # mlx-whisper too on a Mac: '.[mlx]'
pytest

# Translation needs a llama.cpp server with the model (release b11430 here):
llama-server -m Qwen_Qwen3.5-9B-Q4_K_M.gguf --port 8091 -ngl 99 -fa on --reasoning off -c 16384

subtitler plan /path/to/package
subtitler run  /path/to/package --out /path/to/scratch --asr-model large-v3 --batch-size 16 --length-budget
```

The bench reads `local.toml` (untracked; `local.example.toml` shows it):
local copies of a few packages, ten minutes of audio each. Heavy runs go
under `taskpolicy -b` on a shared Mac; speeds are measured in short runs at
normal priority, as background QoS runs on the efficiency cores.

```sh
python bench/asr_run.py --model DIR --compute-type int8 --threads 6 --out results-dir clip.wav:en:OFFSET ...
python bench/asr_score.py results-dir
python bench/mt_run.py --engine llm --tag NAME --sentences [--budget] TITLE:en:de TITLE:en:zh-Hans ...
python bench/mt_run.py --engine madlad --tag NAME --sentences TITLE:en:de ...
python bench/mt_run.py --engine llm --tag NAME --sentences --from-asr results/asr-cues/RUN TITLE:en:de
python bench/mt_score.py
python bench/comet_score.py RESULTS                    # its own environment: unbabel-comet pins old numpy
python bench/opencc_demo.py
python bench/samples.py TAG TAG ...
python bench/summary.py
```

## The GPU run

Prepared, not run: it needs the platform owner's go. Like the live-rung
spike's GPU run it belongs in CI: the image built by the pipeline, the Job
applied by a manual job; these are the steps that job takes.

```sh
docker build -f deploy/Dockerfile -t IMAGE .      # llama.cpp b11430 for sm_86, FFmpeg n7.1.5 (BtbN), CUDA 12.3.2
# fill NAMESPACE, IMAGE, MEDIA_CLAIM and the package paths in deploy/bench-job.yaml
kubectl apply -f deploy/bench-job.yaml
kubectl -n NAMESPACE logs -f job/subtitler-bench > subtitler-gpu-report.txt
```

The image has not been built: this machine has no CUDA toolchain, and
llama.cpp's own CI builds with CUDA 12.4 and later (the Dockerfile names a
fallback). The bench itself ran here end to end on CPU and Metal
(`BENCH_DEVICE=cpu`, one source): extraction, both decoding modes, the
scores, llama-batched-bench, chains, MADLAD; not its CUDA settings or the
NVENC load.

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
| Chinese line breaks | jieba word segmentation | MIT | jieba (Python) |
| scoring only | COMET wmt22-comet-da | Apache-2.0 | unbabel-comet |

Left out for their licences: NLLB-200 and SeamlessM4T (CC-BY-NC-4.0),
Tower (CC-BY-NC-SA-4.0), Hunyuan-MT (Tencent's community licence "does not
apply in the European Union, United Kingdom and South Korea"). Gemma-based
models (TranslateGemma) come under Google's Gemma terms, a custom licence
with use restrictions: possible, not chosen. Seed-X (OpenMDW) and
Qwen3-ASR (Apache-2.0, June 2026) are worth a later look.

## Results

Nine library titles, copied locally in parts (about 105 MB): one episode and
eight films; a ten-minute window of the main audio for seven; audio in
English (three), Italian, Mandarin, Korean and one tagged `und` (Japanese).
Speeds come from short runs at normal priority on the spike's machine, an
Apple M5 laptop (4 performance cores, 32 GB); the bulk ran at background
priority. Numbers name titles by their local ids only.

### Speech recognition

Against the title's own subtitle in the audio's language over the window
(WER, and CER for Chinese; SD rate = substitutions and deletions only;
local recall = subtitle words found within 2 s):

| clip | model, mode | WER | SD rate | local recall | time covered |
|---|---|---|---|---|---|
| English episode | large-v3-turbo, sequential | 5.9 % | 5.4 % | 94 % | 91 % |
| | large-v3-turbo, batched | 6.9 % | 6.5 % | 93 % | 91 % |
| | large-v3, batched | 6.5 % | 5.8 % | 94 % | 91 % |
| English film | large-v3-turbo, sequential | 14.9 % | 12.0 % | 88 % | 93 % |
| | large-v3-turbo, batched | 19.8 % | 13.2 % | 87 % | 92 % |
| Italian film (bilingual, dialect) | large-v3-turbo, sequential | 59.8 % | 47.0 % | 57 % | 74 % |
| | large-v3-turbo, batched | 63.8 % | 47.5 % | 54 % | 74 % |
| | large-v3, batched | 51.7 % | 36.2 % | 66 % | 80 % |
| Mandarin film (a dub) | large-v3-turbo, sequential | 58.2 % | 51.7 % | 49 % | 71 % |
| | large-v3-turbo, batched | 54.7 % | 48.4 % | 53 % | 77 % |
| | large-v3, batched | 49.7 % | 43.0 % | 58 % | 79 % |

Cue starts sit within 0.2 s of the subtitle's (median). Language detection
named the `und` title Japanese (p = 0.96) and the `chi` one Mandarin
(0.998). The Italian and Mandarin references overstate the errors: the
Italian subtitle translates the film's English lines and normalises its
Roman dialect, which Whisper writes as spoken; the Mandarin audio is a dub,
whose script differs from the subtitle's. Recognition errors remain on top
(wrong characters for ordinary words, one loop of a repeated character, shouting under music missed).

Speed here (2-minute English excerpt, beam 5): faster-whisper int8 on 4
CPU threads 6.6× real time (turbo) and 1.8× (large-v3); mlx-whisper on the
GPU (Metal, greedy) 8.4× and 4.2×. Published numbers for the RTX 3090's
class: faster-whisper's README has large-v2 (large-v3's size) on an RTX
3070 Ti at 12× sequential and 46× batched (fp16, batch 8); issue #1030 has
large-v3 at 15× and turbo at 41× sequential. A 3090 is about 1.5× a
3070 Ti (bandwidth and fp16 throughput): large-v3 batched about 70× real
time, a two-hour film in about 2 minutes.

### Translation

The windows' own subtitles translated (cue times kept), against the
title's human subtitle in the target language (COMET wmt22-comet-da; chrF
on characters):

| system | pairs | COMET | chrF |
|---|---|---|---|
| Qwen2.5-7B-Instruct, line by line | 10 | 0.738 | 36.3 |
| Qwen3.5-9B, sentences | 10 | 0.758 | 39.0 |
| MADLAD-400 3B, sentences | 2 | 0.747 | 34.0 |
| Qwen3.5-9B on the same 2 | 2 | 0.785 | 37.1 |
| Qwen3.8-27B (IQ4_XS), sentences, en→de | 2 | 0.787 | 50.9 |
| Qwen3.5-9B on the same 2 | 2 | 0.777 | 49.8 |

Qwen3.5-9B by pair (COMET / chrF): en→de 0.769 / 49.7 and 0.785 / 50.0;
en→zh-Hans 0.801 / 24.4 and 0.730 / 15.5; it→de 0.752 / 42.6; it→en
0.750 / 44.5; zh→de 0.665 / 31.7; zh→en 0.714 / 36.9. From the
transcription instead of a subtitle (the case of a title with none):
en→de 0.761 and 0.744, en→zh-Hans 0.723 and 0.699, it→de 0.674, it→en 0.685.
Line by line against sentences (2 pairs): 0.772 against 0.785; the length
budget (4 pairs): 0.773 against 0.777, shorter German.

Reading speed: of the German translated from the two English titles, 50 %
and 63 % of cues run over 17 characters a second once cue ends may extend
into the gap after them (human tracks: 43 % and 44 %); with the length
budget 48 % and 55 %. Chinese: 6–7 % over 9 a second, human 1–5 %.

Speed here, normal priority: Qwen3.5-9B (Q4_K_M) on Metal reads prompts at
608 tokens/s and generates 21.4 (llama-bench); a ten-minute German window
took 192 s end to end. Several chains at once add up: 20.9, 39.9, 63.0 and
101.3 tokens/s in total for 1, 2, 4 and 8 sequences (llama-batched-bench).
A cue costs 15–17 tokens in German, 10–17 in Chinese; a two-hour film
(1,500 cues; 1,000–2,000 in the library) about 22,000 tokens per language,
about 23 minutes per language here. MADLAD-400 on 4 CPU cores: 19.8
tokens/s (beam 4, batches of 16). On a 3090: llama.cpp's CUDA scoreboard
has 162 tokens/s for a 3.6 GB 7B; generation is bandwidth-bound, so about
90 tokens/s for this 6.2 GB file, 4 minutes per language and film in one
chain; parallel chains add up as they do here (the GPU run measures it).

### The command end to end

`subtitler run` on the local copies: the film with only a Traditional
Chinese track got its Simplified one by conversion (1,577 cues, under a
second); the two films with no subtitle at all were transcribed from their
package's audio (large-v3, int8 on CPU at background priority: the Korean
window in 131 s, the Japanese one, tagged `und` and detected, in 559 s) and
translated from the transcript into English, German and Simplified Chinese,
each written as WebVTT with its NOTE and label ("German · AI") and listed in
`subtitles.json` with origin, source, models and tokens.

### Script conversion

OpenCC on whole tracks, against the same title's human Simplified track:
where both are one translation in two scripts (the Mandarin film), `tw2s`
gives chrF 85.8 and 75 % of cues identical to the character; the rest are
the human track's own word choices. On the other two titles the Traditional
and Simplified tracks are separate translations (chrF 15–23), which the
comparison cannot separate from conversion. A track labelled "Hong Kong"
was written Cantonese (6 % Cantonese-only characters), found by its text.
Conversion takes well under a second per track (0.1–1 million characters a
second).

## Findings

- **Translate sentences, not lines.** Asked line by line, the 9B model
  moves German words into the next line's answer (verb-final sentences
  over two cues) and the text runs a cue behind the speech until it catches
  up. Sentence mode (the cues a sentence runs over translated as one,
  placed back as one cue or split by length within the line limits) cannot
  drift and scores higher (COMET +0.013).
- **Batched speech recognition.** Sequential long-form decoding skipped
  40 s of a mixed-language scene that VAD had found; batched decoding (each
  speech chunk on its own) did not, and is 3–4× faster on a GPU.
- **large-v3 over turbo** for anything not English (5–10 points better);
  equal on English. On a 3090 it costs about two minutes a film.
- **The source order.** From the Mandarin film's English subtitle the
  German scores 0.856, from its Chinese one 0.665; from the Italian film's
  English and Italian ones it is equal. English is as good or better, but
  an English subtitle made for viewers who hear the film's English lines
  leaves those scenes out (a 20-cue gap in the Italian film).
  `--source-order english` takes English first when it leaves no scene
  out (no run of 8 original cues without English near them).
- **The catalog knows no original language.** With a dub as the default
  audio, "original first" takes the dub's subtitle; `--original-language`
  carries it once the catalog has it.
- **Library quirks the decision handles:** unflagged signs-only tracks
  (27 cues in a 109-minute film), "Hong Kong" tracks in Cantonese,
  untitled Chinese tracks of either script, bilingual tracks. PGS image
  tracks are 38 % of the sampled subtitle tracks: present for viewers, no
  source without OCR.
- **A larger model writes better German.** Qwen3.8-27B (Apache-2.0,
  IQ4_XS) on the two English→German pairs: COMET 0.787 against 0.777 and,
  read, about two errors in 20 cues against four to five (a toast, the
  genders and the terms right). It takes 2.5× the
  time and about 17 GB of GPU memory: a quality tier for an idle GPU, not
  for one beside a busy live worker.
- **MADLAD-400 3B repeats short lines** (a two-word line comes back twice)
  and adds words (a "he asked" that nobody said) whatever the decoding settings;
  it is no subtitle translator.
- **Transcripts without punctuation.** Whisper writes Japanese (and often
  Chinese) with little punctuation, so sentence mode sees fragments: the
  Japanese window needed 2–9 retries a language and sentences end up split
  over cues. Grouping by pauses, or asking the model to segment first, is
  the next step.
- **Abbreviations.** "U.S." taken for a sentence end made the model join
  lines: seven times the requests and three times the tokens for one
  window; fixed.

## The ADR's target

*A two-hour film without subtitles gets all three languages in under
15 minutes of GPU time, fit to watch with.* Estimated for the RTX 3090
from published numbers and the token counts measured here; the GPU run
replaces each estimate with a measurement.

| step | English audio (de and zh-Hans made) | other audio (en, de, zh-Hans made) |
|---|---|---|
| transcription, large-v3 batched (about 70× real time) | 2 min | 2 min |
| translations, one chain (22,000 tokens each at about 90 tokens/s) | 2 × 4 min | 3 × 4 min |
| prompts (about 40 new tokens a cue at thousands a second) | under 1 min | under 1 min |
| **one chain** | **about 11 min** | **about 15 min** |
| **four chains in flight** (three times the total, as measured here) | **about 5 min** | **about 6–7 min** |

It holds with parallel chains, and with no margin for audio in other
languages in one chain. Next to a busy GPU, live rungs use NVDEC and NVENC,
which are units of their own, but share memory bandwidth, which bounds
generation; the run measures how much it costs. GPU memory: large-v3 in
fp16 about 4–6 GB batched, unloaded before translation; Qwen3.5-9B
6.2 GB plus about 2 GB of cache for eight chains (8 of its 32 layers keep
one); the live worker 0.6–1.3 GB per job in flight (ADR-0013's run).

Fit to watch, from reading the samples (`results/samples/`, 20 cues a pair):
English→Chinese and Italian→English are fit to watch (about one serious
error in 20 cues, some literal phrasing); English→German carries the
meaning with four to five grammar or word errors in 20 cues and reads fast:
usable, clearly machine-made. From a transcription of English audio the
German is as good as from a subtitle; from hard audio (a dub, a bilingual
film, dialect) the transcription's errors carry through.
