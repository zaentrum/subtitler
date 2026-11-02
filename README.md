# subtitler

Subtitles in each of an instance's languages for titles that have none,
made on the platform with open-weight models (ADR-0014): keep the source's,
give its image subtitles a text copy (OCR), convert the script, translate
the best subtitle the title has, or transcribe the audio first and translate
the transcript.

It began as the spike that measured whether this holds (speech recognition
and translation, their speed and quality on library titles, the decision
per language); its results are in [Results](#results), what they mean in
[Findings](#findings). `subtitler job` runs it over titles that have no
package yet, from their original files, on one GPU beside live playback
([The run](#the-run)).

Nothing here touches a package or an original: the commands write into a
folder of their own and refuse one inside the package. Attaching what they
made to a title is the catalog's business, through its attach endpoint;
`subtitler attach` sends the text copies of image subtitles there, each
checked first against what the title plays and has.

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
- **Image subtitles: Tesseract 5.5.4** (Apache-2.0) on Leptonica 1.87.0
  (BSD-2-Clause), its LSTM engine line by line, with the `tessdata_fast`
  4.1.0 models and `tessdata_best` for Chinese (Apache-2.0); PGS and VobSub
  decoded here, in Python and numpy.

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
| `src/subtitler/pgs.py`, `vobsub.py` | image subtitles: PGS (`.sup`) and VobSub (`.idx` + `.sub`) decoded into what they show, when |
| `src/subtitler/ocr.py` | their text copies: pictures prepared, read by Tesseract, lines, italics, confidence, cues, labels |
| `src/subtitler/recognize.py` | `subtitler ocr`: the text copies of packaged titles, title by title, paced |
| `src/subtitler/attach.py` | `subtitler attach`: a run's text copies attached through the catalog's seam, each checked against the version the title plays and the subtitles it has |
| `src/subtitler/cli.py` | `subtitler plan` and `subtitler run` |
| `bench/` | the local measurements (`asr_run`, `asr_score`, `mt_run`, `mt_score`, `comet_score`, `opencc_demo`, `samples`, `summary`, `ocr_score`) and the GPU run (`gpu_bench`) |
| `src/subtitler/job.py`, `original.py`, `transcribe.py`, `qc.py`, `models.py` | `subtitler job`: titles from their originals, the checks, the models |
| `Dockerfile` | the image of `subtitler job` and `subtitler ocr` |
| `deploy/` | `job.yaml` (a run, placeholders); `ocr-job.yaml` and `ocr-titles.sql` (the image subtitles' backlog); the spike's GPU bench: `Dockerfile`, `bench-job.yaml`, `run-gpu-bench.sh` |
| `tests/` | the decision logic, the text handling and the image subtitles, built in the tests (`synth.py`) (`pytest`) |

## The source's own first

A title with no package plays from its original file, and that file often
carries subtitles people made. `subtitler extract` takes the original's
subtitle streams in the instance's languages out as WebVTT, one file a
stream (SDH and forced tracks each their own, so they keep what they are),
on the CPU, in one read of the file; commentary is left out, and Chinese is
told apart by script (the track title, else its characters). A PGS stream
of a language and kind no text stream is gets its text copy (OCR, below):
its pictures are copied out in a read of the file of their own, after the
text streams', so that a stream ffmpeg cannot copy out costs the text
streams nothing; VobSub and DVB streams of an original are listed as left
out. Each file's entry in `extracted.json` says what attaching it takes:
origin `source`, its language, label, SDH and forced; the text copies are
listed apart (`recognized`, origin `recognized`). Nothing is attached by
the tool.

```sh
subtitler extract --titles titles.json --out /work --languages en,de,zh-Hans --bundle
```

`subtitler job` makes nothing for a language the original has a subtitle of
its own in (a text or image stream, not forced, not commentary: extract
makes an image stream's text copy), and nothing at all for a title that has
one in every language.

## Image subtitles: a text copy (OCR)

Every app is to show a title's subtitles, and an image subtitle (PGS from a
Blu-ray, VobSub from a DVD) is pictures of its lines that some clients draw
and others cannot. So right after the source's own text comes a text copy of
each image subtitle in one of the instance's languages: its lines read by
Tesseract with the model of the track's language, written as WebVTT with
the pictures' times. The copy is the title's own subtitle in text: it counts
as present, and it is a source for the steps after it (a conversion, a
translation), as the source's own text subtitles are.

Which tracks (`plan.copies`): an image subtitle in one of the languages
(Chinese of either script, by the track title, else by reading it; not
written Cantonese), not commentary, of a kind (full, SDH, forced; a track
far sparser than the title's fullest, signs and songs, is a kind of its own)
that the title has no text subtitle of in that language; of several of a
language and kind, the fullest. A forced one is copied where a full text
subtitle of its language is there: it is a kind of its own.

How (`ocr.py`):

1. the pictures (`pgs.py`, `vobsub.py`): each display set's objects, shown
   from its time to that of the set that clears or replaces it; sets that
   show the same again (acquisition points, a fade's palette updates) are
   one picture, read where it is most opaque;
2. each picture laid onto black by its alpha, objects that stand apart (a
   line at the top, one at the bottom) one by one, the upper first, and
   each line on its own (the runs of rows with ink; an accent goes with its
   line, two lines that touch are cut where they have the least ink);
   inverted to dark on light, scaled to about 36 pixels high (a DVD's
   doubled, a Blu-ray's kept), thresholded between the letters' fill and
   their outline (Otsu over the pixels drawn), a margin;
3. Tesseract's LSTM engine reads all lines of a track, each as a single
   line (`--psm 7`; read as blocks of lines, `--psm 6`, Chinese lines lost
   parts, `--psm 13` read worse), a process a core: words with their
   confidence; a line read less surely than 60 is read again in grey
   (Tesseract thresholding it), the surer reading kept; a pipe read for a
   capital I, a lone l for I in English, typographic apostrophes, a lone
   double quote before ? or !, and in French a lone I after a space for "!"
   and a capital Ç within a sentence are put right; an ellipsis at a line's
   start or end is told by the line's pixels (Tesseract leaves it out or
   reads it short); Chinese words are joined without a space unless the
   picture has a wide gap there;
4. a line whose strokes lean 7° or more (the shear that makes its columns
   of ink sharpest) is set in `<i>`;
5. a cue a picture; pictures in a row that read the same are one cue; a cue
   of one or two characters read under a confidence of 50 is left out (what
   a graphic or a stray mark reads as), and a `#` at a line's edge is the
   music note it stands for (Tesseract reads ♪ so).

A copy is not attached, its file written aside
(`not-attached-<track>.<file>`) and the report saying why, and its
language's next candidate makes the language's subtitle, where

- its confidence (Tesseract's, over its characters) is under 80;
- more than a tenth of its pictures read as nothing, as such a stray mark,
  or under 50 (graphics, a font Tesseract does not read; the 45 tracks
  measured below: 2 % at most);
- it reads as another language than its track's, as a track tagged with
  the wrong language reads almost as surely with that language's model
  (German read with the English model at 90): Chinese whose letters are
  under half Chinese characters (a twentieth for a bilingual track); a
  language with a list of its commonest words (English, German, French,
  Spanish, Italian, Dutch, Swedish, Danish, Norwegian, Finnish, Polish,
  Portuguese) whose words another's outnumber in it. Per copy of the tracks
  measured, its own language's are 21-50 % of its words, any other's 25 %
  at most; Danish and Norwegian (35-39 % of each other's) are not told
  apart;
- a copy of its language and kind is made already (two Chinese tracks read
  as the same script: the first one's is kept).

A language without a Tesseract model is not read, and said so; a model the
image does not have is fetched into `--models` (tessdata_fast 4.1.0;
tessdata_best for Chinese) and checked against its SHA-256, which this
repository keeps for every model a language is read with. A track whose
picture claims more than its video's size (or 4096 pixels on a side) is
not decoded.

The label is the track's own name and a plain mark, as a made subtitle's
says how it was made ("German · AI"): "English (Forced) · OCR", or the
language's where the track has none ("English · OCR", "English · SDH ·
OCR"); words of the image format ("PGS") are left out, the copy is text. A
name longer than the catalog keeps (120 characters) is cut, and the mark
kept. Its origin is `recognized` (the catalog's attach seam takes it once
katalog-manager lists it among its origins beside `source`, `provider`,
`transcribed`, `translated` and `converted`); forced and SDH are carried
(`<tag>.forced.vtt`, `<tag>.sdh.vtt`). A copy of a track of signs and songs
(`<tag>.signs.vtt`) is attached as forced, its label saying so ("English ·
Signs · OCR"), which is how the catalog's rows tell it from a forced track.
The NOTE of its file says which track it was read from, by which Tesseract
and model, and with what confidence.

```sh
subtitler ocr --titles titles.json --out /work --languages en,de,zh-Hans --hours 22-17 --read-mbps 20 [--dry-run]
```

```json
[{"id": "ITEM_ID", "name": "S01E06", "package": "/media/PATH/versions/VERSION_ID", "version": "VERSION_ID"}]
```

`subtitler ocr` does it for packaged titles, title by title, on the CPU:
`<out>/<id>/<tag>[.sdh|.forced|.signs].vtt` and `ocr.json` (the version the
copies were read from, each file with what attaching it takes, the copies
not attached with why). It reads the store only within `--hours` (UTC; by
default never 17-22, the evening's playback), each image track of the
languages once and at most `--read-mbps`, a pause between titles; a title
done in a run before is not read again (`--again` reads it), so a run cut
short goes on where it stopped. A title is marked while it is read: one a
run was killed in without warning (out of memory) is recorded as failed by
the next run and left out; a run stopped by SIGTERM (its deadline) takes
the mark away first. `--dry-run` reads package.json alone and says what it
would read. `subtitler run` makes a package's copies first and then the
rest from them; `subtitler extract` copies an original's PGS streams in a
read of their own after its text streams' (`--no-ocr` leaves them out, and
`subtitler plan --no-ocr` plans as before OCR was).

```sh
subtitler attach --out /work --url https://KATALOG_MANAGER --token-file /secrets/token [--dry-run]
```

`subtitler attach` attaches a run's copies through katalog-manager's seam
(origin `recognized`): it asks the seam first, and a copy is not attached,
with why, where the title now plays another version than the one it was
read from, or has a subtitle of the copy's language and kind that covers it
(a copy attached before, or a text subtitle of its own, from its package,
its file or a provider; the kind told as the catalog's rows tell it).
`<out>/<id>/attach.json` says what was attached; a copy attached before is
not sent again.

### How well it reads

Measured against the titles' own text tracks of the same subtitle: 45 PGS
tracks of an instance's titles that carry their language as a text track too,
the two labelled as one subtitle ("English PGS" and "English SRT", "Français
SUP" and "Français SRT", the same label twice), picked at random per
language, 39 in Latin scripts (one of them forced) and 6 Chinese
(`bench/ocr_score.py`, the copies read as `subtitler ocr` reads them); the
files read from the store at night, `cat` alone. Cues pair by time; where
the text track is the same subtitle, made from the image one, its cues keep
the image's times, so the character error rate counts the pairs whose start
and end agree to 50 ms (the track's own shift aside), an SDH track's
annotations and speakers' names and the dialogue dashes out of both (a text
track of the dialogue labelled as the SDH image one, a dash spaced or not)
and the typography (… ’ “ –) as one. A track of another version (its own
translation, cues cut otherwise) shows as few such pairs and does not count.

| language | tracks | of the same subtitle | cues compared | CER | case and spaces aside | median track | confidence |
|---|---|---|---|---|---|---|---|
| English | 12 | 11 | 9,853 | 1.01 % | 0.52 % | 1.17 % | 95.1 |
| German | 4 | 4 | 3,502 | 0.07 % | 0.06 % | 0.06 % | 95.1 |
| Spanish | 3 | 3 | 2,072 | 0.11 % | 0.07 % | 0.08 % | 95.4 |
| French | 3 | 3 | 2,938 | 0.17 % | 0.11 % | 0.18 % | 93.9 |
| Italian | 3 | 2 | 1,519 | 0.06 % | 0.05 % | 0.03 % | 94.8 |
| Dutch | 3 | 2 | 976 | 0.95 % | 0.82 % | 0.97 % | 92.8 |
| Swedish | 2 | 2 | 1,029 | 0.02 % | 0.02 % | 0.02 % | 95.3 |
| Danish | 2 | 1 | 512 | 0.22 % | 0.15 % | 0.22 % | 95.2 |
| Norwegian | 2 | 2 | 1,382 | 0.08 % | 0.07 % | 0.10 % | 95.4 |
| Finnish | 2 | 1 | 531 | 0.13 % | 0.06 % | 0.13 % | 94.9 |
| Polish | 2 | 2 | 2,162 | 0.12 % | 0.06 % | 0.13 % | 94.8 |
| English, forced | 1 | 1 | 32 | 0.52 % | 0.13 % | 0.52 % | 95.4 |
| **Latin scripts** | 39 | 34 | 26,508 | **0.49 %** | 0.27 % | 0.14 % | |

Chinese apart (Simplified with `chi_sim`, Traditional and written Cantonese
with `chi_tra`, both from tessdata_best):

| language | tracks | of the same subtitle | cues compared | CER | case and spaces aside | median track | confidence |
|---|---|---|---|---|---|---|---|
| Chinese (Simplified) | 2 | 2 | 3,137 | 3.96 % | 3.96 % | 3.91 % | 86.6 |
| Chinese (Traditional) | 3 | 2 | 3,171 | 4.30 % | 4.30 % | 4.11 % | 87.5 |
| Cantonese (Traditional characters) | 1 | 1 | 1,558 | 7.22 % | 7.22 % | 7.22 % | 85.8 |

What the error rate still counts is mostly the text tracks' own. Where a
person typed or corrected a text track from the pictures, it differs from
them (names in lower case, an interruption's `--` written `...`, a word's
case changed, speakers' names added), and the copy, which reads what the
picture shows, counts as wrong: English carries the most of it (on the three
English tracks whose text is the pictures' as they stand, 0.04-0.12 %), and
one Dutch text track is a version re-worded at the same times. The copies'
own errors, read where the two differ: a lone l for I or the other way
round, a capital after an ellipsis, an l read before an exclamation mark, a
space lost in a line of capitals; in Chinese, characters read for others of
a similar shape, and the colloquial characters of the Cantonese track. Line
breaks are the pictures' (a text track re-wrapped by hand accounts for the
cues whose lines differ).

Italics, on the 22 tracks whose text track marks them (2,797 of 24,168
cues): the copies set 95.4 % of them in italics, and 97.2 % of the cues
they set in italics are the text track's; the slant's threshold matters
little (5° to 10°: within a point).

The confidence follows the errors: per cue, over every pair (other
versions among them), 42 % of the characters of cues under 50 differ, 18 %
of those of 50-70, 14 % of 70-85, 10 % of 85-90, 3 % of 90 and more. Per
track it is 92-96 in the Latin scripts and 86-88 in Chinese: the gate of
80 passes every one measured.

A Chinese track whose title names no script is read on 40 of its pictures
with both models together (`chi_sim+chi_tra`: each character read by the
model that reads it better), and the script of the characters read tells
it (OpenCC: how many of them a conversion into the other script changes):
right for 24 samples of 24, four of each of the six tracks (the Cantonese
one, written in Traditional characters, Traditional). The two models'
confidence alone, read one after the other, picked the wrong script for a
sample in eight.

VobSub: five of the library's VobSub tracks in the instance's languages
(of four films; the library has 32 in all) decoded and read, with no text
track of the same subtitle to check them by: at a confidence of 94.3-94.7
(English, German, a German forced track of three pictures) and 89.6
(Chinese, read as Simplified). One English VobSub track against the same
film's English PGS track, the same subtitle in another release, both read:
1,826 of 1,828 cues pair, and the two copies differ in 0.16 % of their
characters.

### What it costs

Measured in short runs at normal priority on the spike's machine (Apple M5,
three processes; the bulk ran at background priority, whose times are not
quoted): 15-17 CPU seconds per 1,000 pictures with the fast models, 21-29
for Chinese with the best ones, of which decoding takes 5-11 %, preparing
the lines 2-8 % and Tesseract the rest. That is 7-16 CPU seconds per hour
of a track's runtime: a two-hour film's full track 24-34 CPU seconds, 10-14
seconds of wall time on three processes. The image grows by 38 MB
(Tesseract and its models, "The run" below).

### The backlog

Planned from the catalog (`deploy/ocr-titles.sql`, read-only; not run). It
reads the catalog's rows as `subtitler ocr` reads package.json: a
language by its primary subtag (`ger`, `deu` and `de` are German; a copy
attached as `zh-Hans` is Chinese), Chinese of the script its tag or label
names; the kind by the flags and the label (signs, forced, SDH, full); the
image subtitles of the version a title plays; a text subtitle of the same
language and kind covering one, the package's own, one attached from the
title's file or a provider, or a copy a run before made (one a model made
covers none). A text copy is due for 3,594 image subtitles of 2,409 titles
(2,174 episodes, 235 films), 1,782 hours of runtime, 2,861 hours of tracks;
among them 1,808 of the 1,828 titles whose only subtitles are images (the
other 20 have none in a language of the instance that a version they play
holds).

| language | full | SDH | forced | signs | of them VobSub |
|---|---|---|---|---|---|
| English | 1,085 | 1,187 | 169 | 208 | 3 |
| German | 777 | 28 | 8 | | 2 |
| Chinese | 132 (23 Simplified, 47 Traditional, 62 read to tell) | | | | 2 |

What it would take, from the sample's 722 pictures and 15.3 MB of PGS an
hour of a track's runtime and the cost above: about 2.1 million pictures,
about 10 CPU hours at the spike machine's speed (16.6 CPU seconds per 1,000
pictures, Chinese 21); the cluster's cores are not measured: at half that
speed 20 CPU hours, about 7 hours on three cores, one night within the
hours it may read (22:00 to 17:00 UTC, 19 hours a day). 44 GB read from
the store, each track once, at 20 MB/s at most, which is 36 minutes of
reading spread over the run (under 2 MB/s on average); nothing written to
it.

The run, when the owner says go. `titles.json` from the query, newest first
(ADR-0014: most-watched and newest first; the watch counts are the
player's, to order the list by before the run), read-only and quiet, so
that the output is the JSON alone:

```sh
PGOPTIONS='-c default_transaction_read_only=on' \
  psql -X -q -At -v ON_ERROR_STOP=1 -v langs="'en','de','zh'" -f deploy/ocr-titles.sql katalog > titles.json
oc -n NAMESPACE create configmap subtitler-ocr-titles --from-file=titles.json
```

The backlog's list is some 640 kB, over the 256 KiB a client-side
`oc apply` keeps in its annotation: the ConfigMap is made from the file
(`create --from-file`, or `... --dry-run=client -o yaml | oc apply
--server-side -f -` to replace one; a ConfigMap holds up to 1 MiB). Then
`subtitler ocr --dry-run` over it, then `deploy/ocr-job.yaml` (the image,
CPU only, 3 cores; the media volume read-only, a work claim for the
copies), which reads nothing from 17:00 to 22:00 UTC, each track once at 20
MB/s at most, and goes on where a run before stopped. The copies are
attached by `subtitler attach` over the work claim (origin `recognized`,
which katalog-manager takes from the release that lists it; `--dry-run`
first), and a later GPU run translates from them where a language lacks a
subtitle.

## The run

`subtitler job` takes a list of titles with no subtitle at all and makes
one per language for each, in the list's order:

```sh
subtitler job --titles titles.json --out /work --models /models --languages en,de,zh-Hans \
  --live-metrics http://STREAM_SERVICE/metrics --bundle
```

```json
[{"id": "ITEM_ID", "name": "S01E06", "original": "/media/PATH/original.mkv", "language": "en"}]
```

Per title, one model on the GPU at a time:

1. the original's main audio (`original.py`): the stream in the title's
   language that is not commentary or audio description, the default one
   first; a surround track mixed down centre first, where dialogue sits;
   read once, at most `--readrate` times real time, a title ahead of the GPU;
2. speech recognition by Whisper large-v3 (batched, Silero VAD) in a process
   of its own (`transcribe.py`), which gives the GPU memory back when it
   ends; the transcript is the subtitle in the audio's language (origin
   `transcribed`);
3. translation of the transcript by Qwen3.5-9B behind llama.cpp's server,
   `--chains` parts of the title at once, whole sentences placed back over
   their cues, German asked to condense lines short of time (origin
   `translated`);
4. the checks (`qc.py`): the speech the cues cover, cues with no speech
   under them, speech no cue covers, the transcript against a second model
   (large-v3-turbo) on ten windows, and per language the reading speed,
   line limits, overlaps, empty and untranslated cues; and samples spread
   over the title, every language side by side, for a person to read.

Out: `<out>/<id>/<tag>.vtt` (a NOTE says how it was made and by which model),
`subtitles.json` (label such as "German · AI", origin, model, the GPU
seconds of every step) and `qc.json`. `--bundle` writes them to the log too.

Live playback comes first (ADR-0013): before each GPU step and each
translation request the run reads the live worker's queue from the stream
service's metrics and waits while segments a viewer waits for are being
made (`--live-wait` seconds at most at a time), and it starts a model only
when the GPU has its memory free plus `--vram-reserve` MiB for the live
worker.

The image (`Dockerfile`, built by CI as `ghcr.io/zaentrum/subtitler`) holds
llama.cpp's release build for CUDA 12.8, FFmpeg 7.1, Tesseract and the
Python side; no speech or translation model. `deploy/job.yaml` is a Job for
it with placeholders. Tesseract 5.5.4 and Leptonica 1.87.0 are built from
their releases in a stage of their own (40 s on six cores), for PNM in and
TSV out: 6.7 MB that need libstdc++ and zlib alone, and 31.7 MB of models
(`eng` and `deu` from tessdata_fast, 5.6 MB; `chi_sim` and `chi_tra` from
tessdata_best, 26 MB), 38 MB in all. Ubuntu's package would add 97 MB
(ICU, Pango, Cairo, fonts, curl) and, through libtiff, libjbig0 under the
GPL-2.0.

## The decision per language

For each language of `--languages` (the instance's `subtitles.languages`,
BCP 47 with a script where it matters, default `en,de,zh-Hans`), after the
text copies of its image subtitles (above):

1. **keep**: the package has a complete text subtitle in it (and in its
   script);
2. **recognize**: the text copy of a complete image subtitle in it (with
   `--no-ocr`: a PGS track counts as kept, as the web client draws it);
3. **convert**: the same language in the other script as text, a copy's
   too: OpenCC, `t2s` (`hk2s` for Hong Kong tracks), no model;
4. **translate**: the best complete *text* subtitle in another language,
   copies among them: the original language first, then English, then the
   instance's languages, then widely translated ones; dialogue before SDH,
   the fuller track first, a text subtitle before a copy of the same rank;
5. **transcribe**: speech recognition of the main audio: the subtitle when
   the audio is in the language, else the source of 4. It is the last
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
the label ("German · AI", "Chinese (Simplified) · Converted", "English ·
OCR"), the origin (`recognized`, `transcribed`, `translated`, `converted`),
the source track and language, the model, timings and tokens; the copies
not attached, with why. A candidate that cannot be made (a copy read too
unsurely, one read in the other script) leaves the language to the next.

## Running it locally

```sh
uv venv -p 3.12 .venv && . .venv/bin/activate
uv pip install -e '.[asr,mt,eval,zh,dev]'       # mlx-whisper too on a Mac: '.[mlx]'
pytest

# The text copies of image subtitles need Tesseract 5 (brew install tesseract: English
# alone; other languages' tessdata_fast models in a folder given as --tessdata, or
# fetched into --models):
subtitler ocr --titles titles.json --out /path/to/scratch --hours 0-24 --read-mbps 0

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

## The GPU bench

The spike's bench, prepared, not run: it needs the platform owner's go. Like the live-rung
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
| image subtitles | tessdata_fast 4.1.0 (`eng`, `deu` in the image; others fetched), tessdata_best 4.1.0 (`chi_sim`, `chi_tra`) | Apache-2.0 | Tesseract 5.5.4 (Apache-2.0) on Leptonica 1.87.0 (BSD-2-Clause), built from source |
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
  tracks are 38 % of the sampled subtitle tracks (15 % of the catalog's
  subtitle rows): their text copies (OCR) make them a source.
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
- **Image subtitles read back nearly as written.** Against text tracks of
  the same subtitle, 0.49 % of the characters in the Latin scripts (the
  median track 0.14 %), most of it the text tracks' own edits; Chinese about
  4 % with tessdata_best (5.6 % with tessdata_fast on a sample). Read line
  by line: as blocks of lines, Tesseract left parts of Chinese lines out;
  and it reads an ellipsis at a line's edge short or not at all, which the
  pixels tell instead.

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
