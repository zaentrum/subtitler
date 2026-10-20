#!/bin/sh
# run-gpu-bench - the GPU run of the subtitler spike, as a Job's entrypoint.
# Settings come from the environment (see bench/gpu_bench.py --help):
#   SOURCES       sources file (default /etc/bench/sources.json)
#   MODELS_DIR    model folder, a volume (default /models)
#   PHASES        asr,mt,busy (default all three)
#   ASR_MINUTES   minutes of each source's audio (default 0 = all of it)
#   ASR_MODELS    speech models (default large-v3-turbo,large-v3)
#   ASR_BATCH     batched inference batch size (default 16)
#   LLMS          GGUF models (default qwen3.5-9b-q4km,qwen2.5-7b-q4km)
#   CHAINS        translation chains in flight (default 1,4,8)
#   CUE_LIMIT     cues per source track translated (default 300)
#   MT_SOURCES    sources translated (default 4)
#   BUSY_STREAMS  NVENC streams of the busy load (default 4)
#   BUSY_VIDEO    the busy load's video playlist (default the first source's hls/v0)
#   BUSY_SOURCES  sources of the busy phase (default 3)
#   OUT           report folder (default /tmp/report)
set -eu
exec python3 /opt/subtitler/bench/gpu_bench.py "$@"
