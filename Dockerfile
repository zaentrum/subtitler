# The subtitler image: speech recognition (faster-whisper on CTranslate2,
# CUDA), translation (llama.cpp's server, CUDA) and FFmpeg for the audio of
# original files. A run executes `subtitler job` (src/subtitler/job.py).
#
#   docker build -t IMAGE .
#
# No model is in the image: a run fetches what its model folder lacks from
# Hugging Face (src/subtitler/models.py, about 11 GB), so a volume kept
# between runs saves the download.
#
# Base-image registry prefix. Empty default = public Docker Hub; a private
# mirror passes --build-arg BASE=registry.example/library/.
ARG BASE=

FROM ${BASE}nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS fetch
# llama.cpp's own release build for Linux and CUDA 12.8 (built on Ubuntu
# 24.04; real code for sm_86 and sm_89, PTX for the others), checked
# against the release's digest. Its binaries find their libraries next to
# them ($ORIGIN).
ARG LLAMA_CPP_TAG=b11430
ARG LLAMA_CPP_SHA256=8f79f75093e8fb6d9c65e59167511e9547776794191073e488ce81efd5ade723
# FFmpeg: BtbN's GPL build of the 7.1 release branch (n7.1.5, shared),
# checked against its checksum. It decodes the originals' audio.
ARG FFMPEG_BUILD_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-07-31-14-10/ffmpeg-n7.1.5-12-g1fdbca85aa-linux64-gpl-shared-7.1.tar.xz
ARG FFMPEG_SHA256=df7e15a2d2fe0ee15ae36e0e3b83596dd088ee0a0767874241d9a54380df0add
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_CPP_TAG}/llama-${LLAMA_CPP_TAG}-bin-ubuntu-cuda-12.8-x64.tar.gz" -o /tmp/llama.tar.gz \
    && echo "${LLAMA_CPP_SHA256}  /tmp/llama.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/llama \
    && tar -xzf /tmp/llama.tar.gz -C /opt/llama --strip-components=1 \
    && rm -f /tmp/llama.tar.gz \
    && curl -fsSL "${FFMPEG_BUILD_URL}" -o /tmp/ffmpeg.tar.xz \
    && echo "${FFMPEG_SHA256}  /tmp/ffmpeg.tar.xz" | sha256sum -c - \
    && mkdir -p /opt/ffmpeg \
    && tar -xJf /tmp/ffmpeg.tar.xz -C /opt/ffmpeg --strip-components=1 \
    && rm -rf /tmp/ffmpeg.tar.xz /opt/ffmpeg/doc /opt/ffmpeg/man /opt/ffmpeg/include /opt/ffmpeg/bin/ffplay

FROM ${BASE}nvidia/cuda:12.8.1-runtime-ubuntu24.04
# compute for CUDA, utility for nvidia-smi (the run's GPU memory checks).
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1
# libgomp1 for llama.cpp's CPU backends, libssl3t64 for its server. The
# runtime image has CUDA 12.8's cuBLAS, which CTranslate2 and llama.cpp load.
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv libgomp1 libssl3t64 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-image.txt pyproject.toml README.md LICENSE /opt/subtitler/
COPY src /opt/subtitler/src
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -c /opt/subtitler/requirements-image.txt "/opt/subtitler[asr,zh]"
COPY --from=fetch /opt/llama /opt/llama
COPY --from=fetch /opt/ffmpeg/lib /opt/ffmpeg/lib
COPY --from=fetch /opt/ffmpeg/bin /opt/ffmpeg/bin
ENV PATH=/opt/venv/bin:/opt/llama:/opt/ffmpeg/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/ffmpeg/lib \
    HOME=/tmp \
    HF_HOME=/models/hf
# Everything loads without a GPU (llama.cpp's CUDA backend is loaded when a
# GPU is there): a broken image fails here, not on the GPU node.
RUN llama-server --version \
    && ffmpeg -hide_banner -version | head -n 1 \
    && python3 -c "import faster_whisper, ctranslate2, opencc, jieba, huggingface_hub; print('faster-whisper', faster_whisper.__version__, 'CTranslate2', ctranslate2.__version__)" \
    && subtitler job --help > /dev/null
# OpenShift runs containers under a random uid in group 0; a numeric user
# needs no passwd entry.
USER 1001
WORKDIR /tmp
ENTRYPOINT ["subtitler"]
