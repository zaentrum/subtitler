# The GPU run of the subtitler spike: speech recognition (faster-whisper on
# CTranslate2, CUDA), translation (llama.cpp's server, CUDA; MADLAD-400 on
# CTranslate2) and the bench that measures both, alone and next to
# live-worker-like load (NVDEC -> scale_cuda -> NVENC).
#
#   docker build -f deploy/Dockerfile -t IMAGE .
#
# Models are not in the image: the bench fetches them into MODELS_DIR (a
# volume) on its first run, about 19 GB.
#
# Base-image registry prefix. Empty default = public Docker Hub; a private
# deploy mirror passes --build-arg BASE=registry.example/library/ (it then
# needs the -devel tag below as well as the -runtime one).
ARG BASE=

# llama.cpp from a tagged release, built for the RTX 3090 (sm_86) on the
# runtime's own CUDA and Ubuntu, so the binaries need no newer CUDA or glibc
# than the runtime has. The flags follow llama.cpp's own CUDA image.
# Not built yet: llama.cpp's own CI builds with CUDA 12.4 and later. Should
# 12.3 fail, the release's prebuilt ubuntu-cuda-12.8 binaries run on this
# runtime with NVIDIA's CUDA 12 runtime wheel (nvidia-cuda-runtime-cu12) on
# LD_LIBRARY_PATH next to the cuBLAS one; driver 580 takes CUDA 12.x.
FROM ${BASE}nvidia/cuda:12.3.2-devel-ubuntu22.04 AS llama
ARG LLAMA_CPP_TAG=b11430
ARG CUDA_ARCHITECTURES=86
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 -b "${LLAMA_CPP_TAG}" https://github.com/ggml-org/llama.cpp /src \
    && cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
         -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
         -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
         -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build /build -j"$(nproc)" --target llama-server llama-bench \
    && mkdir -p /out && cp /build/bin/llama-server /build/bin/llama-bench /out/

# FFmpeg as the live-rung spike and the batch transcoder take it: BtbN's
# prebuilt GPL build of the 7.1 release branch (n7.1.5, shared), checked
# against the release's checksum. Here it decodes the packages' audio and
# makes the busy load (NVDEC, scale_cuda, NVENC) with the driver's 580 API.
FROM ${BASE}nvidia/cuda:12.3.2-runtime-ubuntu22.04 AS ffmpeg
ARG FFMPEG_BUILD_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-07-31-14-10/ffmpeg-n7.1.5-12-g1fdbca85aa-linux64-gpl-shared-7.1.tar.xz
ARG FFMPEG_SHA256=df7e15a2d2fe0ee15ae36e0e3b83596dd088ee0a0767874241d9a54380df0add
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "${FFMPEG_BUILD_URL}" -o /tmp/ffmpeg.tar.xz \
    && echo "${FFMPEG_SHA256}  /tmp/ffmpeg.tar.xz" | sha256sum -c - \
    && mkdir -p /opt/ffmpeg \
    && tar -xJf /tmp/ffmpeg.tar.xz -C /opt/ffmpeg --strip-components=1 \
    && rm -rf /tmp/ffmpeg.tar.xz /opt/ffmpeg/doc /opt/ffmpeg/man /opt/ffmpeg/include /opt/ffmpeg/bin/ffplay

FROM ${BASE}nvidia/cuda:12.3.2-runtime-ubuntu22.04
# compute for CUDA, video for NVDEC/NVENC (the busy load), utility for nvidia-smi.
ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# CTranslate2's CUDA wheels load cuBLAS 12 and cuDNN 9 at run time: NVIDIA's
# own wheels (the runtime image has no cuDNN). Ubuntu 22.04's Python 3.10.
COPY pyproject.toml README.md /opt/subtitler/
COPY src /opt/subtitler/src
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install "/opt/subtitler[asr,mt,eval]" "nvidia-cublas-cu12==12.*" "nvidia-cudnn-cu12==9.*"
COPY bench /opt/subtitler/bench
COPY --from=llama /out/ /usr/local/bin/
COPY --from=ffmpeg /opt/ffmpeg/lib /opt/ffmpeg/lib
COPY --from=ffmpeg /opt/ffmpeg/bin /opt/ffmpeg/bin
COPY deploy/run-gpu-bench.sh /usr/local/bin/run-gpu-bench
RUN chmod 0755 /usr/local/bin/run-gpu-bench
ENV PATH=/opt/venv/bin:/opt/ffmpeg/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.10/site-packages/nvidia/cudnn/lib:/opt/ffmpeg/lib \
    HOME=/tmp \
    HF_HOME=/models/hf \
    MODELS_DIR=/models
# OpenShift runs containers under a random uid in group 0; a numeric user
# needs no passwd entry (the CUDA runtime image has no useradd either).
USER 1001
WORKDIR /tmp
ENTRYPOINT ["/usr/local/bin/run-gpu-bench"]
