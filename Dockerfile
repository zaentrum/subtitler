# The subtitler image: speech recognition (faster-whisper on CTranslate2,
# CUDA), translation (llama.cpp's server, CUDA), FFmpeg for the audio of
# original files, and Tesseract for the text of image subtitles (OCR, on the
# CPU). A run executes `subtitler job` (src/subtitler/job.py) or `subtitler
# ocr` (src/subtitler/recognize.py).
#
#   docker build -t IMAGE .
#
# No speech or translation model is in the image: a run fetches what its
# model folder lacks from Hugging Face (src/subtitler/models.py, about
# 11 GB), so a volume kept between runs saves the download. Tesseract's
# models of the default languages are in it (32 MB).
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

# Tesseract (Apache-2.0) on Leptonica (BSD-2-Clause), built from their
# releases, checked against their digests, for what the OCR step asks of
# them: PNM pictures in, TSV out. Leptonica without image libraries but
# zlib, Tesseract without its training tools, curl, libarchive, TIFF and
# OpenMP (the step runs a process a core): it needs libstdc++ and zlib alone.
# Ubuntu's own package would bring about 90 MB and, through libtiff,
# libjbig0 (GPL-2+); this build is 7 MB. ENABLE_NATIVE off: the SIMD code (AVX2, FMA,
# SSE4.1) is chosen when it runs, not by the machine it was built on. The
# models (Apache-2.0) of the default languages: English and German from
# tessdata_fast 4.1.0, the ones Ubuntu packages; Chinese from tessdata_best
# 4.1.0, which reads it with a quarter fewer errors (26 MB of the 32). A run
# fetches another language's into --models.
FROM ${BASE}ubuntu:24.04 AS tesseract
ARG LEPTONICA_VERSION=1.87.0
ARG LEPTONICA_SHA256=c73363397f96eb1295602bf44d708a994ad42046c791bf03ea0505d829bdb6a7
ARG TESSERACT_VERSION=5.5.4
ARG TESSERACT_SHA256=a8dbf8b8c8ee427ff6ad1d99116502feaf59440af1adb893caf60a6576826cf4
ARG TESSDATA_TAG=4.1.0
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake ca-certificates curl zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://github.com/DanBloomberg/leptonica/releases/download/${LEPTONICA_VERSION}/leptonica-${LEPTONICA_VERSION}.tar.gz" -o /tmp/leptonica.tar.gz \
    && echo "${LEPTONICA_SHA256}  /tmp/leptonica.tar.gz" | sha256sum -c - \
    && curl -fsSL "https://github.com/tesseract-ocr/tesseract/archive/refs/tags/${TESSERACT_VERSION}.tar.gz" -o /tmp/tesseract.tar.gz \
    && echo "${TESSERACT_SHA256}  /tmp/tesseract.tar.gz" | sha256sum -c - \
    && mkdir -p /src/leptonica /src/tesseract \
    && tar -xzf /tmp/leptonica.tar.gz -C /src/leptonica --strip-components=1 \
    && tar -xzf /tmp/tesseract.tar.gz -C /src/tesseract --strip-components=1 \
    && cmake -S /src/leptonica -B /build/leptonica -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/tesseract \
         -DBUILD_SHARED_LIBS=ON -DBUILD_PROG=OFF -DSW_BUILD=OFF -DENABLE_ZLIB=ON -DENABLE_PNG=OFF -DENABLE_GIF=OFF \
         -DENABLE_JPEG=OFF -DENABLE_TIFF=OFF -DENABLE_WEBP=OFF -DENABLE_OPENJPEG=OFF \
    && cmake --build /build/leptonica -j"$(nproc)" && cmake --install /build/leptonica \
    && cmake -S /src/tesseract -B /build/tesseract -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/tesseract \
         -DCMAKE_PREFIX_PATH=/opt/tesseract -DBUILD_SHARED_LIBS=ON -DBUILD_TRAINING_TOOLS=OFF -DBUILD_TESTS=OFF \
         -DGRAPHICS_DISABLED=ON -DDISABLE_ARCHIVE=ON -DDISABLE_CURL=ON -DDISABLE_TIFF=ON -DOPENMP_BUILD=OFF \
         -DENABLE_NATIVE=OFF -DSW_BUILD=OFF -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
    && cmake --build /build/tesseract -j"$(nproc)" && cmake --install /build/tesseract \
    && rm -rf /opt/tesseract/include /opt/tesseract/lib/pkgconfig /opt/tesseract/lib/cmake \
         /opt/tesseract/share/tessdata/configs /opt/tesseract/share/tessdata/tessconfigs \
    && strip --strip-unneeded /opt/tesseract/bin/tesseract /opt/tesseract/lib/libtesseract.so.* /opt/tesseract/lib/libleptonica.so.* \
    && cd /opt/tesseract/share/tessdata \
    && for m in fast:eng:7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2 \
                fast:deu:19d219bbb6672c869d20a9636c6816a81eb9a71796cb93ebe0cb1530e2cdb22d \
                best:chi_sim:4fef2d1306c8e87616d4d3e4c6c67faf5d44be3342290cf8f2f0f6e3aa7e735b \
                best:chi_tra:1aa60488574cafa69486d919284f079ca9b68fcc7f6ad8dc1ff1b318dfd97028; do \
         kind=${m%%:*}; rest=${m#*:}; name=${rest%%:*}; sum=${rest#*:}; \
         curl -fsSL "https://github.com/tesseract-ocr/tessdata_${kind}/raw/${TESSDATA_TAG}/${name}.traineddata" -o "${name}.traineddata" \
         && echo "${sum}  ${name}.traineddata" | sha256sum -c - || exit 1; done

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
COPY --from=tesseract /opt/tesseract /opt/tesseract
ENV PATH=/opt/venv/bin:/opt/llama:/opt/ffmpeg/bin:/opt/tesseract/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/ffmpeg/lib \
    TESSDATA_PREFIX=/opt/tesseract/share/tessdata \
    HOME=/tmp \
    HF_HOME=/models/hf
# Everything loads without a GPU (llama.cpp's CUDA backend is loaded when a
# GPU is there): a broken image fails here, not on the GPU node.
RUN llama-server --version \
    && ffmpeg -hide_banner -version | head -n 1 \
    && tesseract --version | head -n 2 && tesseract --list-langs \
    && python3 -c "import faster_whisper, ctranslate2, opencc, jieba, huggingface_hub; print('faster-whisper', faster_whisper.__version__, 'CTranslate2', ctranslate2.__version__)" \
    && subtitler job --help > /dev/null && subtitler ocr --help > /dev/null
# OpenShift runs containers under a random uid in group 0; a numeric user
# needs no passwd entry.
USER 1001
WORKDIR /tmp
ENTRYPOINT ["subtitler"]
