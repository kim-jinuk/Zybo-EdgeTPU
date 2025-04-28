# Build full runtime + dev image for x86_64 (desktop) *and* ARM (cross‑build).
# On ARM host (e.g. Jetson / Raspberry Pi / Zybo rootfs chroot) it runs natively.
# On x86 host use `docker buildx build --platform linux/arm64` for cross‑build.

# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ubuntu:20.04
FROM ${BASE_IMAGE}

# 1) system deps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 python3-pip python3-venv python3-dev build-essential cmake ninja-build \
    pkg-config libopencv-dev wget curl unzip ca-certificates \
    libedgetpu1-std \
    && rm -rf /var/lib/apt/lists/*

# 2) python deps
WORKDIR /workspace
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -U pip \
    && python3 -m pip install --no-cache-dir -r requirements.txt

# 3) project source
COPY . /workspace

# 4) optional: build C++ helper lib (skip if not needed)
RUN mkdir -p build && cd build \
    && cmake -GNinja .. \
    && ninja && ninja install \
    && cd .. && rm -rf build

# 5) entrypoint
ENV PYTHONPATH=/workspace/src:${PYTHONPATH}
WORKDIR /workspace
CMD ["python", "scripts/run_pipeline.py", "--cfg", "config/pipeline.yaml", "--source", "0"]