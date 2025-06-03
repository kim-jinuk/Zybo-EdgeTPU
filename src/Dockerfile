FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config \
    wget curl git \
    libopencv-dev \
    sudo

# Coral EdgeTPU 설치 (TPU 연결 코드 지원)
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" > /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y libedgetpu1-std libedgetpu-dev

# TensorFlow Lite C++ 직접 빌드 (v2.14.0)
RUN git clone --depth 1 --branch v2.14.0 https://github.com/tensorflow/tensorflow.git /tmp/tensorflow && \
    cd /tmp/tensorflow/tensorflow/lite && \
    mkdir build && cd build && \
    cmake .. && make -j$(nproc) && \
    make install && \
    cd / && rm -rf /tmp/tensorflow

# 환경변수/라이브러리 경로
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
WORKDIR /workspace
COPY . .

RUN mkdir -p cpp/build && cd cpp/build && cmake .. && make -j$(nproc)
