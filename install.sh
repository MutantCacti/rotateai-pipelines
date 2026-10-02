#!/bin/bash
set -e

# TFLite Micro, pinned. Bumping this changes every model binary, and the CI
# cache key (it hashes this file) with it.
TFLM_REF=18b9e6f2a8c5a9518e588f59c2ba16ef7ef9d551

if [ ! -d deps/tflite-micro ]; then
    git init -q deps/tflite-micro
    git -C deps/tflite-micro remote add origin https://github.com/tensorflow/tflite-micro.git
    git -C deps/tflite-micro fetch -q --depth 1 origin "$TFLM_REF"
    git -C deps/tflite-micro checkout -q FETCH_HEAD
fi

# No TARGET: TFLM detects the host, and naming another OS is a cross-compile.
make -C deps/tflite-micro -f tensorflow/lite/micro/tools/make/Makefile \
    -j"$(getconf _NPROCESSORS_ONLN)" microlite
