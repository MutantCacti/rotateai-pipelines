TFLM_DIR   = deps/tflite-micro
TFLM_DL    = $(TFLM_DIR)/tensorflow/lite/micro/tools/make/downloads
GEN_DIR    = build/models
MODEL      = models/TASK_NOT_spermwhale_Transformer_best.keras
PARAMS     = models/TASK_NOT_spermwhale_Transformer_preprocess_params.pkl
PYTHON    ?= python


# Host detection mirrors TFLM's own Makefile, which names its gen dir
# <os>_<arch>_default_gcc whatever the compiler actually is
ifeq ($(OS),Windows_NT)
  TFLM_OS  = windows
  EXE      = .exe
  LDFLAGS += -static
else ifeq ($(shell uname -s),Darwin)
  TFLM_OS  = osx
else
  TFLM_OS  = linux
  LDFLAGS += -static-libstdc++ -static-libgcc
endif
TFLM_GEN   = $(TFLM_DIR)/gen/$(TFLM_OS)_$(shell uname -m)_default_gcc
TFLM_LIB   = $(TFLM_GEN)/lib/libtensorflow-microlite.a


CXXFLAGS   = -std=c++17 -DTF_LITE_STATIC_MEMORY \
             -I$(TFLM_DIR) \
             -I$(TFLM_DL)/flatbuffers/include \
             -I$(TFLM_DL)/gemmlowp \
             -I$(TFLM_DL)/kissfft \
             -I$(TFLM_DL)/ruy \
             -I$(GEN_DIR) \
             -Isrc


# Pipelines

baseline: build/baseline$(EXE)
build/baseline$(EXE): src/baseline.cc src/pipeline.h src/protocol.h $(GEN_DIR)/model_data.inc $(TFLM_LIB)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) src/baseline.cc -o $@ $(TFLM_LIB) $(LDFLAGS)


variable: build/variable$(EXE)
build/variable$(EXE): src/variable.cc src/pipeline.h src/protocol.h $(GEN_DIR)/model_data.inc $(TFLM_LIB)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) src/variable.cc -o $@ $(TFLM_LIB) $(LDFLAGS)


surface: build/surface$(EXE)
build/surface$(EXE): src/surface.cc src/pipeline.h src/protocol.h $(GEN_DIR)/model_data.inc $(TFLM_LIB)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) src/surface.cc -o $@ $(TFLM_LIB) $(LDFLAGS)


# -ffp-contract=off stops compilers fusing multiply-adds (arm64 ones do by
# default). Output is still not bit identical across OSes: each libm can
# differ in the last bit of sin/cos/atan2, so compare with a tolerance
prhpredict: build/prhpredict$(EXE)
build/prhpredict$(EXE): src/prhpredict.cc src/protocol.h
	@mkdir -p build
	$(CXX) -std=c++17 -O2 -ffp-contract=off -Isrc src/prhpredict.cc -o $@ $(LDFLAGS)


# Model conversion, needs the Python environment. Not a prerequisite of the
# pipelines, so a build never runs TensorFlow on its own
models:
	$(PYTHON) tools/prepare_model.py --model $(MODEL) --params $(PARAMS) --out $(GEN_DIR)


# TFLite Micro static library
$(TFLM_LIB):
	./install.sh


clean:
	rm -rf build/


.PHONY: baseline variable surface prhpredict models clean

