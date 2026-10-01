# RotateAI Pipelines

[![build](https://github.com/mtmd1/rotateai-pipelines/actions/workflows/build.yml/badge.svg)](https://github.com/mtmd1/rotateai-pipelines/actions/workflows/build.yml)

Inference pipelines for on-tag whale orientation correction. Designed for STM32U5 deployment, tested with [rotateai-simulator](https://github.com/mtmd1/rotateai-simulator).

Each pipeline reads sensor data from stdin and writes corrected orientation angles to stdout using binary float32. They differ in when and how often inference runs.

## I/O shape

- **Input** (per sample): `INPUT_CHANNELS` float32, set by the model in `build/models/model_params.h`, with depth last. A 4-channel model takes `ax, ay, az, p` (accelerometer xyz in g, depth in m, 16 bytes). The 7-channel model takes `ax, ay, az, mx, my, mz, p` (28 bytes). `prhpredict` has no model and always takes the 4-channel layout.
- **Output** (per emitted prediction): 1 flag byte + 3 float32 — `pitch, roll, heading` in radians, in `(-π, π]`. 13 bytes when flag=`0x01`.
- **Skip**: 1 flag byte `0x00`. No payload.

Models are HART-family Transformers ending in `L2NormalizeAngles`, which emits 6 raw `(cos, sin)` pairs. The C binary decodes via `atan2(sin, cos)` per pair before writing, so consumers always see angles, never raw pairs.

## Installation

```sh
./install.sh
```

This clones and builds [TFLite Micro](https://github.com/tensorflow/tflite-micro) at the commit pinned in `install.sh`. It needs GNU make 3.82 or later (on macOS, `brew install make` and put its `gnubin` on `PATH`), `wget`, `unzip`, and a `python3` with `numpy` and `pillow`, which TFLM's Makefile uses while parsing. A Python environment is also required for model preparation:

```sh
python -m venv .venv    # requires Python <= 3.13
source .venv/bin/activate
pip install -r tools/requirements.txt
```

The HART custom Keras layers are vendored under `tools/model/`; no external `PYTHONPATH` setup is needed.

The versions in `tools/requirements.txt` are pinned because the converted model depends on them.

## Model Preparation

Converts a Keras model (and optional preprocessing parameters) into C-compatible files for compilation.

```sh
source .venv/bin/activate
make models
```

This converts the vendored model in `models/`. For any other model:

```sh
python tools/prepare_model.py --model /path/to/model.keras [--params /path/to/params.pkl]
```

`--params` is optional. If passed, the script reads `means`/`stds` from the `.pkl` for input z-scoring; if those keys are absent or `None`, identity normalization is emitted (zero means, unit stds — i.e. raw inputs go straight to the model). Output channel count, window size, and last-row offset are all derived from the model itself.

Generated files (default: `build/models/`):
- `model_data.inc` — TFLite flatbuffer as a hex array
- `model_params.h` — `WINDOW_SIZE`, `INPUT_CHANNELS`, `OUTPUT_CHANNELS`, `OUTPUT_RAW_CHANNELS`, `OUTPUT_TENSOR_SIZE`, `OUTPUT_LAST_ROW_OFFSET`, `INPUT_MEANS[]`, `INPUT_STDS[]`, plus a `REGISTER_OPS()` macro

## Build

```sh
make baseline
make variable
make surface
make prhpredict
```

`prhpredict` has no model, so it needs neither TFLite Micro nor model preparation — just a C++ compiler.

Builds are supported on x86-64 Linux, macOS (Intel) and Windows (MSYS2 UCRT64). Linux binaries link libstdc++ and libgcc statically, and Windows binaries are fully static.

### Prebuilt binaries

CI builds all four pipelines on pushes to `main`, `v*` tags and PRs. Zips are attached to each run, and to GitHub Releases using a `v*` tag. Model conversion runs once on Linux and the same data is embedded on all platforms.

On macOS the binaries are ad-hoc signed. Clear the quarantine flag before running:

```sh
xattr -d com.apple.quarantine baseline variable surface prhpredict
```

Outputs are not bit-identical across operating systems.

## Current Pipelines

| Pipeline | Strategy | Description |
| -------- | -------- | ----------- |
| `baseline` | Every sample | Maximum accuracy and cost. |
| `variable` | Every X samples | Measures a sample window periodically. |
| `surface`  | Event-triggered | Detects surfacing periods and runs inference on them. |
| `prhpredict` | Event-triggered, no model | Mark Johnson's prhpredictor method 1. Get pitch and roll from the mean acceleration while breathing at the surface, heading from the assumption of planarity in pitch on descent and ascent. |

The `surface` binary supports four strategies (`start` / `end` / `bookend` / `average`); `bookend` and `average` average the raw `(cos, sin)` pairs across windows before decoding (circular mean) so wrap-around at ±π is handled correctly. `start` emits as soon as its window fills rather than waiting for the dive, so `--min-samples` is fixed at the window size for that strategy. `bookend` falls back to the end window alone when the surfacing period is shorter than two windows.

`--max-samples` ends a surfacing period after N samples even without a dive. On deployments that never pass `--dive-depth` a period would otherwise never close and nothing would ever be emitted; forcing it shut gives a `variable`-style refresh every N samples. It defaults to 0 (no maximum) and must be at least the window size.
