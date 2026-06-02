'''
tools/prepare_model.py
Converts a model.keras into model_data.inc and model_params.h
for use by pipelines.

I/O shape is derived from the model itself (not the .pkl). The .pkl
is consulted only for input means/stds; if absent or None, the
emitted means/stds default to identity (0/1) so the C-side
normalize() becomes a no-op.

Model output is assumed to be (cos, sin) pairs (the L2NormalizeAngles
convention from RotateAI's HART/Transformer family). The pipeline
emits raw pairs to the C side; a decode_angles() in the C code
converts them to angles via atan2 before writing to stdout.

inputs:
    --model   path to .keras file
    --params  path to .pkl file (optional means/stds)
    --out     output directory (default: build/models/)

outputs:
    build/models/model_data.inc
    build/models/model_params.h

Created: 2026-03-10
Authors: Maxence Morel Dierckx, Claude Opus 4.6, Claude Opus 4.7
'''
import argparse
import os
import pickle
import re
import sys

import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Vendored custom Keras layers (HART / Transformer family)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model.Transformer_model  # noqa: F401  registers @register_keras_serializable classes

import tensorflow as tf
from tensorflow.lite.tools import visualize


def parse_op_name(word):
    """Convert flatbuffer op name to resolver method name."""
    word = word.replace('TFLite', '')
    parts = re.split('_|-', word)
    result = ''
    for part in parts:
        if len(part) > 1:
            if part[0].isalpha():
                result += part[0].upper() + part[1:].lower()
            else:
                result += part.upper()
        else:
            result += part.upper()
    result = result.replace('Lstm', 'LSTM')
    result = result.replace('BatchMatmul', 'BatchMatMul')
    return 'Add' + result


def extract_ops(tflite_bytes):
    """Extract operator names from a .tflite flatbuffer."""
    data = visualize.CreateDictFromFlatbuffer(bytearray(tflite_bytes))
    ops = set()
    for op_code in data['operator_codes']:
        if op_code['custom_code'] is not None:
            name = visualize.NameListToString(op_code['custom_code'])
            print(f'warning: custom op "{name}", skipping', file=sys.stderr)
            continue
        code = max(op_code['builtin_code'], op_code['deprecated_builtin_code'])
        ops.add(visualize.BuiltinCodeToName(code))
    return sorted((op, parse_op_name(op)) for op in ops)


def get_io_layout(model):
    """Inspect model.input_shape and model.output_shape.

    Returns dict with: window_size, in_channels, raw_out_channels,
    raw_out_size, out_last_offset, decoded_out_channels.
    """
    in_shape = model.input_shape
    out_shape = model.output_shape

    if len(in_shape) != 3 or in_shape[0] is not None:
        raise ValueError(f"Unsupported input shape {in_shape}; expected (None, T, C)")
    window_size = int(in_shape[1])
    in_channels = int(in_shape[-1])

    if len(out_shape) == 2:
        raw_out_channels = int(out_shape[1])
        raw_out_size = raw_out_channels
        out_last_offset = 0
    elif len(out_shape) == 3:
        T = int(out_shape[1])
        C = int(out_shape[2])
        raw_out_channels = C
        raw_out_size = T * C
        out_last_offset = (T - 1) * C
    else:
        raise ValueError(f"Unsupported output shape {out_shape}")

    if raw_out_channels % 2 != 0:
        raise ValueError(
            f"Output must be even (cos/sin pairs), got {raw_out_channels}")
    decoded_out_channels = raw_out_channels // 2

    return dict(
        window_size=window_size,
        in_channels=in_channels,
        raw_out_channels=raw_out_channels,
        raw_out_size=raw_out_size,
        out_last_offset=out_last_offset,
        decoded_out_channels=decoded_out_channels,
    )


def load_input_means_stds(params_path, n_channels):
    """Load means/stds from .pkl. If absent or None, return identity (0/1).

    Cross-checks shape; raises if present but mismatched.
    """
    if params_path is None:
        print('warning: no --params passed; emitting identity input means/stds', file=sys.stderr)
        return (np.zeros(n_channels, dtype=np.float32),
                np.ones(n_channels, dtype=np.float32))

    with open(params_path, 'rb') as f:
        params = pickle.load(f)

    means = params.get('means')
    stds = params.get('stds')
    if means is None or stds is None:
        print('warning: .pkl has no input means/stds; emitting identity (no z-score)', file=sys.stderr)
        return (np.zeros(n_channels, dtype=np.float32),
                np.ones(n_channels, dtype=np.float32))

    means = np.asarray(means, dtype=np.float32).flatten()
    stds = np.asarray(stds, dtype=np.float32).flatten()
    if len(means) != n_channels or len(stds) != n_channels:
        raise ValueError(
            f"means/stds length {len(means)}/{len(stds)} != model input channels {n_channels}")
    return means, stds


def convert_to_tflite(model, in_channels, window_size):
    fixed_input = tf.TensorSpec([1, window_size, in_channels], tf.float32, name='input')

    @tf.function(input_signature=[fixed_input])
    def inference(x):
        return model(x, training=False)

    concrete_func = inference.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    return converter.convert()


def write_model_data_inc(tflite_bytes, path):
    with open(path, 'w') as f:
        for i in range(0, len(tflite_bytes), 12):
            chunk = tflite_bytes[i:i+12]
            line = ', '.join(f'0x{b:02x}' for b in chunk)
            if i + 12 < len(tflite_bytes):
                line += ','
            f.write(f'  {line}\n')


def format_float_array(arr):
    def fmt(v):
        s = f'{np.float32(v):.8g}'
        # Ensure C float literal — needs decimal point or exponent
        if not any(c in s for c in '.eEnNiI'):
            s += '.0'
        return s + 'f'
    return ', '.join(fmt(v) for v in arr)


def write_model_params_h(path, layout, input_means, input_stds, ops):
    with open(path, 'w') as f:
        f.write('#ifndef MODEL_PARAMS_H\n')
        f.write('#define MODEL_PARAMS_H\n\n')
        f.write(f'#define WINDOW_SIZE {layout["window_size"]}\n')
        f.write(f'#define INPUT_CHANNELS {layout["in_channels"]}\n')
        f.write(f'#define OUTPUT_CHANNELS {layout["decoded_out_channels"]}\n')
        f.write(f'#define OUTPUT_RAW_CHANNELS {layout["raw_out_channels"]}\n')
        f.write(f'#define OUTPUT_TENSOR_SIZE {layout["raw_out_size"]}\n')
        f.write(f'#define OUTPUT_LAST_ROW_OFFSET {layout["out_last_offset"]}\n')
        f.write(f'#define NUM_OPS {len(ops)}\n\n')
        f.write('#define REGISTER_OPS(resolver) \\\n')
        for i, (_, method) in enumerate(ops):
            slash = ' \\' if i < len(ops) - 1 else ''
            f.write(f'    resolver.{method}();{slash}\n')
        f.write('\n')
        f.write(f'static const float INPUT_MEANS[] = {{{format_float_array(input_means)}}};\n')
        f.write(f'static const float INPUT_STDS[]  = {{{format_float_array(input_stds)}}};\n\n')
        f.write('#endif\n')


def main():
    parser = argparse.ArgumentParser(description='Convert Keras model to C-compatible files')
    parser.add_argument('--model', required=True, help='Path to .keras file')
    parser.add_argument('--params', default=None, help='Path to .pkl file (for input means/stds; optional)')
    parser.add_argument('--out', default='build/models/', help='Output directory')
    args = parser.parse_args()

    try:
        model = tf.keras.models.load_model(args.model)
        print(f'Loaded model from {args.model}')

        layout = get_io_layout(model)
        print(f'I/O layout: window_size={layout["window_size"]}, '
              f'in_channels={layout["in_channels"]}, '
              f'raw_out={layout["raw_out_channels"]} (tensor size {layout["raw_out_size"]}, '
              f'last-row offset {layout["out_last_offset"]}), '
              f'decoded_out={layout["decoded_out_channels"]}')

        input_means, input_stds = load_input_means_stds(args.params, layout['in_channels'])

        tflite_bytes = convert_to_tflite(model, layout['in_channels'], layout['window_size'])
        print(f'Converted to TFLite ({len(tflite_bytes)} bytes)')

        ops = extract_ops(tflite_bytes)
        print(f'Extracted {len(ops)} ops: {", ".join(m for _, m in ops)}')

        os.makedirs(args.out, exist_ok=True)

        inc_path = os.path.join(args.out, 'model_data.inc')
        write_model_data_inc(tflite_bytes, inc_path)
        print(f'Wrote {inc_path}')

        params_path = os.path.join(args.out, 'model_params.h')
        write_model_params_h(params_path, layout, input_means, input_stds, ops)
        print(f'Wrote {params_path}')

    except Exception as e:
        print(f'error: {e}', file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
