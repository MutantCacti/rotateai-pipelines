'''
tools/smoke_test.py
Drives each built pipeline over a synthetic stream and checks the binary
protocol: one flag byte per input sample, 0x00 or 0x01, and OUTPUT_CHANNELS
finite float32s after every 0x01. Checks the protocol, not accuracy.

The accelerometer values contain bytes 0x1A and 0x0A, which a Windows binary
left in text mode would treat as end-of-file and newline.

Channel count and window size are read from model_params.h, in models/
under a build directory or beside the binaries in an unzipped release, so
the expected counts follow whatever model was converted.

usage: python tools/smoke_test.py [BUILD_DIR]

Created: 2026-10-01
Authors: Maxence Morel Dierckx, Claude Opus 5.5
'''
import math
import re
import struct
import subprocess
import sys
from pathlib import Path


# Little-endian bytes 0a 1a 0a 3f, about 0.54 g.
TRAP = struct.unpack('<f', b'\x0a\x1a\x0a\x3f')[0]

# prhpredict reads A + p and writes pitch, roll, heading regardless of the model
PRH_CHANNELS, PRH_OUTPUT = 4, 3


def read_params(path):
    '''Integer #defines from model_params.h.'''
    text = path.read_text()
    return {k: int(v) for k, v in re.findall(r'#define (\w+) (\d+)\s', text)}


def find_params(build_dir):
    '''model_params.h from a build tree or an unzipped release.'''
    for p in (build_dir / 'models' / 'model_params.h', build_dir / 'model_params.h'):
        if p.is_file():
            return p
    sys.exit(f'FAIL: no model_params.h in {build_dir} or {build_dir / "models"}')


def binary(build_dir, name):
    '''The built binary, with or without the Windows .exe suffix.'''
    for p in (build_dir / name, build_dir / f'{name}.exe'):
        if p.is_file():
            return p
    sys.exit(f'FAIL {name}: no binary in {build_dir}')


def stream(depths, channels):
    '''Samples [ax ay az (mx my mz) p] packed as float32. The accelerometer
    wobbles so prhpredict's surfacing scatter is not degenerate.'''
    out = bytearray()
    for i, depth in enumerate(depths):
        a = [TRAP, 0.3 * math.sin(i / 3), 0.8 + 0.1 * math.cos(i / 5)]
        m = [20.0, -5.0, 40.0]
        row = a + m[:channels - 4] + [depth]
        out += struct.pack(f'<{channels}f', *row)
    return bytes(out)


def run(build_dir, name, args, depths, channels, n_out, expect):
    '''Run one binary and check its output. `expect(indices)` returns an
    error string, or None if the emitted sample indices are right.'''
    data = stream(depths, channels)
    cmd = [str(binary(build_dir, name)), *args]
    proc = subprocess.run(cmd, input=data, capture_output=True, timeout=1800)
    if proc.returncode != 0:
        return f'exited {proc.returncode}: {proc.stderr.decode(errors="replace").strip()}'

    out, pos, indices = proc.stdout, 0, []
    for i in range(len(depths)):
        if pos >= len(out):
            return f'{i} flags for {len(depths)} samples'
        flag = out[pos]
        pos += 1
        if flag == 0x01:
            vals = struct.unpack_from(f'<{n_out}f', out, pos)
            pos += 4 * n_out
            if not all(math.isfinite(v) for v in vals):
                return f'non-finite output {vals} at sample {i}'
            indices.append(i)
        elif flag != 0x00:
            return f'invalid flag {flag:#04x} at sample {i}'
    if pos != len(out):
        return f'{len(out) - pos} trailing bytes'
    return expect(indices)


def main():
    build_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'build')
    params = read_params(find_params(build_dir))
    window, channels = params['WINDOW_SIZE'], params['INPUT_CHANNELS']
    n_out = params['OUTPUT_CHANNELS']

    def exactly(want):
        return lambda got: None if got == want else f'emitted at {got}, expected {want}'

    surface_args = ['--strategy', 'start', '--surface-depth', '5', '--dive-depth', '10']
    # 60 samples breathing at the surface (prhpredict needs 50), then a dive:
    # it emits on the sample that crosses dive depth. Both gates off so the
    # synthetic surfacing, at constant depth, cannot be rejected.
    prh_depths = [0.0] * 60 + [20.0 + 0.1 * i for i in range(100)]

    cases = [
        ('baseline', [], [0.0] * 20, channels, n_out, exactly(list(range(20)))),
        ('variable', ['--offset', '10'], [0.0] * 20, channels, n_out, exactly([0, 10])),
        ('surface', surface_args, [0.0] * window, channels, n_out, exactly([window - 1])),
        ('prhpredict', ['--min-aniso', '0', '--min-branch-conf', '0'], prh_depths, PRH_CHANNELS, PRH_OUTPUT,
         exactly([60])),
    ]

    failed = 0
    for name, args, depths, ch, n, expect in cases:
        err = run(build_dir, name, args, depths, ch, n, expect)
        print(f'{"FAIL" if err else "ok  "} {name}' + (f': {err}' if err else ''))
        failed += bool(err)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
