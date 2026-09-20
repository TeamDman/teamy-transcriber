"""Offline Silero VAD oracle, including frame probabilities and exact boundaries.

Uses the supplied local Hub source unchanged and checks extracted native weights
against that model. All timing is single-thread CPU VAD (excludes WAV I/O).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("silero_repo", type=Path)
parser.add_argument("weights", type=Path)
parser.add_argument("inputs", type=Path, help="WAV or JSON array of WAV paths")
parser.add_argument("output", type=Path, help="New output receipt")
parser.add_argument("--repeats", type=int, default=2)
args = parser.parse_args()
assert args.repeats > 0 and not args.output.exists()

import numpy as np
import torch
from safetensors.torch import load_file
from scipy.io import wavfile

started = time.perf_counter()
model, utils = torch.hub.load(str(args.silero_repo), "silero_vad", source="local", onnx=False)
load_ms = (time.perf_counter() - started) * 1000
torch.set_num_threads(1)
native = load_file(str(args.weights))
reference = model._model.state_dict()
assert native.keys() == reference.keys()
assert all(torch.equal(native[key], value) for key, value in reference.items())
get_timestamps = utils[0]


class Capture:
    def __init__(self):
        self.probabilities = []

    def reset_states(self):
        self.probabilities = []
        model.reset_states()

    def __call__(self, audio, rate):
        result = model(audio, rate)
        self.probabilities.append(result.item())
        return result


capture = Capture()
paths = [Path(p) for p in json.loads(args.inputs.read_text(encoding="utf-8-sig"))] if args.inputs.suffix == ".json" else [args.inputs]
runs = []
for path in paths:
    rate, samples = wavfile.read(path)
    assert rate == 16000 and samples.ndim == 1 and samples.dtype in (np.int16, np.float32)
    values = samples.astype(np.float32) / 32768 if samples.dtype == np.int16 else samples
    assert np.isfinite(values).all() and (np.abs(values) <= 1).all()
    audio = torch.from_numpy(values.copy())
    for index in range(args.repeats):
        started = time.perf_counter()
        segments = get_timestamps(audio, capture, sampling_rate=16000,
                                  threshold=0.5, max_speech_duration_s=30)
        vad_ms = (time.perf_counter() - started) * 1000
        runs.append(dict(file=str(path), index=index, samples=len(audio), vad_ms=vad_ms,
                         probabilities=capture.probabilities, segments=segments))
        print(json.dumps(dict(file=path.name, index=index, vad_ms=vad_ms, segments=len(segments))), flush=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

args.output.write_text(json.dumps(dict(backend="silero-reference-torch-cpu", cpu_threads=torch.get_num_threads(),
    scope="VAD only, no audio I/O in request timing; no ASR/alignment/diarization",
    load_ms=load_ms, torch_version=torch.__version__, weights_sha256=sha(args.weights),
    sources={name: sha(args.silero_repo / name) for name in ["hubconf.py", "src/silero_vad/utils_vad.py", "src/silero_vad/data/silero_vad.jit"]},
    inputs={str(path): sha(path) for path in paths}, script_sha256=sha(Path(__file__)), runs=runs)))
