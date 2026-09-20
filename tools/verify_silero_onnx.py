"""Verify local ONNX and native-data VAD assets against the same Torch reference.

Development only. No download, model modification, or inference timing claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument("silero_repo", type=Path)
parser.add_argument("onnx_model", type=Path)
parser.add_argument("weights", type=Path)
parser.add_argument("inputs", type=Path, help="JSON array of local WAV paths")
parser.add_argument("output", type=Path, help="New verification receipt")
args = parser.parse_args()
assert not args.output.exists(), "choose a new receipt"

import numpy as np
import onnxruntime
import torch
from safetensors.torch import load_file
from scipy.io import wavfile

torch.set_num_threads(1)
sys.path.insert(0, str(args.silero_repo / "src"))
from silero_vad.utils_vad import OnnxWrapper, get_speech_timestamps

jit_path = args.silero_repo / "src/silero_vad/data/silero_vad.jit"
reference = torch.jit.load(str(jit_path), map_location="cpu").eval()
native = load_file(str(args.weights))
state = reference._model.state_dict()
assert native.keys() == state.keys()
assert all(torch.equal(native[key], value) for key, value in state.items())
onnx = OnnxWrapper(str(args.onnx_model), force_onnx_cpu=True)

class Capture:
    def __init__(self, model):
        self.model = model
        self.probabilities = []

    def reset_states(self):
        self.probabilities = []
        self.model.reset_states()

    def __call__(self, audio, rate):
        value = self.model(audio, rate)
        self.probabilities.append(value.item())
        return value

captures = [Capture(reference), Capture(onnx)]
rows = []
for name in json.loads(args.inputs.read_text(encoding="utf-8-sig")):
    path = Path(name)
    rate, samples = wavfile.read(path)
    assert rate == 16000 and samples.ndim == 1
    assert samples.dtype in (np.int16, np.float32)
    values = samples.astype(np.float32)/32768 if samples.dtype == np.int16 else samples
    audio = torch.from_numpy(values.copy())
    spans = [get_speech_timestamps(audio, model, sampling_rate=16000,
                                  threshold=0.5, max_speech_duration_s=30) for model in captures]
    error = np.abs(np.array(captures[0].probabilities) - np.array(captures[1].probabilities))
    maximum = float(error.max())
    assert maximum <= 5e-5, (path, maximum)
    assert spans[0] == spans[1], (path, spans)
    rows.append(dict(source=str(path), samples=len(audio), frames=len(error),
                     probability_max_error=maximum, boundaries_match=True, segments=spans[0],
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
args.output.write_text(json.dumps(dict(
    scope="CPU numerical/segmentation check, not a timing benchmark. JIT state equals native weights exactly; ONNX frame probabilities must be within 5e-5 and timestamps exact.",
    torch_version=torch.__version__, onnxruntime_version=onnxruntime.__version__,
    jit_sha256=sha(jit_path), onnx_sha256=sha(args.onnx_model), weights_sha256=sha(args.weights),
    wrapper_sha256=sha(args.silero_repo/'src/silero_vad/utils_vad.py'),
    script_sha256=sha(Path(__file__)), rows=rows), indent=2), encoding="utf-8")
print(json.dumps(dict(recordings=len(rows), frames=sum(r['frames'] for r in rows),
    probability_max_error=max(r['probability_max_error'] for r in rows), all_boundaries_match=True)))
