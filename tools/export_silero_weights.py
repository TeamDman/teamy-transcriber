"""Extract the exact local reference's 16 kHz weights; development only.

The native runtime never reads TorchScript. Do not use an unrelated similarly
named safetensors file: Silero releases can share shapes but differ in weights.
"""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path, help="Existing local silero_vad.jit")
parser.add_argument("output", type=Path, help="New .safetensors file")
args = parser.parse_args()
receipt = args.output.with_suffix(".provenance.json")
assert not args.output.exists() and not receipt.exists(), "choose fresh output paths"

import torch
from safetensors.torch import save_file

model = torch.jit.load(str(args.source), map_location="cpu").eval()._model
assert model.context_size_samples == 64
assert model.stft.hop_length == 128 and model.stft.filter_length == 256
for index, stride in enumerate([1, 2, 2, 1]):
    conv = getattr(model.encoder, str(index)).reparam_conv
    assert list(conv.stride) == [stride] and list(conv.padding) == [1]
    assert list(conv.dilation) == [1] and conv.groups == 1
weights = {key: value.detach().contiguous() for key, value in model.state_dict().items()}
assert len(weights) == 15 and all(value.dtype == torch.float32 for value in weights.values())
source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
save_file(weights, str(args.output), metadata={"architecture": "silero-16k-context64-v1", "source_sha256": source_hash})
receipt.write_text(json.dumps(dict(source_sha256=source_hash,
    weights_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
    bytes=args.output.stat().st_size, torch_version=torch.__version__,
    tensors={key: list(value.shape) for key, value in weights.items()}), indent=2))
print(receipt.read_text())
