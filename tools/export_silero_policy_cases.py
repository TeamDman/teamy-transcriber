"""Create independent synthetic segmentation oracles using local upstream code."""
import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("silero_repo", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
assert not args.output.exists()

import torch
source = args.silero_repo / "src/silero_vad/utils_vad.py"
spec = importlib.util.spec_from_file_location("local_silero_utils", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class Probabilities:
    def __init__(self, values):
        self.values = values
    def reset_states(self):
        self.iterator = iter(self.values)
    def __call__(self, audio, rate):
        return torch.tensor(next(self.iterator), dtype=torch.float32)

generator = random.Random(17933)
cases = []
patterns = [[], [(0., 50)], [(1., 1000)], [(1., 7)], [(1., 8)],
            [(1., 100), (0., 4), (0.4, 20), (1., 900)],
            [(1., 100), (0., 4), (0.4, 20), (1., 100), (0., 4), (0.4, 40), (1., 900)]]
for _ in range(100):
    patterns.append([(generator.choice([0., 0.1, 0.349, 0.35, 0.4, 0.5, 0.501, 0.9, 1.]),
                      generator.randrange(1, 70)) for _ in range(generator.randrange(1, 60))])
for i, pattern in enumerate(patterns):
    probabilities = [value for value, count in pattern for _ in range(count)]
    samples = max(0, len(probabilities) * 512 - (17 if i % 2 else 0))
    maximum = [30., 1., 2.5][i % 3]
    threshold = [0.5, 0.3, 0.7][i % 3]
    segments = module.get_speech_timestamps(torch.zeros(samples), Probabilities(probabilities),
        sampling_rate=16000, threshold=threshold, max_speech_duration_s=maximum)
    cases.append(dict(pattern=pattern, samples=samples, threshold=threshold, max_seconds=maximum, segments=segments))
args.output.write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), cases=cases)))
print(f"Wrote {len(cases)} cases")
