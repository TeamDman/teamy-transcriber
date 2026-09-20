"""Derive reproducible noise probes from a local benchmark manifest.

These are controlled perturbations of real speech, not a conversational/noisy
field-recording benchmark. Nothing is downloaded or uploaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile

p = argparse.ArgumentParser()
p.add_argument("manifest", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--seed", type=int, default=20260919)
args = p.parse_args()
if args.output.exists() and any(args.output.iterdir()):
    p.error("choose an empty output directory to preserve previous fixtures")
args.output.mkdir(parents=True, exist_ok=True)
source = json.loads(args.manifest.read_text())
speakers = {}
for item in source["items"]:
    # VCTK utterance IDs identify the speaker. Joined/silence probes are excluded.
    if item["id"].startswith("p") and "_" in item["id"]:
        speakers.setdefault(item["id"].split("_")[0], item)
assert speakers, "manifest contains no VCTK speaker utterances"
rng = np.random.default_rng(args.seed)
items = []
for speaker, item in sorted(speakers.items()):
    rate, clean = wavfile.read(item["wav"])
    assert rate == 16000 and clean.dtype == np.float32 and clean.ndim == 1
    assert 0 < len(clean) <= 480000 and np.isfinite(clean).all()
    assert hashlib.sha256(Path(item["wav"]).read_bytes()).hexdigest() == item["audio_sha256"]
    noise = rng.standard_normal(len(clean))
    speech_rms = np.sqrt(np.mean(clean.astype(np.float64) ** 2))
    assert speech_rms > 0
    noise /= np.sqrt(np.mean(noise ** 2))
    for snr in (15, 5):
        mixed = clean + noise * speech_rms / (10 ** (snr / 20))
        # Joint scaling prevents clipping while preserving the exact SNR.
        mixed /= max(1., float(np.max(np.abs(mixed))) / 0.95)
        name = f"{item['id']}-noise-{snr}db"
        path = args.output / (name + ".wav")
        wavfile.write(path, rate, mixed.astype(np.float32))
        items.append(dict(id=name, wav=str(path.resolve()), reference=item["reference"],
            seconds=len(clean) / rate, audio_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_id=item["id"], source_sha256=item["audio_sha256"],
            speaker=speaker, snr_db=snr))
result = dict(scope="controlled white-noise stress probes, one utterance per local speaker; not field-recording acceptance",
              seed=args.seed, snr_definition="whole-clip speech RMS / noise RMS; jointly scaled without clipping",
              source_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(), items=items)
(args.output / "manifest.json").write_text(json.dumps(result, indent=2))
(args.output / "wav-list.json").write_text(json.dumps([i["wav"] for i in items], indent=2))
print(json.dumps(dict(clips=len(items), seconds=sum(i["seconds"] for i in items)), indent=2))
