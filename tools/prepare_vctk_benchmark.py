"""Prepare a fixed, local-only multi-speaker ASR corpus; never fetches media."""
import argparse
import hashlib
import json
import random
import wave
from pathlib import Path
import numpy as np
from scipy.io import wavfile

p=argparse.ArgumentParser()
p.add_argument("corpus",type=Path)
p.add_argument("output",type=Path)
p.add_argument("--sample-count", type=int, default=0, help="Seeded sample instead of the fixed canary set")
p.add_argument("--seed", type=int, default=20260919)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
ids=["p225_001","p225_003","p225_010","p226_002","p226_010","p227_003","p227_010","p228_002","p229_003","p230_385","p230_010","p230_020"]
if a.sample_count:
    candidates = sorted(path.stem for path in (a.corpus/"wav48").glob("*/*.wav")
                        if (a.corpus/"txt"/path.parent.name/(path.stem+".txt")).is_file())
    ids = random.Random(a.seed).sample(candidates, a.sample_count)
items=[]
joined=[]
joined_text=[]
total=0
for name in ids:
    speaker=name.split("_")[0]
    source=a.corpus/"wav48"/speaker/(name+".wav")
    reference=(a.corpus/"txt"/speaker/(name+".txt")).read_text().strip()
    with wave.open(str(source),"rb") as r:
        assert r.getsampwidth()==2
        rate=r.getframerate()
        data=np.frombuffer(r.readframes(r.getnframes()),dtype="<i2").astype(np.float32).reshape(-1,r.getnchannels()).mean(axis=1)/32768
    assert rate==48000, "This fixed corpus uses exact integer-rate downsampling"
    samples=data[::3].copy()
    assert len(samples) <= 480000, "A sampled clip exceeds 30 seconds; use explicit long-form chunking"
    destination=a.output/(name+".wav")
    wavfile.write(destination,16000,samples)
    items.append(dict(id=name,wav=str(destination.resolve()),reference=reference,audio_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),seconds=len(samples)/16000))
    if total+len(samples)+3200<=480000:
        joined.extend([samples,np.zeros(3200,dtype=np.float32)])
        joined_text.append(reference)
        total+=len(samples)+3200
for name,samples,reference in [("joined",np.concatenate(joined)," ".join(joined_text)),("silence",np.zeros(16000,dtype=np.float32),"")]:
    destination=a.output/(name+".wav")
    wavfile.write(destination,16000,samples)
    items.append(dict(id=name,wav=str(destination.resolve()),reference=reference,audio_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),seconds=len(samples)/16000))
(a.output/"manifest.json").write_text(json.dumps(dict(scope=f"{len(ids)} VCTK utterances, bounded concatenation, silence",seed=a.seed if a.sample_count else None,normalization="48kHz PCM16 to 16kHz FP32; select every third sample",items=items),indent=2))
(a.output/"wav-list.json").write_text(json.dumps([i["wav"] for i in items],indent=2))
print(json.dumps(dict(clips=len(items),seconds=sum(i["seconds"] for i in items)),indent=2))
