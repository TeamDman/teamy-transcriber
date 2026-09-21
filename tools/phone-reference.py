"""Export corrected PhoneticXeus reference logits using a local reviewed clone.
Python is a validation dependency only; the application does not execute this.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.io import wavfile
from safetensors.torch import load_file
p=argparse.ArgumentParser()
p.add_argument('--reference', required=True)
p.add_argument('--model', required=True)
p.add_argument('--wav', required=True)
p.add_argument('--output', required=True)
p.add_argument('--seconds', type=float, default=3)
a=p.parse_args()
sys.path.insert(0,str(Path(a.reference)/'hub'))
from pxeus.model.xeusphoneme.builders import build_xeus_pr
from pxeus.recipe.phone_recognition.greedy_ctc_strategy import ctc_collapse_vectorized
root=Path(a.model);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(8)
t=time.perf_counter()
m=build_xeus_pr(str(Path(a.reference)/'hub/pxeus/model/xeusphoneme/resources/xeus_config.yaml'),vocab_file=str(root/'ipa_vocab.json'),interctc_layer_idx=[4,8,12],interctc_use_conditioning=True).eval()
w=load_file(str(root/'model.safetensors'))
m.load_state_dict({k.removeprefix('model.'):v for k,v in w.items()},strict=True)
del w
load_seconds=time.perf_counter()-t
rate,audio=wavfile.read(a.wav)
assert rate==16000 and audio.ndim==1
if audio.dtype==np.int16: audio=audio.astype(np.float32)/32768
else: audio=audio.astype(np.float32)
audio=audio[:int(a.seconds*rate)].copy();wavfile.write(str(out/'input.wav'),rate,audio)
x=torch.from_numpy(audio)[None];lengths=torch.tensor([len(audio)])
with torch.inference_mode():
    t=time.perf_counter();y,_=m.encode(x,lengths)
    if isinstance(y,tuple): y=y[0]
    logits=m.ctc.ctc_lo(y);elapsed=time.perf_counter()-t
ids=ctc_collapse_vectorized(logits.argmax(-1),m.blank_id)[0]
tokens=[m.token_list[i] for i in ids]
logits.numpy().tofile(out/'reference-logits.f32')
report={'phones':tokens,'frames':logits.shape[1],'load_seconds':load_seconds,'inference_seconds':elapsed,'reference':'corrected local PhoneticXeus, conditioning 4/8/12','device':'cpu'}
(out/'reference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True))
