"""Check a native diagnostic dump against the original Rust mel and HF graph.

All inputs are explicit local files; no model, audio, or code is downloaded.
"""
import argparse
import json
import os
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import WhisperForConditionalGeneration

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("reference_mel")
p.add_argument("native_mel")
p.add_argument("native_logits")
a = p.parse_args()
torch.set_num_threads(8)
reference = np.fromfile(a.reference_mel, dtype="<f4")
mel = np.fromfile(a.native_mel, dtype="<f4")
mel_error = float(np.max(np.abs(mel - reference)))
model = WhisperForConditionalGeneration.from_pretrained(a.model, local_files_only=True, attn_implementation="eager").eval()
tok = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
prompt = [tok.token_to_id(t) for t in ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]]
with torch.inference_mode():
    oracle = model(input_features=torch.from_numpy(mel.reshape(1, model.config.num_mel_bins, 3000)), decoder_input_ids=torch.tensor([prompt]), use_cache=False).logits[0, -1].numpy()
native = np.fromfile(a.native_logits, dtype="<f4")
error = np.abs(oracle-native)
result = dict(mel_max_abs_error=mel_error, logits_max_abs_error=float(error.max()), logits_rms_error=float(np.sqrt(np.mean(error**2))), native_argmax=int(native.argmax()), oracle_argmax=int(oracle.argmax()), finite=bool(np.isfinite(native).all()), passed=bool(mel_error < 5e-5 and error.max() < 2e-3 and native.argmax() == oracle.argmax() and np.isfinite(native).all()))
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["passed"] else 1)
