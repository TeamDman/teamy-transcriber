"""Explicit-input ASR kernel reference; does not claim full WhisperX timing.

Convert the same canonical model with CT2's TransformersConverter beforehand.
Run with an existing CT2-capable Python environment; never installs/downloads.
"""
import time

process_start = time.perf_counter()
import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("mel", type=Path)
parser.add_argument("--repeats", type=int, default=6)
parser.add_argument("--compute-type", choices=["float32", "float16"], default="float32")
parser.add_argument("--dll-dir", action="append", default=[])
parser.add_argument("--mel-prefix", help="For a corpus manifest: prefix of <prefix>-<id>.mel.f32 dumps")
parser.add_argument("--wav-input", action="store_true", help="Include WAV reads and WhisperX's real CPU frontend")
parser.add_argument("--cpu-threads", type=int, default=8)
args = parser.parse_args()
dll_handles = [os.add_dll_directory(p) for p in args.dll_dir] if os.name == "nt" else []
import ctranslate2
import numpy as np
from tokenizers import Tokenizer
if args.wav_input:
    import torch
    from scipy.io import wavfile
    from whisperx.audio import log_mel_spectrogram
    torch.set_num_threads(args.cpu_threads)
    bins = json.loads((args.model / "preprocessor_config.json").read_text())["feature_size"]

import_ms = (time.perf_counter() - process_start) * 1000
started = time.perf_counter()
model = ctranslate2.models.Whisper(str(args.model), device="cuda", compute_type=args.compute_type)
tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
load_ms = (time.perf_counter() - started) * 1000
prompt = [tokenizer.token_to_id(t) for t in ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]]
suppressed = [tokenizer.token_to_id(t) for t in ["<|startoftranscript|>", "<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>", "<|en|>"]]
suppressed = [t for t in suppressed if t is not None]
if args.mel.suffix == ".json":
    assert args.wav_input or args.mel_prefix, "mel corpus manifests require --mel-prefix"
    inputs = [(item["id"], Path(item["wav"]) if args.wav_input else Path(f"{args.mel_prefix}-{item['id']}.mel.f32")) for item in json.loads(args.mel.read_text())["items"]]
else:
    inputs = [(args.mel.stem, args.mel)]
runs = []
first_result_ms = None
for name, path in inputs:
    if not args.wav_input:
        mel = np.fromfile(path, dtype="<f4").reshape(1, -1, 3000)
        assert mel.shape[1] in (80, 128), "expected Whisper 80/128-bin mel features"
    for index in range(args.repeats):
        request_start = time.perf_counter()
        if args.wav_input:
            rate, samples = wavfile.read(path)
            assert rate == 16000 and samples.ndim == 1 and len(samples) <= 480000
            if samples.dtype == np.int16:
                samples = samples.astype(np.float32) / 32768.0
            assert samples.dtype == np.float32, "expected FP32 or PCM16 WAV"
            mel = log_mel_spectrogram(samples, n_mels=bins, padding=480000-len(samples)).numpy()[None]
        frontend_ms = (time.perf_counter() - request_start) * 1000
        started = time.perf_counter()
        encoded = model.encode(ctranslate2.StorageView.from_array(mel))
        result = model.generate(encoded, [prompt], beam_size=1, patience=1, length_penalty=1, max_length=448, suppress_blank=False, suppress_tokens=suppressed)[0]
        inference_ms = (time.perf_counter() - started) * 1000
        tokens = result.sequences_ids[0]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        total_ms = (time.perf_counter() - request_start) * 1000
        if first_result_ms is None:
            first_result_ms = (time.perf_counter() - process_start) * 1000
        runs.append(dict(id=name, index=index, frontend_ms=frontend_ms, inference_ms=inference_ms, total_ms=total_ms, tokens=tokens, text=text))
with (args.model / "model.bin").open("rb") as f:
    digest = hashlib.sha256()
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    model_hash = digest.hexdigest()
print(json.dumps(dict(schema=2, backend="python-whisperx-asr" if args.wav_input else "python-ctranslate2", version=ctranslate2.__version__, device=model.device, compute_type=model.compute_type, cpu_threads=args.cpu_threads if args.wav_input else None, scope=("WAV-to-text, WhisperX CPU frontend and CT2 ASR" if args.wav_input else "mel-to-text CT2 ASR") + "; greedy English, batch one; excludes VAD/alignment/diarization", model_sha256=model_hash, input_sha256=hashlib.sha256(args.mel.read_bytes()).hexdigest(), import_ms=import_ms, load_ms=load_ms, first_result_ms=first_result_ms, runs=runs), indent=2))
