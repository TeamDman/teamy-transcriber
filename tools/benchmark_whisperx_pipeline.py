"""Measure real WhisperX VAD, batching, decoding and result assembly.

Requires existing local CT2 weights, a local Silero Hub checkout and an installed
WhisperX environment. Does not download, patch upstream algorithms, align words
or diarize speakers. Silero's local loader avoids network/cache lookup in timing.
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
parser.add_argument("inputs", type=Path, help="JSON array of complete audio paths")
parser.add_argument("output", type=Path, help="New receipt path")
parser.add_argument("--silero-repo", type=Path, required=True)
parser.add_argument("--generation-config", type=Path, required=True)
parser.add_argument("--dll-dir", action="append", default=[])
parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
parser.add_argument("--repeats", type=int, default=2)
parser.add_argument("--cpu-threads", type=int, default=1,
                    help="Torch frontend/VAD threads; local Silero defaults to one")
args = parser.parse_args()
assert not args.output.exists(), "choose a new receipt path"
assert args.repeats > 0 and all(size > 0 for size in args.batch_sizes)
assert (args.silero_repo / "hubconf.py").is_file(), "local Silero source is required"
dll_handles = [os.add_dll_directory(path) for path in args.dll_dir] if os.name == "nt" else []

import torch
import whisperx
import ctranslate2
from whisperx.asr import load_model
from whisperx.audio import load_audio
from whisperx.vads.silero import Silero
from whisperx.vads.vad import Vad


class LocalSilero(Silero):
    """Only model resolution differs; inference and merging remain upstream."""

    def __init__(self, root):
        Vad.__init__(self, 0.5)
        self.vad_onset = 0.5
        self.chunk_size = 30
        self.vad_pipeline, utils = torch.hub.load(
            str(root), "silero_vad", source="local", onnx=False
        )
        self.get_speech_timestamps, _, self.read_audio, _, _ = utils
        self.elapsed_ms = 0.0

    def __call__(self, audio, **kwargs):
        start = time.perf_counter()
        result = super().__call__(audio, **kwargs)
        self.elapsed_ms += (time.perf_counter() - start) * 1000
        return result


def file_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


paths = [Path(path) for path in json.loads(args.inputs.read_text(encoding="utf-8-sig"))]
assert paths
generation = json.loads(args.generation_config.read_text())
assert generation["begin_suppress_tokens"] == [220, 50257], "expected canonical blank/EOT suppression"
import_ms = (time.perf_counter() - process_start) * 1000
load_start = time.perf_counter()
vad = LocalSilero(args.silero_repo)
torch.set_num_threads(args.cpu_threads)
pipeline = load_model(
    str(args.model), device="cuda", compute_type="float16", language="en",
    local_files_only=True, threads=args.cpu_threads, vad_model=vad,
    asr_options={"beam_size": 1, "best_of": 1, "suppress_blank": True,
                 "suppress_tokens": generation["suppress_tokens"],
                 "condition_on_previous_text": False, "suppress_numerals": False},
)
load_ms = (time.perf_counter() - load_start) * 1000
assert pipeline.model.model.device == "cuda", "reference ASR must run on CUDA"
runs = []
first_result_ms = None
for path in paths:
    for batch_size in args.batch_sizes:
        for index in range(args.repeats):
            started = time.perf_counter()
            audio = load_audio(str(path))
            audio_ms = (time.perf_counter() - started) * 1000
            vad.elapsed_ms = 0
            decode_start = time.perf_counter()
            result = pipeline.transcribe(audio, batch_size=batch_size, chunk_size=30)
            pipeline_ms = (time.perf_counter() - decode_start) * 1000
            total_ms = (time.perf_counter() - started) * 1000
            if first_result_ms is None:
                first_result_ms = (time.perf_counter() - process_start) * 1000
            runs.append(dict(source=str(path), batch_size=batch_size, index=index,
                             audio_seconds=len(audio) / 16000, audio_ms=audio_ms,
                             vad_ms=vad.elapsed_ms, pipeline_ms=pipeline_ms,
                             total_ms=total_ms, result=result))
            print(json.dumps(dict(source=path.name, batch_size=batch_size, index=index,
                                  total_ms=total_ms, vad_ms=vad.elapsed_ms,
                                  segments=len(result["segments"]))), flush=True)
source_files = [args.silero_repo / "hubconf.py", args.silero_repo / "src/silero_vad/utils_vad.py",
                args.silero_repo / "src/silero_vad/data/silero_vad.jit"]
receipt = dict(
    scope="Real WhisperX load_audio + Silero CPU VAD + merge + batched CUDA ASR + result assembly; greedy English. Excludes word alignment, diarization and file export. Native fixed-chunk workflow has different features/boundaries: no direct end-to-end speed claim.",
    backend="python-whisperx-pipeline", torch_version=torch.__version__,
    ctranslate2_version=ctranslate2.__version__, asr_device=pipeline.model.model.device,
    asr_compute_type=pipeline.model.model.compute_type, vad_device="cpu",
    cpu_threads=torch.get_num_threads(), import_ms=import_ms, load_ms=load_ms,
    first_result_ms=first_result_ms, silero_files=[dict(name=str(path.relative_to(args.silero_repo)), sha256=file_hash(path)) for path in source_files],
    model_sha256=file_hash(args.model / "model.bin"),
    generation_config_sha256=file_hash(args.generation_config),
    input_hashes=[dict(name=str(path), sha256=file_hash(path)) for path in paths],
    script_sha256=file_hash(Path(__file__)), runs=runs, full_goal_acceptance=False,
)
args.output.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
