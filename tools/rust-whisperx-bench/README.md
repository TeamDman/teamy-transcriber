# Rust WhisperX ASR comparison

This development adapter calls the unmodified public ASR library used by
`moritzbrantner/native-whisperx`, pinned to
`moenarch-audio-analysis-transcription = 0.1.15`. It uses the reusable Candle
provider, CUDA device zero, explicit precision, English greedy decoding and
the model's canonical generation configuration. The lockfile is required:
newer transitive packages have incompatible contract migrations.

Build in a Visual Studio developer shell with CUDA available:

```powershell
$env:CUDA_COMPUTE_CAP = '89' # select your GPU's compute capability
$env:NVCC_PREPEND_FLAGS = '-Xcompiler=/Zc:preprocessor'
cargo build --release --locked --manifest-path tools/rust-whisperx-bench/Cargo.toml
./tools/rust-whisperx-bench/target/release/rust-whisperx-bench.exe <canonical-model> <wav-or-list.json> 4 fp32 openai/whisper-large-v3
```

The timer includes WAV reading, the reference's own frontend, inference and
text assembly. Repeated requests share one model; the receipt records model
reuse, CUDA diagnostics, load time, first-result time, all transcripts, model
and configuration hashes. Only explicitly supplied local assets are used.
Inputs must be mono 16 kHz FP32/PCM16 WAVs of at most 30 seconds. The reference
does not expose generated token IDs or EOT termination through this API;
reports must leave those checks unavailable, rather than infer a pass.

This is an **ASR component comparison**, not the complete Rust WhisperX CLI.
VAD, alignment and diarization are outside this adapter. Timestamp generation
is disabled through the public API; chunk timing is retained. Use the same
canonical `generation_config.json` with native `wav_bench` and Python
`benchmark_ct2.py --generation-config` to match suppression policies.
No dependency source or model tensor is patched to make a reference run.

The complete `tools/compare-asr.ps1` runner accepts `-RustReferenceExe`,
`-CanonicalModel`, `-ModelId`, and `-GenerationConfig` to rotate all three
engines through fresh-process rounds. An unsupported precision or failed
reference stops the run; it never counts as a performance win.

## Complete speech-detection and ASR pipeline

The optional `rust-whisperx-pipeline-bench` binary calls the same pinned library's
public `run_transcription_pipeline_with_observer` function. The library performs
WAV loading, single-thread CPU ONNX Silero inference, segmentation/merging,
resident CUDA Whisper inference and transcript assembly. The adapter only sets
public request controls and records results; upstream sources remain unchanged.
This covers the transcriber's speech-detection/ASR profile, with alignment and
diarization disabled explicitly. It does not emulate the stock CLI's English
timestamp-decoding defaults. No word-timing or speaker-quality claim follows.

```powershell
cargo build --release --locked --manifest-path tools/rust-whisperx-bench/Cargo.toml --features pipeline --bin rust-whisperx-pipeline-bench
$env:ORT_DYLIB_PATH = '<onnxruntime.dll>'
./tools/rust-whisperx-bench/target/release/rust-whisperx-pipeline-bench.exe <canonical-model> <silero.onnx> <wav-list.json> 2 8 fp32 openai/whisper-large-v3
```

The lockfile pins ONNX bindings to `ort 2.0.0-rc.12`, matching the upstream
release. They require ONNX Runtime API 24; supply a compatible CPU runtime
explicitly. Model files and runtime DLLs stay outside this repository.
For a batch above one, the adapter selects `ActiveRowTensorBatch` and checks
that multi-window requests actually use it. Merely setting `batch_chunks` leaves
the upstream default decoder serial. Receipts retain the effective batch size,
active-row compaction, model reuse and phase timings. FP16 has failed on the
tested canonical model with an upstream F32/F16 convolution mismatch; use the
working FP32 reference and report that precision difference.

Verify the ONNX asset against the same local Torch model and native tensor data
before comparing pipelines:

```powershell
python tools/verify_silero_onnx.py <silero-source> <silero.onnx> <silero.safetensors> <wav-list.json> <new-receipt.json>
```

The check requires exact native/JIT weights, frame probabilities within 0.00005,
and identical speech boundaries. The ONNX runtime used for this numerical check
may differ from the Rust runtime; compare actual Rust boundaries as well.

`tools/compare-pipelines.ps1` rotates the native application, Python WhisperX and
this Rust pipeline through fresh processes. It takes explicit local executables,
models, runtimes, audio and a new output directory; its parameter list is the
invocation contract. Keep CUDA DLLs available on the calling process's PATH.
It hashes inputs/artifacts, rejects differing native/canonical weights or
generation configurations, records GPU state, and stops on any failed engine.
The CT2 package must have been converted from the supplied canonical weights;
the hashes record identity but do not prove that conversion by themselves.
The application includes recording persistence and export; both references end
after transcript assembly. Model/file caches are retained. Repetitions expose
warm behavior, while process totals include all requests and teardown. Check
source ranges, repeated output and reference-word accuracy before using timings.

The pipeline adapter records failed requests and continues collecting the corpus,
then exits unsuccessfully if any request failed. Error entries use
`failed_elapsed_ms`, never a successful transcription time. In the tested upstream
version, silent audio produces an empty VAD result that ASR rejects. Preserve this
failure in quality reports; do not remove silence or count it as a timing win.
The rotating runner stops on this nonzero exit, leaving the diagnostic receipt.
