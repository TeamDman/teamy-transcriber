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
