# Native Whisper inference

This experimental backend defines Whisper's encoder and decoder in Rust and
uses custom CUDA kernels plus cuBLAS. Model files contain numerical tensors,
dimensions, and tokenizer data. The native build has no LibTorch, TorchScript,
Python, or cuDNN runtime dependency.

The default application build retains its existing `tch-native` backend while
the performance and quality comparison is underway. Build the native option:

```powershell
./tools/build-cuda-native.ps1
./target/release/teamy-transcriber.exe model prepare --source-dir <canonical-model> --output-dir <prepared-model>
```

The CUDA toolkit must be installed and `CUDA_PATH` must identify it. The build
defaults to compute capability 8.9 with forward-compatible PTX; override
`TEAMY_WHISPER_CUDA_ARCH` to select another toolkit-supported target. Current
empirical validation is Windows x64 on RTX 4090. Other GPU/platform combinations
have not been validated.

Use the existing recording workflow with `--model-dir <prepared-model>`. The
`cuda-native` feature selects the CUDA implementation for safetensors packages.
`TEAMY_TRANSCRIBER_CUDA_DEVICE` selects a device index (default zero). A combined
build can use `TEAMY_TRANSCRIBER_BACKEND=tch` for the retained implementation;
its existing `TEAMY_TRANSCRIBER_TORCH_DEVICE=-1` CPU selection is preserved.
TorchScript and Burn packages retain their previous feature requirements.

Native application builds use TF32 tensor-core matrix multiplication with FP32
storage and reductions. Set `TEAMY_TRANSCRIBER_CUDA_MATH=fp32` for the full-FP32
numerical oracle path (`tf32` selects the default). This is a computation setting;
the safetensors weights stay unchanged.

Weights are loaded once, from single-file or indexed safetensors, into FP32
device storage. Sequential file reads, SIMD FP16 conversion and bounded staging
groups avoid a separate GPU allocation/transfer for every tensor. Decoder K/V
buffers and workspaces are allocated once; the prompt is evaluated as one causal
batch, and requests reset their logical cache position. Only the selected token crosses back to
the CPU during generation. The application owns a bounded inference worker:
model creation, inference, and destruction happen on the same thread.

The resident frontend reuses its FFT plan and skips zero filter coefficients
and wholly padded frames. It accepts complete windows up to 30 seconds; the
existing application workflow remains responsible for chunking longer audio.

## Verification tools

```powershell
cargo test --release --manifest-path native/Cargo.toml --lib
# Explicit GPU kernel checks against scalar numerical oracles:
cargo test --release --manifest-path native/Cargo.toml --lib -- --ignored
cargo run --release --manifest-path native/Cargo.toml --bin wav_bench -- <prepared-model> <mono-16k.wav> 6 - tf32
```

`wav_bench` also accepts a JSON array of WAV paths, keeping the same model alive
across all inputs. It reports frontend, encoder, decoder, load, and complete
WAV-to-text timings alongside generated tokens and termination. A diagnostic
dump prefix adds raw mel and prompt-logit files for independent parity checks.

`tools/prepare_vctk_benchmark.py` prepares a fixed local corpus without downloads.
`tools/check_native_parity.py` compares diagnostic dumps with the original Rust
frontend and an independently executed Hugging Face model.
`tools/benchmark_ct2.py` measures the CTranslate2 ASR component used by WhisperX,
with explicit CUDA precision and greedy decoding. It accepts the same supplied
mel features, or `--wav-input` uses WhisperX's own CPU frontend for complete
WAV-to-text comparison. `tools/summarize_asr.py` checks repeated output stability,
termination, token matches, word errors, cold process startup and warm latency.
`examples/whisper_bench.rs` preserves the existing Rust/tch comparison.
`tools/compare-asr.ps1` alternates fresh native/Python processes over several
rounds, captures GPU state and binary/script hashes, and produces local summaries.
Python is used only by these development/reference tools.

These tools do not establish a full WhisperX speed claim: VAD, beam-search
defaults, word alignment, diarization, long-form quality and comparison to a
confirmed Rust WhisperX target remain acceptance work. Numerical parity has
been checked on Whisper tiny and large-v3, including the 128-bin frontend.
The development corpus is local VCTK speech; it does not establish accuracy on
noisy, multilingual or conversational workloads. Existing clip-range timestamp
exports continue to use the application workflow.
