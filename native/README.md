# Native Whisper inference

The inference backend defines Whisper's encoder and decoder in Rust and
uses custom CUDA kernels plus cuBLAS. Model files contain numerical tensors,
dimensions, and tokenizer data. The native build has no LibTorch, TorchScript,
Python, or cuDNN runtime dependency.

Build the native release and stage its runtime DLLs:

```powershell
./tools/build-cuda-native.ps1
./target/release/teamy-transcriber.exe model prepare --source-dir <canonical-model> --output-dir <prepared-model>
```

The CUDA toolkit must be installed and `CUDA_PATH` must identify it. The build
defaults to compute capability 8.9 with forward-compatible PTX; override
`TEAMY_WHISPER_CUDA_ARCH` to select another toolkit-supported target. Current
empirical validation is Windows x64 on RTX 4090. Other GPU/platform combinations
have not been validated.

Install the native release into an explicit directory with its CUDA runtime and
cuBLAS DLLs:

```powershell
./update.ps1 -Root <install-directory>
```

The executable is `<install-directory>/bin/teamy-transcriber.exe`. Put that
directory before other installations on your terminal's `PATH` to select it.
The script does not change `PATH` or install weights. It refuses to replace a
different CUDA DLL in a shared directory; use a separate install directory in
that case. With no `-Root`, `./update.ps1` installs into `CARGO_INSTALL_ROOT`,
then `CARGO_HOME`, or the user's `.cargo` directory, in that order. Pass `-Root`
explicitly if you use Cargo's `install.root` configuration.

The application requires an NVIDIA GPU. CPU Whisper, TorchScript, LibTorch,
Burnpack and packed-NPY backends have been removed; the local `pre-cuda-only` tag
preserves them. Existing recordings and transcript history remain readable.
Supply canonical safetensors weights for this runtime. The build has no inference
backend feature switch, and the installer has no `-Backend` option.

Use the recording workflow with `--model-dir <prepared-model>`.
`TEAMY_TRANSCRIBER_CUDA_DEVICE` selects a device index (default zero). Model
preparation and inference remain offline.

Native application builds use TF32 tensor-core matrix multiplication with FP32
storage and reductions. Set `TEAMY_TRANSCRIBER_CUDA_MATH=fp32` for the full-FP32
numerical oracle path (`tf32` selects the default). This is a computation setting;
the safetensors weights stay unchanged.

Single-token decoder projections use a custom FP32 matrix-vector kernel that
fuses bias, GELU and residual addition when requested. It uses vectorized reads
for aligned buffers and a scalar fallback for other shapes/suballocations.
Encoder and multi-token prompt matrix multiplications continue to use cuBLAS.

Weights are loaded once, from single-file or indexed safetensors, into FP32
device storage. Sequential file reads, SIMD FP16 conversion and bounded staging
groups avoid a separate GPU allocation/transfer for every tensor. Decoder K/V
buffers and workspaces are allocated once; the prompt is evaluated as one causal
batch, and requests reset their logical cache position. Only the selected token crosses back to
the CPU during generation. The application owns a bounded inference worker:
model creation, inference, and destruction happen on the same thread.
The GUI retains one `TranscriptionSession` across recordings, so subsequent
transcriptions reuse the loaded model. Choosing a model again, preparing a new
model, or closing the application releases the session. A failed request clears
the cache so repaired model files can be retried. Cancellation retains completed
clip transcripts and leaves the session available for the next request. The
one-shot recording CLI loads a model for each process.

`Engine::transcribe_mel_batch` decodes up to eight independent windows using the
same resident weights. Each window keeps separate self/cross-attention caches
and end-of-text state; results retain input order. Encoders and prompt prefills
run serially, then decoder projections run as matrices and a custom FP32 kernel
fuses each head's attention scores, softmax and weighted values. One small token
array crosses to the CPU per step. Finished slots remain in GPU computation
until the batch finishes but cannot append further output tokens.

Batch storage is allocated lazily and reused, including for smaller subsequent
batches. It does not duplicate weights. Eight large-v3 slots add approximately
5.11 GB of GPU cache/scratch storage beyond the ordinary engine; four add half
that. `batch_workspace_bytes` reports the exact additional allocation. The native
application worker prepares up to eight clips at a time and chooses a batch that
fits currently free GPU memory, reserving 256 MiB. Set
`TEAMY_TRANSCRIBER_CUDA_BATCH_SIZE=1` through `8` to cap the batch (default eight).
One uses the original serial path without additional batch caches. Allocation
failures remain explicit errors; another process can consume VRAM after sizing.

The recording workflow saves completed transcripts in clip order, acknowledging
each save before another result is delivered. CUDA checks cancellation between
decoder steps and encoder invocations; a running kernel and initial model loading
are not interrupted. The bounded audio/frontend preparation also checks for stop
requests. Cancelled clips restore their latest transcript state, including edits,
or return to pending when no transcript exists. Failure preserves completed
transcripts and records failure for the remaining started clips. The resident
worker drains cancelled requests before reuse and joins when its owner closes.
Clip boundaries and export ordering are unchanged by batching.

The desktop event loop sleeps when idle. Worker and tray messages wake it
directly, and text/progress/input changes request a new frame. Microphone
animation retains a bounded timer while recording. Inference no longer shares
the CPU with continuous rasterization of an unchanged transcript.

The resident frontend reuses its FFT plan and skips zero filter coefficients
and wholly padded frames. It accepts complete windows up to 30 seconds; the
existing application workflow remains responsible for chunking longer audio.

## Verification tools

```powershell
cargo test --release --manifest-path native/Cargo.toml --lib
# Explicit GPU kernel checks against scalar numerical oracles:
cargo test --release --manifest-path native/Cargo.toml --lib cuda::tests -- --ignored
cargo test --release --manifest-path native/Cargo.toml --lib loader_tests -- --ignored
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
`tools/compare-asr.ps1` alternates fresh native/Python processes over several
rounds, captures GPU state and binary/script hashes, and produces local summaries.
It can also rotate the pinned [Rust WhisperX ASR adapter](../tools/rust-whisperx-bench/README.md).
The adapter calls the public upstream inference library without modifying it;
it does not time the complete Rust WhisperX CLI.
Python is used only by these development/reference tools.

Benchmark manifests may use `null` for unavailable reference transcripts.
Such audio remains in timing, engine-parity and repetition checks, while word
error scores explicitly report the subset with available references. An empty
reference string means known silence and counts hallucinated words as errors.

`tools/benchmark_whisperx_pipeline.py` measures the larger Python execution path:
audio loading, Silero speech detection, chunk merging, batched CUDA recognition
and result assembly. Supply complete audio recordings, existing CT2 weights,
the canonical generation configuration and a local Silero Hub checkout. It
uses WhisperX's actual VAD and pipeline methods; only Silero's model resolution
uses the supplied local checkout to avoid a network lookup during timing.
The default comparison uses batches of 1 and 16, greedy English decoding and
one Torch CPU thread. Receipts record the model/source hashes and individual
speech-detection times. This exposes batching and segmentation costs that the
per-chunk ASR harness excludes. Word alignment, diarization and file export
remain outside this harness. Compare with the native application's optional
speech-detection path using the same local Silero weights and suppression policy;
account for application persistence/export separately.

```powershell
python tools/benchmark_whisperx_pipeline.py <ct2-model> <wav-list.json> <new-receipt.json> --silero-repo <local-silero-source> --generation-config <canonical-generation-config.json> --dll-dir <ct2-runtime-directory>
```

`examples/workflow_bench.rs` exercises actual import, normalization, chunking,
transcription, persistence and timestamped export. It accepts a model, one WAV
or a JSON array of paths, a new output directory, repetition count, and
`resident` or `cold` session mode. Each request creates a fresh recording;
resident mode reuses only the inference session. Output distinguishes source
clip coverage from transcription accuracy; cold totals include model release,
reported separately as `release_ms`. A prepared `vad/` package selects speech
windows; without it the harness uses fixed 30-second windows. Receipts include
the segmentation kind, speech-weight checksum, silence outcome and elapsed time
to the first committed clip. Run with release/native features.

`tests/transcription_session.rs` is an opt-in real CUDA regression covering
cancellation, persisted partial results, callback errors/panics, model repair at
the same path, changed model selection, and subsequent recordings. Set `TEAMY_TRANSCRIBER_TEST_MODEL`
to a small prepared single-file model and `TEAMY_TRANSCRIBER_TEST_WAV` to a short
speech WAV, then run:

```powershell
cargo test --release --test transcription_session -- --ignored --test-threads=1
```

On Windows, `gui::runtime_test::hidden_gui_reuses_model_and_closes_during_transcription`
is an opt-in release GUI test. Set `TEAMY_TRANSCRIBER_TEST_MODEL`, a speech WAV
longer than 60 seconds in `TEAMY_TRANSCRIBER_TEST_WAV`, and a new directory in
`TEAMY_TRANSCRIBER_GUI_TEST_HOME`. It opens an invisible Winit/Vulkan window,
resizes it, uses real import/transcribe handlers, checks two complete requests, then closes
during a third. It checks projected/persisted text, idle event counts and session
release, and saves `gui-receipt.json` in that isolated home. No tray, hotkey,
microphone capture or dialogs are enabled. The hidden HWND needs explicit
delivery of requested redraws; no desktop screenshot is taken and this is not
a visual-appearance test. The deadline helper does not poll the application
during inference, so missing worker wakeups cannot pass accidentally.

```powershell
cargo test --release --lib gui::runtime_test::hidden_gui_reuses_model_and_closes_during_transcription -- --ignored --exact --test-threads=1
```

For standard Whisper suppression, pass the canonical `generation_config.json`
as the last `wav_bench` argument after `fp32` or `tf32`, and pass the same file
to Python's `--generation-config` (or the runner's `-GenerationConfig`).
`Engine::configure_greedy` applies those suppression fields, including blank/EOT
suppression only on the first token and timestamp suppression for plain text.
It does not interpret beam-search, sampling or language settings from that file.
The application also reads these two suppression fields from an optional
`generation_config.json` inside its prepared model directory. `model prepare`
copies that file when supplied by the canonical source. Malformed policies and
out-of-vocabulary tokens fail explicitly; packages without it retain the legacy
policy. Language remains fixed English with greedy decoding. Benchmark
summaries reject mismatched suppression; unavailable reference token IDs remain
unavailable instead of being counted as a token match.

`batch_bench` validates batched decoding on the same WAV corpus and canonical
generation configuration. It repeats the complete corpus, includes WAV reading
and frontend work, and exercises a partial final batch. Per-batch timing is
reported separately from per-input token/text results. Use explicit release
builds and serialize performance runs:

```powershell
cargo build --release --manifest-path native/Cargo.toml --bin batch_bench
./native/target/release/batch_bench.exe <prepared-model> <wav-list.json> 2 8 tf32 <generation_config.json>
```

The CUDA tests include batched attention against an independent FP64 oracle,
cache strides/tails, vocabulary suppression and device-token embedding. A real
model regression checks changing batch sizes/order, unequal output lengths,
early termination, invalid-input retry and subsequent serial calls. Use a small
prepared model and a short speech file for this test:

```powershell
$env:WHISPER_BATCH_TEST_MODEL = '<prepared-model>'
$env:WHISPER_BATCH_TEST_WAV = '<mono-16k.wav>'
cargo test --release --manifest-path native/Cargo.toml --lib batch::tests -- --ignored --test-threads=1
```

The complete-pipeline runner `tools/compare-pipelines.ps1` compares the real
application workflow with Python WhisperX and the pinned Rust WhisperX public
pipeline, using matching speech detection and greedy English decoding. The Rust
adapter selects active-row batching through public controls; it does not measure
the stock CLI's timestamp-decoding defaults. The application also persists and
exports recordings, while reference timers end after text assembly.

This covers speech detection, transcription and source clip ranges. Beam search,
word alignment, diarization, translation and multilingual accuracy are outside
the validated profile. Numerical checks cover Whisper tiny and large-v3,
including its 128-bin frontend; quality checks include local VCTK, seeded noise
and continuous English earnings-call excerpts. Use the complete pipeline for
end-to-end comparisons; kernel/ASR-only tools describe components.

## Source-defined speech detection

`vad::Silero` implements the 16 kHz, 512-sample Silero model in Rust: context,
reflection padding, spectral projection, four convolutions, recurrent LSTM state
and the final speech probability. CPU dot products select AVX2/FMA when available
and otherwise use scalar code. The runtime reads 15 FP32 safetensors tensors;
it does not interpret a saved graph. The external weights occupy about 1.24 MB.
The [Silero MIT notice](licenses/silero.txt) covers the upstream-derived model
and segmentation behavior.

Each `probabilities` call resets recording state and zero-pads its final frame.
Streaming callers use `step` with 512 normalized mono samples and call `reset`
between recordings. Inputs must be finite and within [-1, 1]. The timestamp
helper implements the upstream default minimum speech/silence durations,
padding and longest-silence splitting, with configurable threshold and maximum
duration. It returns sample ranges; merging retains silence inside each span.
Only 16 kHz is currently supported.

Native CLI/GUI builds can use this detector in the normal recording workflow.
VAD weights are not downloaded or installed automatically. To prepare an exact
reference pair, use the local Silero version that the Python benchmark loads:

```powershell
python tools/export_silero_weights.py <local-silero.jit> <new-silero.safetensors>
./target/release/teamy-transcriber.exe model prepare-vad --model-dir <prepared-whisper-model> --source-weights <new-silero.safetensors>
python tools/benchmark_silero.py <local-silero-source> <silero.safetensors> <wav-list.json> <new-reference.json>
cargo build --release --manifest-path native/Cargo.toml --bin vad_bench --bin pipeline_bench
./native/target/release/vad_bench.exe <silero.safetensors> <wav-list.json> 2
./native/target/release/pipeline_bench.exe <prepared-whisper-model> <silero.safetensors> <wav-list.json> 2 <generation_config.json> 8
```

`model prepare-vad` requires a canonical safetensors Whisper package. It validates
the detector tensors, stages a checksum/size/architecture manifest and MIT notice,
then publishes the `vad/` directory with one rename. It refuses to overwrite an
existing detector. Runtime loading verifies the manifest and weights before
scanning audio, with cancellation checked every 512 samples. Silence is a
successful `no_speech` result and does not initialize Whisper or CUDA.

For a fresh recording, omitting `--chunk-duration-ms` (GUI `AUTO`) detects and
merges speech into windows of at most 30 seconds. Without a prepared detector,
the workflow uses fixed 30-second windows. An explicit duration selects fixed
windows instead. Existing saved clip boundaries, splits and deletions remain
authoritative; changing the detector does not resegment those clips. An empty
silence plan can be retried or replaced by an explicit fixed-duration request.
Every initial plan is one replayable event, so a failed manifest write cannot
leave only part of the planned clips. Old recordings remain readable; new
`ClipsPlanned` events require this version or newer when reopening a recording.

Speech windows retain original source-time offsets and can contain gaps. The
application extracts with integer floor-start/ceil-end sample positions, keeping
the last partial sample. The reference harness follows Python's float-seconds
truncation, which can produce a one-sample difference. Neither path provides word
alignment or speaker diarization. A repeated transcription adds new raw-ASR
versions while preserving earlier edit history.

The opt-in `transcription_speech` test validates package integrity, cancellation,
policy copying, retry and persisted silence. It uses `SILERO_TEST_WEIGHTS` and a
small canonical prepared model in `TEAMY_TRANSCRIBER_TEST_MODEL`; it can run with
an invalid CUDA device index to verify that silence avoids GPU initialization:

```powershell
cargo test --release --test transcription_speech -- --ignored --test-threads=1
```

The exporter needs Python/Torch only during artifact preparation, records
source/output hashes, and refuses to overwrite existing outputs. A similarly
named weights-only file may belong to another Silero revision even if its
shapes match; the reference harness checks every tensor against its loaded model.
`vad_bench` excludes audio I/O from VAD timing and emits all frame probabilities
and boundaries. `pipeline_bench` includes PCM16 WAV loading, VAD, merging, native
CUDA ASR and result assembly, with an optional batch size from one to eight
(default one). It follows WhisperX's
seconds-to-samples truncation for comparable input slices. It excludes alignment,
diarization, application persistence and export, so it is not full-goal acceptance.

Additional opt-in CPU regression tests use explicit external fixtures:

```powershell
python tools/export_silero_policy_cases.py <local-silero-source> <new-policy-cases.json>
$env:SILERO_TEST_WEIGHTS = '<silero.safetensors>'
$env:SILERO_POLICY_CASES = '<policy-cases.json>'
cargo test --release --manifest-path native/Cargo.toml --lib vad:: -- --include-ignored --test-threads=1
```

The policy cases come from upstream code with independent synthetic probability
sequences. They cover threshold hysteresis, short speech, silence, final partial
frames, and maximum-duration cuts at competing silences. Other tests check
recording-state isolation, retry after invalid input, scalar/SIMD agreement,
and malformed weights. Real-audio frame and timestamp parity is checked by the
benchmark receipts rather than by committing model or audio data.
