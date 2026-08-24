# teamy-transcriber implementation plan

Status: active implementation slice; the GUI now owns the first end-to-end local workflow, and the direct native Whisper path has bounded real-model evidence. Human-operated file-picker/capture, parity/JIT, and broader quality evidence remain open.
Plan owner: Teamy
Plan path: G:\Programming\Repos\teamy-transcriber\PLAN.md
Public repository: https://github.com/TeamDman/teamy-transcriber
Last updated: 2026-08-24
Current focus: [~] finish direct native Whisper evidence: parity/JIT comparison, long-input controls, and broader quality checks

This file is the living work contract. A fresh agent should be able to resume from it without reconstructing the project intent from conversation history.

## Progress log

2026-08-06: Initialized the repository from G:\Programming\Repos\teamy-rust-cli using its own init command. Adapted package metadata, MPL-2.0 identity, Teamy-specific path overrides, logging target, profiler target, and CLI documentation. Removed the template-only init command, added the app-specific doctor command, and verified the full check-all.ps1 gate.

2026-08-06: Created the public GitHub repository TeamDman/teamy-transcriber and pushed baseline commit 3f5bb81 to main. GitHub reports the repository as public and MPL-2.0.

2026-08-06: Implemented typed domain state, replayable event records, manifest/NDJSON persistence, local model inventory, a deterministic fake transcription backend, and the doctor/model/recording CLI surfaces. The current model policy assumes locally supplied files and defers CDN acquisition.

2026-08-06: Added a generated stereo-WAV fixture and a replaceable WavMediaAdapter. The adapter inspects metadata, downmixes, resamples to 16 kHz mono, writes a deterministic derived WAV, and is exposed through `recording prepare`. Evidence is bounded to generated WAV fixtures; video, ffmpeg, and microphone paths remain open.

2026-08-06: Added a local WhisperX JSONL worker boundary with `local_files_only=True`, request correlation, model-directory configuration, and a `recording transcribe` command that persists raw ASR results. Full repository validation passes; real inference is unverified because no Python, WhisperX, ffmpeg, or ffprobe executable is available on PATH in the current environment.

2026-08-06: Added persisted `recording clip add` boundaries and normalized WAV clip extraction. Partial clips now become immutable derived artifacts before transcription, and out-of-range requests are rejected instead of being silently clamped.

2026-08-06: Used the local G:\Datasets\VCTK\VCTK-Corpus-smaller corpus as an empirical media check without copying it into the repository. `p225_001.wav` created and normalized successfully to 16 kHz mono (2,051,500 microseconds, 32,824 frames). The corpus includes local ODC-By attribution text; actual WhisperX inference remains unverified because the configured `python` command is missing.

2026-08-06: Added a replaceable `ffmpeg`/`ffprobe` media adapter for non-WAV audio and video sources, with explicit executable environment overrides and probe/normalization diagnostics. It is compile- and parser-tested but not empirically executed because `ffmpeg` and `ffprobe` are absent on PATH in this environment.

2026-08-06: Added typed runtime readiness for Python, the WhisperX worker, and the model directory. Python executable resolution now supports PATH commands such as `python` instead of requiring a filesystem path; `doctor` reports the readiness states and configured ffmpeg tools.

2026-08-06: Added active-clip overlap validation and explicit transcript export. The domain rejects overlapping active ranges, and `recording export` writes the latest committed transcript per active clip to an atomic text artifact with provenance labels.

2026-08-06: Added Windows Core Audio microphone inventory and an explicitly bounded WASAPI capture command. `microphone list` empirically enumerated two active 48 kHz endpoints on this device; capture lifecycle events now persist `created → recording → saved` or `failed` with a replayable failure reason. Capture code was not started implicitly, so a real saved microphone fixture remains pending an explicit capture run.

2026-08-06: Added typed clip transcription lifecycle events. `recording transcribe` now persists `pending/failed → processing → transcribed`, records local-worker failure reasons before returning the error, and `recording show` projects clip status/failure diagnostics. The VCTK recording empirically reached `failed` with the honest missing-`python` reason; no hosted inference was attempted.

2026-08-06: Added a deterministic fixed-duration chunk planner and `recording transcribe --chunk-duration-ms`. Chunk ranges are contiguous and non-overlapping, become immutable clip records before work starts, and retain pending/failed statuses for resumable retries. A 2.0515-second VCTK smoke with 500 ms chunks produced five persisted ranges; the first failed honestly on missing `python` and the remaining four stayed pending.

2026-08-06: Added renderer-neutral presentation state with stable UI/action IDs, contextual key resolution, selected-clip/transcript projection, and failure diagnostics. Headless tests verify that transcript and clip state are projected without depending on a window or renderer; actual GUI, tray, and hotkey integration remain deferred.

2026-08-07: Added the pure-Rust native Whisper vertical slice from the burnt-apple Burn implementation: Whisper log-mel frontend, Burn encoder/decoder, Burnpack loading, tokenizer prompt construction, greedy decoding, and legacy packed-NPY inspection. `recording transcribe` now uses this backend directly; Python and the WhisperX worker are no longer application prerequisites. `cargo check --all-targets` and `cargo test --all-targets` pass. Real model-backed inference remains unverified because no local `model.bpk`/`dims.json`/`tokenizer.json` package was found in the searched workspace/model locations.

2026-08-07: Compared model lifecycle conventions with `G:\Programming\Repos\teamy-tts`. Both projects should share stable model IDs, revision-keyed prepared directories, explicit manifests, artifact hashes, and acquisition receipts. Whisper keeps its task-specific `model.bpk`, `dims.json`, and `tokenizer.json` contract; TTS keeps its role-specific Burnpack/frontend assets. Neither project should treat the other task's tensors or sidecars as interchangeable.

2026-08-08: Closed the GUI media-tool configuration gap. `TOOLS` now lets a user select local `ffmpeg` and `ffprobe` executables, persists those paths, and routes GUI non-WAV/video preparation through the same explicit adapter configuration as the CLI. No model CDN or download flow is added yet; local model files remain an intentional prerequisite.

2026-08-08: Added a visible GUI `NEXT` guidance line to the renderer-neutral
status panel. It now points first-use users to `MODEL`, `IMPORT`/microphone,
`PREPARE`, `TRANSCRIBE`, editing, or `EXPORT` based on actual state, and
reports the appropriate wait/cancel instruction during active operations.
This keeps the zero-to-first-use path actionable without requiring CLI
knowledge; model acquisition and microphone evidence remain separate open
boundaries.

2026-08-08: Audited the local Hugging Face cache and found a
`Systran/faster-whisper-large-v3` snapshot containing a 3.1 GB CTranslate2
`model.bin`, tokenizer, and config. It is not compatible with the active pure
Rust Burn runtime, which requires `model.bpk` + `dims.json` + `tokenizer.json`
or the legacy packed-NPY layout. No CTranslate2-to-Burn converter exists in
this repository, so the GUI reports the native package contract rather than
silently reintroducing a Python/CLI prerequisite.

2026-08-08: Added a GUI-only local preparation path for Whisper PyTorch
checkpoints. MODEL now offers native-package selection or asynchronous
checkpoint conversion; the latter reads checkpoint dimensions and
`model_state_dict`, remaps the known Whisper/Burn naming differences, copies a
local tokenizer, writes `model.bpk` + `dims.json` + `tokenizer.json`, validates
the result, and selects it as the active model. Existing output directories
are never overwritten, no network access is used, and CTranslate2/faster-
whisper `model.bin` remains explicitly unsupported.

2026-08-08: Hardened GUI model preparation as a transaction boundary. Invalid
tokenizers are rejected before checkpoint import, generated packages are
loaded back through the native Burn runtime before activation, and failed
preparations remove only their own newly created partial directory so a retry
does not inherit misleading artifacts.

2026-08-08: Made native model inspection reject tokenizer/model vocabulary
mismatches and missing Whisper control tokens. Existing packages selected
through MODEL now fail visibly before TRANSCRIBE can be enabled instead of
waiting for a decoder-time token lookup or embedding failure.

2026-08-08: Extended first-use recovery guidance for media preparation. When
local `ffmpeg`/`ffprobe` execution fails, the GUI status panel now points the
user to TOOLS to select the missing local executable rather than repeating an
opaque PREPARE step.

2026-08-08: Added cooperative GUI transcription cancellation. The GUI exposes `CANCEL` and `Escape` while recording/transcribing; transcription observes the request at clip boundaries, retains completed clip receipts, and reports `cancelled` explicitly in the shared and CLI reports. The native per-clip decoder remains synchronous, so cancellation does not interrupt an already-running clip.

2026-08-08: Connected persisted clip reordering to the GUI. `LEFT`/`RIGHT` now dispatch the existing replayable `MoveClip` command through shared workflow orchestration, retain the selected clip across reload, and report the new position without introducing an NLE-style timeline editor.

2026-08-08: Added GUI-selectable reversible audio profiles. `AUDIO` cycles the original normalized signal, +6 dB gain, a conservative amplitude noise gate, and a simple voice-EQ filter; derived WAVs and JSON parameter receipts remain separate from the authoritative normalized artifact, and GUI transcription uses the selected derived path.

2026-08-08: Replaced the prepared-media synthetic waveform with a bounded streaming peak envelope from the selected derived WAV. Live microphone capture retains the animated level view until its artifact is available; media tests cover deterministic peak extraction without loading the whole file.

2026-08-08: Re-ran the Ash/Vulkan GUI startup smoke after the workflow controls were added. Under a temporary writable app home, the executable remained responsive for three seconds and exposed a nonzero native window handle; this verifies startup/event-loop/window creation, not a human-operated file-picker or model-backed inference session.

2026-08-08: Added a fontdue-backed CPU text path to the GUI canvas. It rasterizes the visible transcript/status strings through a runtime system-font candidate and retains the deterministic bitmap alphabet as a bounded fallback when no supported font is available. A headless test covers non-ASCII text; embedded-font fixtures, artifact manifests, and the optional Slug GPU path remain open.

2026-08-08: Added bounded per-clip transcription progress to the shared workflow. The GUI now reports `completed/total` clip progress while native Whisper runs, retains the existing cooperative cancellation boundary, and keeps the callback non-blocking and separate from persisted transcript state.

2026-08-08: Added a confirmation-gated GUI `DELETE` action for the selected clip. Deletion is a replayable soft-delete through the shared domain workflow; source and derived audio stay on disk, and the GUI reloads the active clip projection after the event.

2026-08-08: Hardened GUI startup for constrained machines. The platform app home remains preferred, but if it cannot be resolved or created the GUI tries an app-owned LocalAppData, working-directory, then temporary fallback; an empirical smoke with a deliberately unusable configured home still reached a responsive Ash/Vulkan window.

2026-08-08: Retested WASAPI with the endpoint's exact `GetMixFormat()` pointer, a valid closest-match output pointer, and the standard one-second shared buffer. `IsFormatSupported` returned `S_OK`, but `Initialize` still returned `E_INVALIDARG` on the active default endpoint; the capture error now explains the privacy/exclusive-control recovery checks instead of implying a format conversion failure.

2026-08-08: Made shared WASAPI initialization ask the audio engine for its
smallest valid period, and added a standards-compliant fresh-client exclusive
mode probe after shared `E_INVALIDARG`. Both active endpoints still reject the
session here: shared initialization returns `0x80070057`, while exclusive
format support returns `0x88890008` after reporting a 100,000 100-ns default
period and 30,000 100-ns minimum period. The GUI preserves this as a
recoverable recording failure; a saved microphone fixture remains unverified.

2026-08-08: Confirmed Windows microphone consent is `Allow`, but the local
DirectShow inventory reports no audio-only devices in this managed session.
This strengthens the interpretation that the missing saved microphone fixture
is an environment/device capability gap, not a GUI action that silently needs
the CLI.

2026-08-08: Added bounded transcript scrolling to the GUI. Mouse-wheel input over the transcript panel and PageUp/PageDown now select visible wrapped lines without changing the committed transcript or edit provenance; headless state coverage verifies the scroll position cannot move above the beginning.

2026-08-08: Implemented the Windows convenience projection for tray and hotkey
behavior. The GUI now owns a tray icon, restores it after `TaskbarCreated`,
routes `Ctrl+Shift+Space` and tray start/stop/show/exit actions through the
existing reducer, exposes a persisted `HOTKEY ON/OFF` control, and reports
registration conflicts without blocking startup. Runtime interaction with a
human-operated tray menu and a conflict/restart matrix remains open.

2026-08-08: Extended the tiny native Burnpack fixture through the full greedy
decode path. The test now loads the package, constructs Whisper features,
decodes until the end-of-text token, and checks stop reason, dimensions, and
tokenizer output. This proves the native decoder path with a deterministic
fixture; a real Whisper checkpoint and quality transcript remain unverified.

2026-08-08: Retested bounded microphone capture after moving the worker's Core
Audio COM apartment from MTA to STA, following the Windows `IAudioClient`
activation contract. Both enumerated endpoints still reject shared
initialization with `0x80070057`; the exclusive-mode format probe returns
`0x88890008`. The GUI therefore preserves a recoverable failure with explicit
diagnostics; this environment still provides no saved microphone fixture.

2026-08-08: Widened GUI-driven Whisper checkpoint preparation to accept the
two common local weight-container keys, `model_state_dict` and `state_dict`.
Each candidate is loaded into a fresh Burn model, validated for missing/unused
tensors, and only then allowed to create the transactional native package.

2026-08-08: Preserved native-model validation details in the GUI recovery
status. Selecting a malformed model now shows the specific validation reason
alongside the instruction to choose another folder, rather than discarding the
diagnostic behind a generic `MODEL INVALID` label.

2026-08-08: Made the built executable GUI-first for zero-argument launches.
Double-clicking `teamy-transcriber.exe` now opens the same Ash/Vulkan window as
`teamy-transcriber gui`; all explicit CLI subcommands remain unchanged for
scripts and diagnostics. A bounded no-argument process smoke reached a
responsive native window handle.

2026-08-08: Added drag-and-drop media import to the same GUI action path as the
IMPORT dialog. Winit file-drop events now validate supported audio/video
extensions, create the recording, and start the existing asynchronous prepare
workflow; unsupported dropped files receive an explicit status diagnostic.

2026-08-08: Widened the GUI's local PyTorch checkpoint picker to advertise the
common `.bin` and `.ckpt` names in addition to `.pt` and `.pth`. The converter
still validates the checkpoint contents and native Burnpack round trip, so a
CTranslate2 `model.bin` remains an explicit incompatible-format diagnostic.

2026-08-08: Closed the GUI copy half of the transcript output surface. The
transcript panel now exposes `COPY`, copying the visible committed text or edit
draft through the native clipboard and reporting clipboard failures in the
status line; `EXPORT` remains the explicit persisted output path.

2026-08-08: Made button labels responsive to their hit-target width, scaling
and centering text at the 720-pixel minimum window size so the compact
COPY/EXPORT/DELETE output row remains usable after resize.

2026-08-08: Added a CPAL/WASAPI recovery route after the direct endpoint
capture path fails. It preserves friendly-name endpoint selection, converts
the device's native sample format to mono f32, and keeps the same bounded or
stop-controlled WAV artifact contract. A rebuilt capture smoke exercised the
fallback here; CPAL also receives `0x80070057` during stream creation, so this
managed session still cannot provide a saved microphone fixture.

2026-08-08: Added a visible GUI `REVEAL RECORDING` action. It opens the active
recording's persisted directory through the platform file manager, exposing
the source/derived audio, event receipt, manifest, and transcript artifacts
without requiring the CLI; the action is backed by the same stable
`RecordingStore::recording_dir` path used by persistence.

2026-08-08: Added replayable GUI clip editing boundaries. `S` replaces the
selected active clip with two midpoint source ranges; `A` replaces the selected
clip and its next active source-time-adjacent clip with one combined range.
Both operations retain replaced clips and derived artifacts, start the new
clip(s) pending, reload the active projection, and require explicit
re-transcription. Shared workflow tests cover split-then-append replay.

2026-08-08: Made the split/append operations visible bottom-row GUI controls
and added hit-target coverage. Hardened close behavior so an active recording
or transcription receives its stop/cancel request and the event loop waits
for the terminal worker message before exiting; this preserves the persisted
completion/failure receipt during window shutdown.

2026-08-08: Added an ffmpeg metadata fallback for media inspection when the
selected ffprobe executable is missing or fails. Using the local VCTK
`p225_001.wav` speech sample, ShareX's local ffmpeg generated a temporary MP4;
the shared recording-prepare workflow then successfully extracted and
normalized its audio to 16 kHz mono (981,312 microseconds, 15,701 frames) with
the same ffmpeg binary supplied for both tool paths. The temporary video and
recording home were removed after the smoke.

2026-08-08: Made the GUI ffprobe selection optional after selecting ffmpeg.
Cancelling that second dialog now persists the same ffmpeg path as the probe
path and reports that the metadata fallback is enabled, so the local video
workflow remains discoverable without a CLI step.

2026-08-07: Added the first native GUI slice using the Ash 0.38, ash-window 0.13,
raw-window-handle 0.6, and Winit 0.30 stack already used by cursor-latency and
teamy-terminal. `cargo run -- gui` now creates the Winit window, Vulkan surface,
physical device, logical device, swapchain, transfer command buffer, and redraw
loop. The CPU reference layout follows the supplied microphone-centered sketch
and includes a clickable recording-state toggle, waveform, and transcript panel.
`cargo check --all-targets`, `cargo build`, and focused GUI state/bitmap tests
pass; a live launch reached event-loop, window, and Vulkan-renderer startup and
remained responsive. Capture, persistence, and action/domain dispatch are now
implemented through the shared workflow module; human-operated workflow and
model-backed inference evidence remain open.

2026-08-08: Connected the native GUI to shared workflow orchestration. The GUI
can choose and inspect a local native model directory, import supported media
through a file picker and auto-prepare it, select microphones, capture until a
visible stop action, transcribe asynchronously with native Rust Whisper, review
and edit committed text with user-edit provenance, and export through a save
dialog. Settings and the most recent recording are restored on restart. The CLI
transcription/export/bounded-capture commands now call the same workflow module;
shared workflow/domain/export tests pass. A real GUI capture and model-backed
inference run remain explicit empirical evidence gaps.

2026-08-08: Exercised bounded capture against both enumerated active microphone
endpoints. Each advertises a 48 kHz, two-channel float mix format and passes the
shared-engine format probe, but WASAPI initialization returns `0x80070057` in
this managed audio session. A temporary CPAL probe reproduced the same failure
on both devices, so this is recorded as an unverified device/session limitation
rather than a successful capture claim; native errors now include activation,
format, initialization, and stream-start context.

2026-08-08: The repository's `check-all.ps1` quality gate passes after the
cancellable WASAPI helper was moved under the correct unsafe-lint boundary. The
gate covers nightly formatting, clippy with warnings denied, an all-feature
build, and the full test suite. A built GUI executable also reached the event
loop and stayed responsive with an explicit writable
`TEAMY_TRANSCRIBER_HOME_DIR` smoke directory; the default profile's OS config
location is inaccessible in this managed shell, so that launch was not used as
evidence against normal user-machine behavior.

2026-08-08: Extended the GUI to expose the persisted clip workflow instead of
forcing full-recording transcription. Full, 10-second, 30-second, and
60-second chunk presets now persist in GUI settings; previous/next clip review,
saved-recording cycling, preferred-recording recovery, model structure
validation, and close-while-recording stop signaling are implemented. The
shared tests and full quality gate pass again.

2026-08-08: Added compact active-model, recording-source, clip, operation, and
model-readiness labels to the canvas so recovery does not depend on CLI
inspection. The GUI-only path remains blocked only on empirical execution with
a real local model/device fixture, not on missing control wiring.

2026-08-08: Re-ran the full quality gate and a launch smoke after the recovery
labels/control-safety changes. `check-all.ps1` passes, and the GUI process
reaches the event loop and remains responsive with a writable app-home override.

2026-08-23: Audited and removed only an interrupted, uncompilable local-LLM GUI
WIP that referenced undefined actions/configuration; the existing Ash/Vulkan
GUI, local native-model preparation, and `REVEAL RECORDING` work remain intact.
The native model contract is unchanged and no CDN/model acquisition was added.

2026-08-23: Added the portable `fixtures/vctk-p230-385.json` descriptor and
typed `verify speech vctk-p230-385` command. The descriptor records the
user-owned VCTK 0.80 attribution, relative sample path, expected source hash
`1580043a209675e1a5caa32fe41e1cb4725e496728ed62f7488cc6c72772f9f0`, reference
text, and deterministic 16 kHz mono normalization policy; no corpus media is
tracked. Missing corpus produces `unavailable` with `passed: false`.

2026-08-23: Ran the real VCTK sample through import, deterministic WAV
normalization, persisted manifest/events, and replay evidence. The normalized
artifact is 153,176 bytes with SHA-256
`692424b61419a0fd8a59bb0e6237c5f21cbc732fede91ec6c3651fcb5c259ef1`; a
missing native model produces an honest `failed` receipt at the model stage
while replay reports one event, next sequence 2, and a matching materialized
recording. A cached `Systran/faster-whisper-large-v3` CTranslate2 directory was
fingerprinted but also fails honestly because it contains `model.bin` rather
than the required native Burnpack/legacy-NPY layout. The native model-backed
transcript and CER/WER oracle remain unverified pending a compatible local
model package.

2026-08-23: `cargo fmt --all`, `cargo check --all-targets`, and
`cargo test --all-targets` pass (38 library tests plus all integration suites).

2026-08-23: The native ASR engine decision is now aligned with `teamy-tts`:
target Rust `tch` 0.24.0 over LibTorch 2.11.0+cu128, with no Python runtime.
The matching local LibTorch installation exists at
`G:\Programming\Caches\teamy-tts-libtorch-2.11.0-cu128\libtorch`, but
`LIBTORCH` is not currently exported. The installed GLaDOS TorchScript model
is TTS-only and cannot serve as a Whisper recognizer.

2026-08-23: Audited local model locations using a narrowly scoped
`whisper-model-audit-20260823` teamy-mft profile plus targeted filesystem
searches. The machine has the 3.1 GB WhisperX/faster-whisper CTranslate2
`model.bin` and tokenizer, but no discovered Whisper PyTorch/TorchScript ASR
checkpoint or native Burn package. The WhisperX `whisperx/assets/pytorch_model.bin`
file is a 17 MB alignment asset, not the primary recognizer. No USN-backed MFT
indexes were available, so the profile could not query indexed paths; no drive
journals were enabled for this audit.

2026-08-23: The supplied long-form WAV is a 44.1 kHz, stereo, 16-bit PCM file
of approximately 29 minutes 03 seconds. The current native frontend is capped
at Whisper's 30-second context, so long-form work must use persisted chunks;
the current no-chunk path would not be an acceptable full-file transcription
route. A first multi-speaker fixture can concatenate user-owned VCTK clips with
known speaker/time metadata, but this will test segmentation/label plumbing,
not prove diarization until a native speaker/VAD model is added.

2026-08-23: Added the preferred tch/LibTorch TorchScript Whisper path. The
project now pins `tch` 0.24.0, recognizes `model.pt` + `dims.json` +
`tokenizer.json`, loads an exported graph's `encoder`/`decoder` methods, and
uses `TEAMY_TRANSCRIBER_TORCH_DEVICE` with the same CUDA/CPU policy as
`teamy-tts`. The matching local LibTorch build compiles successfully and all
Rust tests pass. No compatible Whisper TorchScript model was found, so this is
runtime/package evidence, not ASR quality evidence. Omitted transcription
chunking now persists Whisper-sized 30-second ranges rather than allowing a
long file to be silently truncated to its first window.
The default per-window decode budget is now the Whisper text-context limit
(448 generated tokens) and still stops early on end-of-text; callers can lower
it explicitly when bounded latency is more important than completeness.
The LibTorch model and validated artifacts are now cached for the lifetime of a
transcription job instead of being loaded once per chunk, and inference runs
under `no_grad`.

2026-08-23: Refined the model strategy: the final preparation path should be
Rust-native and accept canonical Whisper `safetensors`/config/tokenizer
artifacts, remapping them into the direct `tch` model implementation. The
WhisperX `faster-whisper`/CTranslate2 `model.bin` remains a distinct converted
format; it cannot be losslessly reversed into LibTorch weights by `tch`. Python
may be used as a logits/JIT-export oracle during development, but never as a
runtime dependency. TorchScript is an optional optimized artifact after eager
`tch` parity, not a hard product requirement.

2026-08-23: Implemented the first Rust-native canonical model package path.
`model prepare --source-dir <hf-dir> --output-dir <native-dir>` accepts a
single `model.safetensors` plus matching `config.json` and `tokenizer.json`, derives and writes
`dims.json`, memory-maps and validates the safetensor manifest, and refuses to
overwrite an existing output. The direct LibTorch runtime loads F32/F16/BF16
weights into resident `tch` tensors and runs the Whisper encoder, causal
decoder self-attention, cross-attention, tied/output projection, and greedy
decoding without Python. Decoder self-attention K/V are now cached per layer
and cross-attention K/V are precomputed once per window; the cached logits
match the full-prefix reference on a deterministic synthetic model. A
synthetic encoder/decoder shape test and the full repository gate pass.
Single-file and indexed-shard safetensors are both accepted, with manifest
path-safety and cross-shard duplicate checks. TorchScript export/JIT
benchmarking, logits parity against Python, and real model-backed VCTK quality
remain pending.

2026-08-24: Prepared a real public `openai/whisper-tiny` Hugging Face package
(`model.safetensors`, `config.json`, and `tokenizer.json`) with the Rust
`model prepare` command. The resulting native package was loaded by direct
`tch`/LibTorch inference on CPU, with the model resident for the transcription
job and cached decoder attention state. The real VCTK canary
`vctk-p230-385` traversed import, deterministic 16 kHz mono normalization,
persistence, model inspection, direct native inference, raw-ASR commit, and
replay. Receipt `target/vctk-p230-385-tiny-2.json` reports exact text
`If you can get it.`, CER 0, WER 0, successful replay, and 10,859 ms total
runtime (3,858 ms model load, 6,944 ms transcription). This is bounded
empirical evidence for the Python-free ASR path, not evidence for large-v3
quality, long-form completeness, alignment, diarization, GPU performance, or
Python logits/JIT parity.

2026-08-24: Fixed two real-model issues exposed by that canary: final logits
are flattened before host conversion, and Whisper `fc1.bias` validation expects
the feed-forward width (`4 * state`) rather than the hidden state width. With
`LIBTORCH=G:\\Programming\\Caches\\teamy-tts-libtorch-2.11.0-cu128\\libtorch`,
`check-all.ps1` passes formatting, clippy, build, 44 library tests, and all
integration suites. The working tree contains only these two correctness fixes
until they are committed.

## Plan operating rules

1. Keep the requirements ledger and traceability current as decisions change.
2. Use [ ] pending, [~] in progress, [x] complete, and [!] blocked or needing an explicit decision.
3. Maintain at most one current focus.
4. Record evidence strength honestly: exhaustive, bounded, symbolic, queried, sampled, empirical, experimental, research-only, or unverified.
5. Treat local-first as a product boundary: no audio, transcript, or prompt is sent to a hosted service by default.
6. Prefer a narrow end-to-end vertical slice over broad scaffolding.
7. Do not claim real-time quality, GPU speed, formal coverage, or clean-machine installability until the corresponding evidence exists.
8. When a plan item changes, update its validation and completion criteria at the same time.

## Native model artifact convention

The active native Whisper runtime is Rust `tch` over the same pinned LibTorch
distribution selected by `teamy-tts`: `tch` 0.24.0 over LibTorch
`2.11.0+cu128`. The preferred Whisper package must be self-contained and
identify the exact LibTorch runtime/model revision rather than relying on
Python. `LIBTORCH` is a build-time input; `TEAMY_TRANSCRIBER_TORCH_DEVICE`
selects CUDA device `0` by default or CPU with `-1`.

The preferred preparation input is the canonical Whisper model package:

- a Hugging Face `model.safetensors` file or
  `model.safetensors.index.json` plus its shard files;
- `config.json`/processor metadata from the same model revision;
- `tokenizer.json` and any required tokenizer sidecars.

Rust preparation should validate and remap those tensors into the project's
versioned native package. The preferred runtime package is:

- direct `tch` weights and dimensions, with an optional `model.pt` TorchScript
  cache for measured hot paths;
- `dims.json`: Whisper dimensions and context limits;
- `tokenizer.json`: the tokenizer used for the language/task prompt and text
  decoding.

Until the migration passes the VCTK canary, the existing Burn package remains
readable as a compatibility/reference and conversion experiment. Its current
package is:

- `model.bpk`: Burnpack weights for the handwritten Burn Whisper model;
- `dims.json`: the dimensions needed to instantiate that model;
- `tokenizer.json`: the tokenizer used for the language/task prompt and text
  decoding.

The older `encoder/` and `decoder/` packed-NPY layout remains readable during
migration. The GUI's existing checkpoint preparation still creates the Burn
compatibility package; the CLI's safetensors preparation creates the direct
LibTorch package. The application does not download assets or pretend to
convert CTranslate2/faster-whisper `model.bin` directories. Transcription only
consumes an already prepared native package.

The shared cross-project registry shape follows `teamy-tts`: stable model ID,
revision, prepared-directory path, package status, source/archive fingerprint,
manifest version, and per-file hashes. A future `model-manifest.json` (or the
equivalent project-specific manifest) should identify task, model family,
converter version, backend, LibTorch build, and artifact roles.
`teamy-transcriber` should adopt the shared registry metadata shape while
retaining Whisper-specific artifacts; `teamy-tts` should retain its separate
ForwardTacotron, HiFiGAN, phonemizer, and voice artifacts. This gives the
projects compatible native-runtime acquisition and verification tooling without
falsely claiming weight or tokenizer compatibility.

## User guidance ledger

| ID | Guidance or constraint | Status | Traceability |
|---|---|---|---|
| U1 | Build teamy-transcriber in G:\Programming\Repos\teamy-transcriber. | Confirmed | Scope, W1 |
| U2 | Transcribe imported audio files. | Confirmed | Scope, W3, W6, W9 |
| U3 | Transcribe imported video files. | Confirmed | Scope, W3, W6, W9 |
| U4 | Support microphone recording. | Confirmed | Scope, W5, W10 |
| U5 | Keep the product general and easy to use, but narrow enough to be coherent. | Confirmed | Product boundary, W1, W6 |
| U6 | Run WhisperX locally and account for model/runtime files on a new device. | Confirmed | G4, G5, W6, W9 |
| U7 | Use a local large language model, with teamy-llm-service as relevant prior art. | Confirmed, role open | G3, W20 |
| U8 | Learn from previous Rust and Burn work without assuming a Burn rewrite is the first milestone. | Confirmed | W1, W6, R2 |
| U9 | Provide a GUI centered on a skeuomorphic microphone and visible transcription text. | Confirmed | G1, W15 |
| U10 | Save recordings somewhere predictable and recoverable. | Confirmed | G2, W2, W5 |
| U11 | Emit clips of audio predictably. | Confirmed | G8, W2, W3, W5 |
| U12 | Include convenience operations such as noise reduction and equalization. | Confirmed, scope to gate | G8, W19 |
| U13 | Support moving or reordering chunks without pretending to be an NLE. | Confirmed | Scope, W4, W19 |
| U14 | Focus the project on a coherent transcription utility, not a DAW or nonlinear editor. | Confirmed | Non-goals |
| U15 | Reuse lessons from Ash/Vulkan window work and cursor-latency measurement. | Confirmed | G1, W11, W18 |
| U16 | Reuse system-tray lessons from piing, tb, and related Windows projects. | Confirmed | G1, W16 |
| U17 | Make text rendering correctness a first-class concern. | Confirmed | G10, W17, W18 |
| U18 | Compare CPU fontdue rendering with GPU Slug rendering and investigate artifacts. | Confirmed | G10, W17, W18 |
| U19 | Use Teamy Studio and Teamy Terminal as implementation evidence and prior art. | Confirmed | Foundation, W1, W2, W18 |
| U20 | Use the formal-methods lessons from Poche. | Confirmed | W4, W21, W22, evidence policy |
| U21 | Use the supplied planning skill and keep a resumable implementation plan. | Confirmed | This document |
| U22 | Keep microphone, transcript, and external-app output boundaries explicit. | Confirmed | W2, W5, W7 |
| U23 | Preserve diagnostics, progress, cancellation, and timing evidence for long-running work. | Confirmed from prior work | W7, W9, W12, W22 |
| U24 | Use action-first UI semantics, contextual keyboard bindings, stable IDs, and action-backed widgets. | Transfer requirement | W14, W16 |
| U25 | Separate domain state, commands/events, projection, renderer, and transport. | Transfer requirement | Architecture, W2, W7, W14, W18 |
| U26 | Use typed boundaries, replayable events, machine-readable receipts, and explicit stop conditions where practical. | Transfer requirement | W4, W7, W21, W22 |
| U27 | Distinguish proven facts, hypotheses, deferred decisions, and non-claims. | Transfer requirement | Gates, risks, W22, acceptance matrix |
| U28 | For the current implementation, assume model files are already available locally; defer CDN acquisition and distribution. | Confirmed | G5, W8, W9, W23 |
| U29 | The GUI must cover the usable workflow from zero; CLI commands must not be required for model preparation or other auxiliary setup. | Confirmed | W8, W15 |
| U30 | Begin implementation in the public MPL-2.0 teamy-transcriber repository; CDN/model distribution is later. | Confirmed | W1, W8, W23 |
| U31 | Use the VCTK corpus when useful for local media validation, without copying it into the repository. | Confirmed | W3, W6, W9 |

## Source and implementation evidence

### Local verified foundation

| Area | Evidence | Transferable lesson |
|---|---|---|
| Teamy Studio | G:\Programming\Repos\Teamy-Studio\docs\spec\product\audio-input.md and G:\Programming\Repos\Teamy-Studio\docs\notes\audio-input-inbox-plan.md | Rust should own capture, normalization, buffering, feature preparation, result staging, and explicit output routing; the active native Whisper path keeps inference local and in-process. |
| Teamy Studio Whisper work | G:\Programming\Repos\Teamy-Studio\docs\notes\whisperx-optimization-plan.md | Long inputs need bounded chunking, ordered assembly, progress, timing, and resource-aware worker selection. |
| Teamy Studio architecture | G:\Programming\Repos\Teamy-Studio\Cargo.toml and AGENTS.md | Crate boundaries and preservation-first refactors are useful, but this project must avoid inheriting Teamy Studio's experimental breadth. |
| whisper-burn | G:\Programming\Repos\whisper-burn\README.md | A Rust/Burn path is valuable as a future backend or verification experiment; it has nontrivial model conversion and model-file assumptions. |
| whisperX | G:\Programming\Repos\whisperX\README.md and pyproject.toml | WhisperX is a multi-component runtime involving faster-whisper, VAD, alignment, optional diarization, CUDA/CPU choices, and model downloads/cache state. |
| teamy-llm-service | G:\Programming\Repos\teamy-llm-service\README.md | Persistent local model service, model registry/cache, cancellation, and per-model scheduling are useful patterns; current GPU-only behavior is not a product guarantee. |
| Teamy Terminal | G:\Programming\Repos\teamy-terminal\README.md and renderer/font crates | Vulkan, Ash, offscreen rendering, CPU references, Slug contracts, and renderer/transport separation are relevant to the text surface. |
| Teamy Terminal SFM brief | C:\Users\Teamy\.codex\attachments\d462b59d-93a4-48ab-a74d-f014951f7340\pasted-text.txt | Use action-first semantics, stable IDs, contextual bindings, typed addresses, complete renderer/transport tuples, push-based delivery, and machine-readable visual evidence. |
| Poche | GitHub repository TeamDman/Poche, including PLAN.md and docs/explicit-checker.md | Use typed transitions, explicit finite scopes, replay, deterministic checking, honest evidence labels, acceptance matrices, and non-claims. |
| Poche transfer brief | C:\Users\Teamy\.codex\attachments\011d4bad-66d6-4a69-a446-fe957b4a38c6\pasted-text.txt | Freeze vocabulary first; build a pure kernel and one human-observable slice; keep session/identity/authority/transport/projection/rendering separate. |
| voice2text | GitHub repository TeamDman/voice2text | Existing push-to-talk typing proves the use case, but blind typing into a focused external app is too risky as the default output behavior. |
| piing and tb | G:\Programming\Repos\piing and G:\Programming\Repos\tb | Tray lifecycle, hidden console behavior, global hotkey, logs, config, and taskbar affordance patterns are available prior art. |
| teamy-subs | G:\Programming\Repos\teamy-subs | Keep pure media/subtitle domain logic separate from subprocess adapters; use fixtures, golden tests, and explicit unsupported syntax decisions. |
| cursor-latency | G:\Programming\Repos\cursor-latency | User-perceived latency should be measured end to end rather than inferred from renderer timing. |
| VCTK local corpus | G:\Datasets\VCTK\VCTK-Corpus-smaller\README and COPYING | Use as a user-owned speech-quality/media corpus for empirical checks; preserve its ODC-By attribution and keep it out of Git. |

### Source limitations

- Poche was inspected through its GitHub repository because no local Poche checkout was found.
- tv.exe was not found on PATH with Get-Command tv.exe. Its origin is therefore unverified and is not a dependency assumption.
- Repository contents and prior plans describe intent and experiments. They are not proof that every behavior is production-ready.

## Product definition

### Purpose

teamy-transcriber captures or imports speech, creates durable audio clips, runs local transcription, and lets a person inspect, edit, reorder, and export the resulting text with clear provenance.

### First-release scope

1. Windows desktop GUI with a microphone-centered recording surface.
2. File import for common audio and video formats through a replaceable media adapter.
3. Microphone capture with an explicit armed/recording/stopped state.
4. Authoritative recording and clip manifests stored under an app-owned data directory.
5. Local native Whisper ASR backend with model/package doctor, preparation, progress, cancellation, and staged results.
6. Transcript presentation with source/clip provenance and explicit commit/export actions.
7. Predictable clip split, move, append, and delete operations with undo or replayable history.
8. Small, reversible audio preparation profiles: gain, noise reduction, equalization, and resampling where required by the backend.
9. Optional local LLM actions that are visibly separate from transcription, such as cleanup or summarization; no silent overwrite of the raw transcript.
10. Tray presence and hotkeys only after the core action model and privacy boundaries are stable.

### Explicit non-goals

- A digital audio workstation.
- A nonlinear video editor.
- Cloud transcription or cloud LLM inference in the default product path.
- Blind typing into whichever external window currently has focus.
- A requirement that the first native slice include full WhisperX alignment, diarization, or timestamp behavior.
- Diarization, translation, speaker labeling, or word-level alignment as first-slice blockers. They may be capabilities added behind explicit gates.
- A claim that Vulkan/Slug is required for the first usable transcription flow.

## Architecture contract

The design separates semantic truth from presentation and device-specific mechanisms.

~~~mermaid
flowchart LR
    A[Audio or video file] --> M[Media adapter]
    B[Microphone] --> C[Capture adapter]
    M --> N[Normalize and prepare]
    C --> N
    N --> S[Immutable source and clip store]
    S --> T[Native Whisper backend]
    T --> R[Typed transcript results]
    R --> P[Transcript projection]
    R --> L[Optional local LLM action]
    P --> V[GUI, tray, CLI, export]
    E[Typed commands and events] --> K[Domain kernel]
    K --> S
    K --> P
    K --> D[Diagnostics and receipts]
~~~

### Domain vocabulary

The initial kernel should define explicit types for:

- AssetId, RecordingId, ClipId, TranscriptId, JobId, ModelId, SessionId, and ActionId.
- TimeRange in source time and SampleRange in normalized audio time.
- AssetKind: audio file, video file, microphone recording.
- RecordingState: idle, armed, recording, stopping, saved, failed.
- ClipState: pending, ready, processing, transcribed, edited, deleted.
- JobState: queued, preparing, running, cancelling, completed, failed, cancelled.
- Transcript provenance: raw ASR, aligned ASR, user edit, local LLM derivative, imported text.
- Runtime state: unavailable, checking, ready, preparing, degraded, failed.

No subsystem should exchange unvalidated path strings, floating-point time ranges, or untyped status strings when a domain type can express the contract.

### Commands, events, and projections

Commands represent requested intent: import asset, arm microphone, start recording, stop recording, split clip, move clip, prepare model, transcribe clip, cancel job, edit transcript, run local transform, export transcript, and reveal recording.

Events represent accepted state changes. They must carry stable IDs, monotonic sequence numbers, schema version, timestamp, and enough data to replay the semantic state. Canonical NDJSON is the initial receipt format.

The GUI, CLI, tray, and future automation entry points must converge on the same action executor. Presentation code may propose an action but may not mutate the domain directly.

### Storage layout proposal

The default app home should be configurable but deterministic:

~~~text
app-home/
  recordings/
    <recording-id>/
      manifest.json
      source/
      audio/
      transcripts/
      events.ndjson
      receipts/
  models/
  runtimes/
  logs/
  cache/
~~~

The manifest records source provenance, normalization, clip boundaries, checksums, model/runtime identifiers, transcript versions, and derived-file relationships. Raw recordings and raw ASR output are retained when the user has not explicitly removed them.

### Backend boundary

The first backend contract should expose:

1. doctor: report runtime, model, device, and dependency readiness;
2. describe: report capabilities, expected sample format, model identity, and limits;
3. prepare: acquire or validate runtime/model assets with progress;
4. submit: accept an immutable clip or prepared feature payload;
5. stream: emit ordered partial/staged results and diagnostics;
6. cancel: stop work without corrupting the source or committed transcript;
7. shutdown: release resources and report final metrics.

The current reference backend is a pure-Rust Burn Whisper implementation. The
target backend is Rust `tch`/LibTorch, using the same pinned runtime family as
`teamy-tts` and a separately prepared Whisper ASR model. WhisperX-style VAD,
alignment, diarization, timestamps, and accelerator strategies remain separate
capabilities to add around the native ASR core rather than hidden Python
prerequisites.

### Rendering and transport boundary

The semantic transcript projection must be renderer-neutral. The first visual backend may use native controls or a simple renderer if that closes the end-to-end slice sooner.

If Ash/Vulkan/Slug is used, renderer and transport are separate axes. A complete render tuple includes renderer, transport, dimensions, generation, and payload format; switching any member creates a fresh generation and a full resync. GPU output must be compared with a CPU fontdue reference and validated with fixtures before performance claims are made.

## Design gates

These are decisions or evidence gates, not implementation chores. A gate may be closed by a documented decision with rationale, or by evidence where the question is empirical.

| ID | Gate | Current position | Exit evidence |
|---|---|---|---|
| G1 | GUI/window technology | Ash/Vulkan is a candidate; native/simple presentation is allowed for the first slice. | One documented choice with a visible window, close behavior, and input routing. |
| G2 | Recording/data home | App-owned deterministic home with configurable root. | Manifest round-trip and recovery test on a fresh directory. |
| G3 | LLM role | Optional post-ASR/local transformation, never the authoritative ASR and never a silent overwrite. | Typed command/result contract and a raw-vs-derived transcript example. |
| G4 | WhisperX capability set | Start with local ASR for imported/recorded normalized audio; alignment/diarization are later capabilities. | Capability descriptor and one successful local transcription. |
| G5 | Runtime/model bootstrap | Current slice assumes local model files and only inspects their directory; no downloader or CDN path yet. Runtime validation and future acquisition remain separate. | Local model inventory, runtime readiness contract, and later clean-machine policy. |
| G6 | Media decoding | Use a replaceable adapter; likely ffmpeg/ffprobe or a documented library path. | Supported-format matrix, fixture corpus, and failure diagnostics. |
| G7 | Microphone contract | Rust owns capture and buffering; explicit sample format, device identity, permission/failure states. | Device list and saved recording fixture, with no implicit external output. |
| G8 | Clip semantics and processing | Immutable source plus derived clips; operations are reversible or replayable. | Boundary/ordering tests and before/after artifact receipts. |
| G9 | Authoritative audio format | Choose source preservation plus normalized transcription format. | Chosen format, sample-rate/downmix policy, and checksums. |
| G10 | Text rendering acceptance | CPU fontdue reference first; Slug GPU path only after artifact fixtures and bounds are stable. | Pixel/contract tests with known artifact cases and explicit tolerance. |
| G11 | CLI and diagnostics surface | CLI is a diagnostic/control surface, not a separate semantic implementation. | doctor, model, recording, and transcribe commands backed by same actions. |
| G12 | Formal scope | Bounded checker for the domain kernel, not a claim about model quality or all UI behavior. | Named finite scope, state/edge counts, limits, and shortest counterexample. |
| G13 | Distribution and licenses | Verify WhisperX, model, ffmpeg, CUDA, and font/runtime licensing before packaging. | Recorded license inventory and redistribution decision. |
| G14 | Quality corpus | Small checked-in fixtures plus user-owned local corpus policy. | Golden transcripts, timing expectations, and known non-claims. |

## Work breakdown

### Phase 1: contract and workspace

#### W1 [~] Freeze the product contract

Work: Close or explicitly defer G1-G14. Write the first capability matrix, vocabulary, privacy boundary, supported formats, and first-slice definition.

Validation: A second reader can map every U-row and transfer lesson to a scope item, gate, work item, or non-goal.

Completion: This plan contains the decision, rationale, evidence strength, owner, and next review date for every gate.

#### W2 [~] Scaffold the semantic workspace

Work: Extend the template-backed Rust application shell into a small workspace with domain, storage, media, transcription-client, application, and UI-facing crates only when each boundary is justified. Preserve the template's CLI, path, logging, cancellation, lint, test, and profiling conventions.

Validation: cargo check, unit tests, and a crate-dependency review show no UI-to-model or renderer-to-storage shortcuts.

Completion: A pure domain crate can load a manifest, apply valid commands, reject invalid transitions, and emit replayable events. The current template-backed shell is complete; the domain/storage extension remains.

#### W3 [~] Establish a fixture corpus

Work: Add short licensed or generated audio fixtures, at least one video fixture, silence/noise/speech cases, and expected normalized metadata. Keep large/private recordings out of Git.

Validation: The generated stereo WAV fixture and the user-owned VCTK sample
exercise metadata, downmixing, resampling, output duration, and the typed
import-to-persist canary without a microphone, GPU, or hosted service. The
portable descriptor and source checksum are checked in, while corpus media is
not; missing-corpus and incompatible-model outcomes are empirically non-passing
receipts. Video, silence/noise cases, and a native-model-backed oracle remain
pending.

Completion: A fresh checkout can exercise import, normalization metadata, clip boundaries, and a fake transcription backend.

### Phase 2: domain, storage, and media

#### W4 [~] Implement typed recording and clip state

Work: Implement IDs, time/sample ranges, recording/job/clip state machines, commands, events, validation, replay, and deterministic serialization. The current kernel rejects invalid ranges, duplicate IDs, deleted-transcript commits, and overlapping active clip ranges while permitting adjacent ranges.

Validation: Unit tests cover invalid ranges, overlap/boundary rules, ordering, duplicate IDs, deleted clips, and replay equivalence. Job-state and cancellation transitions remain pending.

Completion: A complete headless session can import an asset, create clips, move them, and recover from its event receipt.

#### W5 [ ] Implement the recording manifest and artifact store

Work: Write atomic manifests, immutable source references, derived clip records, transcript versions, checksums, and receipts. Define retention and deletion behavior.

Validation: Round-trip tests cover interrupted writes, missing derived files, path relocation, duplicate preparation, and recovery after restart.

Completion: The application can show exactly which source and clip produced a transcript and where the recording is stored.

#### W6 [~] Add media import and normalization

Work: Define the media adapter, supported-format matrix, ffmpeg/ffprobe or library integration, mono/sample-rate policy, and source-time to normalized-time mapping. The bounded implementation now includes a WAV adapter and an ffmpeg/ffprobe adapter with an ffmpeg diagnostic fallback for non-WAV audio/video; support remains dependent on local tool availability and fixture coverage.

Validation: The generated WAV fixture and VCTK sample compare duration, channels, sample count, and normalized output metadata. ffprobe parser tests, ffmpeg-fallback parser tests, missing-tool diagnostics, and one local video normalization smoke pass; source-time offset fixtures and broader format coverage remain pending.

Completion: Imported audio and video produce normalized clips with deterministic metadata and no model dependency.

### Phase 3: local runtime and capture

#### W7 [~] Define the transcription backend protocol

Work: Implement a fake backend, capability descriptors, typed native-model
readiness, and a native Rust submit/decode boundary. Full doctor/describe/
prepare/stream/cancel/shutdown messages, schema versions, and bounded progress
events remain pending.

Validation: Fake backend behavior, local-only capability reporting, native
model configuration rejection, Burn frontend reference behavior, model-shape
checks, greedy-decoder behavior, and empty-result handling are covered at the
current boundary. Long-running lifecycle behavior and a real model fixture
remain unverified.

Completion: The app can run a fully observable transcription job without real inference and can replay its receipt.

#### W8 [~] Build model and runtime preparation

Work: Separate executable/runtime cache from model cache. For this phase, validate a locally supplied model directory, define versioned manifests, checksums, device selection, CUDA/CPU policy, and readiness diagnostics. The GUI may prepare a compatible local PyTorch checkpoint into the native package, but must not add a downloader or CDN dependency.

Validation: `model show`, local inventory, `doctor`, backend readiness, and the GUI MODEL flow report the configured local model/runtime paths without downloading assets; canonical single-file and indexed-shard safetensor preparation is explicit, local, validated, and non-overwriting. A real `openai/whisper-tiny` package has been prepared and loaded with the pinned teamy-tts LibTorch runtime. Detailed runtime compatibility across model families, accelerator matrix, permission diagnostics, and clean-machine installation remain pending.

Completion: The application explains which local model/runtime assets are present or missing, and no installed executable requires the source checkout. CDN acquisition remains explicitly deferred.

#### W9 [~] Integrate native Whisper ASR

Work: Implement the first native Whisper ASR path behind the backend protocol,
consuming Rust-prepared direct `tch` weights, dimensions, and tokenizer from the
configured local model directory. An optional TorchScript `model.pt` path may
serve measured hot subgraphs, and the existing Burn package remains a
compatibility path. The current path accepts normalized 16 kHz mono input,
builds Whisper log-mel features in Rust, runs greedy decoder steps, and returns
raw transcript text. VAD/alignment remain later capabilities.

Validation: Rust unit and integration tests pass, including deterministic
frontend, direct-tch synthetic model-shape checks, cached/full-prefix decoder
agreement, and indexed-shard validation. The real VCTK canary verifies input
hash, import, normalization, persisted artifacts, model fingerprinting,
direct-safetensors inference, committed raw-ASR text, CER/WER, timing, and
replay. Long-input ordering, bounded work, cancellation, Python logits parity,
TorchScript/JIT performance, and a broader quality/timing matrix remain
pending.

Completion: One imported audio fixture and one imported video fixture produce a local transcript with provenance and honest capability reporting.

#### W10 [~] Add microphone capture

Work: Enumerate devices, show stable endpoint identity, capture an explicitly bounded interval through WASAPI, save native-rate mono-f32 audio, detect start/stop/failure states, and route captured audio into the same artifact path as imports. The current slice provides active endpoint inventory, shared bounded capture with fresh-client exclusive-mode diagnostics after shared initialization failure, GUI stop-controlled capture, and replayable recording lifecycle states; real-device GUI capture evidence remains pending.

Validation: Device inventory works without capture and empirically reported two active endpoints. The bounded capture path compiles and rejects zero-duration requests without opening a device; domain tests cover saved/failed lifecycle replay. A real saved recording, device disconnect, and permission errors remain pending an explicit capture run.

Completion: Microphone recording is a normal source kind in the domain model and can be transcribed by the same backend.

### Phase 4: headless transcription workflow

#### W11 [~] Complete one file-to-transcript vertical slice

Work: Connect WAV normalization, persisted 30-second clip extraction, native
Whisper submission, raw transcript commit, and structured
CLI/GUI output through the same event-backed recording. The current slice also
persists clip processing/failure transitions, exports the latest transcript,
and projects them through `recording show`; import/video fixture execution,
ordered result staging, progress, and cancellation remain pending.

Validation: The typed no-GUI canary traverses the real VCTK import, 16 kHz mono
normalization, persisted recording/events, canonical safetensor model
inspection, direct native Whisper inference, raw-ASR commit, CER/WER, timing,
and replay receipt; missing corpus and incompatible local models are explicitly
non-passing. Native frontend/model tests and the full repository gate pass.
The successful tiny-model canary is bounded to one short fixture; video,
long-form completeness, cancellation, alignment, diarization, and a broader
quality matrix remain pending.

Completion: The first user-value path works end to end for a fixture and is documented as the reference slice.

#### W12 [~] Add long-input chunking and quality controls

Work: Implement bounded chunking, speech-aware boundaries where available, ordered assembly, partial results, retry/cancel semantics, and quality metadata. The current slice provides deterministic fixed-duration ranges, stable clip IDs, ordered per-clip reports, and resumable failure states; speech-aware boundaries, cancellation, and timing quality metadata remain pending. Preserve source-to-chunk offsets.

Validation: Unit coverage verifies no gaps or overlap in synthetic plans; the
real short VCTK canary completes one ordered native-inference window. Long-input
worker limits, multi-window ordering, cancellation, and timing evidence remain
pending.

The next fixture extension can concatenate clips from distinct VCTK speakers
with a checked-in logical descriptor and an ignored generated WAV. Its truth
manifest should retain source speaker IDs and clip offsets, and its first gate
should assert ordered segmentation and label plumbing. It must not be called a
diarization-quality oracle until the native VAD/speaker-embedding path exists;
simple concatenation does not exercise overlap, reverberation, or crosstalk.

Completion: Long files yield a coherent transcript with a machine-readable chunk map and no unreported approximation.

#### W13 [~] Add explicit transcript editing and output routing

Work: Add staged versus committed transcript versions, user edits, local derivative actions, copy/export/save, and an explicit future integration boundary for external apps. The current slice exports the latest committed transcript per active clip, lets the GUI edit committed text with `user_edit` provenance, and keeps external-target actions deferred.

Validation: Raw ASR provenance remains in the manifest and export labels; export is explicit and writes a recording-owned or user-selected file. Shared workflow/domain tests cover user edits and latest-version export; LLM derivatives and external-target safety tests remain pending.

Completion: A user can review and export a transcript without losing the source or raw result.

### Phase 5: GUI, actions, and tray

#### W14 [~] Build renderer-neutral presentation state

Work: Define stable UI IDs, action IDs, focus/context state, narration/diagnostics, waveform/transcript projections, and contextual keyboard precedence. The current slice provides these as a pure `presentation` module and maps GUI controls to shared workflow actions, including chunk presets, clip navigation, recovery, and keyboard editing; a fuller projection/transport convergence remains pending.

Validation: Headless tests verify deterministic Escape, transcript, timeline, and recording-control key precedence plus renderer-neutral transcript/diagnostic projection. GUI tests cover microphone hit targets, clip navigation, chunk labels, and workflow edit/export persistence. Pointer/palette/tray adapters and conflict logging remain pending.

Completion: A headless presentation test can drive the first slice without depending on a window or renderer.

#### W15 [~] Build the microphone-centered window

Work: Present a skeuomorphic microphone/record control, armed/recording/stopped state, level or waveform view, clip timeline, staged transcript area, model/runtime status, and obvious save/export actions. The Ash/Vulkan/Winit shell now owns the GUI-only first workflow: local model selection/readiness, media import/prepare, microphone capture, native transcription, transcript edit, and export, with asynchronous status/error reporting, bounded per-clip progress, and persisted preferences.

Validation: Window lifecycle and first-frame Vulkan startup are empirically verified on this device; the bounded waveform, fontdue CPU text path with bitmap fallback, microphone hit-test, clip navigation/reordering, chunk/profile controls, and cancellation paths are covered by focused tests; shared workflow/domain tests cover persistence, user-edit provenance, profile artifacts, reordering, and export. A human-operated file-picker/model/capture/transcription run, accessibility/narration, and local model-backed inference remain open. The view shows staged text only after a committed transcript and does not imply uncommitted edits are final.

Completion: A user can import or record, see the current state, transcribe, inspect text, and save/export from one coherent window.

#### W16 [~] Add tray and hotkey behavior

Work: Add tray presence, explicit global hotkey registration, notification policy, and a compact action menu only after W14/W15 are stable. The Windows implementation now creates a hidden tray window, installs the embedded icon, restores it after Explorer restart, routes tray and `Ctrl+Shift+Space` actions through the GUI reducer, and persists the hotkey-enabled preference.

Validation: Headless hit-testing covers the GUI toggle and compile/clippy gates cover the Win32 boundary. A human-operated tray-menu/hotkey check, conflict behavior, and restart persistence without losing an active recording remain open.

Completion: Tray and hotkeys are convenience projections of the same action model, not a second control path.

### Phase 6: text rendering correctness

#### W17 [~] Define text rendering contracts and evidence

Work: Specify glyph origin, atlas/texture layout, band/glyph metadata, clipping, baseline, generation, transport, pixel format, and bounds. Add known artifact fixtures.

Validation: CPU fontdue output is the reference for selected cases; renderer tests can run offscreen and emit manifests plus images.

Completion: A failing render can be classified as semantic, transport, rasterization, or presentation error.

#### W18 [~] Integrate CPU reference and optional Slug GPU path

Work: Implement the CPU reference and then the Ash/Vulkan/Slug path if G1 selects it. The GUI now uses fontdue for CPU reference rasterization with a bitmap fallback; complete renderer/transport tuples, fresh generations, full resync, artifact fixtures, and the optional Slug path remain open. Use bounded push-based delivery.

Validation: Compare fixtures at multiple sizes, scripts, clipping cases, and stale-generation transitions. Measure end-to-end latency before any speed claim.

Completion: Text artifacts are below the documented tolerance or the affected path remains explicitly experimental and is not the default.

### Phase 7: convenience processing and local LLM

#### W19 [~] Add reversible audio preparation profiles

Work: Implement gain, noise reduction, equalization, resampling, and clip move/split/append/delete as derived operations. Preserve original source and parameter receipts. The current slice implements GUI-selectable gain, a conservative noise gate, voice EQ, replayable clip movement, midpoint split, adjacent append, and confirmation-gated soft deletion; richer profile quality validation remains open.

Validation: Golden audio metadata and transcript comparisons cover each profile; processing failure leaves the prior artifact usable.

Completion: Convenience processing is predictable, inspectable, and never changes the authoritative source silently. The active GUI path now exposes the complete move/split/append/delete clip operation set and retains replaced artifacts in replayable history.

#### W20 [ ] Add optional local LLM actions

Work: Reuse the teamy-llm-service lessons for model registry, readiness, cancellation, and per-model scheduling. Define cleanup, formatting, and summary as derived transcript actions.

Validation: The LLM path is unavailable without blocking ASR; prompts and outputs stay local; raw transcript is immutable; model identity and prompt/action receipts are saved.

Completion: A user can invoke a visible local transformation and compare raw, edited, and derived text.

### Phase 8: formal and empirical evidence

#### W21 [ ] Add property, replay, and bounded-checker coverage

Work: Test typed state transitions, clip ordering, event replay, stale jobs, cancellation, manifest recovery, and action routing. Add a deterministic explicit checker over a named finite scope.

Validation: Report scope identifiers, assumptions, state/edge counts, exhaustion or limit status, shortest counterexamples, tool versions, and non-claims.

Completion: The checker demonstrates the stated bounded properties without being described as proof of model accuracy, UI correctness, or all runtime behavior.

#### W22 [ ] Add acceptance matrix and diagnostics

Work: Track feature, tool/runtime/model version, evidence type, confidence, support fixture, known limits, and next action. Emit structured diagnostics beside visual evidence.

Validation: Every overall criterion below has a row with evidence or an explicit gap. Screenshots are never the only evidence for a semantic claim.

Completion: A release review can answer what is proven, sampled, empirical, experimental, deferred, and unverified.

### Phase 9: packaging and handoff

#### W23 [ ] Rehearse clean-device packaging

Work: Package the executable, local runtime/model readiness diagnostics, licenses, config migration, logs, and uninstall/retention behavior. CDN acquisition is a later project.

Validation: Rehearse on a clean Windows environment with no checkout, pre-existing model cache, or hidden developer dependency. Record all failures.

Completion: A new user can install, run doctor, prepare the selected local model, record/import, transcribe, and locate outputs.

#### W24 [ ] Maintain operator and developer documentation

Work: Document privacy, storage, supported formats, model licenses, troubleshooting, keyboard/tray behavior, evidence limits, and a fresh-agent continuation section.

Validation: Follow the docs from a clean checkout and update the plan when actual behavior diverges.

Completion: The repository tells a user how to use the product and an agent how to continue the next safe slice.

## Overall acceptance criteria

| ID | Criterion | Required evidence |
|---|---|---|
| A1 | Audio file import works locally. | Fixture receipt, transcript provenance, timing, and failure case. |
| A2 | Video file import works locally. | Empirical local MP4 receipt showing extracted/normalized audio; broader source-time mapping fixtures remain pending. |
| A3 | Microphone recording works. | Device identity, saved artifact, replayable manifest, disconnect/permission diagnostics. |
| A4 | WhisperX runs locally. | Runtime/model/device receipt and raw transcript from at least one fixture. |
| A5 | New-device model/runtime setup is understandable and repeatable. | Doctor/prepare output and clean-machine rehearsal. |
| A6 | GUI is coherent and microphone-centered. | Human-observable vertical slice plus action/presentation tests. |
| A7 | Clips and transcript versions are predictable. | Boundary, ordering, replay, and provenance tests. |
| A8 | Noise reduction, equalization, and movement are safe. | Derived artifacts, parameter receipts, and no source mutation. |
| A9 | Local LLM actions are explicit and optional. | Raw/derived comparison, local-only receipt, unavailable-service behavior. |
| A10 | Text rendering is correct enough for the chosen default. | CPU reference plus GPU artifact evidence if GPU is enabled. |
| A11 | Reliability claims are bounded. | Chunking, cancellation, recovery, latency, and structured diagnostics. |
| A12 | Formal-methods claims are scoped. | Named finite checker scope, counts, limits, counterexamples, and non-claims. |

## Risk register

| ID | Risk | Mitigation and stop condition |
|---|---|---|
| R1 | WhisperX setup requires several model/runtime assets or gated downloads. | Build doctor/prepare first; stop packaging claims until clean-machine rehearsal passes. |
| R2 | Burn rewrite expands scope and delays user value. | Keep Burn behind the backend contract; stop if it blocks the first local WhisperX slice. |
| R3 | WhisperX quality or latency is insufficient on the target device. | Measure fixture quality and end-to-end latency; report capability limits rather than hiding them. |
| R4 | Media preprocessing shifts timestamps. | Preserve mappings and test source-time/normalized-time conversions. |
| R5 | GPU text artifacts consume the project. | CPU fontdue reference is the acceptance baseline; pause GPU defaulting when artifact fixtures fail. |
| R6 | GUI and tray paths diverge semantically. | One typed action executor, stable IDs, and replay tests. |
| R7 | Microphone capture leaks to an external app or records unexpectedly. | Explicit armed state, visible indicator, no blind typing, disable-able hotkeys, diagnostics. |
| R8 | LLM transforms overwrite authoritative text. | Immutable raw transcript, derived versions, explicit action and provenance. |
| R9 | Formal checker creates false confidence. | Named bounded scope and evidence labels on every claim. |
| R10 | Packaging relies on the developer checkout or hidden PATH tools. | Clean-device rehearsal and doctor diagnostics; tv.exe is not assumed. |
| R11 | Feature breadth turns the utility into a DAW/NLE. | Keep the scope centered on capture, clips, transcription, review, and export; defer editing features that do not serve that loop. |

## Intent audit

### Pass 1: extraction

Completed 2026-08-06. Re-read the original request and both supplied transfer briefs. U1-U28 were extracted into the ledger. The audit specifically checked for audio, video, microphone, local WhisperX, local LLM, model bootstrap, saved recordings, clips, noise reduction, equalization, movement, GUI, skeuomorphic microphone, tray/hotkeys, Ash/Vulkan, cursor-latency, fontdue/Slug, Teamy Studio, Teamy Terminal, Burn, Poche, action-first UI, typed state/events, renderer/transport separation, diagnostics, evidence language, and stop conditions.

### Pass 2: traceability

Completed 2026-08-06. Every ledger item maps to at least one product-boundary statement, gate, work item, acceptance criterion, risk, or explicit non-goal. The transfer lessons are represented in W4, W7, W14, W17, W21, W22, and the architecture contract.

### Pass 3: adversarial omission review

Completed 2026-08-06. Checked the likely failure modes:

- Local-only could have been weakened by the LLM or model bootstrap: it is an explicit default boundary and receipt requirement.
- ASR, LLM cleanup, user edits, and export could have been conflated: they have separate provenance and commands.
- Microphone capture could have become blind external typing: that is an explicit non-goal and acceptance condition.
- GUI ambitions could have hidden missing persistence: storage and recovery precede the GUI.
- WhisperX could have been treated as a single binary: runtime/model/doctor/prepare are separate work, and the current phase assumes local model files without adding a downloader.
- Renderer work could have swallowed the product: CPU reference and an allowed non-GPU first slice are explicit.
- Poche evidence could have been overstated: bounded scope, tool versions, limits, and non-claims are required.
- “General and easy to use” could have become DAW/NLE scope: non-goals and R11 constrain it.

Remaining open gates are intentional decisions, not omitted requirements.

## Decision log

| Date | Decision | Reason | Revisit trigger |
|---|---|---|---|
| 2026-08-06 | Start with a typed Rust domain/storage/backend contract and a fake backend. | Produces a testable vertical slice before device/model/rendering complexity. | Revisit if the chosen GUI framework forces a different boundary. |
| 2026-08-07 | Make the native Burn Whisper path the active ASR backend; retain WhisperX behavior as future capability work. | The reusable burnt-apple frontend/model/Burnpack loader provides a pure-Rust path and removes a Python runtime prerequisite. | Revisit if model parity or quality evidence requires a different core. |
| 2026-08-06 | Keep raw ASR immutable and LLM output derived. | Protects trust and makes local transformations auditable. | Revisit only with an explicit versioning design. |
| 2026-08-06 | CPU fontdue is the text-rendering reference. | Prior Slug artifacts require a stable comparator and honest GPU evidence. | Revisit after artifact fixtures pass. |

## Next safe implementation slice

The next implementation turn should close the remaining GUI evidence and lifecycle gaps:

1. Run `recording prepare` and `recording transcribe` against a local native model package, recording model revision, artifact hashes, CPU timing, and output checksum.
2. Add a portable video/audio fixture strategy and source-time mapping, likely behind an ffmpeg/ffprobe adapter or a documented native library.
3. Exercise GUI microphone start/stop, failure recovery, chunked transcription, edit, and export with a real local fixture.
4. Expand the backend lifecycle with staged results and structured timing/quality receipts; cooperative cancellation and bounded per-clip progress are now present at the workflow/GUI boundary.
5. Keep local model inventory and native artifact validation separate from any future CDN acquisition.

Do not add CDN acquisition or make the experimental Slug renderer the default
until the GUI-only core workflow has a real model/device evidence run. Tray and
hotkey behavior is implemented; its human-operated conflict/restart check is
still an evidence task rather than a second control path.

## Plan completion rule

This planning phase is complete when the gates have explicit owners/decisions, the first vertical slice is executable, and the acceptance matrix has no unowned ambiguity. Implementation is complete only when the acceptance criteria and documented evidence support a release decision.
