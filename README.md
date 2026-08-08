# teamy-transcriber

An intended Windows-first, local-first transcription utility for audio files, video files, and microphone recordings.

The first release is deliberately narrow:

- capture or import audio;
- save an authoritative recording and clip manifest;
- transcribe locally through a pure-Rust Burn Whisper runtime;
- present staged transcript text without silently typing into another application;
- provide predictable clip movement and a small set of reversible audio-preparation operations;
- keep the GUI, tray behavior, renderer, and model runtime observable and testable.

This is not intended to become a digital audio workstation or nonlinear editor. The active design contract is in [PLAN.md](G:/Programming/Repos/teamy-transcriber/PLAN.md).

The project is informed by Teamy-Studio, teamy-llm-service, teamy-terminal, whisper-burn, whisperX, voice2text, teamy-subs, piing, tb, cursor-latency, and the Poche/SFM transfer briefs. Those sources are evidence and prior art, not permission to copy their unfinished assumptions into this project.

## Current command surface

The initial repository baseline is a template-backed diagnostic CLI:

~~~powershell
cargo run -- doctor
cargo run -- microphone list
# Capture one explicit microphone interval (Windows):
cargo run -- microphone record --duration-ms 5000
cargo run -- model show
cargo run -- home show
cargo run -- cache show
# Open the native Ash/Vulkan desktop surface:
cargo run -- gui
cargo run -- recording create example.wav
# Then use the UUID returned by recording create:
cargo run -- recording prepare <recording-id>
# Optional: add a source-time clip before transcribing it:
cargo run -- recording clip add <recording-id> 0 30000000
# With a local native model package already installed:
cargo run -- recording transcribe <recording-id> --model-dir C:\path\to\models
# Optional deterministic fixed-duration chunks:
cargo run -- recording transcribe <recording-id> --chunk-duration-ms 30000
# Export committed transcript text after transcription:
cargo run -- recording export <recording-id>
~~~

The doctor command reports the resolved application, cache, and local model
paths. `recording prepare` normalizes WAV sources directly and routes other
audio/video sources through local `ffmpeg` into 16 kHz mono audio. In the GUI,
`TOOLS` opens file pickers for local `ffmpeg` and `ffprobe` executables; those
paths are persisted with the other GUI settings.
`recording transcribe` invokes the native Burn Whisper encoder/decoder and
commits raw ASR text through the same event receipt; persisted partial clips are
materialized as separate normalized WAV artifacts first. `--chunk-duration-ms`
creates contiguous, non-overlapping clip records and resumes from their stable
IDs after a failure. Video fixture verification, runtime installation, and
model/CDN acquisition remain later slices. During
chunked transcription, `CANCEL`/`Escape` cooperatively stop after the active
clip and retain completed clip transcripts.

The renderer-neutral presentation model in `src/presentation.rs` keeps stable
UI/action IDs, contextual key resolution, transcript projection, and diagnostics
separate from the future window, tray, and GPU renderer.

`cargo run -- gui` creates the native Winit window, Ash/Vulkan surface and
swapchain, and a CPU-rasterized reference layout with fontdue text (falling back
to a small deterministic bitmap alphabet when no supported system font is
available), a microphone control, microphone/save-directory selectors, waveform,
and transcript panel. The GUI is
the complete first workflow surface: choose a local model directory, import an
audio/video file (which automatically prepares normalized audio), record from a
selected microphone, transcribe locally, review/edit the committed transcript,
and export it through native file dialogs. Long-running capture, preparation,
transcription, edit, and export work runs off the window thread and reports
bounded preparation/transcription progress and success/failure back into the
visible status line. Press Space to start/stop
microphone capture, Escape to stop capture, cancel transcription, or cancel
transcript editing, and
Ctrl+E to open transcript export. The GUI also exposes full-recording or
10/30/60-second chunk presets, previous/next clip review, and cycling through
persisted recordings. `LEFT`/`RIGHT` reorder the selected clip through the
replayable recording history; those choices survive restart.
`AUDIO` cycles the original normalized signal, gain, a conservative noise gate,
and a simple voice-EQ profile. Derived WAVs are written beside the original
normalized artifact with a JSON parameter receipt. Prepared recordings render a
bounded peak envelope from the selected WAV; live capture keeps a level view
until the recording is saved.

The GUI and diagnostic CLI share the workflow orchestration in
`src/workflow.rs`; both persist the same recording lifecycle and transcript
provenance events. Restarting the GUI reopens the selected persisted recording
(falling back to the available recordings) and restores the selected model,
microphone, and export directory
from its app-owned settings file. Selecting MODEL validates the tokenizer,
dimensions, and Burnpack/legacy layout before TRANSCRIBE is enabled; no CLI
model-preparation command is required for a locally supplied package.

The native model package is assumed to be available locally for this
implementation slice; the application does not download or convert it. The
preferred model directory contains `model.bpk`, `dims.json`, and
`tokenizer.json`. The runtime also recognizes the older packed-NPY
`encoder/`/`decoder/` layout during migration.

For local media validation, a user-owned VCTK sample corpus can be used when
available at `G:\Datasets\VCTK\VCTK-Corpus-smaller\`. It is not required for
builds or automated tests, and it must not be copied into this repository.

## Development

Run the repository quality gate:

~~~powershell
.\check-all.ps1
~~~

The gate runs nightly formatting, clippy with warnings denied, a build, and the
test suite. The repository also retains the template's optional Tracy profiling
harness for later end-to-end latency work.

The path environment overrides are:

- TEAMY_TRANSCRIBER_HOME_DIR
- TEAMY_TRANSCRIBER_CACHE_DIR
- TEAMY_TRANSCRIBER_MODEL_DIR
- TEAMY_TRANSCRIBER_FFMPEG
- TEAMY_TRANSCRIBER_FFPROBE
- RUST_LOG

## License

This repository is distributed under the Mozilla Public License 2.0. See
[LICENSE](G:/Programming/Repos/teamy-transcriber/LICENSE).
