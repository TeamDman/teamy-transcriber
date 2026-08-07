# teamy-transcriber

An intended Windows-first, local-first transcription utility for audio files, video files, and microphone recordings.

The first release is deliberately narrow:

- capture or import audio;
- save an authoritative recording and clip manifest;
- transcribe locally through a managed WhisperX runtime;
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
cargo run -- recording create example.wav
# Then use the UUID returned by recording create:
cargo run -- recording prepare <recording-id>
# Optional: add a source-time clip before transcribing it:
cargo run -- recording clip add <recording-id> 0 30000000
# With local Python/WhisperX and model files already installed:
cargo run -- recording transcribe <recording-id> --model-dir C:\path\to\models
# Optional deterministic fixed-duration chunks:
cargo run -- recording transcribe <recording-id> --chunk-duration-ms 30000
# Export committed transcript text after transcription:
cargo run -- recording export <recording-id>
~~~

The doctor command reports the resolved application, cache, and local model
paths. `recording prepare` normalizes WAV sources directly and routes other
audio/video sources through local `ffmpeg` into 16 kHz mono audio.
`recording transcribe` invokes the local one-shot WhisperX worker and
commits raw ASR text through the same event receipt; persisted partial clips are
materialized as separate normalized WAV artifacts first. `--chunk-duration-ms`
creates contiguous, non-overlapping clip records and resumes from their stable
IDs after a failure. Video decoding, GUI
controls, runtime installation, and model/CDN acquisition remain later slices.

The renderer-neutral presentation model in `src/presentation.rs` keeps stable
UI/action IDs, contextual key resolution, transcript projection, and diagnostics
separate from the future window, tray, and GPU renderer.

The local worker is documented in [runtime/README.md](G:/Programming/Repos/teamy-transcriber/runtime/README.md).
Model files are assumed to be available locally for this implementation slice;
the application does not download them.

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
