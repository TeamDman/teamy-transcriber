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
cargo run -- model show
cargo run -- home show
cargo run -- cache show
cargo run -- recording create example.wav
~~~

The doctor command reports the resolved application, cache, and local model
paths. The recording command creates a durable manifest and event receipt; it
does not decode or transcribe the source yet.

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
- RUST_LOG

## License

This repository is distributed under the Mozilla Public License 2.0. See
[LICENSE](G:/Programming/Repos/teamy-transcriber/LICENSE).
