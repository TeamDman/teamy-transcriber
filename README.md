# teamy-transcriber

An intended Windows-first, local-first transcription utility for audio files, video files, and microphone recordings.

The first release is deliberately narrow:

- capture or import audio;
- save an authoritative recording and clip manifest;
- transcribe locally through source-defined Rust/CUDA Whisper on an NVIDIA GPU;
- present staged transcript text without silently typing into another application;
- provide predictable clip movement and a small set of reversible audio-preparation operations;
- keep the GUI, tray behavior, renderer, and model runtime observable and testable.

This is not intended to become a digital audio workstation or nonlinear editor. [PLAN.md](PLAN.md) records the recording and workflow design; the CUDA runtime instructions below supersede its historical backend choices.

The project is informed by Teamy-Studio, teamy-llm-service, teamy-terminal, whisper-burn, whisperX, voice2text, teamy-subs, piing, tb, cursor-latency, and the Poche/SFM transfer briefs. Those sources are evidence and prior art, not permission to copy their unfinished assumptions into this project.

## Current command surface

For one-file transcription, install the executable with `./update.ps1`, download
the [ready model package](https://huggingface.co/TeamDman/teamy-transcriber-whisper-large-v3)
once, then select it:

~~~powershell
hf download TeamDman/teamy-transcriber-whisper-large-v3 --revision v1 --quiet
teamy-transcriber model prepare
teamy-transcriber transcribe "example.webm"
~~~

`hf download --quiet` prints the downloaded folder. `model prepare` with no
arguments revalidates the configured model or discovers the released package in
the local HF cache. It remembers that folder in application configuration; it
does not copy or re-encode ready weights, and never downloads anything. To select
a folder explicitly, use `model prepare --source-dir <downloaded-folder>`.
Keep the selected folder available: removing it from the HF cache makes the
model unavailable until it is downloaded again. `model show` reports the selection.

`transcribe FILE` creates a recording, prepares audio, transcribes it, prints
plain text, and removes that recording's intermediate files after successful
output. It leaves the source file and model untouched. Use `--output result.txt`
to also save a new text file, `--keep-recording` to retain intermediates on
success, or `--output-format json` before the command for a structured report.
Existing output files are never overwritten.
Speech detection can miss singing or speech mixed with music. If it reports
no speech, use `transcribe "example.webm" --no-vad` to transcribe the entire
file in windows of up to 30 seconds.

If a step fails or is cancelled, the error identifies the retained recording
and prints `teamy-transcriber transcribe --resume <recording-id>`. Fix the cause
and run that command, repeating any needed model/output options. Resume reuses
prepared audio, completed transcripts and edits, and transcribes only unfinished
clips. Failed output also retains the recording. Cleanup failures report that
some files may remain.

For manual control, `recording create FILE` only saves a source reference and
returns its UUID. `recording prepare UUID` extracts or normalizes its audio to
16 kHz mono; neither command transcribes. Both `recording create` and the shortcut
infer video from common extensions, including `.webm`, `.mp4`, `.mkv` and `.mov`,
case-insensitively. `--kind audio` or `--kind video` overrides that inference.
The manual recording commands retain their files until explicitly removed.
Use `teamy-transcriber recording list` to find saved recording UUIDs, source
paths, status and clip/transcript counts. It reads existing state without loading
a model. `teamy-transcriber --output-format json recording list` returns an array
for scripts (empty when there are no saved recordings). Pass a listed UUID to
`recording show` for details or `transcribe --resume` to continue retained work.
`recording clean --dry-run` previews completed microphone recordings eligible
for removal; `recording clean` removes them. Unfinished microphone windows and
imported audio/video recordings are preserved.

The remaining command surface includes diagnostics, capture, GUI and manual
recording operations:

~~~powershell
# With no arguments, the executable opens the GUI directly:
teamy-transcriber.exe
# The explicit command remains available for scripts and development:
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
cargo run -- recording transcribe <recording-id> --chunk-duration-ms 5000
# Export committed transcript text after transcription:
cargo run -- recording export <recording-id>
~~~

The doctor command reports the resolved application, cache, and local model
paths. `recording prepare` normalizes WAV sources directly and routes other
audio/video sources through local `ffmpeg` into 16 kHz mono audio. In the GUI,
`TOOLS` opens file pickers for local `ffmpeg` and `ffprobe` executables; those
paths are persisted with the other GUI settings. If the selected `ffprobe`
executable is unavailable or rejects the probe request, the adapter falls back
to parsing the selected `ffmpeg` binary's stream diagnostics; cancelling the
GUI's optional ffprobe picker selects that fallback explicitly.
`recording transcribe` invokes the source-defined CUDA Whisper encoder/decoder
with canonical safetensors weights and commits raw ASR text through the same
event receipt. Persisted partial clips are materialized as
separate normalized WAV artifacts first. `--chunk-duration-ms` creates
contiguous, non-overlapping clip records and resumes from their stable IDs
after a failure; omitted chunking reuses existing clips or creates windows of
at most 30 seconds. The native runtime can use prepared local Silero
weights for automatic speech windows and skip silence without loading Whisper;
see [native inference and model preparation](native/README.md). During
chunked transcription, `CANCEL`/`Escape` cooperatively stop after the active
clip and retain completed clip transcripts.
The default decoder budget is the Whisper text-context limit; use
`--max-decode-tokens` to choose a smaller bound for faster exploratory runs.

The renderer-neutral presentation model in `src/presentation.rs` keeps stable
UI/action IDs, contextual key resolution, transcript projection, and diagnostics
separate from the future window, tray, and GPU renderer.

`cargo run -- gui` creates the native Winit window, Ash/Vulkan surface and
swapchain, and a CPU-rasterized reference layout with fontdue text (falling back
to a small deterministic bitmap alphabet when no supported system font is
available), a microphone control, microphone/save-directory selectors, waveform,
and transcript panel. The GUI is
the complete first workflow surface: choose a local model directory, import an
audio/video file by dialog or drag-and-drop (which automatically prepares normalized audio), record from a
selected microphone, transcribe locally, review/edit the committed transcript,
copy it to the system clipboard, and export it through native file dialogs. Long-running capture, preparation,
transcription, edit, and export work runs off the window thread and reports
bounded preparation/transcription progress and success/failure back into the
visible status line. The status panel also shows a state-derived `NEXT`
instruction, so a first-use workflow can be followed entirely from the GUI.
Press Space to start/stop
microphone capture, Escape to stop capture, cancel transcription, or cancel
transcript editing, and
Ctrl+E to open transcript export. Mouse-wheel or PageUp/PageDown scrolling
keeps long transcripts reviewable inside the text panel. The GUI also exposes
automatic or
10/30-second chunk presets, previous/next clip review, and cycling through
persisted recordings. `LEFT`/`RIGHT` reorder the selected clip through the
replayable recording history; those choices survive restart.
`DELETE` removes the selected clip from the active manifest after confirmation
while retaining its source and derived audio for recovery.
Press `S` to replace the selected clip with two midpoint-split source ranges,
or `A` to append it with the next active source-time-adjacent clip; both
operations retain the replaced clips and require the resulting clip(s) to be
transcribed again.
`AUDIO` cycles the original normalized signal, gain, a conservative noise gate,
and a simple voice-EQ profile. Derived WAVs are written beside the original
normalized artifact with a JSON parameter receipt. Prepared recordings render a
bounded peak envelope from the selected WAV; live capture keeps a level view
until the recording is saved.

On Windows, the GUI also installs a tray icon and enables `Ctrl+Shift+Space`
by default. The hotkey starts or stops the same microphone action as the main
window; the `HOTKEY ON/OFF` control and tray menu can disable it, and the
setting is persisted. A registration conflict is reported in the GUI status
line instead of preventing the window from starting. Tray exit follows the
same close/cancellation lifecycle as the window.

The GUI and diagnostic CLI share the workflow orchestration in
`src/workflow.rs`; both persist the same recording lifecycle and transcript
provenance events. Restarting the GUI reopens the selected persisted recording
(falling back to the available recordings) and restores the selected model,
microphone, and export directory
from its app-owned settings file. Selecting MODEL validates the tokenizer,
dimensions and safetensors layout before TRANSCRIBE is enabled.
The GUI
also offers a local-only preparation path: choose `No` in the model setup
dialog, select a canonical safetensors folder containing `config.json` and
`tokenizer.json`, and choose an output parent directory. Preparation runs asynchronously and selects the resulting native package
after validation. The headless equivalent is:

~~~powershell
teamy-transcriber model prepare `
  --source-dir C:\path\to\canonical-whisper `
  --output-dir C:\path\to\teamy-transcriber-model
~~~

With an explicit output directory, this command accepts one `model.safetensors` file or an indexed
`model.safetensors.index.json` package plus matching `config.json` and
`tokenizer.json`; it performs no download and refuses to overwrite an
existing directory. It copies tensor files unchanged, writes dimensions and
selects the new package. Users of the ready HF package do not need this step.

The native model package or source checkpoint must be available
locally; the application does not download
model assets. The preferred package contains canonical safetensors plus
`dims.json` and `tokenizer.json`; the Rust preparation path has been exercised
with real `openai/whisper-tiny` and `openai/whisper-large-v3` packages. The default
CUDA build defines computation in Rust and CUDA source and reads the tensor
weights directly. `TEAMY_TRANSCRIBER_CUDA_DEVICE` selects the NVIDIA device
(default zero). Build and installer scripts stage the CUDA runtime and cuBLAS
DLLs beside the executable.

TorchScript, LibTorch/CPU, Burnpack and packed-NPY inference have been removed.
The local `pre-cuda-only` Git tag retains the previous implementation. Existing
recordings and transcript history remain readable; legacy model files need to
be replaced with canonical safetensors. CTranslate2 `model.bin` is also unsupported.
No automatic model conversion or download occurs.

The preparation path validates a single safetensors file or indexed shards and
packages config, tokenizer and sidecar dimensions for direct CUDA inference.

For local media validation, a user-owned VCTK sample corpus can be used when
available at `G:\Datasets\VCTK\VCTK-Corpus-smaller\`. It is not required for
builds or automated tests, and it must not be copied into this repository.

The bounded native-model canary is run explicitly with the supplied corpus
and a locally prepared native Whisper model:

~~~powershell
.\target\debug\teamy-transcriber.exe --output-format json `
  verify speech vctk-p230-385 `
  --vctk-root G:\Datasets\VCTK\VCTK-Corpus-smaller `
  --model-dir G:\path\to\native-whisper `
  --receipt artifacts\verification\vctk-p230-385.json
~~~

The command never downloads the corpus or model. It writes a versioned
receipt for `passed`, `failed`, or `unavailable` outcomes and only reports
`passed` when the real WAV is imported, normalized to 16 kHz mono, persisted,
transcribed by the native backend, committed as raw ASR, and matched to the
descriptor reference. The model directory must contain canonical safetensors
weights (single file or indexed shards), `dims.json` and `tokenizer.json`.
Unsupported model layouts produce a non-passing diagnostic. Version 4 receipts
record the native CUDA device/math/batch settings instead of LibTorch diagnostics.

## Development

Native CUDA is the only inference backend. Model topology is Rust
code, numerical kernels are CUDA, and weights remain separate safetensors files.
Run `./tools/build-cuda-native.ps1` to build and stage runtime DLLs, or
`./update.ps1` to install in the usual Cargo bin directory. Use
`./update.ps1 -Root <install-directory>` for an isolated installation. The installer
stages CUDA DLLs beside the executable and verifies it starts with only Windows
system directories on PATH. It preserves model settings and does not download weights.
See [native inference](native/README.md) for prerequisites, model preparation,
comparison tools and validation limits. Current measured hardware is Windows x64
with an RTX 4090; the native build requires a compatible CUDA toolkit and GPU.

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

## Live microphone transcription

```powershell
teamy-transcriber microphone transcribe
teamy-transcriber microphone transcribe --duration-ms 10000
teamy-transcriber microphone transcribe --chunk-duration-ms 500
teamy-transcriber microphone transcribe --keep-recording
```

Streaming Silero VAD submits audio after roughly 320 ms of silence following
speech. The default five-second window is a maximum for continuous speech;
`--chunk-duration-ms` changes that cap. Models without VAD use fixed windows.
Transcripts are flushed to stdout while capture continues; status goes to stderr.
One inference session stays resident between submissions. Initial model
loading adds latency to the first result. Fixed window boundaries can split words;
this command does not emit partial word hypotheses.

Press Ctrl+C once to stop capture, transcribe the final partial window, drain
queued work and exit successfully. Reaching `--duration-ms` does the same.
Two rapid Ctrl+C presses force exit through the normal cancellation handler.
Capture and inference queues are bounded; overload reports an error rather than
silently dropping audio. Each completed window is removed after its text or
phones reach stdout. Use `--keep-recording` to retain successful windows.
Failed or unfinished windows remain available through `recording list` for
recovery. To remove older completed microphone windows, preview with
`recording clean --dry-run`, then run `recording clean`.
Use `microphone list` and `--device-id` to select a microphone.

## Direct IPA phone recognition

Phone recognition uses PhoneticXeus, a separate model from Whisper. Its CNN,
E-Branchformer, self-conditioned CTC and greedy phone decoder are implemented in
Rust/CUDA. Only canonical safetensors weights and JSON configuration/vocabulary
are loaded. Python and Hugging Face remote code are not application dependencies.

Download once, then validate and select the folder:

```powershell
hf download changelinglab/PhoneticXeus model.safetensors config.json ipa_vocab.json --revision 3a8d860fa68f8936ceb4196651221215bab9dae4 --local-dir ./phone-model
teamy-transcriber model prepare-phones --source-dir ./phone-model
teamy-transcriber phones ./speech.wav
teamy-transcriber --output-format json phones ./speech.wav
teamy-transcriber microphone transcribe --phones
```

The weights are about 2.3 GB. Selection does not convert them or download anything;
it checks the model by loading it on CUDA. `phones --model-dir` and microphone
`--phone-model-dir` override selection. `TEAMY_TRANSCRIBER_PHONE_MODEL_DIR` is also
supported. Without selection, the CLI can discover the pinned revision in the
local `hf` cache.

Text output is joined IPA. JSON keeps the phone tokens separate (one token may
contain several Unicode characters). Direct `phones` results and prepared audio
stay in the recording directory as `phones.json`; use `recording list` to find
that directory's recording ID. Live phone transcription removes successful
windows by default, or retains them with `--keep-recording`. Phone results are
kept separately from Whisper text transcripts.
For live mode the phone model stays loaded, existing VAD pause submission is reused,
and Ctrl+C stops capture and drains pending phone recognition. VAD comes from the
selected Whisper package, or the package passed to microphone `--model-dir`;
without VAD it falls back to fixed windows with a warning.

This is utterance/chunk recognition, not a causal streaming acoustic model. Files
are split at 30 seconds to bound attention memory. Live mode uses the usual
five-second maximum and VAD pauses. Chunk boundaries can affect predictions.
JSON chunk times describe audio windows, not individually aligned phone times.

Reference: [PhoneticXeus](https://github.com/changelinglab/PhoneticXeus),
[model and Apache-2.0 license](https://huggingface.co/changelinglab/PhoneticXeus).
The native port includes the corrected conditioning at layers 4, 8 and 12.
To reproduce numerical validation, use `tools/phone-reference.py` with a local
reference clone, then run the ignored `phone_parity` test with `PHONE_TEST_MODEL`
and `PHONE_TEST_REFERENCE`. Python is used only for this independent comparison.
