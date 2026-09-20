use crate::cli::output::CliOutput;
use crate::cli::output::CliOutputValue;
use crate::cli::output::OutputFormat;
use crate::cli::recording::create::RecordingKind;
use crate::domain::AppState;
use crate::domain::Command;
use crate::domain::RecordingId;
use crate::domain::SourceAsset;
use crate::media::AudioProfile;
use crate::media::MediaAdapter;
use crate::media::WavMediaAdapter;
use crate::paths::AppHome;
use crate::paths::ModelHome;
use crate::storage::RecordingStore;
use crate::workflow::TranscriptionOptions;
use crate::workflow::TranscriptionSession;
use crate::workflow::asset_kind_for_path;
use crate::workflow::audio_path_for_profile;
use crate::workflow::prepare_recording;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use eyre::ensure;
use facet::Facet;
use figue as args;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use teamy_cancellation::CancellationToken;

/// Import, prepare and transcribe media; remove intermediate work after successful output.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct TranscribeArgs {
    /// Audio/video file. Omit when using --resume.
    #[facet(args::positional)]
    pub source: Option<String>,
    /// Resume a retained recording UUID instead of creating another recording.
    #[facet(args::named)]
    pub resume: Option<String>,
    /// Override the source kind inferred from its extension.
    #[facet(args::named)]
    pub kind: Option<RecordingKind>,
    /// Override the model selected by model prepare.
    #[facet(args::named)]
    pub model_dir: Option<String>,
    /// Also write plain text to a new file; existing files are never overwritten.
    #[facet(args::named)]
    pub output: Option<String>,
    /// Keep the recording and intermediate audio after success.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub keep_recording: bool,
    /// Transcribe every 30-second window, bypassing speech detection (useful for singing/music).
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub no_vad: bool,
    /// Maximum generated tokens per clip (default 448).
    #[facet(args::named)]
    pub max_decode_tokens: Option<usize>,
}

#[derive(Debug, Facet)]
struct TranscribeReport {
    recording_id: String,
    text: String,
    no_speech: bool,
    chunk_count: usize,
    cleanup_requested: bool,
    output_path: Option<String>,
}

#[derive(Debug)]
struct Recovery {
    store: RecordingStore,
    id: RecordingId,
}

impl Recovery {
    fn error(&self, stage: &str, error: eyre::Report) -> eyre::Report {
        error.wrap_err(format!(
            "{stage} failed. Recording {} was NOT cleaned up; any saved audio and completed transcripts remain at {}. Fix the cause, then retry: teamy-transcriber transcribe --resume {} (repeat any --model-dir/--output options you need)",
            self.id, self.store.recording_dir(self.id).display(), self.id
        ))
    }

    fn checked_directory(&self) -> Result<PathBuf> {
        let root = self.store.root().canonicalize()?;
        let expected = root.join("recordings").join(self.id.to_string());
        let actual = self.store.recording_dir(self.id).canonicalize()?;
        ensure!(
            actual == expected,
            "refusing cleanup of a redirected recording directory"
        );
        let recording = self.store.load_recording(self.id)?;
        if let Ok(source) = Path::new(&recording.source.path).canonicalize() {
            ensure!(
                !source.starts_with(&actual),
                "refusing cleanup: original source is inside the recording directory"
            );
        }
        Ok(actual)
    }
}

impl TranscribeArgs {
    /// # Errors
    /// Returns a stage-specific error, retaining created recordings for --resume.
    pub fn invoke(self, token: CancellationToken) -> Result<CliOutput> {
        ensure!(
            self.source.is_some() != self.resume.is_some(),
            "supply a media file OR --resume <recording-id>"
        );
        ensure!(
            self.resume.is_none() || self.kind.is_none(),
            "--kind applies only to a new media file"
        );
        let store = RecordingStore::new(AppHome::resolve()?.0);
        let id = if let Some(id) = &self.resume {
            RecordingId::parse(id).wrap_err("--resume must be a recording UUID")?
        } else {
            RecordingId::new()
        };
        let recovery = Recovery { store, id };
        if let Some(source) = &self.source {
            let path = Path::new(source)
                .canonicalize()
                .wrap_err("source file is unavailable; no recording was created")?;
            ensure!(
                path.is_file(),
                "source must be a file; no recording was created"
            );
            let kind = self
                .kind
                .map_or_else(|| asset_kind_for_path(&path), RecordingKind::asset_kind);
            let operation = (|| {
                let source = SourceAsset::new(kind, path)?;
                recovery.store.apply_command(
                    &mut AppState::new(),
                    Command::CreateRecording {
                        recording_id: id,
                        source,
                    },
                )?;
                Ok(())
            })();
            operation.map_err(|error| {
                recovery.error("recording creation (state may be partial)", error)
            })?;
        } else {
            recovery
                .store
                .load_recording(id)
                .map_err(|error| recovery.error("resume", error.into()))?;
        }
        let report = self
            .run(&recovery, &token)
            .map_err(|error| recovery.error("transcription workflow", error))?;
        Ok(CliOutput::custom(TranscribeOutput {
            report,
            recovery,
            token,
        }))
    }

    fn run(&self, recovery: &Recovery, token: &CancellationToken) -> Result<TranscribeReport> {
        token.bail_if_cancelled()?;
        let model_dir = self.model_dir.as_ref().map_or_else(
            || ModelHome::resolve().map(|home| home.0),
            |path| Ok(PathBuf::from(path)),
        )?;
        crate::native_whisper::model::inspect_model_dir(&model_dir).wrap_err(
            "model validation: run teamy-transcriber model prepare or supply --model-dir",
        )?;
        let audio = audio_path_for_profile(&recovery.store, recovery.id, AudioProfile::Original);
        let incomplete = recovery
            .store
            .recording_dir(recovery.id)
            .join("preparation-incomplete");
        if audio.exists() && !incomplete.exists() {
            WavMediaAdapter
                .inspect(&audio)
                .wrap_err("saved prepared audio is invalid")?;
        } else {
            // A failed decoder may leave a valid but truncated WAV. Persist the
            // incomplete stage so retries replace it instead of accepting it.
            std::fs::write(&incomplete, b"audio preparation has not completed")?;
            prepare_recording(&recovery.store, recovery.id).wrap_err("audio preparation")?;
            std::fs::remove_file(&incomplete)?;
        }
        token.bail_if_cancelled()?;
        let bridge = CancelBridge::new(token.clone());
        let options = TranscriptionOptions {
            model_dir,
            max_decode_tokens: self.max_decode_tokens.unwrap_or(448),
            chunk_duration_us: self.no_vad.then_some(30_000_000),
            profile: AudioProfile::Original,
        };
        let report = if self.resume.is_some() {
            crate::workflow::resume_recording(
                &recovery.store,
                recovery.id,
                options,
                Some(&bridge.cancelled),
            )
        } else {
            TranscriptionSession::default().transcribe(
                &recovery.store,
                recovery.id,
                options,
                Some(&bridge.cancelled),
                None,
            )
        }
        .wrap_err("model inference")?;
        ensure!(
            !report.cancelled && !token.is_cancelled(),
            "transcription cancelled"
        );
        let text = report
            .chunks
            .iter()
            .map(|chunk| chunk.text.trim())
            .collect::<Vec<_>>()
            .join("\n");
        if !self.keep_recording {
            recovery.checked_directory()?;
        }
        if let Some(output) = &self.output {
            let path = Path::new(output);
            let parent = path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            let parent = parent
                .canonicalize()
                .wrap_err("output parent directory is unavailable")?;
            ensure!(
                !parent.starts_with(recovery.store.recording_dir(recovery.id).canonicalize()?),
                "output must be outside the recording directory"
            );
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .wrap_err(
                    "creating output file (choose a new path; existing files are preserved)",
                )?;
            let saved = file
                .write_all(text.as_bytes())
                .and_then(|()| file.sync_all());
            drop(file);
            if let Err(error) = saved {
                let _ = std::fs::remove_file(path);
                return Err(error).wrap_err("saving transcript output");
            }
        }
        Ok(TranscribeReport {
            recording_id: recovery.id.to_string(),
            text,
            no_speech: report.no_speech,
            chunk_count: report.chunks.len(),
            cleanup_requested: !self.keep_recording,
            output_path: self.output.clone(),
        })
    }
}

#[derive(Debug)]
struct TranscribeOutput {
    report: TranscribeReport,
    recovery: Recovery,
    token: CancellationToken,
}

impl CliOutputValue for TranscribeOutput {
    fn render(&self, format: OutputFormat, _: bool) -> Result<Option<String>> {
        Ok(Some(match format {
            OutputFormat::Text => self.report.text.clone(),
            OutputFormat::Json => facet_json::to_string_pretty(&self.report)?,
            OutputFormat::Csv => facet_csv::to_string(&self.report)?,
        }))
    }
    fn default_format(&self) -> Option<OutputFormat> {
        Some(OutputFormat::Text)
    }
    fn before_emit(&self) -> Result<()> {
        self.token.bail_if_cancelled()?;
        if self.report.no_speech {
            writeln!(
                std::io::stderr(),
                "No speech was detected. For singing or music, retry the file with --no-vad to transcribe every window."
            )?;
        }
        Ok(())
    }
    fn output_error(&self, error: eyre::Report) -> eyre::Report {
        self.recovery.error("output", error)
    }
    fn after_emit(&self) -> Result<()> {
        self.token
            .bail_if_cancelled()
            .map_err(|error| self.recovery.error("cleanup cancelled", error))?;
        if self.report.cleanup_requested {
            let directory = self
                .recovery
                .checked_directory()
                .map_err(|error| self.recovery.error("cleanup validation", error))?;
            std::fs::remove_dir_all(&directory).wrap_err_with(|| format!("Transcript was emitted, but cleanup of recording {} at {} failed; some intermediate files may remain", self.recovery.id, directory.display()))?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct CancelBridge {
    cancelled: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl CancelBridge {
    fn new(token: CancellationToken) -> Self {
        let cancelled = Arc::new(AtomicBool::new(token.is_cancelled()));
        let finished = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&cancelled);
        let done = Arc::clone(&finished);
        let thread = std::thread::spawn(move || {
            while !done.load(Ordering::Relaxed) {
                if token.is_cancelled() {
                    flag.store(true, Ordering::Relaxed);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        });
        Self {
            cancelled,
            finished,
            thread: Some(thread),
        }
    }
}

impl Drop for CancelBridge {
    fn drop(&mut self) {
        self.finished.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct BrokenPipe;
    impl Write for BrokenPipe {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn prepared_output() -> (PathBuf, PathBuf, RecordingId, CliOutput) {
        let root = std::env::temp_dir().join(format!("transcribe-output-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("source.wav");
        std::fs::write(&source, b"original").unwrap();
        let store = RecordingStore::new(root.join("app"));
        let id =
            crate::workflow::create_recording(&store, crate::domain::AssetKind::AudioFile, &source)
                .unwrap();
        let report = TranscribeReport {
            recording_id: id.to_string(),
            text: "transcribed text".into(),
            no_speech: false,
            chunk_count: 1,
            cleanup_requested: true,
            output_path: None,
        };
        let output = CliOutput::custom(TranscribeOutput {
            report,
            recovery: Recovery { store, id },
            token: CancellationToken::new(),
        });
        (root, source, id, output)
    }

    #[test]
    fn failed_output_retains_recording_and_reports_resume() {
        let (root, source, id, output) = prepared_output();
        let error = output.emit_to(None, false, &mut BrokenPipe).unwrap_err();
        assert!(
            error
                .to_string()
                .contains(&format!("transcribe --resume {id}"))
        );
        assert!(root.join("app/recordings").join(id.to_string()).exists());
        assert_eq!(std::fs::read(source).unwrap(), b"original");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn successful_plain_output_cleans_only_recording() {
        let (root, source, id, output) = prepared_output();
        let mut text = Vec::new();
        output.emit_to(None, false, &mut text).unwrap();
        assert_eq!(text, b"transcribed text\n");
        assert!(!root.join("app/recordings").join(id.to_string()).exists());
        assert_eq!(std::fs::read(source).unwrap(), b"original");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cleanup_refuses_to_remove_an_original_source_inside_the_recording() {
        let root = std::env::temp_dir().join(format!("transcribe-guard-{}", uuid::Uuid::new_v4()));
        let store = RecordingStore::new(&root);
        let id = RecordingId::new();
        let directory = store.recording_dir(id);
        std::fs::create_dir_all(&directory).unwrap();
        let source = directory.join("source.wav");
        std::fs::write(&source, b"original").unwrap();
        store
            .apply_command(
                &mut AppState::new(),
                Command::CreateRecording {
                    recording_id: id,
                    source: SourceAsset::new(crate::domain::AssetKind::AudioFile, &source).unwrap(),
                },
            )
            .unwrap();
        let error = Recovery { store, id }.checked_directory().unwrap_err();
        assert!(error.to_string().contains("original source is inside"));
        assert_eq!(std::fs::read(source).unwrap(), b"original");
        std::fs::remove_dir_all(root).unwrap();
    }
}
