use super::transcribe::MicrophoneTranscribeArgs;
use crate::capture::AudioCaptureReport;
use crate::capture::AudioInputDevice;
use crate::capture::list_audio_input_devices;
use crate::cli::output::CliOutput;
use crate::cli::positive_milliseconds::PositiveMilliseconds;
use crate::paths::AppHome;
use crate::paths::CacheHome;
use crate::storage::RecordingStore;
use crate::workflow::record_microphone_for_duration;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use figue as args;
use std::path::PathBuf;
use std::time::Duration;
use teamy_cancellation::CancellationScope;

/// Inspect local microphone input devices.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct MicrophoneArgs {
    /// The microphone subcommand to run.
    #[facet(args::subcommand)]
    pub command: MicrophoneCommand,
}

#[derive(Facet, Arbitrary, Debug, PartialEq)]
#[repr(u8)]
pub enum MicrophoneCommand {
    /// List active Windows Core Audio capture endpoints without starting capture.
    List(MicrophoneListArgs),
    /// Capture a bounded microphone interval into a persisted recording.
    Record(MicrophoneRecordArgs),
    /// Transcribe microphone audio live; Ctrl+C stops capture and drains queued work.
    Transcribe(MicrophoneTranscribeArgs),
}

#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct MicrophoneListArgs;

#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct MicrophoneRecordArgs {
    /// Capture endpoint ID; defaults to the Windows default microphone.
    #[facet(args::named)]
    pub device_id: Option<String>,
    /// Capture duration in milliseconds.
    #[facet(args::named)]
    pub duration_ms: PositiveMilliseconds,
    /// WAV destination; defaults to the recording-owned source directory.
    #[facet(args::named)]
    pub output: Option<String>,
}

#[derive(Facet, Debug)]
struct MicrophoneListReport {
    devices: Vec<AudioInputDevice>,
}

impl MicrophoneArgs {
    /// # Errors
    ///
    /// Returns an error when microphone inventory cannot be queried.
    pub async fn invoke(
        self,
        app_home: &AppHome,
        cache_home: &CacheHome,
        cancellation: &CancellationScope,
    ) -> Result<CliOutput> {
        match self.command {
            MicrophoneCommand::List(args) => args.invoke().await,
            MicrophoneCommand::Record(args) => args.invoke(app_home).await,
            MicrophoneCommand::Transcribe(args) => args.invoke(app_home, cache_home, cancellation),
        }
    }
}

#[derive(Facet, Debug)]
struct MicrophoneRecordReport {
    recording_id: String,
    device_id: Option<String>,
    capture: AudioCaptureReport,
}

impl MicrophoneRecordArgs {
    /// # Errors
    ///
    /// Returns an error when the duration is invalid, the recording manifest
    /// cannot be persisted, or the Windows capture session fails.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self, app_home: &AppHome) -> Result<CliOutput> {
        let store = RecordingStore::new(app_home.as_ref());
        let report = record_microphone_for_duration(
            &store,
            self.device_id.as_deref(),
            self.output.map(PathBuf::from),
            self.duration_ms.duration(),
        )?;

        Ok(CliOutput::facet(
            MicrophoneRecordReport {
                recording_id: report.recording_id.to_string(),
                device_id: self.device_id,
                capture: report.capture,
            },
            Some(
                self.duration_ms
                    .duration()
                    .saturating_add(Duration::from_secs(5)),
            ),
        ))
    }
}

impl MicrophoneListArgs {
    /// # Errors
    ///
    /// Returns an error when microphone inventory cannot be queried.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        Ok(CliOutput::facet(
            MicrophoneListReport {
                devices: list_audio_input_devices()?,
            },
            Some(Duration::from_secs(3)),
        ))
    }
}
