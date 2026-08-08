use crate::capture::AudioInputDevice;
use crate::capture::list_audio_input_devices;
use crate::cli::output::CliOutput;
use crate::storage::RecordingStore;
use crate::workflow::record_microphone_for_duration;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use figue as args;
use std::path::PathBuf;
use std::time::Duration;

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
    pub duration_ms: Option<u64>,
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
    pub async fn invoke(self) -> Result<CliOutput> {
        match self.command {
            MicrophoneCommand::List(args) => args.invoke().await,
            MicrophoneCommand::Record(args) => args.invoke().await,
        }
    }
}

#[derive(Facet, Debug)]
struct MicrophoneRecordReport {
    recording_id: String,
    device_id: Option<String>,
    capture: crate::capture::AudioCaptureReport,
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
    pub async fn invoke(self) -> Result<CliOutput> {
        let duration_ms = self
            .duration_ms
            .ok_or_else(|| eyre::eyre!("--duration-ms is required"))?;
        if duration_ms == 0 {
            eyre::bail!("--duration-ms must be greater than zero");
        }
        let app_home = crate::paths::AppHome::resolve()?;
        let store = RecordingStore::new(app_home.0);
        let report = record_microphone_for_duration(
            &store,
            self.device_id.as_deref(),
            self.output.map(PathBuf::from),
            Duration::from_millis(duration_ms),
        )?;

        Ok(CliOutput::facet(MicrophoneRecordReport {
            recording_id: report.recording_id.to_string(),
            device_id: self.device_id,
            capture: report.capture,
        }))
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
        Ok(CliOutput::facet(MicrophoneListReport {
            devices: list_audio_input_devices()?,
        }))
    }
}
