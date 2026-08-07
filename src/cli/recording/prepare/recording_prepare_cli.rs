use crate::cli::output::CliOutput;
use crate::domain::RecordingId;
use crate::media::MediaAdapter;
use crate::media::WavMediaAdapter;
use crate::storage::RecordingStore;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use facet::Facet;
use figue as args;
use std::path::Path;

#[derive(Facet, Debug)]
struct RecordingPrepareReport {
    recording_id: String,
    normalized_path: String,
    duration_us: u64,
    sample_rate_hz: u32,
    channels: u16,
    frame_count: u64,
}

/// Normalize a persisted WAV source into mono 16 kHz audio.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct RecordingPrepareArgs {
    /// Recording UUID returned by recording create.
    #[facet(args::positional)]
    pub recording_id: String,
}

impl RecordingPrepareArgs {
    /// # Errors
    ///
    /// Returns an error when the recording cannot be loaded or the WAV source
    /// cannot be normalized.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let recording_id =
            RecordingId::parse(&self.recording_id).wrap_err("recording ID must be a UUID")?;
        let app_home = crate::paths::AppHome::resolve()?;
        let store = RecordingStore::new(app_home.0);
        let recording = store
            .load_recording(recording_id)
            .wrap_err("failed to load recording manifest")?;
        let source = Path::new(&recording.source.path);
        let output_dir = store.recording_dir(recording_id).join("audio");
        let prepared = WavMediaAdapter
            .prepare_audio(source, &output_dir)
            .wrap_err("failed to normalize WAV source")?;

        Ok(CliOutput::facet(RecordingPrepareReport {
            recording_id: recording_id.to_string(),
            normalized_path: prepared.path.display().to_string(),
            duration_us: prepared.metadata.duration_us,
            sample_rate_hz: prepared.metadata.sample_rate_hz,
            channels: prepared.metadata.channels,
            frame_count: prepared.metadata.frame_count,
        }))
    }
}
