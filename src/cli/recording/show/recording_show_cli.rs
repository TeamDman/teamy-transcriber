use crate::cli::output::CliOutput;
use crate::domain::ClipStatus;
use crate::domain::RecordingId;
use crate::domain::RecordingStatus;
use crate::storage::RecordingStore;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use facet::Facet;
use figue as args;

#[derive(Facet, Debug)]
struct RecordingShowReport {
    recording_id: String,
    source_path: String,
    status: RecordingStatus,
    failure: Option<String>,
    clip_count: usize,
    transcript_count: usize,
    clips: Vec<RecordingClipReport>,
}

#[derive(Facet, Debug)]
struct RecordingClipReport {
    clip_id: String,
    start_us: u64,
    end_us: u64,
    status: ClipStatus,
    failure: Option<String>,
}

/// Show one persisted recording manifest.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct RecordingShowArgs {
    /// Recording UUID returned by recording create.
    #[facet(args::positional)]
    pub recording_id: String,
}

impl RecordingShowArgs {
    /// # Errors
    ///
    /// Returns an error when the recording ID is invalid or its manifest cannot be loaded.
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

        Ok(CliOutput::facet(RecordingShowReport {
            recording_id: recording.id.to_string(),
            source_path: recording.source.path,
            status: recording.status,
            failure: recording.failure,
            clip_count: recording.clips.len(),
            transcript_count: recording.transcripts.len(),
            clips: recording
                .clips
                .into_iter()
                .map(|clip| RecordingClipReport {
                    clip_id: clip.id.to_string(),
                    start_us: clip.source_range.start_us,
                    end_us: clip.source_range.end_us,
                    status: clip.status,
                    failure: clip.failure,
                })
                .collect(),
        }))
    }
}
