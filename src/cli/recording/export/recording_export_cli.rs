use crate::cli::output::CliOutput;
use crate::domain::RecordingId;
use crate::storage::RecordingStore;
use crate::workflow::export_recording;
use crate::workflow::export_recording_with_timestamps;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use facet::Facet;
use figue as args;
use std::path::PathBuf;

#[derive(Facet, Debug)]
struct RecordingExportReport {
    recording_id: String,
    output_path: String,
    transcript_count: usize,
    byte_count: usize,
}

/// Export the latest committed transcript for each active clip.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct RecordingExportArgs {
    /// Recording UUID returned by recording create.
    #[facet(args::positional)]
    pub recording_id: String,
    /// Destination text file; defaults to the recording transcript directory.
    #[facet(args::named)]
    pub output: Option<String>,
    /// Prefix each persisted transcription segment with its source-time range.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub timestamps: bool,
}

impl RecordingExportArgs {
    /// # Errors
    ///
    /// Returns an error when the recording has no transcripts or the export
    /// destination cannot be written.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let recording_id =
            RecordingId::parse(&self.recording_id).wrap_err("recording ID must be a UUID")?;
        let app_home = crate::paths::AppHome::resolve()?;
        let store = RecordingStore::new(app_home.0);
        let output = self.output.map(PathBuf::from);
        let report = if self.timestamps {
            export_recording_with_timestamps(&store, recording_id, output)?
        } else {
            export_recording(&store, recording_id, output)?
        };

        Ok(CliOutput::facet(RecordingExportReport {
            recording_id: recording_id.to_string(),
            output_path: report.output_path.display().to_string(),
            transcript_count: report.transcript_count,
            byte_count: report.byte_count,
        }))
    }
}
