use crate::cli::output::CliOutput;
use crate::domain::RecordingId;
use crate::paths::AppHome;
use crate::storage::RecordingStore;
use crate::workflow::export_recording;
use crate::workflow::export_recording_with_timestamps;
use arbitrary::Arbitrary;
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
    pub recording_id: RecordingId,
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
    pub async fn invoke(self, app_home: &AppHome) -> Result<CliOutput> {
        let store = RecordingStore::new(app_home.as_ref());
        let output = self.output.map(PathBuf::from);
        let report = if self.timestamps {
            export_recording_with_timestamps(&store, self.recording_id, output)?
        } else {
            export_recording(&store, self.recording_id, output)?
        };

        Ok(CliOutput::facet(
            RecordingExportReport {
                recording_id: self.recording_id.to_string(),
                output_path: report.output_path.display().to_string(),
                transcript_count: report.transcript_count,
                byte_count: report.byte_count,
            },
            None, /* Recording history and export size are unbounded. */
        ))
    }
}
