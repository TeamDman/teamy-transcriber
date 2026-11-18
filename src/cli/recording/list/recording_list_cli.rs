use crate::cli::output::CliOutput;
use crate::cli::output::CliOutputValue;
use crate::cli::output::OutputFormat;
use crate::cli::output::write_facet;
use crate::domain::AssetKind;
use crate::domain::RecordingStatus;
use crate::paths::AppHome;
use crate::storage::RecordingStore;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use facet::Facet;
use std::io::Write;
use std::time::Duration;

#[derive(Facet, Debug)]
struct RecordingListEntry {
    recording_id: String,
    source_path: String,
    source_kind: AssetKind,
    status: RecordingStatus,
    failure: Option<String>,
    clip_count: usize,
    transcript_count: usize,
}

/// List saved recordings in UUID order without decoding audio or loading a model.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct RecordingListArgs;

#[derive(Debug)]
struct RecordingListOutput {
    entries: Vec<RecordingListEntry>,
}

impl CliOutputValue for RecordingListOutput {
    fn write_to(
        &self,
        format: OutputFormat,
        stdout_is_terminal: bool,
        writer: &mut dyn Write,
    ) -> Result<()> {
        match format {
            OutputFormat::Text if self.entries.is_empty() => {
                writer.write_all(b"No saved recordings.")?;
                Ok(())
            }
            OutputFormat::Csv => {
                // The CSV serializer accepts one flat record at a time.
                writer.write_all(b"recording_id,source_path,source_kind,status,failure,clip_count,transcript_count\n")?;
                for entry in &self.entries {
                    write_facet(entry, format, stdout_is_terminal, writer)?;
                }
                Ok(())
            }
            _ => write_facet(&self.entries, format, stdout_is_terminal, writer),
        }
    }
}

impl RecordingListArgs {
    /// # Errors
    /// Returns an error when application storage cannot be read or a recording is malformed.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self, app_home: &AppHome) -> Result<CliOutput> {
        let store = RecordingStore::new(app_home.as_ref());
        let recordings = store
            .list_recordings()
            .wrap_err("failed to list saved recordings")?;
        let entries: Vec<_> = recordings
            .into_iter()
            .map(|recording| RecordingListEntry {
                recording_id: recording.id.to_string(),
                source_path: recording.source.path,
                source_kind: recording.source.kind,
                status: recording.status,
                failure: recording.failure,
                clip_count: recording.clips.len(),
                transcript_count: recording.transcripts.len(),
            })
            .collect();
        Ok(CliOutput::custom(
            RecordingListOutput { entries },
            Some(Duration::from_secs(5)),
        ))
    }
}
