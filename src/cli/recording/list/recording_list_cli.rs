use crate::cli::output::CliOutput;
use crate::cli::output::CliOutputValue;
use crate::cli::output::OutputFormat;
use crate::domain::AssetKind;
use crate::domain::RecordingStatus;
use crate::paths::AppHome;
use crate::storage::RecordingStore;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use facet::Facet;
use facet_pretty::ColorMode;
use facet_pretty::PrettyPrinter;

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
    fn render(&self, format: OutputFormat, stdout_is_terminal: bool) -> Result<Option<String>> {
        let rendered = match format {
            OutputFormat::Text if self.entries.is_empty() => "No saved recordings.".to_owned(),
            OutputFormat::Text => PrettyPrinter::new()
                .with_colors(if stdout_is_terminal {
                    ColorMode::Always
                } else {
                    ColorMode::Never
                })
                .format(&self.entries),
            OutputFormat::Json => facet_json::to_string_pretty(&self.entries)?,
            OutputFormat::Csv => {
                // The CSV serializer accepts one flat record at a time.
                let mut csv = "recording_id,source_path,source_kind,status,failure,clip_count,transcript_count\n".to_owned();
                for entry in &self.entries {
                    csv.push_str(&facet_csv::to_string(entry)?);
                }
                csv
            }
        };
        Ok(Some(rendered))
    }
}

impl RecordingListArgs {
    /// # Errors
    /// Returns an error when application storage cannot be read or a recording is malformed.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let store = RecordingStore::new(AppHome::resolve()?.0);
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
        Ok(CliOutput::custom(RecordingListOutput { entries }))
    }
}
