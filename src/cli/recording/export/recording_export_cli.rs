use crate::cli::output::CliOutput;
use crate::domain::RecordingId;
use crate::domain::TranscriptProvenance;
use crate::storage::RecordingStore;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use eyre::bail;
use facet::Facet;
use figue as args;
use std::fmt::Write as _;
use std::fs::File;
use std::io::Write;
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
        let recording = store
            .load_recording(recording_id)
            .wrap_err("failed to load recording manifest")?;
        if recording.transcripts.is_empty() {
            bail!("recording has no committed transcripts to export");
        }

        let mut text = String::new();
        let mut transcript_count = 0;
        for clip in recording
            .clips
            .iter()
            .filter(|clip| clip.status != crate::domain::ClipStatus::Deleted)
        {
            let Some(transcript) = recording
                .transcripts
                .iter()
                .rev()
                .find(|transcript| transcript.clip_id == clip.id)
            else {
                continue;
            };
            if !text.is_empty() {
                text.push_str("\n\n");
            }
            let _ = write!(
                text,
                "[clip {} | {}]\n{}",
                clip.id,
                provenance_label(transcript.provenance),
                transcript.text.trim()
            );
            transcript_count += 1;
        }
        if transcript_count == 0 {
            bail!("recording has no transcripts for active clips");
        }

        let output_path = self.output.map_or_else(
            || {
                store
                    .recording_dir(recording_id)
                    .join("transcripts")
                    .join("transcript.txt")
            },
            PathBuf::from,
        );
        write_atomic_text(&output_path, &text)?;

        Ok(CliOutput::facet(RecordingExportReport {
            recording_id: recording_id.to_string(),
            output_path: output_path.display().to_string(),
            transcript_count,
            byte_count: text.len(),
        }))
    }
}

fn provenance_label(provenance: TranscriptProvenance) -> &'static str {
    match provenance {
        TranscriptProvenance::RawAsr => "raw_asr",
        TranscriptProvenance::UserEdit => "user_edit",
        TranscriptProvenance::LocalLlm => "local_llm",
        TranscriptProvenance::Imported => "imported",
    }
}

fn write_atomic_text(path: &PathBuf, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let temporary_path = path.with_extension("txt.tmp");
    let mut file = File::create(&temporary_path)?;
    file.write_all(text.as_bytes())?;
    file.write_all(b"\n")?;
    file.flush()?;
    drop(file);
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    std::fs::rename(temporary_path, path)?;
    Ok(())
}
