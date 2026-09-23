use crate::cli::output::CliOutput;
use crate::paths::AppHome;
use crate::recording_cleanup;
use crate::storage::RecordingStore;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use facet::Facet;
use figue as args;

/// Remove completed microphone recordings, preserving unfinished work and imported media.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct RecordingCleanArgs {
    /// Show how many recordings would be removed without deleting anything.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub dry_run: bool,
}

#[derive(Facet, Debug)]
struct RecordingCleanReport {
    dry_run: bool,
    eligible: usize,
    removed: usize,
    preserved: usize,
}

impl RecordingCleanArgs {
    /// # Errors
    /// Returns an error if recordings cannot be inspected or an eligible directory cannot be removed.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let store = RecordingStore::new(AppHome::resolve()?.0);
        let recordings = store
            .list_recordings()
            .wrap_err("failed to inspect recordings before cleanup")?;
        let eligible: Vec<_> = recordings
            .iter()
            .filter(|recording| recording_cleanup::is_completed_microphone(&store, recording))
            .map(|recording| recording.id)
            .collect();
        let mut removed = 0;
        if !self.dry_run {
            for id in &eligible {
                recording_cleanup::remove_completed_microphone(&store, *id).wrap_err_with(|| {
                    format!(
                        "recording clean removed {removed} recording(s) before stopping at {id}; rerun after resolving the error"
                    )
                })?;
                removed += 1;
            }
        }
        Ok(CliOutput::facet(RecordingCleanReport {
            dry_run: self.dry_run,
            eligible: eligible.len(),
            removed,
            preserved: recordings.len() - eligible.len(),
        }))
    }
}
