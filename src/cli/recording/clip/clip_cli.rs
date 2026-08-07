use crate::cli::output::CliOutput;
use crate::domain::ClipId;
use crate::domain::Command;
use crate::domain::RecordingId;
use crate::domain::TimeRange;
use crate::storage::RecordingStore;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use facet::Facet;
use figue as args;

/// Manage immutable recording clips.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ClipArgs {
    /// The clip subcommand to run.
    #[facet(args::subcommand)]
    pub command: ClipCommand,
}

#[derive(Facet, Arbitrary, Debug, PartialEq)]
#[repr(u8)]
pub enum ClipCommand {
    /// Add one source-time clip to a recording.
    Add(ClipAddArgs),
}

#[derive(Facet, Debug)]
struct ClipAddReport {
    recording_id: String,
    clip_id: String,
    start_us: u64,
    end_us: u64,
}

/// Add an immutable source-time clip to a persisted recording.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ClipAddArgs {
    /// Recording UUID returned by recording create.
    #[facet(args::positional)]
    pub recording_id: String,
    /// Inclusive start position in microseconds.
    #[facet(args::positional)]
    pub start_us: u64,
    /// Exclusive end position in microseconds.
    #[facet(args::positional)]
    pub end_us: u64,
}

impl ClipArgs {
    /// # Errors
    ///
    /// Returns an error when the clip command cannot be persisted.
    pub async fn invoke(self) -> Result<CliOutput> {
        match self.command {
            ClipCommand::Add(args) => args.invoke().await,
        }
    }
}

impl ClipAddArgs {
    /// # Errors
    ///
    /// Returns an error when the recording ID or range is invalid, or the
    /// recording event receipt cannot be updated.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let recording_id =
            RecordingId::parse(&self.recording_id).wrap_err("recording ID must be a UUID")?;
        let source_range = TimeRange::new(self.start_us, self.end_us)?;
        let app_home = crate::paths::AppHome::resolve()?;
        let store = RecordingStore::new(app_home.0);
        let mut state = store
            .load_state(recording_id)
            .wrap_err("failed to load recording event state")?;
        let clip_id = ClipId::new();
        store
            .apply_command(
                &mut state,
                Command::AddClip {
                    recording_id,
                    clip_id,
                    source_range,
                },
            )
            .wrap_err("failed to persist clip")?;

        Ok(CliOutput::facet(ClipAddReport {
            recording_id: recording_id.to_string(),
            clip_id: clip_id.to_string(),
            start_us: source_range.start_us,
            end_us: source_range.end_us,
        }))
    }
}
