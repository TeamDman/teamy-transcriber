use crate::cli::output::CliOutput;
use crate::cli::recording::clean::RecordingCleanArgs;
use crate::cli::recording::clip::ClipArgs;
use crate::cli::recording::create::RecordingCreateArgs;
use crate::cli::recording::export::RecordingExportArgs;
use crate::cli::recording::list::RecordingListArgs;
use crate::cli::recording::prepare::RecordingPrepareArgs;
use crate::cli::recording::show::RecordingShowArgs;
use crate::cli::recording::transcribe::RecordingTranscribeArgs;
use crate::paths::AppHome;
use crate::paths::CacheHome;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use figue as args;

/// Recording and clip commands.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct RecordingArgs {
    /// The recording subcommand to run.
    #[facet(args::subcommand)]
    pub command: RecordingCommand,
}

#[derive(Facet, Arbitrary, Debug, PartialEq)]
#[repr(u8)]
pub enum RecordingCommand {
    /// Manage immutable recording clips.
    Clip(ClipArgs),
    /// Remove completed microphone recording intermediates.
    Clean(RecordingCleanArgs),
    /// Create a durable recording manifest for an audio, video, or microphone source.
    Create(RecordingCreateArgs),
    /// Export committed transcript text for a recording.
    Export(RecordingExportArgs),
    /// List saved recordings and their source, status, and clip/transcript counts.
    List(RecordingListArgs),
    /// Normalize a WAV recording into local 16 kHz mono audio.
    Prepare(RecordingPrepareArgs),
    /// Show a recording manifest and its current clip/transcript counts.
    Show(RecordingShowArgs),
    /// Transcribe a prepared recording clip with native local Whisper.
    Transcribe(RecordingTranscribeArgs),
}

impl RecordingArgs {
    /// # Errors
    ///
    /// This function returns an error if the recording subcommand fails.
    pub async fn invoke(self, app_home: &AppHome, cache_home: &CacheHome) -> Result<CliOutput> {
        match self.command {
            RecordingCommand::Clip(args) => args.invoke(app_home).await,
            RecordingCommand::Clean(args) => args.invoke(app_home).await,
            RecordingCommand::Create(args) => args.invoke(app_home).await,
            RecordingCommand::Export(args) => args.invoke(app_home).await,
            RecordingCommand::List(args) => args.invoke(app_home).await,
            RecordingCommand::Prepare(args) => args.invoke(app_home).await,
            RecordingCommand::Show(args) => args.invoke(app_home).await,
            RecordingCommand::Transcribe(args) => args.invoke(app_home, cache_home).await,
        }
    }
}
