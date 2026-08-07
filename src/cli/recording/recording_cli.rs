use crate::cli::output::CliOutput;
use crate::cli::recording::clip::ClipArgs;
use crate::cli::recording::create::RecordingCreateArgs;
use crate::cli::recording::export::RecordingExportArgs;
use crate::cli::recording::prepare::RecordingPrepareArgs;
use crate::cli::recording::show::RecordingShowArgs;
use crate::cli::recording::transcribe::RecordingTranscribeArgs;
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
    /// Create a durable recording manifest for an audio, video, or microphone source.
    Create(RecordingCreateArgs),
    /// Export committed transcript text for a recording.
    Export(RecordingExportArgs),
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
    pub async fn invoke(self) -> Result<CliOutput> {
        match self.command {
            RecordingCommand::Clip(args) => args.invoke().await,
            RecordingCommand::Create(args) => args.invoke().await,
            RecordingCommand::Export(args) => args.invoke().await,
            RecordingCommand::Prepare(args) => args.invoke().await,
            RecordingCommand::Show(args) => args.invoke().await,
            RecordingCommand::Transcribe(args) => args.invoke().await,
        }
    }
}
