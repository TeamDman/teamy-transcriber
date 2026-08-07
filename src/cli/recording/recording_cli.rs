use crate::cli::output::CliOutput;
use crate::cli::recording::create::RecordingCreateArgs;
use crate::cli::recording::show::RecordingShowArgs;
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
    /// Create a durable recording manifest for an audio, video, or microphone source.
    Create(RecordingCreateArgs),
    /// Show a recording manifest and its current clip/transcript counts.
    Show(RecordingShowArgs),
}

impl RecordingArgs {
    /// # Errors
    ///
    /// This function returns an error if the recording subcommand fails.
    pub async fn invoke(self) -> Result<CliOutput> {
        match self.command {
            RecordingCommand::Create(args) => args.invoke().await,
            RecordingCommand::Show(args) => args.invoke().await,
        }
    }
}
