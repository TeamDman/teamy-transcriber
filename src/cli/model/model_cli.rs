use crate::cli::model::prepare::ModelPrepareArgs;
use crate::cli::model::prepare_vad::ModelPrepareVadArgs;
use crate::cli::model::show::ModelShowArgs;
use crate::cli::output::CliOutput;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use figue as args;

/// Local model inspection commands.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ModelArgs {
    /// The model subcommand to run.
    #[facet(args::subcommand)]
    pub command: ModelCommand,
}

#[derive(Facet, Arbitrary, Debug, PartialEq)]
#[repr(u8)]
pub enum ModelCommand {
    /// Show the assumed local model directory and inventory.
    Show(ModelShowArgs),
    /// Prepare a local canonical Hugging Face safetensors Whisper package.
    Prepare(ModelPrepareArgs),
    /// Prepare local data-only speech detection weights without downloading.
    PrepareVad(ModelPrepareVadArgs),
}

impl ModelArgs {
    /// # Errors
    ///
    /// This function will return an error if the model subcommand fails.
    pub async fn invoke(self) -> Result<CliOutput> {
        match self.command {
            ModelCommand::Show(args) => args.invoke().await,
            ModelCommand::Prepare(args) => args.invoke().await,
            ModelCommand::PrepareVad(args) => args.invoke().await,
        }
    }
}
