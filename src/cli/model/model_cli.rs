use super::prepare_phones::ModelPreparePhonesArgs;
use crate::cli::model::prepare::ModelPrepareArgs;
use crate::cli::model::prepare_vad::ModelPrepareVadArgs;
use crate::cli::model::show::ModelShowArgs;
use crate::cli::output::CliOutput;
use crate::paths::AppHome;
use crate::paths::CacheHome;
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
    /// Select data-only `PhoneticXeus` phone recognition weights.
    PreparePhones(ModelPreparePhonesArgs),
}

impl ModelArgs {
    /// # Errors
    ///
    /// This function will return an error if the model subcommand fails.
    pub async fn invoke(self, app_home: &AppHome, cache_home: &CacheHome) -> Result<CliOutput> {
        match self.command {
            ModelCommand::Show(args) => args.invoke(app_home, cache_home).await,
            ModelCommand::Prepare(args) => args.invoke(app_home, cache_home).await,
            ModelCommand::PrepareVad(args) => args.invoke().await,
            ModelCommand::PreparePhones(args) => args.invoke(app_home),
        }
    }
}
