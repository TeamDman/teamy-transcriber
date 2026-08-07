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
}

impl ModelArgs {
    /// # Errors
    ///
    /// This function will return an error if the model subcommand fails.
    pub async fn invoke(self) -> Result<CliOutput> {
        match self.command {
            ModelCommand::Show(args) => args.invoke().await,
        }
    }
}
