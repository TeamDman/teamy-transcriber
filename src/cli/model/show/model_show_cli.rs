use crate::cli::output::CliOutput;
use crate::transcription::LocalModelInventory;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;

#[derive(Facet, Debug)]
struct ModelShowReport {
    path: String,
    exists: bool,
    file_count: usize,
    download_policy: String,
}

/// Show the local model directory without downloading or modifying it.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ModelShowArgs;

impl ModelShowArgs {
    /// # Errors
    ///
    /// This function returns an error when the model directory cannot be inspected.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let model_home = crate::paths::ModelHome::resolve()?;
        let inventory = LocalModelInventory::inspect(model_home.0)?;

        Ok(CliOutput::facet(ModelShowReport {
            path: inventory.root.display().to_string(),
            exists: inventory.exists,
            file_count: inventory.file_count,
            download_policy: "assume local files; CDN acquisition is deferred".to_string(),
        }))
    }
}
