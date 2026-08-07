use crate::cli::output::CliOutput;
use crate::transcription::LocalModelInventory;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;

#[derive(Facet, Debug)]
struct DoctorReport {
    app_home: String,
    app_home_exists: bool,
    cache_home: String,
    cache_home_exists: bool,
    model_home: String,
    model_home_exists: bool,
    model_file_count: usize,
    whisperx: String,
}

/// Report local application paths and the current transcription-runtime placeholder.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct DoctorArgs;

impl DoctorArgs {
    /// # Errors
    ///
    /// This function returns an error if the platform application paths cannot be resolved.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let app_home = crate::paths::AppHome::resolve()?;
        let cache_home = crate::paths::CacheHome::resolve()?;
        let model_home = crate::paths::ModelHome::resolve()?;
        let model_inventory = LocalModelInventory::inspect(model_home.0.clone())?;

        Ok(CliOutput::facet(DoctorReport {
            app_home: app_home.display().to_string(),
            app_home_exists: app_home.exists(),
            cache_home: cache_home.display().to_string(),
            cache_home_exists: cache_home.exists(),
            model_home: model_home.display().to_string(),
            model_home_exists: model_inventory.exists,
            model_file_count: model_inventory.file_count,
            whisperx: "local model files are assumed; runtime integration is planned in W9"
                .to_string(),
        }))
    }
}
