use crate::cli::output::CliOutput;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;

#[derive(Facet, Debug)]
struct DoctorReport {
    app_home: String,
    app_home_exists: bool,
    cache_home: String,
    cache_home_exists: bool,
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

        Ok(CliOutput::facet(DoctorReport {
            app_home: app_home.display().to_string(),
            app_home_exists: app_home.exists(),
            cache_home: cache_home.display().to_string(),
            cache_home_exists: cache_home.exists(),
            whisperx: "not configured yet; runtime preparation is planned in W8-W9".to_string(),
        }))
    }
}
