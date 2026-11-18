use crate::cli::output::CliOutput;
use crate::paths::AppHome;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use facet::Facet;
use std::time::Duration;

/// Open the home path in the platform file manager.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct HomeOpenArgs;

impl HomeOpenArgs {
    /// # Errors
    ///
    /// This function will return an error if the home directory cannot be created
    /// or the file manager cannot be launched.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self, app_home: &AppHome) -> Result<CliOutput> {
        app_home.ensure_dir()?;
        open::that_detached(app_home.as_os_str())
            .wrap_err_with(|| format!("Failed to open {} in file manager", app_home.display()))?;
        Ok(CliOutput::none(Some(Duration::from_secs(3))))
    }
}
