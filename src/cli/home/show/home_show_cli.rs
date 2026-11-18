use crate::cli::output::CliOutput;
use crate::paths::AppHome;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use std::time::Duration;

#[derive(Facet, Debug)]
struct HomeShowReport {
    path: String,
}

/// Show the home path.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct HomeShowArgs;

impl HomeShowArgs {
    /// # Errors
    ///
    /// This function does not return any errors.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self, app_home: &AppHome) -> Result<CliOutput> {
        Ok(CliOutput::facet(
            HomeShowReport {
                path: app_home.display().to_string(),
            },
            Some(Duration::from_secs(1)),
        ))
    }
}
