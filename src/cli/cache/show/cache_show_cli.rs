use crate::cli::output::CliOutput;
use crate::paths::CacheHome;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use std::time::Duration;

#[derive(Facet, Debug)]
struct CacheShowReport {
    path: String,
}

/// Show the cache path.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct CacheShowArgs;

impl CacheShowArgs {
    /// # Errors
    ///
    /// This function does not return any errors.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self, cache_home: &CacheHome) -> Result<CliOutput> {
        Ok(CliOutput::facet(
            CacheShowReport {
                path: cache_home.display().to_string(),
            },
            Some(Duration::from_secs(1)),
        ))
    }
}
