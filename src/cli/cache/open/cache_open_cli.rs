use crate::cli::output::CliOutput;
use crate::paths::CacheHome;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use facet::Facet;
use std::time::Duration;

/// Open the cache path in the platform file manager.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct CacheOpenArgs;

impl CacheOpenArgs {
    /// # Errors
    ///
    /// This function will return an error if the cache directory cannot be created
    /// or the file manager cannot be launched.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self, cache_home: &CacheHome) -> Result<CliOutput> {
        std::fs::create_dir_all(cache_home)?;
        open::that_detached(cache_home.as_os_str())
            .wrap_err_with(|| format!("Failed to open {} in file manager", cache_home.display()))?;
        Ok(CliOutput::none(Some(Duration::from_secs(3))))
    }
}
