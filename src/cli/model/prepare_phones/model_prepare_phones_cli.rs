use crate::cli::output::CliOutput;
use crate::paths::AppHome;
use crate::paths::PhoneModelDir;
use crate::paths::PhoneModelDirSelection;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use figue as args;
use std::time::Duration;
/// Validate and select canonical `PhoneticXeus` weights without conversion or downloads.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct ModelPreparePhonesArgs {
    /// Existing model folder; omitted discovers the pinned model in the hf cache.
    #[facet(args::named)]
    pub source_dir: Option<String>,
}
impl ModelPreparePhonesArgs {
    /// # Errors
    /// Returns model validation, CUDA availability or selection persistence errors.
    pub fn invoke(self, app_home: &AppHome) -> Result<CliOutput> {
        let path = PhoneModelDir::resolve(app_home, self.source_dir.as_deref())?;
        let path = path.select(app_home)?;
        Ok(CliOutput::facet(
            path.to_string_lossy().into_owned(),
            Some(Duration::from_secs(10)),
        ))
    }
}
