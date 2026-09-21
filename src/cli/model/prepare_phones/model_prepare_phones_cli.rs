use crate::cli::output::CliOutput;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use figue as args;
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
    pub fn invoke(self) -> Result<CliOutput> {
        let path = crate::phone_runtime::resolve(self.source_dir.as_deref())?;
        let path = crate::phone_runtime::select(&path)?;
        Ok(CliOutput::facet(path.to_string_lossy().into_owned()))
    }
}
