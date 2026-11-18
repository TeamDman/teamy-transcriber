use crate::cli::cuda_whisper_options::resolve_cuda_whisper_options;
use crate::cli::output::CliOutput;
use crate::paths::AppHome;
use crate::paths::CacheHome;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;

/// Open the native Teamy-Transcriber desktop window.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct GuiArgs;

impl GuiArgs {
    /// # Errors
    ///
    /// Returns an error when the window, Vulkan loader, surface, or swapchain
    /// cannot be initialized.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self, app_home: &AppHome, cache_home: &CacheHome) -> Result<CliOutput> {
        crate::gui::run(app_home, cache_home, resolve_cuda_whisper_options()?)?;
        Ok(CliOutput::none(
            None, /* User-interactive desktop lifetime. */
        ))
    }
}
