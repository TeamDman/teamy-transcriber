use crate::cli::output::CliOutput;
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
    pub async fn invoke(self) -> Result<CliOutput> {
        crate::gui::run()?;
        Ok(CliOutput::none())
    }
}
