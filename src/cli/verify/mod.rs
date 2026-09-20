use crate::cli::output::CliOutput;
use crate::verify::VctkCanaryOptions;
use arbitrary::Arbitrary;
use eyre::Result;
use facet::Facet;
use figue as args;
use std::path::PathBuf;

/// Reproducible local verification commands.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct VerifyArgs {
    /// The verification family to run.
    #[facet(args::subcommand)]
    pub command: VerifyCommand,
}

#[derive(Facet, Arbitrary, Debug, PartialEq)]
#[repr(u8)]
pub enum VerifyCommand {
    /// Speech-dataset verification commands.
    Speech(SpeechArgs),
}

#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct SpeechArgs {
    /// The speech verification fixture to run.
    #[facet(args::subcommand)]
    pub command: SpeechCommand,
}

#[derive(Facet, Arbitrary, Debug, PartialEq)]
#[repr(u8)]
pub enum SpeechCommand {
    /// Verify the short VCTK `p230_385` speech canary.
    VctkP230_385(VctkCanaryArgs),
}

#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct VctkCanaryArgs {
    /// Root of the user-owned VCTK corpus; never downloaded by this command.
    #[facet(args::named)]
    pub vctk_root: Option<String>,
    /// Receipt destination. The parent directory is created when needed.
    #[facet(args::named)]
    pub receipt: Option<String>,
    /// Local canonical safetensors Whisper model directory.
    #[facet(args::named)]
    pub model_dir: Option<String>,
    /// Maximum decoder tokens for the canary.
    #[facet(args::named)]
    pub max_decode_tokens: Option<usize>,
}

impl VerifyArgs {
    /// # Errors
    ///
    /// Returns an error when the verification command cannot be invoked.
    pub async fn invoke(self) -> Result<CliOutput> {
        match self.command {
            VerifyCommand::Speech(args) => args.invoke().await,
        }
    }
}

impl SpeechArgs {
    /// # Errors
    ///
    /// Returns an error when the selected speech verification command cannot
    /// be invoked.
    pub async fn invoke(self) -> Result<CliOutput> {
        match self.command {
            SpeechCommand::VctkP230_385(args) => args.invoke().await,
        }
    }
}

impl VctkCanaryArgs {
    /// # Errors
    ///
    /// Returns an error only when the receipt itself cannot be written or the
    /// command configuration cannot be resolved. Dataset/model/inference
    /// outcomes are returned as typed non-passing reports with receipts.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let options = VctkCanaryOptions {
            vctk_root: self.vctk_root.map(PathBuf::from),
            receipt_path: self.receipt.map(PathBuf::from),
            model_dir: self.model_dir.map(PathBuf::from),
            max_decode_tokens: self.max_decode_tokens,
        };
        let report = crate::verify::run_vctk_canary(options)?;
        Ok(CliOutput::facet(report))
    }
}
