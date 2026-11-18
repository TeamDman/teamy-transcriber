pub mod cache;
pub(crate) mod cuda_whisper_options;
pub mod decode_token_limit;
pub mod doctor;
pub mod facet_shape;
pub mod global_args;
pub mod gui;
pub mod home;
pub mod microphone;
pub mod model;
pub mod output;
pub mod phones;
pub mod positive_milliseconds;
pub mod recording;
pub mod recording_chunk_milliseconds;
pub mod transcribe;
pub mod verify;
pub use crate::microphone_chunk_milliseconds;

use crate::cli::cache::CacheArgs;
use crate::cli::doctor::DoctorArgs;
use crate::cli::global_args::GlobalArgs;
use crate::cli::gui::GuiArgs;
use crate::cli::home::HomeArgs;
use crate::cli::microphone::MicrophoneArgs;
use crate::cli::model::ModelArgs;
use crate::cli::output::CliOutput;
use crate::cli::recording::RecordingArgs;
use crate::cli::verify::VerifyArgs;
use crate::paths::AppHome;
use crate::paths::CacheHome;
use arbitrary::Arbitrary;
use eyre::Context;
use facet::Facet;
use figue::FigueBuiltins;
use figue::{self as args};
use teamy_cancellation::CancellationScope;
use tokio::runtime::Builder as RuntimeBuilder;

/// Local-first audio and video transcription utility.
///
/// Environment variables:
/// - `TEAMY_TRANSCRIBER_HOME_DIR` overrides the resolved application home directory.
/// - `TEAMY_TRANSCRIBER_CACHE_DIR` overrides the resolved cache directory.
/// - `RUST_LOG` provides a tracing filter when `--log-filter` is omitted.
#[derive(Facet, Arbitrary, Debug)]
pub struct Cli {
    /// Global arguments (`debug`, `log_filter`, `log_file`).
    #[facet(flatten)]
    pub global_args: GlobalArgs,

    /// Standard CLI options (help, version, completions).
    #[facet(flatten)]
    #[arbitrary(default)]
    pub builtins: FigueBuiltins,

    /// The command to run.
    #[facet(args::subcommand)]
    pub command: Command,
}

impl PartialEq for Cli {
    fn eq(&self, other: &Self) -> bool {
        // Ignore builtins in comparison since FigueBuiltins doesn't implement PartialEq
        self.global_args == other.global_args && self.command == other.command
    }
}

impl Cli {
    /// # Errors
    ///
    /// This function will return an error if the tokio runtime cannot be built or if the command fails.
    pub fn invoke(
        self,
        app_home: &AppHome,
        cache_home: &CacheHome,
        cancellation: &CancellationScope,
    ) -> eyre::Result<CliOutput> {
        let runtime = RuntimeBuilder::new_multi_thread()
            .enable_all()
            .build()
            .wrap_err("Failed to build tokio runtime")?;
        runtime.block_on(async move {
            self.command
                .invoke(app_home, cache_home, cancellation)
                .await
        })
    }
}

/// Top-level teamy-transcriber commands.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
#[facet(rename_all = "kebab-case")]
#[repr(u8)]
pub enum Command {
    /// Cache-related commands.
    Cache(CacheArgs),
    /// Report local paths and transcription-runtime readiness.
    Doctor(DoctorArgs),
    /// Home-related commands.
    Home(HomeArgs),
    /// Open the native Teamy-Transcriber desktop window.
    Gui(GuiArgs),
    /// Local model inspection commands.
    Model(ModelArgs),
    /// Local microphone inventory commands.
    Microphone(MicrophoneArgs),
    /// Recording and clip commands.
    Recording(RecordingArgs),
    /// Transcribe a media file, printing text and cleaning up successful work.
    Transcribe(transcribe::TranscribeArgs),
    /// Reproducible local speech verification commands.
    Verify(VerifyArgs),
    /// Recognize IPA phones directly from a media file.
    Phones(phones::PhonesArgs),
}

impl Command {
    /// # Errors
    /// Returns an error if Facet cannot reflect the active command variant.
    pub fn name(&self) -> eyre::Result<&'static str> {
        let command = facet::Peek::new(self)
            .into_enum()
            .map_err(|error| eyre::eyre!("Could not reflect command enum: {error}"))?;
        let variant = command
            .active_variant()
            .map_err(|error| eyre::eyre!("Could not reflect active command variant: {error}"))?;
        Ok(variant.effective_name())
    }

    /// # Errors
    ///
    /// This function will return an error if the subcommand fails.
    pub async fn invoke(
        self,
        app_home: &AppHome,
        cache_home: &CacheHome,
        cancellation: &CancellationScope,
    ) -> eyre::Result<CliOutput> {
        cancellation.bail_if_cancelled()?;
        match self {
            Command::Cache(args) => args.invoke(cache_home).await,
            Command::Doctor(args) => args.invoke(app_home, cache_home).await,
            Command::Home(args) => args.invoke(app_home).await,
            Command::Gui(args) => args.invoke(app_home, cache_home).await,
            Command::Model(args) => args.invoke(app_home, cache_home).await,
            Command::Microphone(args) => args.invoke(app_home, cache_home, cancellation).await,
            Command::Recording(args) => args.invoke(app_home, cache_home).await,
            Command::Transcribe(args) => args.invoke(app_home, cache_home, cancellation),
            Command::Verify(args) => args.invoke(app_home, cache_home).await,
            Command::Phones(args) => args.invoke(app_home, cancellation),
        }
    }
}
