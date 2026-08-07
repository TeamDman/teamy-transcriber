pub mod cache;
pub mod doctor;
pub mod facet_shape;
pub mod global_args;
pub mod home;
pub mod microphone;
pub mod model;
pub mod output;
pub mod recording;

use crate::cli::cache::CacheArgs;
use crate::cli::doctor::DoctorArgs;
use crate::cli::global_args::GlobalArgs;
use crate::cli::home::HomeArgs;
use crate::cli::microphone::MicrophoneArgs;
use crate::cli::model::ModelArgs;
use crate::cli::output::CliOutput;
use crate::cli::recording::RecordingArgs;
use arbitrary::Arbitrary;
use eyre::Context;
use facet::Facet;
use figue::FigueBuiltins;
use figue::{self as args};
use teamy_cancellation::CancellationToken;

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
    pub fn invoke(self, cancellation_token: CancellationToken) -> eyre::Result<CliOutput> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .wrap_err("Failed to build tokio runtime")?;
        runtime.block_on(async move { self.command.invoke(cancellation_token).await })
    }
}

/// Top-level teamy-transcriber commands.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
#[repr(u8)]
pub enum Command {
    /// Cache-related commands.
    Cache(CacheArgs),
    /// Report local paths and transcription-runtime readiness.
    Doctor(DoctorArgs),
    /// Home-related commands.
    Home(HomeArgs),
    /// Local model inspection commands.
    Model(ModelArgs),
    /// Local microphone inventory commands.
    Microphone(MicrophoneArgs),
    /// Recording and clip commands.
    Recording(RecordingArgs),
}

impl Command {
    /// # Errors
    ///
    /// This function will return an error if the subcommand fails.
    pub async fn invoke(self, cancellation_token: CancellationToken) -> eyre::Result<CliOutput> {
        cancellation_token.bail_if_cancelled()?;
        match self {
            Command::Cache(args) => args.invoke().await,
            Command::Doctor(args) => args.invoke().await,
            Command::Home(args) => args.invoke().await,
            Command::Model(args) => args.invoke().await,
            Command::Microphone(args) => args.invoke().await,
            Command::Recording(args) => args.invoke().await,
        }
    }
}
