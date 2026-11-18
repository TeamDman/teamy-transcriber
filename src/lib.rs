#![deny(clippy::disallowed_methods)]
#![deny(clippy::disallowed_macros)]

pub mod capture;
pub mod cli;
pub mod domain;
mod elapsed_duration;
pub mod gui;
pub(crate) mod live_microphone;
pub mod logging_init;
pub mod media;
pub mod microphone_chunk_milliseconds;
pub mod native_whisper;
pub mod paths;
pub mod presentation;
mod recording_cleanup;
pub mod storage;
pub mod transcription;
pub mod verify;
#[cfg(windows)]
mod windows_startup;
pub mod workflow;

use crate::cli::Cli;
use crate::cli::Command;
use crate::cli::cuda_whisper_options::resolve_cuda_whisper_options;
use crate::cli::global_args::GlobalArgs;
use crate::cli::microphone::MicrophoneCommand;
use crate::cli::output::OutputFormat;
use crate::paths::AppHome;
use crate::paths::CacheHome;
use chrono::DateTime;
use chrono::Local;
use chrono::Utc;
use std::ffi::OsString;
use std::time::Instant;
use teamy_cancellation::CtrlCHandler;

/// Version string combining package version, git repository metadata, and build time.
fn version() -> String {
    let built_at = option_env!("BUILD_TIMESTAMP_UNIX")
        .and_then(|value| value.parse::<i64>().ok())
        .and_then(|timestamp| DateTime::<Utc>::from_timestamp(timestamp, 0))
        .map_or_else(
            || "unknown build time".to_string(),
            |timestamp| {
                timestamp
                    .with_timezone(&Local)
                    .format("%Y-%m-%d %H:%M:%S %Z")
                    .to_string()
            },
        );

    format!(
        "{} (repo {}, branch {}, rev {}, worktree {}, built {})",
        env!("CARGO_PKG_VERSION"),
        env!("GIT_REPOSITORY_URL"),
        env!("GIT_BRANCH"),
        env!("GIT_REVISION"),
        env!("GIT_WORKTREE_STATUS"),
        built_at,
    )
}

/// Entrypoint for the program.
///
/// # Errors
///
/// This function will return an error if `color_eyre` installation, CLI parsing, logging initialization, command execution, or command output rendering fails.
///
/// # Panics
///
/// Panics if the CLI schema is invalid (should never happen with correct code).
pub fn main() -> eyre::Result<()> {
    // Install color_eyre for better error reports
    color_eyre::install()?;
    let cancellation_token = CtrlCHandler::default().install()?;

    #[cfg(windows)]
    {
        // Enable ANSI support on Windows
        // This fails in a pipe scenario, so we ignore the error
        let _ = windows_startup::enable_ansi_support();

        // Warn if UTF-8 is not enabled on Windows
        #[cfg(windows)]
        windows_startup::warn_if_utf8_not_enabled();
    };

    // Parse command line arguments using figue
    // unwrap() is figue's intended CLI entry behavior:
    // it exits with proper codes for --help/--version/completions/parse-errors.
    let command_line_arguments: Vec<OsString> = std::env::args_os().skip(1).collect();
    let app_home = AppHome::resolve()?;
    let cache_home = CacheHome::resolve()?;
    if launch_gui_without_command(&command_line_arguments) {
        let global_args = GlobalArgs::default();
        logging_init::init_logging(&global_args, cancellation_token.clone())?;
        gui::run(&app_home, &cache_home, resolve_cuda_whisper_options()?)?;
        cancellation_token.with_scope(|scope| scope.bail_if_cancelled())?;
        return Ok(());
    }
    let version = version();

    let cli: Cli = figue::Driver::new(
        figue::builder::<Cli>()
            .expect("schema should be valid")
            .cli(move |cli| cli.args_os(command_line_arguments).strict())
            .help(move |help| {
                help.version(version)
                    .include_implementation_source_file(true)
                    .include_implementation_github_url(
                        "TeamDman/teamy-transcriber",
                        env!("GIT_REVISION"),
                    )
            })
            .build(),
    )
    .run()
    .unwrap();

    // For live microphone transcription cancellation stops the producer; the
    // consumer must finish queued audio and return success after draining.
    let drains_capture = matches!(&cli.command, Command::Microphone(args)
        if matches!(&args.command, MicrophoneCommand::Transcribe(_)));
    if drains_capture
        && cli
            .global_args
            .output_format
            .is_some_and(|format| format != OutputFormat::Text)
    {
        eyre::bail!("live microphone transcription emits plain text; use --output-format text");
    }

    let _stop_after_duration_thread = cli
        .global_args
        .stop_after
        .start_stop_after_duration_thread(cancellation_token.clone())?;

    // Initialize logging
    logging_init::init_logging(&cli.global_args, cancellation_token.clone())?;

    let command_name = cli.command.name()?;
    tracing::debug!(command = command_name, "Dispatching command");
    let result = cancellation_token.with_scope(|cancellation| -> eyre::Result<()> {
        let _span = tracing::info_span!("command", command = command_name).entered();
        let requested_output_format = cli.global_args.output_format;
        let started = Instant::now();
        let output = cli.invoke(&app_home, &cache_home, &cancellation)?;
        let threshold = output.elapsed_duration_warn_threshold;
        let emitted = output.emit(requested_output_format, &cancellation);
        elapsed_duration::warn_if_exceeded(command_name, started.elapsed(), threshold);
        emitted?;
        if !drains_capture {
            cancellation.bail_if_cancelled()?;
        }
        Ok(())
    });
    tracing::debug!(
        command = command_name,
        success = result.is_ok(),
        "Command finished"
    );
    result
}

fn launch_gui_without_command(arguments: &[OsString]) -> bool {
    arguments.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_command_arguments_request_the_gui() {
        let arguments: Vec<OsString> = Vec::new();
        assert!(launch_gui_without_command(&arguments));
        assert!(!launch_gui_without_command(&[OsString::from("doctor")]));
    }
}
pub(crate) mod phone_runtime;
