use arbitrary::Arbitrary;
use eyre::Context;
use facet::Facet;
use facet_pretty::ColorMode;
use facet_pretty::PrettyPrinter;
use std::io::IsTerminal;
use std::io::Write;
use std::io::{self};
use std::time::Duration;
use teamy_cancellation::CancellationScope;

#[derive(Arbitrary, Facet, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[facet(rename_all = "kebab-case")]
#[repr(u8)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    Csv,
}

pub struct CliOutput {
    value: Option<Box<dyn CliOutputValue>>,
    /// Execution metadata; never serialized into command results.
    pub elapsed_duration_warn_threshold: Option<Duration>,
}

impl core::fmt::Debug for CliOutput {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CliOutput")
            .field("has_value", &self.value.is_some())
            .field(
                "elapsed_duration_warn_threshold",
                &self.elapsed_duration_warn_threshold,
            )
            .finish()
    }
}

pub(crate) trait CliOutputValue {
    fn render(
        &self,
        format: OutputFormat,
        stdout_is_terminal: bool,
    ) -> eyre::Result<Option<String>>;

    fn default_format(&self) -> Option<OutputFormat> {
        None
    }
    fn before_emit(&self) -> eyre::Result<()> {
        Ok(())
    }
    fn after_emit(&self) -> eyre::Result<()> {
        Ok(())
    }
    fn output_error(&self, error: eyre::Report) -> eyre::Report {
        error
    }
}

struct FacetCliOutput<T> {
    value: T,
}

impl CliOutput {
    pub(crate) fn custom(
        value: impl CliOutputValue + 'static,
        elapsed_duration_warn_threshold: Option<Duration>,
    ) -> Self {
        Self {
            value: Some(Box::new(value)),
            elapsed_duration_warn_threshold,
        }
    }

    /// Check cancellation while preserving a command's recovery information.
    /// # Errors
    /// Returns a contextual cancellation error.
    pub fn check_cancellation(&self, cancellation: &CancellationScope) -> eyre::Result<()> {
        cancellation.bail_if_cancelled().map_err(|error| {
            self.value.as_ref().map_or_else(
                || eyre::eyre!("{error:#}"),
                |output| output.output_error(eyre::eyre!("{error:#}")),
            )
        })
    }
    #[must_use]
    pub const fn none(elapsed_duration_warn_threshold: Option<Duration>) -> Self {
        Self {
            value: None,
            elapsed_duration_warn_threshold,
        }
    }

    #[must_use]
    pub fn facet<T>(value: T, elapsed_duration_warn_threshold: Option<Duration>) -> Self
    where
        T: Facet<'static> + 'static,
    {
        Self {
            value: Some(Box::new(FacetCliOutput { value })),
            elapsed_duration_warn_threshold,
        }
    }

    /// # Errors
    ///
    /// This function will return an error if the selected output format cannot be rendered
    /// or if the rendered output cannot be written to stdout.
    pub fn emit(self, requested_format: Option<OutputFormat>) -> eyre::Result<()> {
        self.emit_to(
            requested_format,
            io::stdout().is_terminal(),
            &mut io::stdout().lock(),
        )
    }

    pub(crate) fn emit_to(
        self,
        requested_format: Option<OutputFormat>,
        stdout_is_terminal: bool,
        stdout: &mut dyn Write,
    ) -> eyre::Result<()> {
        let Some(output) = self.value else {
            return Ok(());
        };

        let format = requested_format
            .or_else(|| output.default_format())
            .unwrap_or(if stdout_is_terminal {
                OutputFormat::Text
            } else {
                OutputFormat::Json
            });
        let emitted = (|| {
            output.before_emit()?;
            if let Some(rendered) = output.render(format, stdout_is_terminal)? {
                stdout
                    .write_all(rendered.as_bytes())
                    .wrap_err("failed to write command output")?;
                if !rendered.ends_with('\n') {
                    stdout.write_all(b"\n")?;
                }
                stdout.flush().wrap_err("failed to flush command output")?;
            }
            Ok(())
        })();
        emitted.map_err(|error| output.output_error(error))?;
        output.after_emit()
    }
}

impl<T> CliOutputValue for FacetCliOutput<T>
where
    T: Facet<'static> + 'static,
{
    fn render(
        &self,
        format: OutputFormat,
        stdout_is_terminal: bool,
    ) -> eyre::Result<Option<String>> {
        let rendered = match format {
            OutputFormat::Text => PrettyPrinter::new()
                .with_colors(if stdout_is_terminal {
                    ColorMode::Always
                } else {
                    ColorMode::Never
                })
                .format(&self.value),
            OutputFormat::Json => facet_json::to_string_pretty(&self.value)
                .wrap_err("failed to serialize command output as JSON")?,
            OutputFormat::Csv => facet_csv::to_string(&self.value)
                .wrap_err("failed to serialize command output as CSV")?,
        };
        Ok(Some(rendered))
    }
}

#[cfg(test)]
mod budget_tests;
