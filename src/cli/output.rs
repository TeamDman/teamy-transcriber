use arbitrary::Arbitrary;
use eyre::Context;
use facet::Facet;
use facet_json::SerializeOptions;
use facet_pretty::ColorMode;
use facet_pretty::PrettyPrinter;
use std::io::IsTerminal;
use std::io::Write;
use std::io::{self};
use std::time::Duration;
use teamy_cancellation::CancellationScope;

mod output_writer;
use output_writer::OutputWriter;

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
    fn write_to(
        &self,
        format: OutputFormat,
        stdout_is_terminal: bool,
        writer: &mut dyn Write,
    ) -> eyre::Result<()>;

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
    pub fn emit(
        self,
        requested_format: Option<OutputFormat>,
        cancellation: &CancellationScope,
    ) -> eyre::Result<()> {
        self.emit_to(
            requested_format,
            io::stdout().is_terminal(),
            &mut io::stdout().lock(),
            cancellation,
        )
    }

    pub(crate) fn emit_to(
        self,
        requested_format: Option<OutputFormat>,
        stdout_is_terminal: bool,
        stdout: &mut dyn Write,
        cancellation: &CancellationScope,
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
            cancellation.bail_if_cancelled()?;
            output.before_emit()?;
            let mut writer = OutputWriter::new(stdout, cancellation);
            output.write_to(format, stdout_is_terminal, &mut writer)?;
            writer.finish()
        })();
        emitted.map_err(|error| output.output_error(error))?;
        output.after_emit()
    }
}

impl<T> CliOutputValue for FacetCliOutput<T>
where
    T: Facet<'static> + 'static,
{
    fn write_to(
        &self,
        format: OutputFormat,
        stdout_is_terminal: bool,
        writer: &mut dyn Write,
    ) -> eyre::Result<()> {
        write_facet(&self.value, format, stdout_is_terminal, writer)
    }
}

pub(crate) fn write_facet<T: Facet<'static>>(
    value: &T,
    format: OutputFormat,
    stdout_is_terminal: bool,
    writer: &mut dyn Write,
) -> eyre::Result<()> {
    match format {
        OutputFormat::Text => {
            let rendered = std::fmt::from_fn(|formatter| {
                PrettyPrinter::new()
                    .with_colors(if stdout_is_terminal {
                        ColorMode::Always
                    } else {
                        ColorMode::Never
                    })
                    .format_to(value, formatter)
            });
            write!(writer, "{rendered}").wrap_err("failed to write command output")
        }
        OutputFormat::Json => {
            let options = SerializeOptions::new().pretty();
            facet_json::to_writer_std_with_options(&mut *writer, value, &options)
                .wrap_err("failed to serialize command output as JSON")
        }
        OutputFormat::Csv => {
            let rendered = facet_csv::to_string(value)
                .wrap_err("failed to serialize command output as CSV")?;
            writer
                .write_all(rendered.as_bytes())
                .wrap_err("failed to write command output")
        }
    }
}

#[cfg(test)]
mod budget_tests;
