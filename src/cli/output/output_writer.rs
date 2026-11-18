use eyre::Context;
use std::io;
use std::io::Write;
use teamy_cancellation::CancellableWriter;
use teamy_cancellation::CancellationScope;

pub(super) struct OutputWriter<'a> {
    writer: CancellableWriter<'a, dyn Write + 'a>,
    last_byte: Option<u8>,
}

impl<'a> OutputWriter<'a> {
    pub(super) fn new(writer: &'a mut dyn Write, cancellation: &'a CancellationScope) -> Self {
        Self {
            writer: CancellableWriter::new(writer, cancellation),
            last_byte: None,
        }
    }

    pub(super) fn finish(&mut self) -> eyre::Result<()> {
        if self.last_byte != Some(b'\n') {
            self.write_all(b"\n")
                .wrap_err("failed to terminate command output")?;
        }
        self.flush().wrap_err("failed to flush command output")
    }
}

impl Write for OutputWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let written = self.writer.write(buf)?;
        if written != 0 {
            self.last_byte = Some(buf[written - 1]);
        }
        Ok(written)
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.writer.write_all(buf)?;
        if let Some(last_byte) = buf.last() {
            self.last_byte = Some(*last_byte);
        }
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}
