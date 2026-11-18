use super::TranscriptStream;
use super::TranscriptText;
use eyre::Context;
use eyre::Result;
use std::io;
use std::io::Write;
use std::sync::Mutex;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::RecvTimeoutError;
use std::sync::mpsc::Sender;
use std::sync::mpsc::channel;
use std::thread;
use std::time::Duration;
use std::time::Instant;
use teamy_cancellation::CancellationScope;

#[derive(Clone, Copy)]
enum ChunkOutput {
    Idle,
    Writing,
    FinishedAt(Instant),
}

struct OutputState<'a> {
    writer: &'a mut (dyn Write + Send),
    chunk: ChunkOutput,
}

pub(super) struct TranscriptOutput<'state, 'writer> {
    state: &'state Mutex<OutputState<'writer>>,
    wake: Option<Sender<()>>,
}

impl TranscriptOutput<'_, '_> {
    pub(super) fn run(
        writer: &mut (dyn Write + Send),
        inactivity: Option<Duration>,
        cancellation: &CancellationScope,
        run: impl FnOnce(&TranscriptOutput<'_, '_>) -> Result<()>,
    ) -> Result<()> {
        let state = Mutex::new(OutputState {
            writer,
            chunk: ChunkOutput::Idle,
        });
        let Some(inactivity) = inactivity else {
            return run(&TranscriptOutput {
                state: &state,
                wake: None,
            });
        };

        thread::scope(|scope| {
            let (wake, updates) = channel();
            let state = &state;
            let timer = scope
                .spawn(move || write_inactivity_markers(state, &updates, inactivity, cancellation));
            let result = run(&TranscriptOutput {
                state,
                wake: Some(wake),
            });
            let marker = timer
                .join()
                .map_err(|_panic| eyre::eyre!("inactivity timer panicked"))?;
            result?;
            marker.wrap_err("could not write microphone inactivity marker")
        })
    }

    pub(super) fn write_line(
        &self,
        stream: TranscriptStream,
        text: TranscriptText<'_>,
    ) -> io::Result<()> {
        let mut state = self.state.lock().unwrap();
        emit(state.writer, stream, text)?;
        state.chunk = ChunkOutput::Writing;
        Ok(())
    }

    pub(super) fn finish_chunk(&self) {
        let Some(wake) = &self.wake else {
            return;
        };
        let mut state = self.state.lock().unwrap();
        if matches!(state.chunk, ChunkOutput::Writing) {
            state.chunk = ChunkOutput::FinishedAt(Instant::now());
            let _ = wake.send(());
        }
    }
}

fn write_inactivity_markers(
    output: &Mutex<OutputState<'_>>,
    updates: &Receiver<()>,
    inactivity: Duration,
    cancellation: &CancellationScope,
) -> io::Result<()> {
    loop {
        let chunk = output.lock().unwrap().chunk;
        let ChunkOutput::FinishedAt(last_output) = chunk else {
            if updates.recv().is_err() {
                return Ok(());
            }
            continue;
        };

        match updates.recv_timeout(inactivity.saturating_sub(last_output.elapsed())) {
            Ok(()) => {}
            Err(RecvTimeoutError::Disconnected) => return Ok(()),
            Err(RecvTimeoutError::Timeout) => {
                let mut state = output.lock().unwrap();
                if matches!(state.chunk, ChunkOutput::FinishedAt(at) if at.elapsed() >= inactivity)
                {
                    if let Err(error) = emit(
                        state.writer,
                        TranscriptStream::Unnumbered,
                        TranscriptText::new("..."),
                    ) {
                        cancellation.request_cancel(error.to_string());
                        return Err(error);
                    }
                    state.chunk = ChunkOutput::Idle;
                }
            }
        }
    }
}

fn emit(
    output: &mut dyn Write,
    stream: TranscriptStream,
    text: TranscriptText<'_>,
) -> io::Result<()> {
    let prefix = stream.prefix();
    let text = text.as_str();
    if prefix.is_empty() || text.is_empty() {
        writeln!(output, "{prefix}{text}")?;
        return output.flush();
    }
    for line in text.lines() {
        writeln!(output, "{prefix}{line}")?;
        output.flush()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use teamy_cancellation::CancellationToken;

    struct FlushedOutput {
        bytes: Vec<u8>,
        lines: Sender<Vec<u8>>,
    }

    impl Write for FlushedOutput {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.bytes.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            self.lines
                .send(std::mem::take(&mut self.bytes))
                .map_err(io::Error::other)
        }
    }

    #[test]
    fn prints_once_after_both_lines_even_when_the_second_decode_is_slow() -> Result<()> {
        let (lines, received) = channel();
        let mut output = FlushedOutput {
            bytes: Vec::new(),
            lines,
        };
        let gap = Duration::from_millis(20);
        let wait = Duration::from_secs(2);
        CancellationToken::new().with_scope(|cancellation| {
            TranscriptOutput::run(&mut output, Some(gap), &cancellation, |output| {
                assert_eq!(
                    received.recv_timeout(gap * 2),
                    Err(RecvTimeoutError::Timeout)
                );

                output.write_line(TranscriptStream::Phonemes, TranscriptText::new("fəʊn"))?;
                assert_eq!(received.recv_timeout(wait)?, "1| fəʊn\n".as_bytes());
                assert_eq!(
                    received.recv_timeout(gap * 2),
                    Err(RecvTimeoutError::Timeout)
                );

                output.write_line(TranscriptStream::Words, TranscriptText::new("phone"))?;
                output.finish_chunk();
                assert_eq!(received.recv_timeout(wait)?, b"2| phone\n");
                assert_eq!(received.recv_timeout(wait)?, b"...\n");
                assert_eq!(
                    received.recv_timeout(gap * 2),
                    Err(RecvTimeoutError::Timeout)
                );

                output.write_line(TranscriptStream::Unnumbered, TranscriptText::new("nɛkst"))?;
                output.finish_chunk();
                assert_eq!(received.recv_timeout(wait)?, "nɛkst\n".as_bytes());
                assert_eq!(received.recv_timeout(wait)?, b"...\n");
                Ok(())
            })
        })
    }

    #[test]
    fn combined_output_keeps_flushing_after_cancellation() -> Result<()> {
        let mut bytes = Vec::new();
        CancellationToken::new().with_scope(|cancellation| {
            TranscriptOutput::run(&mut bytes, None, &cancellation, |output| {
                output.write_line(TranscriptStream::Phonemes, TranscriptText::new("wʌn"))?;
                cancellation.request_cancel("simulated Ctrl+C");
                output.write_line(TranscriptStream::Words, TranscriptText::new("one\r\ntwo"))?;
                output.write_line(
                    TranscriptStream::Unnumbered,
                    TranscriptText::new("unprefixed"),
                )?;
                Ok(())
            })
        })?;
        assert_eq!(
            std::str::from_utf8(&bytes)?,
            "1| wʌn\n2| one\n2| two\nunprefixed\n"
        );
        Ok(())
    }
}
