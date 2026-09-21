//! Bounded microphone packets for live transcription; no inference on the audio callback.
use eyre::Result;
use std::sync::atomic::AtomicBool;
use std::time::Duration;
use teamy_cancellation::CancellationToken;

#[cfg(windows)]
#[expect(
    clippy::too_many_lines,
    reason = "sample format dispatch keeps all CPAL callback ownership in one scope"
)]
pub(crate) fn capture(
    endpoint_id: Option<&str>,
    duration: Option<Duration>,
    token: &CancellationToken,
    abort: &AtomicBool,
    on_samples: &mut dyn FnMut(&[f32], u32) -> Result<()>,
) -> Result<()> {
    use cpal::traits::DeviceTrait;
    use cpal::traits::HostTrait;
    use cpal::traits::StreamTrait;
    use eyre::Context;
    use eyre::ensure;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::Ordering;
    use std::sync::mpsc;
    use std::time::Instant;

    let host = cpal::default_host();
    let device = if let Some(id) = endpoint_id {
        let name = super::list_audio_input_devices()?
            .into_iter()
            .find(|device| device.id == id)
            .ok_or_else(|| eyre::eyre!("selected microphone is unavailable"))?
            .name;
        let mut matches = host
            .input_devices()?
            .filter(|device| device.name().ok().as_ref() == Some(&name));
        let device = matches
            .next()
            .ok_or_else(|| eyre::eyre!("selected microphone is unavailable to CPAL"))?;
        ensure!(
            matches.next().is_none(),
            "multiple microphones share this friendly name"
        );
        device
    } else {
        host.default_input_device()
            .ok_or_else(|| eyre::eyre!("no default microphone is available"))?
    };
    let supported = device.default_input_config()?;
    let config = supported.config();
    ensure!(
        config.channels > 0 && config.sample_rate.0 > 0,
        "invalid microphone format"
    );
    let channels = usize::from(config.channels);
    let rate = config.sample_rate.0;
    let (sender, receiver) = mpsc::sync_channel(64);
    let overflow = Arc::new(AtomicBool::new(false));
    let stream_error = Arc::new(Mutex::new(None::<String>));
    macro_rules! stream {
        ($sample:ty) => {{
            let sender = sender.clone();
            let overflow = Arc::clone(&overflow);
            let stream_error = Arc::clone(&stream_error);
            device.build_input_stream(
                &config,
                move |data: &[$sample], _| {
                    let mono: Vec<f32> = data
                        .chunks_exact(channels)
                        .map(|frame| {
                            frame
                                .iter()
                                .copied()
                                .map(<f32 as cpal::FromSample<$sample>>::from_sample_)
                                .sum::<f32>()
                                / f32::from(config.channels)
                        })
                        .collect();
                    if sender.try_send(mono).is_err() {
                        overflow.store(true, Ordering::Relaxed);
                    }
                },
                move |error| {
                    if let Ok(mut slot) = stream_error.lock() {
                        *slot = Some(error.to_string());
                    }
                },
                None,
            )?
        }};
    }
    let stream = match supported.sample_format() {
        cpal::SampleFormat::I8 => stream!(i8),
        cpal::SampleFormat::I16 => stream!(i16),
        cpal::SampleFormat::I32 => stream!(i32),
        cpal::SampleFormat::I64 => stream!(i64),
        cpal::SampleFormat::U8 => stream!(u8),
        cpal::SampleFormat::U16 => stream!(u16),
        cpal::SampleFormat::U32 => stream!(u32),
        cpal::SampleFormat::U64 => stream!(u64),
        cpal::SampleFormat::F32 => stream!(f32),
        cpal::SampleFormat::F64 => stream!(f64),
        format => eyre::bail!("unsupported microphone sample format: {format}"),
    };
    drop(sender);
    let deadline = duration
        .map(|duration| {
            Instant::now()
                .checked_add(duration)
                .ok_or_else(|| eyre::eyre!("capture duration is too large"))
        })
        .transpose()?;
    stream.play().wrap_err("starting microphone capture")?;
    eprintln!("Recording microphone. Ctrl+C stops capture and drains transcription.");
    let result = (|| {
        while !token.is_cancelled()
            && !abort.load(Ordering::Relaxed)
            && deadline.is_none_or(|deadline| Instant::now() < deadline)
        {
            ensure!(
                !overflow.load(Ordering::Relaxed),
                "microphone packet queue overflowed; capture stopped to avoid silently losing audio"
            );
            if let Some(error) = stream_error
                .lock()
                .map_err(|error| eyre::eyre!("microphone error lock poisoned: {error}"))?
                .as_ref()
            {
                eyre::bail!("microphone stream failed: {error}");
            }
            match receiver.recv_timeout(Duration::from_millis(10)) {
                Ok(samples) => on_samples(&samples, rate)?,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    eyre::bail!("microphone stream disconnected")
                }
            }
        }
        Ok(())
    })();
    drop(stream);
    // Packets already captured before stop belong to the final window.
    for samples in receiver.try_iter() {
        on_samples(&samples, rate)?;
    }
    result?;
    ensure!(
        !overflow.load(Ordering::Relaxed),
        "microphone packet queue overflowed"
    );
    if let Some(error) = stream_error
        .lock()
        .map_err(|error| eyre::eyre!("microphone error lock poisoned: {error}"))?
        .as_ref()
    {
        eyre::bail!("microphone stream failed: {error}");
    }
    Ok(())
}

#[cfg(not(windows))]
pub(crate) fn capture(
    _endpoint_id: Option<&str>,
    _duration: Option<Duration>,
    _token: &CancellationToken,
    _abort: &AtomicBool,
    _on_samples: &mut dyn FnMut(&[f32], u32) -> Result<()>,
) -> Result<()> {
    eyre::bail!("live microphone capture is currently implemented for Windows only")
}
