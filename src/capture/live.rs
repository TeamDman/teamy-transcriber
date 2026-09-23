//! Bounded microphone packets for live transcription; no inference on the audio callback.
use eyre::Result;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Duration;
use teamy_cancellation::CancellationToken;

const MAX_QUEUED_AUDIO_SECONDS: usize = 30;

fn enqueue_packet(
    sender: &mpsc::Sender<Vec<f32>>,
    queued_samples: &AtomicUsize,
    max_samples: usize,
    packet: Vec<f32>,
) -> bool {
    let len = packet.len();
    if queued_samples
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            current.checked_add(len).filter(|&next| next <= max_samples)
        })
        .is_err()
    {
        return false;
    }
    if sender.send(packet).is_err() {
        queued_samples.fetch_sub(len, Ordering::AcqRel);
        return false;
    }
    true
}

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
    // A packet-count limit gives very different buffering at different device
    // callback sizes. Bound actual audio time instead, so short disk/VAD stalls
    // do not drop microphone packets.
    let max_samples = usize::try_from(rate)?
        .checked_mul(MAX_QUEUED_AUDIO_SECONDS)
        .ok_or_else(|| eyre::eyre!("microphone sample rate is too large"))?;
    let (sender, receiver) = mpsc::channel();
    let queued_samples = Arc::new(AtomicUsize::new(0));
    let overflow = Arc::new(AtomicBool::new(false));
    let stream_error = Arc::new(Mutex::new(None::<String>));
    macro_rules! stream {
        ($sample:ty) => {{
            let sender = sender.clone();
            let queued_samples = Arc::clone(&queued_samples);
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
                    if !enqueue_packet(&sender, &queued_samples, max_samples, mono) {
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
                "microphone audio queue exceeded 30 seconds; capture stopped to avoid silently losing audio"
            );
            if let Some(error) = stream_error
                .lock()
                .map_err(|error| eyre::eyre!("microphone error lock poisoned: {error}"))?
                .as_ref()
            {
                eyre::bail!("microphone stream failed: {error}");
            }
            match receiver.recv_timeout(Duration::from_millis(10)) {
                Ok(samples) => {
                    queued_samples.fetch_sub(samples.len(), Ordering::AcqRel);
                    on_samples(&samples, rate)?;
                }
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
        queued_samples.fetch_sub(samples.len(), Ordering::AcqRel);
        on_samples(&samples, rate)?;
    }
    result?;
    ensure!(
        !overflow.load(Ordering::Relaxed),
        "microphone audio queue exceeded 30 seconds"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_callback_packets_are_bounded_by_audio_duration() {
        let (sender, receiver) = mpsc::channel();
        let queued_samples = AtomicUsize::new(0);
        // A 1000 Hz device with a one-second budget can queue 100 ten-sample
        // callbacks. The old 64-packet limit would lose audio at callback 65.
        for _ in 0..100 {
            assert!(enqueue_packet(&sender, &queued_samples, 1000, vec![0.; 10]));
        }
        assert_eq!(queued_samples.load(Ordering::Acquire), 1000);
        assert!(!enqueue_packet(&sender, &queued_samples, 1000, vec![0.; 1]));
        let packet = receiver.recv().unwrap();
        queued_samples.fetch_sub(packet.len(), Ordering::AcqRel);
        assert!(enqueue_packet(&sender, &queued_samples, 1000, vec![0.; 10]));
    }
}
