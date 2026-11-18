//! Queue microphone packets for live transcription without running inference on the audio callback.
use eyre::Result;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::Duration;
use teamy_cancellation::CancellationScope;

#[cfg(windows)]
const QUEUED_AUDIO_WARNING_SECONDS: usize = 30;
#[cfg(windows)]
const QUEUED_AUDIO_WARNING_INTERVAL: Duration = Duration::from_secs(2);

#[cfg(any(windows, test))]
fn enqueue_packet(sender: &mpsc::Sender<Vec<f32>>, queued_samples: &AtomicUsize, packet: Vec<f32>) {
    let len = packet.len();
    queued_samples.fetch_add(len, Ordering::AcqRel);
    if sender.send(packet).is_err() {
        queued_samples.fetch_sub(len, Ordering::AcqRel);
    }
}

#[cfg(windows)]
#[expect(
    clippy::too_many_lines,
    reason = "sample format dispatch keeps all CPAL callback ownership in one scope"
)]
pub(crate) fn capture(
    endpoint_id: Option<&str>,
    duration: Option<Duration>,
    cancellation: &CancellationScope,
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
    let samples_per_second = usize::try_from(rate)?;
    let warning_threshold = samples_per_second.saturating_mul(QUEUED_AUDIO_WARNING_SECONDS);
    let (sender, receiver) = mpsc::channel();
    let queued_samples = Arc::new(AtomicUsize::new(0));
    let stream_error = Arc::new(Mutex::new(None::<String>));
    macro_rules! stream {
        ($sample:ty) => {{
            let sender = sender.clone();
            let queued_samples = Arc::clone(&queued_samples);
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
                    enqueue_packet(&sender, &queued_samples, mono);
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
    tracing::info!("Recording microphone. Ctrl+C stops capture and drains transcription.");
    std::thread::scope(|scope| {
        let (stop_monitor, monitor) = mpsc::channel::<()>();
        let samples = &queued_samples;
        // The packet consumer can stall in on_samples; monitor it independently.
        scope.spawn(move || {
            while let Err(mpsc::RecvTimeoutError::Timeout) =
                monitor.recv_timeout(QUEUED_AUDIO_WARNING_INTERVAL)
            {
                let buffered = samples.load(Ordering::Acquire);
                if buffered > warning_threshold {
                    tracing::warn!(
                        buffered_seconds = buffered / samples_per_second,
                        queued_samples = buffered,
                        "Microphone audio backlog is over 30 seconds; capture is continuing."
                    );
                }
            }
        });
        let result = (|| {
            while !cancellation.is_cancelled()
                && deadline.is_none_or(|deadline| Instant::now() < deadline)
            {
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
        drop(stop_monitor);
        result?;
        if let Some(error) = stream_error
            .lock()
            .map_err(|error| eyre::eyre!("microphone error lock poisoned: {error}"))?
            .as_ref()
        {
            eyre::bail!("microphone stream failed: {error}");
        }
        Ok(())
    })
}

#[cfg(not(windows))]
pub(crate) fn capture(
    _endpoint_id: Option<&str>,
    _duration: Option<Duration>,
    _cancellation: &CancellationScope,
    _on_samples: &mut dyn FnMut(&[f32], u32) -> Result<()>,
) -> Result<()> {
    eyre::bail!("live microphone capture is currently implemented for Windows only")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_and_drains_packets_beyond_thirty_seconds() {
        let (sender, receiver) = mpsc::channel();
        let queued_samples = AtomicUsize::new(0);
        // 1000 samples per second and 10 samples per callback.
        for _ in 0..3_001 {
            enqueue_packet(&sender, &queued_samples, vec![0.; 10]);
        }
        drop(sender);
        assert_eq!(queued_samples.load(Ordering::Acquire), 30_010);
        let drained_samples: usize = receiver.into_iter().map(|packet| packet.len()).sum();
        assert_eq!(drained_samples, 30_010);
    }
}
