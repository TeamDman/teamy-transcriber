//! Persist bounded audio windows, transcribe concurrently, and drain on capture stop.
use crate::domain::AppState;
use crate::domain::AssetKind;
use crate::domain::Command;
use crate::domain::RecordingId;
use crate::domain::SourceAsset;
use crate::media::AudioProfile;
use crate::storage::RecordingStore;
use crate::workflow::TranscriptionOptions;
use crate::workflow::TranscriptionSession;
use crate::workflow::prepare_recording;
use eyre::Context;
use eyre::Result;
use eyre::ensure;
use std::io::Write;
#[cfg(test)]
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Sender;
use std::sync::mpsc::channel;

mod endpoint;

type SampleSink<'a> = dyn FnMut(&[f32], u32) -> Result<()> + 'a;

pub(crate) fn transcribe<C>(
    store: &RecordingStore,
    model_dir: &std::path::Path,
    chunk_ms: u64,
    capture: C,
    output: &mut dyn Write,
) -> Result<()>
where
    C: FnOnce(&AtomicBool, &mut SampleSink<'_>) -> Result<()> + Send,
{
    let mut session = TranscriptionSession::default();
    let detector = endpoint::Detector::load(model_dir)?;
    pipeline(store, chunk_ms, detector, capture, |id| {
        let result: Result<()> = (|| {
            prepare_recording(store, id)?;
            let report = session.transcribe(store, id, TranscriptionOptions {
                model_dir: model_dir.to_path_buf(), max_decode_tokens: 448,
                chunk_duration_us: None, profile: AudioProfile::Original,
            }, None, None)?;
            for chunk in report.chunks {
                writeln!(output, "{}", chunk.text.trim())?;
                output.flush()?;
            }
            Ok(())
        })();
        result.wrap_err_with(|| recovery(id))
    }).wrap_err("live microphone transcription stopped; captured windows remain available through recording list")
}

pub(crate) fn transcribe_phones<C>(
    store: &RecordingStore,
    vad_model_dir: &std::path::Path,
    phone_model_dir: &std::path::Path,
    chunk_ms: u64,
    capture: C,
    output: &mut dyn Write,
) -> Result<()>
where
    C: FnOnce(&AtomicBool, &mut SampleSink<'_>) -> Result<()> + Send,
{
    let model = teamy_whisper_native::phones::PhoneModel::load(phone_model_dir, 0)
        .map_err(|error| eyre::eyre!("{error:#}"))?;
    let detector = endpoint::Detector::load(vad_model_dir)?;
    pipeline(store, chunk_ms, detector, capture, |id| {
        let run = (|| -> Result<()> {
            let audio = prepare_recording(store, id)?;
            let chunks = crate::phone_runtime::recognize_wav(&model, &audio.normalized_path, None)?;
            std::fs::write(
                store.recording_dir(id).join("phones.json"),
                facet_json::to_string_pretty(&chunks)?,
            )?;
            for chunk in chunks {
                if !chunk.ipa.is_empty() {
                    writeln!(output, "{}", chunk.ipa)?;
                    output.flush()?;
                }
            }
            Ok(())
        })();
        run.wrap_err_with(|| format!("Phone recording {id} retained. Retry with teamy-transcriber phones <saved-source-wav> --model-dir <phone-model-folder>."))
    })
}
fn recovery(id: RecordingId) -> String {
    format!(
        "Recording {id} was retained. Retry: teamy-transcriber transcribe --resume {id} --keep-recording"
    )
}

fn pipeline<C, F>(
    store: &RecordingStore,
    chunk_ms: u64,
    detector: Option<endpoint::Detector>,
    capture: C,
    mut consume: F,
) -> Result<()>
where
    C: FnOnce(&AtomicBool, &mut SampleSink<'_>) -> Result<()> + Send,
    F: FnMut(RecordingId) -> Result<()>,
{
    ensure!(
        (500..=30000).contains(&chunk_ms),
        "audio windows must be 500..30000 ms"
    );
    let abort = AtomicBool::new(false);
    // Only recording IDs enter this queue; each audio window is already on
    // disk. Let inference/output lag temporarily and drain every saved window
    // after capture stops instead of failing when eight IDs are pending.
    let (sender, receiver) = channel();
    std::thread::scope(|scope| {
        let producer = scope.spawn(|| {
            let mut recorder = WindowRecorder {
                store,
                chunk_ms,
                sender,
                detector,
                rate: None,
                samples: Vec::new(),
            };
            let captured = capture(&abort, &mut |samples, rate| recorder.push(samples, rate));
            // Flush already accepted audio even when capture reports an error.
            let flushed = recorder.finish();
            captured.and(flushed)
        });
        let mut consumed = Ok(());
        for id in &receiver {
            if let Err(error) = consume(id) {
                abort.store(true, Ordering::Relaxed);
                consumed = Err(error);
                break;
            }
        }
        drop(receiver);
        let captured = producer
            .join()
            .map_err(|_panic| eyre::eyre!("microphone capture thread panicked"))?;
        consumed?;
        captured
    })
}

struct WindowRecorder<'a> {
    store: &'a RecordingStore,
    chunk_ms: u64,
    sender: Sender<RecordingId>,
    rate: Option<u32>,
    samples: Vec<f32>,
    detector: Option<endpoint::Detector>,
}

impl WindowRecorder<'_> {
    fn push(&mut self, samples: &[f32], rate: u32) -> Result<()> {
        ensure!(
            rate > 0 && samples.iter().all(|sample| sample.is_finite()),
            "invalid microphone PCM"
        );
        ensure!(
            *self.rate.get_or_insert(rate) == rate,
            "microphone sample rate changed during capture"
        );
        let size = usize::try_from(u64::from(rate) * self.chunk_ms / 1000)?;
        ensure!(size > 0, "audio window is empty");
        for &sample in samples {
            self.samples.push(sample);
            let endpoint = match self.detector.as_mut() {
                Some(detector) => detector.push(sample, rate)?,
                None => false,
            };
            if endpoint || self.samples.len() == size {
                self.finish()?;
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        if self.samples.is_empty() {
            return Ok(());
        }
        if let Some(detector) = &mut self.detector {
            detector.boundary();
        }
        let samples = std::mem::take(&mut self.samples);
        let id = RecordingId::new();
        let path = self.store.recording_dir(id).join("source/microphone.wav");
        let mut state = AppState::new();
        let saved: Result<()> = (|| {
            self.store.apply_command(
                &mut state,
                Command::CreateRecording {
                    recording_id: id,
                    source: SourceAsset::new(AssetKind::MicrophoneRecording, &path)?,
                },
            )?;
            self.store
                .apply_command(&mut state, Command::StartRecording { recording_id: id })?;
            std::fs::create_dir_all(
                path.parent()
                    .ok_or_else(|| eyre::eyre!("missing capture directory"))?,
            )?;
            let mut writer = hound::WavWriter::create(
                &path,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: self
                        .rate
                        .ok_or_else(|| eyre::eyre!("missing capture sample rate"))?,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )?;
            for sample in samples {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
            self.store
                .apply_command(&mut state, Command::CompleteRecording { recording_id: id })?;
            Ok(())
        })();
        saved.wrap_err_with(|| recovery(id))?;
        self.sender.send(id).wrap_err_with(|| {
            format!(
                "transcription consumer stopped before accepting saved audio; {}",
                recovery(id)
            )
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;
    use teamy_cancellation::CancellationToken;

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!("live-microphone-{}", uuid::Uuid::new_v4())))
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn consumes_before_stop_and_drains_final_partial_window_in_order() -> Result<()> {
        let fixture = Fixture::new();
        let store = RecordingStore::new(&fixture.0);
        let token = &CancellationToken::new();
        let (done, visible) = mpsc::channel();
        let stopped = &AtomicBool::new(false);
        let mut lengths = Vec::new();
        pipeline(
            &store,
            500,
            None,
            move |_, sink| {
                sink(&vec![0.; 1200], 1000)?;
                visible.recv_timeout(Duration::from_secs(5))?;
                ensure!(token.is_cancelled());
                stopped.store(true, Ordering::Relaxed);
                Ok(())
            },
            |id| {
                let recording = store.load_recording(id)?;
                lengths.push(hound::WavReader::open(recording.source.path)?.duration());
                if lengths.len() == 1 {
                    ensure!(
                        !stopped.load(Ordering::Relaxed),
                        "output waited for capture to stop"
                    );
                    token.request_cancel("Ctrl+C");
                    done.send(())?;
                }
                Ok(())
            },
        )?;
        assert_eq!(lengths, [500, 500, 200]);
        assert_eq!(store.list_recordings()?.len(), 3);
        Ok(())
    }

    #[test]
    fn consumer_failure_stops_and_joins_capture_and_retains_tail() {
        let fixture = Fixture::new();
        let store = RecordingStore::new(&fixture.0);
        let joined = AtomicBool::new(false);
        let error = pipeline(
            &store,
            500,
            None,
            |abort, sink| {
                sink(&vec![0.; 700], 1000)?;
                let deadline = std::time::Instant::now() + Duration::from_secs(5);
                while !abort.load(Ordering::Relaxed) && std::time::Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(1));
                }
                ensure!(abort.load(Ordering::Relaxed));
                joined.store(true, Ordering::Relaxed);
                Ok(())
            },
            |_| Err(eyre::eyre!("broken stdout")),
        )
        .unwrap_err();
        assert!(error.to_string().contains("broken stdout"));
        assert!(joined.load(Ordering::Relaxed));
        assert_eq!(store.list_recordings().unwrap().len(), 2);
    }

    #[test]
    fn slow_consumer_drains_all_saved_windows() -> Result<()> {
        let fixture = Fixture::new();
        let store = RecordingStore::new(&fixture.0);
        let (done, blocked) = mpsc::channel();
        let mut waited = false;
        let mut consumed = 0;
        pipeline(
            &store,
            500,
            None,
            |_, sink| {
                sink(&vec![0.; 10000], 1000)?;
                done.send(())?;
                Ok(())
            },
            |_| {
                if !waited {
                    blocked.recv_timeout(Duration::from_secs(5))?;
                    waited = true;
                }
                consumed += 1;
                Ok(())
            },
        )?;
        assert_eq!(consumed, 20);
        assert_eq!(store.list_recordings()?.len(), 20);
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA, TEST_MODEL and a speech TEST_WAV"]
    fn speech_is_flushed_before_stop_and_tail_is_drained() -> Result<()> {
        struct Output {
            bytes: Vec<u8>,
            visible: mpsc::Sender<()>,
            sent: bool,
        }
        impl Write for Output {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.bytes.extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                if !self.sent {
                    self.visible.send(()).map_err(std::io::Error::other)?;
                    self.sent = true;
                }
                Ok(())
            }
        }
        let fixture = Fixture::new();
        let store = RecordingStore::new(&fixture.0);
        let model = PathBuf::from(std::env::var("TEST_MODEL")?);
        let mut reader = hound::WavReader::open(std::env::var("TEST_WAV")?)?;
        let spec = reader.spec();
        ensure!(spec.channels == 1);
        let count = spec.sample_rate as usize * 3;
        let mut samples: Vec<f32> = if spec.sample_format == hound::SampleFormat::Float {
            reader
                .samples::<f32>()
                .take(count)
                .collect::<std::result::Result<_, _>>()?
        } else {
            ensure!(spec.bits_per_sample == 16);
            reader
                .samples::<i16>()
                .take(count)
                .map(|sample| sample.map(|value| f32::from(value) / 32768.0))
                .collect::<std::result::Result<_, _>>()?
        };
        ensure!(samples.len() == count);
        samples.extend(std::iter::repeat_n(0., spec.sample_rate as usize));
        let (visible, wait) = mpsc::channel();
        let mut output = Output {
            bytes: Vec::new(),
            visible,
            sent: false,
        };
        let token = CancellationToken::new();
        transcribe(
            &store,
            &model,
            5000,
            move |_, sink| {
                sink(&samples, spec.sample_rate)?;
                // Capture cannot finish until actual inference has printed text.
                wait.recv_timeout(Duration::from_mins(2))?;
                sink(&samples[..spec.sample_rate as usize / 2], spec.sample_rate)?;
                token.request_cancel("simulated Ctrl+C after live output");
                ensure!(token.is_cancelled());
                Ok(())
            },
            &mut output,
        )?;
        ensure!(!std::str::from_utf8(&output.bytes)?.trim().is_empty());
        assert!(store.list_recordings()?.len() >= 2);
        Ok(())
    }
    #[test]
    #[ignore = "requires CUDA, PHONE_TEST_MODEL, TEST_MODEL for VAD and TEST_WAV"]
    fn phones_are_flushed_before_stop_and_tail_is_drained() -> Result<()> {
        struct Output {
            bytes: Vec<u8>,
            visible: mpsc::Sender<()>,
            sent: bool,
        }
        impl Write for Output {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.bytes.extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                if !self.sent {
                    self.visible.send(()).map_err(std::io::Error::other)?;
                    self.sent = true;
                }
                Ok(())
            }
        }
        let fixture = Fixture::new();
        let store = RecordingStore::new(&fixture.0);
        let model = PathBuf::from(std::env::var("PHONE_TEST_MODEL")?);
        let vad = PathBuf::from(std::env::var("TEST_MODEL")?);
        let mut reader = hound::WavReader::open(std::env::var("TEST_WAV")?)?;
        let spec = reader.spec();
        ensure!(spec.channels == 1);
        let count = spec.sample_rate as usize * 3;
        let mut samples: Vec<f32> = if spec.sample_format == hound::SampleFormat::Float {
            reader
                .samples::<f32>()
                .take(count)
                .collect::<std::result::Result<_, _>>()?
        } else {
            ensure!(spec.bits_per_sample == 16);
            reader
                .samples::<i16>()
                .take(count)
                .map(|sample| sample.map(|value| f32::from(value) / 32768.0))
                .collect::<std::result::Result<_, _>>()?
        };
        ensure!(samples.len() == count);
        samples.extend(std::iter::repeat_n(0., spec.sample_rate as usize));
        let (visible, wait) = mpsc::channel();
        let mut output = Output {
            bytes: Vec::new(),
            visible,
            sent: false,
        };
        let token = CancellationToken::new();
        transcribe_phones(
            &store,
            &vad,
            &model,
            5000,
            move |_, sink| {
                sink(&samples, spec.sample_rate)?;
                // Capture cannot finish until actual inference has printed text.
                wait.recv_timeout(Duration::from_mins(2))?;
                sink(&samples[..spec.sample_rate as usize / 2], spec.sample_rate)?;
                token.request_cancel("simulated Ctrl+C after live output");
                ensure!(token.is_cancelled());
                Ok(())
            },
            &mut output,
        )?;
        ensure!(!std::str::from_utf8(&output.bytes)?.trim().is_empty());
        assert!(store.list_recordings()?.len() >= 2);
        Ok(())
    }
}
