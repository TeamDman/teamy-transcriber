//! Persist bounded audio windows, transcribe concurrently, and drain on capture stop.
use crate::domain::AppState;
use crate::domain::AssetKind;
use crate::domain::Command;
use crate::domain::RecordingId;
use crate::domain::SourceAsset;
use crate::media::AudioProfile;
use crate::microphone_chunk_milliseconds::MicrophoneChunkMilliseconds;
use crate::paths::PhoneModelDir;
#[cfg(test)]
use crate::paths::PhoneModelDirSelection;
use crate::paths::WhisperModelDir;
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
use std::sync::mpsc::Sender;
use std::sync::mpsc::channel;
use teamy_cancellation::CancellationScope;
use teamy_whisper_native::CudaDeviceIndex;

mod endpoint;
mod transcript_output;
mod transcript_stream;
mod transcript_text;
mod transcription_options;

use transcript_output::TranscriptOutput;
use transcript_stream::TranscriptStream;
use transcript_text::TranscriptText;
pub(crate) use transcription_options::LiveTranscriptionOptions;

type SampleSink<'a> = dyn FnMut(&[f32], u32) -> Result<()> + 'a;

pub(crate) enum Recognition {
    Words,
    Phonemes(PhoneModelDir),
    WordsAndPhonemes(PhoneModelDir),
}

impl Recognition {
    pub(crate) fn includes_words(&self) -> bool {
        matches!(self, Self::Words | Self::WordsAndPhonemes(_))
    }

    pub(crate) fn includes_phonemes(&self) -> bool {
        self.phone_model_dir().is_some()
    }

    fn phone_model_dir(&self) -> Option<&PhoneModelDir> {
        match self {
            Self::Words => None,
            Self::Phonemes(path) | Self::WordsAndPhonemes(path) => Some(path),
        }
    }
}

pub(crate) fn transcribe<C>(
    store: &RecordingStore,
    model_dir: &WhisperModelDir,
    recognition: &Recognition,
    options: LiveTranscriptionOptions,
    cancellation: &CancellationScope,
    capture: C,
    output: &mut (dyn Write + Send),
) -> Result<()>
where
    C: FnOnce(&CancellationScope, &mut SampleSink<'_>) -> Result<()> + Send,
{
    let LiveTranscriptionOptions {
        cuda,
        chunk_ms,
        keep_recording,
        inactivity,
    } = options;
    let mut session = TranscriptionSession::default();
    let phone_model = recognition
        .phone_model_dir()
        .map(|path| path.load(CudaDeviceIndex::FIRST))
        .transpose()
        .map_err(|error| eyre::eyre!("{error:#}"))?;
    let detector = endpoint::Detector::load(model_dir)?;
    let (phoneme_stream, word_stream) =
        if recognition.includes_words() && recognition.includes_phonemes() {
            (TranscriptStream::Phonemes, TranscriptStream::Words)
        } else {
            (TranscriptStream::Unnumbered, TranscriptStream::Unnumbered)
        };
    // Capture stops on cancellation while recognition drains saved windows.
    TranscriptOutput::run(output, inactivity, cancellation, |output| {
        pipeline(store, chunk_ms, keep_recording, detector, cancellation, capture, |id| {
            let run = (|| -> Result<()> {
                let audio = prepare_recording(store, id)?;
                if let Some(model) = &phone_model {
                    let chunks = crate::phone_runtime::recognize_wav(model, &audio.normalized_path, None)?;
                    std::fs::write(
                        store.recording_dir(id).join("phones.json"),
                        facet_json::to_string_pretty(&chunks)?,
                    )?;
                    for chunk in chunks {
                        if !chunk.ipa.is_empty() {
                            output.write_line(phoneme_stream, TranscriptText::new(&chunk.ipa))?;
                        }
                    }
                }
                if recognition.includes_words() {
                    let report = session.transcribe(
                        store,
                        id,
                        TranscriptionOptions {
                            cuda,
                            model_dir: model_dir.to_path_buf(),
                            max_decode_tokens: 448,
                            chunk_duration_us: None,
                            profile: AudioProfile::Original,
                        },
                        None,
                        None,
                    )?;
                    for chunk in report.chunks {
                        let text = chunk.text.trim();
                        if inactivity.is_none() || !text.is_empty() {
                            output.write_line(word_stream, TranscriptText::new(text))?;
                        }
                    }
                }
                output.finish_chunk();
                Ok(())
            })();
            run.wrap_err_with(|| match recognition {
                Recognition::Words => recovery(id),
                Recognition::Phonemes(_) => format!("Phone recording {id} retained. Retry with teamy-transcriber phones <saved-source-wav> --model-dir <phone-model-folder>."),
                Recognition::WordsAndPhonemes(_) => format!("{} To retry phones, use teamy-transcriber phones <saved-source-wav> --model-dir <phone-model-folder>.", recovery(id)),
            })
        }).wrap_err("live microphone transcription stopped; failed or unprocessed windows remain available through recording list")
    })
}

fn recovery(id: RecordingId) -> String {
    format!(
        "Recording {id} was retained. Retry: teamy-transcriber transcribe --resume {id} --keep-recording"
    )
}

fn pipeline<C, F>(
    store: &RecordingStore,
    chunk_ms: MicrophoneChunkMilliseconds,
    keep_recording: bool,
    detector: Option<endpoint::Detector>,
    cancellation: &CancellationScope,
    capture: C,
    mut consume: F,
) -> Result<()>
where
    C: FnOnce(&CancellationScope, &mut SampleSink<'_>) -> Result<()> + Send,
    F: FnMut(RecordingId) -> Result<()>,
{
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
            let captured = capture(cancellation, &mut |samples, rate| {
                recorder.push(samples, rate)
            });
            // Flush already accepted audio even when capture reports an error.
            let flushed = recorder.finish();
            captured.and(flushed)
        });
        let mut consumed = Ok(());
        for id in &receiver {
            if let Err(error) = consume(id) {
                cancellation.request_cancel(error.to_string());
                consumed = Err(error);
                break;
            }
            if !keep_recording
                && let Err(error) = crate::recording_cleanup::remove_completed_microphone(store, id)
                    .wrap_err_with(|| format!("output was emitted but cleanup of recording {id} failed; the recording was retained"))
            {
                cancellation.request_cancel(error.to_string());
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
    chunk_ms: MicrophoneChunkMilliseconds,
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
        let size = usize::try_from(u64::from(rate) * self.chunk_ms.get() / 1000)?;
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
    use crate::paths::AppHome;
    use std::io::Error;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;
    use std::sync::mpsc;
    use std::time::Duration;
    use std::time::Instant;
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
        let (done, visible) = mpsc::channel();
        let stopped = &AtomicBool::new(false);
        let mut lengths = Vec::new();
        CancellationToken::new().with_scope(|cancellation| {
            pipeline(
                &store,
                MicrophoneChunkMilliseconds::try_from(500).unwrap(),
                true,
                None,
                &cancellation,
                move |cancellation, sink| {
                    sink(&vec![0.; 1200], 1000)?;
                    visible.recv_timeout(Duration::from_secs(5))?;
                    ensure!(cancellation.is_cancelled());
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
                        cancellation.request_cancel("Ctrl+C");
                        done.send(())?;
                    }
                    Ok(())
                },
            )
        })?;
        assert_eq!(lengths, [500, 500, 200]);
        assert_eq!(store.list_recordings()?.len(), 3);
        Ok(())
    }

    #[test]
    fn consumer_failure_stops_and_joins_capture_and_retains_tail() {
        let fixture = Fixture::new();
        let store = RecordingStore::new(&fixture.0);
        let joined = AtomicBool::new(false);
        let error = CancellationToken::new()
            .with_scope(|cancellation| {
                pipeline(
                    &store,
                    MicrophoneChunkMilliseconds::try_from(500).unwrap(),
                    false,
                    None,
                    &cancellation,
                    |cancellation, sink| {
                        sink(&vec![0.; 700], 1000)?;
                        let deadline = Instant::now() + Duration::from_secs(5);
                        while !cancellation.is_cancelled() && Instant::now() < deadline {
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        ensure!(cancellation.is_cancelled());
                        joined.store(true, Ordering::Relaxed);
                        Ok(())
                    },
                    |_| Err(eyre::eyre!("broken stdout")),
                )
            })
            .unwrap_err();
        assert!(error.to_string().contains("broken stdout"));
        assert!(joined.load(Ordering::Relaxed));
        assert_eq!(store.list_recordings().unwrap().len(), 2);
    }

    #[test]
    fn successful_output_removes_completed_windows() -> Result<()> {
        let fixture = Fixture::new();
        let store = RecordingStore::new(&fixture.0);
        let mut emitted = Vec::new();
        CancellationToken::new().with_scope(|cancellation| {
            pipeline(
                &store,
                MicrophoneChunkMilliseconds::try_from(500).unwrap(),
                false,
                None,
                &cancellation,
                |_, sink| sink(&vec![0.; 700], 1000),
                |id| {
                    ensure!(store.recording_dir(id).is_dir());
                    std::fs::write(store.recording_dir(id).join("phones.json"), "[]")?;
                    writeln!(&mut emitted, "recognized")?;
                    emitted.flush()?;
                    Ok(())
                },
            )
        })?;
        assert_eq!(std::str::from_utf8(&emitted)?, "recognized\nrecognized\n");
        assert!(store.list_recordings()?.is_empty());
        Ok(())
    }

    #[test]
    fn slow_consumer_drains_all_saved_windows() -> Result<()> {
        let fixture = Fixture::new();
        let store = RecordingStore::new(&fixture.0);
        let (done, blocked) = mpsc::channel();
        let mut waited = false;
        let mut consumed = 0;
        CancellationToken::new().with_scope(|cancellation| {
            pipeline(
                &store,
                MicrophoneChunkMilliseconds::try_from(500).unwrap(),
                true,
                None,
                &cancellation,
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
            )
        })?;
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
                    self.visible.send(()).map_err(Error::other)?;
                    self.sent = true;
                }
                Ok(())
            }
        }
        let fixture = Fixture::new();
        let store = RecordingStore::new(&fixture.0);
        let model = WhisperModelDir::from(PathBuf::from(std::env::var("TEST_MODEL")?));
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
        CancellationToken::new().with_scope(|cancellation| {
            transcribe(
                &store,
                &model,
                &Recognition::Words,
                LiveTranscriptionOptions {
                    cuda: Default::default(),
                    chunk_ms: MicrophoneChunkMilliseconds::default(),
                    keep_recording: false,
                    inactivity: None,
                },
                &cancellation,
                move |cancellation, sink| {
                    sink(&samples, spec.sample_rate)?;
                    // Capture cannot finish until actual inference has printed text.
                    wait.recv_timeout(Duration::from_mins(2))?;
                    sink(&samples[..spec.sample_rate as usize / 2], spec.sample_rate)?;
                    cancellation.request_cancel("simulated Ctrl+C after live output");
                    ensure!(cancellation.is_cancelled());
                    Ok(())
                },
                &mut output,
            )
        })?;
        ensure!(!std::str::from_utf8(&output.bytes)?.trim().is_empty());
        assert!(store.list_recordings()?.is_empty());
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
                    self.visible.send(()).map_err(Error::other)?;
                    self.sent = true;
                }
                Ok(())
            }
        }
        let fixture = Fixture::new();
        let store = RecordingStore::new(&fixture.0);
        let model = PhoneModelDir::resolve(
            &AppHome(fixture.0.clone()),
            Some(&std::env::var("PHONE_TEST_MODEL")?),
        )?;
        let vad = WhisperModelDir::from(PathBuf::from(std::env::var("TEST_MODEL")?));
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
        CancellationToken::new().with_scope(|cancellation| {
            transcribe(
                &store,
                &vad,
                &Recognition::Phonemes(model),
                LiveTranscriptionOptions {
                    cuda: Default::default(),
                    chunk_ms: MicrophoneChunkMilliseconds::default(),
                    keep_recording: false,
                    inactivity: None,
                },
                &cancellation,
                move |cancellation, sink| {
                    sink(&samples, spec.sample_rate)?;
                    // Capture cannot finish until actual inference has printed text.
                    wait.recv_timeout(Duration::from_mins(2))?;
                    sink(&samples[..spec.sample_rate as usize / 2], spec.sample_rate)?;
                    cancellation.request_cancel("simulated Ctrl+C after live output");
                    ensure!(cancellation.is_cancelled());
                    Ok(())
                },
                &mut output,
            )
        })?;
        ensure!(!std::str::from_utf8(&output.bytes)?.trim().is_empty());
        assert!(store.list_recordings()?.is_empty());
        Ok(())
    }
}
