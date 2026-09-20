use super::*;
use crate::transcription::BackendCapabilities;
use crate::transcription::FakeTranscriptionBackend;
use crate::transcription::TranscriptionResult;
use std::cell::RefCell;
use std::sync::atomic::Ordering;

#[derive(Clone, Copy)]
enum Mode {
    Normal,
    FailAfterFirst,
    OutOfOrder,
    Omit,
}

struct Backend {
    mode: Mode,
    calls: RefCell<Vec<usize>>,
    block_manifest: Option<PathBuf>,
}
impl Backend {
    fn new(mode: Mode) -> Self {
        Self {
            mode,
            calls: RefCell::new(Vec::new()),
            block_manifest: None,
        }
    }
}
impl TranscriptionBackend for Backend {
    fn capabilities(&self) -> BackendCapabilities {
        FakeTranscriptionBackend::default().capabilities()
    }
    fn transcribe(
        &self,
        request: &TranscriptionRequest,
    ) -> std::result::Result<TranscriptionResult, TranscriptionError> {
        FakeTranscriptionBackend::default().transcribe(request)
    }
    fn batch_capacity(&self) -> usize {
        3
    }
    fn transcribe_batch(
        &self,
        requests: &[TranscriptionRequest],
        should_stop: &mut dyn FnMut() -> bool,
        on_complete: &mut dyn FnMut(
            usize,
            TranscriptionResult,
        ) -> std::result::Result<(), TranscriptionError>,
    ) -> std::result::Result<bool, TranscriptionError> {
        self.calls.borrow_mut().push(requests.len());
        if let Some(path) = &self.block_manifest {
            std::fs::create_dir(path).unwrap();
        }
        for (index, request) in requests.iter().enumerate() {
            if should_stop() {
                return Ok(true);
            }
            match self.mode {
                Mode::FailAfterFirst if index == 1 => {
                    return Err(TranscriptionError::Inference(
                        "injected inference failure".into(),
                    ));
                }
                Mode::Omit => return Ok(false),
                _ => {}
            }
            on_complete(
                if matches!(self.mode, Mode::OutOfOrder) {
                    index + 1
                } else {
                    index
                },
                self.transcribe(request)?,
            )?;
        }
        Ok(should_stop())
    }
}

struct Fixture {
    root: PathBuf,
    store: RecordingStore,
    id: RecordingId,
}
impl Fixture {
    fn new() -> Result<Self> {
        let root = std::env::temp_dir().join(format!("transcriber-batch-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&root)?;
        let wav = root.join("source.wav");
        let mut writer = hound::WavWriter::create(
            &wav,
            hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )?;
        for _ in 0..64_000 {
            writer.write_sample(1000i16)?;
        }
        writer.finalize()?;
        let store = RecordingStore::new(root.join("home"));
        let id = create_recording(&store, AssetKind::AudioFile, wav)?;
        prepare_recording(&store, id)?;
        Ok(Self { root, store, id })
    }
    fn run(
        &self,
        backend: &dyn TranscriptionBackend,
        stop: Option<&AtomicBool>,
        progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> Result<TranscriptionReport> {
        transcribe_recording_inner(
            &self.store,
            self.id,
            Some(1_000_000),
            AudioProfile::Original,
            stop,
            progress,
            backend,
        )
    }
    fn statuses(&self) -> Result<Vec<ClipStatus>> {
        Ok(self
            .store
            .load_recording(self.id)?
            .clips
            .iter()
            .map(|c| c.status)
            .collect())
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        // Only this fixture's freshly-created UUID directory is removed.
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

#[test]
fn cancellation_persists_a_prefix_and_restores_previous_edits() -> Result<()> {
    let f = Fixture::new()?;
    let backend = Backend::new(Mode::Normal);
    let stop = AtomicBool::new(false);
    let mut progress = |completed, _| {
        if completed == 1 {
            assert_eq!(f.store.load_recording(f.id).unwrap().transcripts.len(), 1);
            stop.store(true, Ordering::Relaxed);
        }
    };
    let partial = f.run(&backend, Some(&stop), Some(&mut progress))?;
    assert!(partial.cancelled);
    assert_eq!(partial.chunks.len(), 1);
    assert_eq!(
        f.statuses()?,
        [
            ClipStatus::Transcribed,
            ClipStatus::Pending,
            ClipStatus::Pending,
            ClipStatus::Pending
        ]
    );
    assert_eq!(*backend.calls.borrow(), [3]);
    let resumed = f.run(&backend, None, None)?;
    assert!(!resumed.cancelled);
    assert_eq!(resumed.chunks.len(), 4);
    assert_eq!(*backend.calls.borrow(), [3, 3, 1]);
    let edited = resumed.chunks[2].clip_id;
    commit_transcript_edit(&f.store, f.id, edited, "preserved user edit".into())?;
    stop.store(false, Ordering::Relaxed);
    let mut progress = |completed, _| {
        if completed == 1 {
            stop.store(true, Ordering::Relaxed);
        }
    };
    assert!(f.run(&backend, Some(&stop), Some(&mut progress))?.cancelled);
    assert_eq!(
        f.statuses()?,
        [
            ClipStatus::Transcribed,
            ClipStatus::Transcribed,
            ClipStatus::Edited,
            ClipStatus::Transcribed
        ]
    );
    let reloaded = f.store.load_recording(f.id)?;
    assert_eq!(
        reloaded
            .transcripts
            .iter()
            .rev()
            .find(|t| t.clip_id == edited)
            .unwrap()
            .text,
        "preserved user edit"
    );
    assert!(reloaded.clips.iter().all(|c| c.failure.is_none()));
    Ok(())
}

#[test]
fn batch_failure_keeps_completed_work_and_clears_processing() -> Result<()> {
    let f = Fixture::new()?;
    let error = f
        .run(&Backend::new(Mode::FailAfterFirst), None, None)
        .unwrap_err();
    assert!(format!("{error:?}").contains("injected inference failure"));
    assert_eq!(
        f.statuses()?,
        [
            ClipStatus::Transcribed,
            ClipStatus::Failed,
            ClipStatus::Failed,
            ClipStatus::Pending
        ]
    );
    assert_eq!(f.store.load_recording(f.id)?.transcripts.len(), 1);
    assert_eq!(
        f.run(&Backend::new(Mode::Normal), None, None)?.chunks.len(),
        4
    );
    Ok(())
}

#[test]
fn invalid_completion_sequences_never_commit_another_clips_text() -> Result<()> {
    for mode in [Mode::OutOfOrder, Mode::Omit] {
        let f = Fixture::new()?;
        let _ = f.run(&Backend::new(mode), None, None).unwrap_err();
        assert!(f.store.load_recording(f.id)?.transcripts.is_empty());
        assert!(!f.statuses()?.contains(&ClipStatus::Processing));
    }
    Ok(())
}

#[test]
fn pre_cancel_and_audio_preparation_failure_leave_no_processing_clips() -> Result<()> {
    let f = Fixture::new()?;
    let backend = Backend::new(Mode::Normal);
    let stop = AtomicBool::new(true);
    assert!(f.run(&backend, Some(&stop), None)?.cancelled);
    assert!(backend.calls.borrow().is_empty());
    assert_eq!(f.statuses()?, vec![ClipStatus::Pending; 4]);
    // A file where derived audio's directory belongs makes preparation fail.
    std::fs::write(
        f.store.recording_dir(f.id).join("audio").join("clips"),
        "blocked",
    )?;
    let _ = f.run(&backend, None, None).unwrap_err();
    assert!(backend.calls.borrow().is_empty());
    assert_eq!(
        f.statuses()?,
        [
            ClipStatus::Failed,
            ClipStatus::Pending,
            ClipStatus::Pending,
            ClipStatus::Pending
        ]
    );
    Ok(())
}

#[test]
fn manifest_failure_after_event_append_does_not_duplicate_event_sequences() -> Result<()> {
    let f = Fixture::new()?;
    let blocked = f.store.recording_dir(f.id).join("manifest.json.tmp");
    let backend = Backend {
        mode: Mode::Normal,
        calls: RefCell::new(Vec::new()),
        block_manifest: Some(blocked.clone()),
    };
    let error = f.run(&backend, None, None).unwrap_err();
    assert!(error.to_string().contains("failed to save clip lifecycle"));
    // The commit and both cleanup events reached the receipt before their
    // manifest writes failed. Every sequence must still replay exactly once.
    assert_eq!(f.store.load_recording(f.id)?.transcripts.len(), 1);
    assert_eq!(
        f.statuses()?,
        [
            ClipStatus::Transcribed,
            ClipStatus::Failed,
            ClipStatus::Failed,
            ClipStatus::Pending
        ]
    );
    std::fs::remove_dir(blocked)?;
    assert_eq!(
        f.run(&Backend::new(Mode::Normal), None, None)?.chunks.len(),
        4
    );
    Ok(())
}
