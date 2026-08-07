use crate::domain::ClipId;
use crate::domain::RecordingId;
use crate::domain::TranscriptProvenance;
use std::path::Path;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendCapabilities {
    pub backend_id: String,
    pub local_only: bool,
    pub accepts_normalized_audio: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscriptionRequest {
    pub recording_id: RecordingId,
    pub clip_id: ClipId,
    pub audio_path: PathBuf,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TranscriptionResult {
    pub provenance: TranscriptProvenance,
    pub text: String,
}

pub trait TranscriptionBackend {
    /// Describe the backend without loading a model.
    fn capabilities(&self) -> BackendCapabilities;

    /// Transcribe one immutable clip.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend cannot process the request.
    fn transcribe(
        &self,
        request: &TranscriptionRequest,
    ) -> Result<TranscriptionResult, TranscriptionError>;
}

#[derive(Clone, Debug)]
pub struct FakeTranscriptionBackend {
    text: String,
}

impl FakeTranscriptionBackend {
    #[must_use]
    pub fn with_text(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

impl Default for FakeTranscriptionBackend {
    fn default() -> Self {
        Self::with_text("[fake transcription]")
    }
}

impl TranscriptionBackend for FakeTranscriptionBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            backend_id: "fake".to_string(),
            local_only: true,
            accepts_normalized_audio: true,
        }
    }

    fn transcribe(
        &self,
        request: &TranscriptionRequest,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        if request.audio_path.as_os_str().is_empty() {
            return Err(TranscriptionError::EmptyAudioPath);
        }
        if self.text.trim().is_empty() {
            return Err(TranscriptionError::EmptyResult);
        }
        Ok(TranscriptionResult {
            provenance: TranscriptProvenance::RawAsr,
            text: self.text.clone(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalModelInventory {
    pub root: PathBuf,
    pub exists: bool,
    pub file_count: usize,
}

impl LocalModelInventory {
    /// # Errors
    ///
    /// Returns an error when the model directory exists but cannot be read.
    pub fn inspect(root: impl Into<PathBuf>) -> Result<Self, std::io::Error> {
        let root = root.into();
        if !root.exists() {
            return Ok(Self {
                root,
                exists: false,
                file_count: 0,
            });
        }

        let file_count = count_files(&root)?;
        Ok(Self {
            root,
            exists: true,
            file_count,
        })
    }
}

fn count_files(root: &Path) -> Result<usize, std::io::Error> {
    let mut count = 0;
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            count += count_files(&entry.path())?;
        } else if entry.file_type()?.is_file() {
            count += 1;
        }
    }
    Ok(count)
}

#[derive(Debug, Error)]
pub enum TranscriptionError {
    #[error("audio path cannot be empty")]
    EmptyAudioPath,
    #[error("transcription returned empty text")]
    EmptyResult,
}
