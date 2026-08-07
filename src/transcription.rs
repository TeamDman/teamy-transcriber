use crate::domain::ClipId;
use crate::domain::RecordingId;
use crate::domain::TranscriptProvenance;
use crate::native_whisper::frontend::whisper_log_mel_spectrogram;
use crate::native_whisper::model::MODEL_BURNPACK_FILE_NAME;
use crate::native_whisper::model::MODEL_DIMS_FILE_NAME;
use crate::native_whisper::model::TOKENIZER_FILE_NAME;
use crate::native_whisper::model::inspect_model_dir;
use crate::native_whisper::whisper::greedy_decode_with_model;
use facet::Facet;
use std::path::Path;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, Facet, PartialEq)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum RuntimeAssetStatus {
    Present,
    Missing,
    WrongKind,
    Unavailable,
}

impl std::fmt::Display for RuntimeAssetStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match self {
            Self::Present => "present",
            Self::Missing => "missing",
            Self::WrongKind => "wrong_kind",
            Self::Unavailable => "unavailable",
        };
        f.write_str(value)
    }
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
pub struct NativeWhisperReadiness {
    pub model_dir: RuntimeAssetStatus,
    pub weights: RuntimeAssetStatus,
    pub dims: RuntimeAssetStatus,
    pub tokenizer: RuntimeAssetStatus,
}

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

#[derive(Clone, Debug)]
pub struct NativeWhisperConfig {
    pub model_dir: PathBuf,
    pub max_decode_tokens: usize,
}

#[derive(Clone, Debug)]
pub struct NativeWhisperBackend {
    config: NativeWhisperConfig,
}

impl NativeWhisperBackend {
    #[must_use]
    pub fn new(config: NativeWhisperConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub const fn config(&self) -> &NativeWhisperConfig {
        &self.config
    }

    #[must_use]
    pub fn readiness(&self) -> NativeWhisperReadiness {
        let root = &self.config.model_dir;
        let weights = if file_status(&root.join(MODEL_BURNPACK_FILE_NAME))
            == RuntimeAssetStatus::Present
            || (directory_status(&root.join("encoder")) == RuntimeAssetStatus::Present
                && directory_status(&root.join("decoder")) == RuntimeAssetStatus::Present)
        {
            RuntimeAssetStatus::Present
        } else {
            RuntimeAssetStatus::Missing
        };
        let dims = if file_status(&root.join(MODEL_DIMS_FILE_NAME)) == RuntimeAssetStatus::Present
            || weights == RuntimeAssetStatus::Present
                && directory_status(&root.join("encoder")) == RuntimeAssetStatus::Present
        {
            RuntimeAssetStatus::Present
        } else {
            RuntimeAssetStatus::Missing
        };
        NativeWhisperReadiness {
            model_dir: directory_status(root),
            weights,
            dims,
            tokenizer: file_status(&root.join(TOKENIZER_FILE_NAME)),
        }
    }

    fn validate_configuration(
        &self,
        request: &TranscriptionRequest,
    ) -> Result<(), TranscriptionError> {
        if request.audio_path.as_os_str().is_empty() {
            return Err(TranscriptionError::EmptyAudioPath);
        }
        if !request.audio_path.is_file() {
            return Err(TranscriptionError::Configuration(format!(
                "normalized audio is missing: {}",
                request.audio_path.display()
            )));
        }
        if self.config.max_decode_tokens == 0 {
            return Err(TranscriptionError::Configuration(
                "max decode tokens must be greater than zero".to_string(),
            ));
        }
        let readiness = self.readiness();
        if readiness.model_dir != RuntimeAssetStatus::Present {
            return Err(TranscriptionError::Configuration(format!(
                "native model directory is {status}: {}",
                self.config.model_dir.display(),
                status = readiness.model_dir
            )));
        }
        if readiness.tokenizer != RuntimeAssetStatus::Present
            || readiness.weights != RuntimeAssetStatus::Present
            || readiness.dims != RuntimeAssetStatus::Present
        {
            return Err(TranscriptionError::Configuration(format!(
                "native model package is incomplete (weights={}, dims={}, tokenizer={})",
                readiness.weights, readiness.dims, readiness.tokenizer
            )));
        }
        inspect_model_dir(&self.config.model_dir)
            .map(|_| ())
            .map_err(|error| TranscriptionError::Configuration(error.to_string()))
    }
}

impl TranscriptionBackend for NativeWhisperBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            backend_id: "whisper-burn-native-cpu".to_string(),
            local_only: true,
            accepts_normalized_audio: true,
        }
    }

    fn transcribe(
        &self,
        request: &TranscriptionRequest,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        self.validate_configuration(request)?;
        let artifacts = inspect_model_dir(&self.config.model_dir)
            .map_err(|error| TranscriptionError::Configuration(error.to_string()))?;
        let samples = read_normalized_wav(&request.audio_path)?;
        let features = whisper_log_mel_spectrogram(&samples);
        let result = greedy_decode_with_model(&artifacts, &features, self.config.max_decode_tokens)
            .map_err(|error| TranscriptionError::Inference(error.to_string()))?;
        let text = result.text.trim().to_string();
        if text.is_empty() {
            return Err(TranscriptionError::EmptyResult);
        }
        Ok(TranscriptionResult {
            provenance: TranscriptProvenance::RawAsr,
            text,
        })
    }
}

fn read_normalized_wav(path: &Path) -> Result<Vec<f32>, TranscriptionError> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|error| TranscriptionError::Audio(error.to_string()))?;
    let spec = reader.spec();
    if spec.sample_rate != crate::media::WHISPER_SAMPLE_RATE_HZ || spec.channels != 1 {
        return Err(TranscriptionError::Audio(format!(
            "native Whisper expects 16 kHz mono WAV, found {} Hz / {} channels",
            spec.sample_rate, spec.channels
        )));
    }
    match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| TranscriptionError::Audio(error.to_string())),
        hound::SampleFormat::Int => {
            if spec.bits_per_sample > 16 {
                return Err(TranscriptionError::Audio(format!(
                    "native Whisper does not accept integer WAVs wider than 16 bits (found {})",
                    spec.bits_per_sample
                )));
            }
            let scale = 2_f32.powi(i32::from(spec.bits_per_sample.saturating_sub(1)));
            reader
                .samples::<i16>()
                .map(|sample| {
                    sample
                        .map(|sample| f32::from(sample) / scale)
                        .map_err(|error| TranscriptionError::Audio(error.to_string()))
                })
                .collect()
        }
    }
}

fn file_status(path: &Path) -> RuntimeAssetStatus {
    if !path.exists() {
        RuntimeAssetStatus::Missing
    } else if path.is_file() {
        RuntimeAssetStatus::Present
    } else {
        RuntimeAssetStatus::WrongKind
    }
}

fn directory_status(path: &Path) -> RuntimeAssetStatus {
    if !path.exists() {
        RuntimeAssetStatus::Missing
    } else if path.is_dir() {
        RuntimeAssetStatus::Present
    } else {
        RuntimeAssetStatus::WrongKind
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
    #[error("native Whisper configuration is invalid: {0}")]
    Configuration(String),
    #[error("native Whisper audio decoding failed: {0}")]
    Audio(String),
    #[error("native Whisper inference failed: {0}")]
    Inference(String),
}
