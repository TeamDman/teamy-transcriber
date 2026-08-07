use crate::domain::ClipId;
use crate::domain::RecordingId;
use crate::domain::TranscriptProvenance;
use facet::Facet;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;
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
pub struct LocalWhisperXReadiness {
    pub python: RuntimeAssetStatus,
    pub worker_script: RuntimeAssetStatus,
    pub model_dir: RuntimeAssetStatus,
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
pub struct LocalWhisperXConfig {
    pub python_executable: PathBuf,
    pub worker_script: PathBuf,
    pub model_dir: PathBuf,
    pub model_name: String,
    pub device: String,
    pub compute_type: String,
    pub batch_size: u32,
}

#[derive(Clone, Debug)]
pub struct LocalWhisperXBackend {
    config: LocalWhisperXConfig,
}

impl LocalWhisperXBackend {
    #[must_use]
    pub fn new(config: LocalWhisperXConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub fn config(&self) -> &LocalWhisperXConfig {
        &self.config
    }

    #[must_use]
    pub fn readiness(&self) -> LocalWhisperXReadiness {
        LocalWhisperXReadiness {
            python: executable_status(&self.config.python_executable),
            worker_script: file_status(&self.config.worker_script),
            model_dir: directory_status(&self.config.model_dir),
        }
    }

    fn validate_configuration(
        &self,
        request: &TranscriptionRequest,
    ) -> Result<(), TranscriptionError> {
        let readiness = self.readiness();
        if readiness.python != RuntimeAssetStatus::Present {
            return Err(TranscriptionError::Configuration(format!(
                "Python executable is {status}: {}",
                self.config.python_executable.display(),
                status = readiness.python
            )));
        }
        if readiness.worker_script != RuntimeAssetStatus::Present {
            return Err(TranscriptionError::Configuration(format!(
                "WhisperX worker is {status}: {}",
                self.config.worker_script.display(),
                status = readiness.worker_script
            )));
        }
        if readiness.model_dir != RuntimeAssetStatus::Present {
            return Err(TranscriptionError::Configuration(format!(
                "model directory is {status}: {}",
                self.config.model_dir.display(),
                status = readiness.model_dir
            )));
        }
        if !request.audio_path.is_file() {
            return Err(TranscriptionError::Configuration(format!(
                "normalized audio is missing: {}",
                request.audio_path.display()
            )));
        }
        if self.config.model_name.trim().is_empty() {
            return Err(TranscriptionError::Configuration(
                "model name cannot be empty".to_string(),
            ));
        }
        if self.config.device.trim().is_empty() {
            return Err(TranscriptionError::Configuration(
                "device cannot be empty".to_string(),
            ));
        }
        if self.config.compute_type.trim().is_empty() {
            return Err(TranscriptionError::Configuration(
                "compute type cannot be empty".to_string(),
            ));
        }
        if self.config.batch_size == 0 {
            return Err(TranscriptionError::Configuration(
                "batch size must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

fn executable_status(executable: &Path) -> RuntimeAssetStatus {
    if executable.exists() {
        return if executable.is_file() {
            RuntimeAssetStatus::Present
        } else {
            RuntimeAssetStatus::WrongKind
        };
    }
    let result = Command::new(executable)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    match result {
        Ok(status) if status.success() => RuntimeAssetStatus::Present,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => RuntimeAssetStatus::Missing,
        Ok(_) | Err(_) => RuntimeAssetStatus::Unavailable,
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

impl TranscriptionBackend for LocalWhisperXBackend {
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            backend_id: "whisperx-local".to_string(),
            local_only: true,
            accepts_normalized_audio: true,
        }
    }

    fn transcribe(
        &self,
        request: &TranscriptionRequest,
    ) -> Result<TranscriptionResult, TranscriptionError> {
        self.validate_configuration(request)?;
        let worker_request = WorkerRequest {
            operation: "transcribe".to_string(),
            request_id: format!("{}-{}", request.recording_id, request.clip_id),
            audio_path: request.audio_path.display().to_string(),
            model_dir: self.config.model_dir.display().to_string(),
            model_name: self.config.model_name.clone(),
            device: self.config.device.clone(),
            compute_type: self.config.compute_type.clone(),
            batch_size: self.config.batch_size,
        };
        let request_json = facet_json::to_string(&worker_request)
            .map_err(|error| TranscriptionError::Protocol(error.to_string()))?;
        let mut child = Command::new(&self.config.python_executable)
            .arg(&self.config.worker_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| TranscriptionError::Process(error.to_string()))?;
        {
            let stdin = child.stdin.as_mut().ok_or_else(|| {
                TranscriptionError::Process("worker stdin unavailable".to_string())
            })?;
            stdin
                .write_all(request_json.as_bytes())
                .and_then(|()| stdin.write_all(b"\n"))
                .map_err(|error| TranscriptionError::Process(error.to_string()))?;
        };
        drop(child.stdin.take());
        let output = child
            .wait_with_output()
            .map_err(|error| TranscriptionError::Process(error.to_string()))?;
        if !output.status.success() {
            return Err(TranscriptionError::Process(
                String::from_utf8_lossy(&output.stderr).trim().to_string(),
            ));
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let response_line = stdout
            .lines()
            .rev()
            .find(|line| !line.trim().is_empty())
            .ok_or_else(|| {
                TranscriptionError::Protocol("worker returned no response".to_string())
            })?;
        let response: WorkerResponse = facet_json::from_str(response_line)
            .map_err(|error| TranscriptionError::Protocol(error.to_string()))?;
        if response.request_id != worker_request.request_id {
            return Err(TranscriptionError::Protocol(
                "worker response request ID did not match".to_string(),
            ));
        }
        if !response.ok {
            return Err(TranscriptionError::WorkerRejected(
                response
                    .error
                    .unwrap_or_else(|| "worker rejected the request".to_string()),
            ));
        }
        let text = response
            .text
            .filter(|text| !text.trim().is_empty())
            .ok_or(TranscriptionError::EmptyResult)?;
        Ok(TranscriptionResult {
            provenance: TranscriptProvenance::RawAsr,
            text,
        })
    }
}

#[derive(Debug, Facet)]
struct WorkerRequest {
    operation: String,
    request_id: String,
    audio_path: String,
    model_dir: String,
    model_name: String,
    device: String,
    compute_type: String,
    batch_size: u32,
}

#[derive(Debug, Facet)]
struct WorkerResponse {
    ok: bool,
    request_id: String,
    text: Option<String>,
    error: Option<String>,
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
    #[error("local WhisperX configuration is invalid: {0}")]
    Configuration(String),
    #[error("local WhisperX worker process failed: {0}")]
    Process(String),
    #[error("local WhisperX worker protocol failed: {0}")]
    Protocol(String),
    #[error("local WhisperX worker rejected the request: {0}")]
    WorkerRejected(String),
}
