use crate::domain::AppState;
use crate::domain::Command;
use crate::domain::DomainError;
use crate::domain::EVENT_SCHEMA_VERSION;
use crate::domain::EventRecord;
use crate::domain::Recording;
use crate::domain::RecordingId;
use facet::Facet;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Clone, Debug)]
pub struct RecordingStore {
    root: PathBuf,
}

impl RecordingStore {
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn recording_dir(&self, recording_id: RecordingId) -> PathBuf {
        self.root.join("recordings").join(recording_id.to_string())
    }

    #[must_use]
    pub fn events_path(&self, recording_id: RecordingId) -> PathBuf {
        self.recording_dir(recording_id).join("events.ndjson")
    }

    #[must_use]
    pub fn manifest_path(&self, recording_id: RecordingId) -> PathBuf {
        self.recording_dir(recording_id).join("manifest.json")
    }

    /// Load every recording whose manifest or event receipt is present.
    ///
    /// Directory names that are not recording UUIDs are ignored so cache and
    /// temporary files cannot prevent the GUI from opening.
    ///
    /// # Errors
    ///
    /// Returns an error when the recordings directory cannot be read or a
    /// discovered recording is malformed.
    pub fn list_recordings(&self) -> Result<Vec<Recording>, StorageError> {
        let recordings_root = self.root.join("recordings");
        if !recordings_root.exists() {
            return Ok(Vec::new());
        }
        let mut recordings = Vec::new();
        for entry in std::fs::read_dir(recordings_root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let Some(directory_name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            let Ok(recording_id) = RecordingId::parse(&directory_name) else {
                continue;
            };
            recordings.push(self.load_recording(recording_id)?);
        }
        recordings.sort_by_key(|recording| recording.id);
        Ok(recordings)
    }

    /// Apply a command, append its event, and materialize the affected manifest.
    ///
    /// # Errors
    ///
    /// Returns a domain, serialization, or filesystem error. The supplied state
    /// is changed only after the event and manifest have been written.
    pub fn apply_command(
        &self,
        state: &mut AppState,
        command: Command,
    ) -> Result<EventRecord, StorageError> {
        let mut candidate = state.clone();
        let record = candidate.execute(command)?;
        let recording_id = record.event.recording_id();
        self.append_event(&record)?;
        let recording = candidate
            .recording(recording_id)
            .ok_or(StorageError::MissingRecording(recording_id))?;
        self.write_manifest(recording)?;
        *state = candidate;
        Ok(record)
    }

    /// Append one canonical event record as one NDJSON line.
    ///
    /// # Errors
    ///
    /// Returns an error when the recording directory or event file cannot be written.
    pub fn append_event(&self, record: &EventRecord) -> Result<(), StorageError> {
        let recording_id = record.event.recording_id();
        let directory = self.recording_dir(recording_id);
        std::fs::create_dir_all(&directory)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(directory.join("events.ndjson"))?;
        let json =
            facet_json::to_string(record).map_err(|error| StorageError::Json(error.to_string()))?;
        file.write_all(json.as_bytes())?;
        file.write_all(b"\n")?;
        file.flush()?;
        Ok(())
    }

    /// Load a recording by replaying its event receipt.
    ///
    /// # Errors
    ///
    /// Returns an error when the receipt is missing, malformed, or invalid.
    pub fn load_recording(&self, recording_id: RecordingId) -> Result<Recording, StorageError> {
        let state = self.load_state(recording_id)?;
        state
            .recording(recording_id)
            .cloned()
            .ok_or(StorageError::MissingRecording(recording_id))
    }

    /// Load the replayable application state for one recording.
    ///
    /// A manifest-only recording is represented as a one-recording state so a
    /// subsequent command can continue the same event stream.
    ///
    /// # Errors
    ///
    /// Returns an error when the receipt is missing, malformed, or invalid.
    pub fn load_state(&self, recording_id: RecordingId) -> Result<AppState, StorageError> {
        let events_path = self.events_path(recording_id);
        if !events_path.is_file() {
            let recording = self.load_manifest(recording_id)?;
            let mut state = AppState::new();
            state.recordings.insert(recording.id, recording);
            return Ok(state);
        }

        let file = File::open(events_path)?;
        let reader = BufReader::new(file);
        let mut records = Vec::new();
        for (line_number, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record = facet_json::from_str::<EventRecord>(&line).map_err(|source| {
                StorageError::InvalidEventLine {
                    line: line_number + 1,
                    detail: source.to_string(),
                }
            })?;
            if record.event.recording_id() != recording_id {
                return Err(StorageError::WrongRecording {
                    expected: recording_id,
                    actual: record.event.recording_id(),
                });
            }
            records.push(record);
        }

        Ok(AppState::replay(records)?)
    }

    fn load_manifest(&self, recording_id: RecordingId) -> Result<Recording, StorageError> {
        let path = self.manifest_path(recording_id);
        if !path.is_file() {
            return Err(StorageError::MissingRecording(recording_id));
        }
        let contents = std::fs::read_to_string(path)?;
        let manifest: RecordingManifest = facet_json::from_str(&contents)
            .map_err(|error| StorageError::Json(error.to_string()))?;
        if manifest.schema_version != EVENT_SCHEMA_VERSION {
            return Err(StorageError::UnsupportedManifestSchema {
                expected: EVENT_SCHEMA_VERSION,
                actual: manifest.schema_version,
            });
        }
        if manifest.recording.id != recording_id {
            return Err(StorageError::WrongRecording {
                expected: recording_id,
                actual: manifest.recording.id,
            });
        }
        Ok(manifest.recording)
    }

    fn write_manifest(&self, recording: &Recording) -> Result<(), StorageError> {
        let directory = self.recording_dir(recording.id);
        std::fs::create_dir_all(&directory)?;
        let manifest = RecordingManifest {
            schema_version: EVENT_SCHEMA_VERSION,
            recording: recording.clone(),
        };
        let temporary_path = directory.join("manifest.json.tmp");
        let mut file = File::create(&temporary_path)?;
        let json = facet_json::to_string_pretty(&manifest)
            .map_err(|error| StorageError::Json(error.to_string()))?;
        file.write_all(json.as_bytes())?;
        file.write_all(b"\n")?;
        file.flush()?;
        drop(file);

        if self.manifest_path(recording.id).exists() {
            std::fs::remove_file(self.manifest_path(recording.id))?;
        }
        std::fs::rename(temporary_path, self.manifest_path(recording.id))?;
        Ok(())
    }
}

#[derive(Debug, Facet)]
struct RecordingManifest {
    schema_version: u16,
    recording: Recording,
}

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("filesystem operation failed")]
    Io(#[from] std::io::Error),
    #[error("JSON serialization failed: {0}")]
    Json(String),
    #[error("domain transition failed")]
    Domain(#[from] DomainError),
    #[error("recording {0} was not found in storage")]
    MissingRecording(RecordingId),
    #[error("event line {line} is malformed")]
    InvalidEventLine { line: usize, detail: String },
    #[error("event belongs to recording {actual}, expected {expected}")]
    WrongRecording {
        expected: RecordingId,
        actual: RecordingId,
    },
    #[error("manifest schema {actual} is not supported; expected {expected}")]
    UnsupportedManifestSchema { expected: u16, actual: u16 },
}
