use facet::Facet;
use std::collections::BTreeMap;
use std::path::Path;
use thiserror::Error;
use uuid::Uuid;

pub const EVENT_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, Facet, Hash, Ord, PartialEq, PartialOrd)]
pub struct RecordingId(Uuid);

impl RecordingId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// # Errors
    ///
    /// Returns an error when the value is not a UUID.
    pub fn parse(value: &str) -> Result<Self, uuid::Error> {
        Uuid::parse_str(value).map(Self)
    }
}

impl Default for RecordingId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RecordingId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Facet, Hash, Ord, PartialEq, PartialOrd)]
pub struct ClipId(Uuid);

impl ClipId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for ClipId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ClipId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Facet, Hash, Ord, PartialEq, PartialOrd)]
pub struct TranscriptId(Uuid);

impl TranscriptId {
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for TranscriptId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TranscriptId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Facet, PartialEq)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum AssetKind {
    AudioFile,
    VideoFile,
    MicrophoneRecording,
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
pub struct SourceAsset {
    pub kind: AssetKind,
    pub path: String,
}

impl SourceAsset {
    /// # Errors
    ///
    /// Returns an error when the source path is empty.
    pub fn new(kind: AssetKind, path: impl AsRef<Path>) -> Result<Self, DomainError> {
        let path = path.as_ref();
        if path.as_os_str().is_empty() {
            return Err(DomainError::EmptySourcePath);
        }
        Ok(Self {
            kind,
            path: path.to_string_lossy().into_owned(),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, Facet, PartialEq)]
pub struct TimeRange {
    pub start_us: u64,
    pub end_us: u64,
}

impl TimeRange {
    /// # Errors
    ///
    /// Returns an error when the end is not after the start.
    pub fn new(start_us: u64, end_us: u64) -> Result<Self, DomainError> {
        let range = Self { start_us, end_us };
        range.validate()?;
        Ok(range)
    }

    /// # Errors
    ///
    /// Returns an error when the end is not after the start.
    pub fn validate(self) -> Result<(), DomainError> {
        if self.end_us <= self.start_us {
            return Err(DomainError::InvalidTimeRange {
                start_us: self.start_us,
                end_us: self.end_us,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, Facet, PartialEq)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum RecordingStatus {
    Created,
    Recording,
    Saved,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, Facet, PartialEq)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum ClipStatus {
    Pending,
    Ready,
    Processing,
    Transcribed,
    Edited,
    Deleted,
}

#[derive(Clone, Copy, Debug, Eq, Facet, PartialEq)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum TranscriptProvenance {
    RawAsr,
    UserEdit,
    LocalLlm,
    Imported,
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
pub struct Clip {
    pub id: ClipId,
    pub source_range: TimeRange,
    pub status: ClipStatus,
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
pub struct TranscriptVersion {
    pub id: TranscriptId,
    pub clip_id: ClipId,
    pub provenance: TranscriptProvenance,
    pub text: String,
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
pub struct Recording {
    pub id: RecordingId,
    pub source: SourceAsset,
    pub status: RecordingStatus,
    pub clips: Vec<Clip>,
    pub transcripts: Vec<TranscriptVersion>,
}

#[derive(Clone, Debug, Default, Eq, Facet, PartialEq)]
pub struct AppState {
    pub recordings: BTreeMap<RecordingId, Recording>,
    pub next_sequence: u64,
}

impl AppState {
    #[must_use]
    pub fn new() -> Self {
        Self {
            recordings: BTreeMap::new(),
            next_sequence: 1,
        }
    }

    /// Apply a command and return its replayable event record.
    ///
    /// # Errors
    ///
    /// Returns a domain error when the requested transition is invalid.
    pub fn execute(&mut self, command: Command) -> Result<EventRecord, DomainError> {
        let event = match command {
            Command::CreateRecording {
                recording_id,
                source,
            } => Event::RecordingCreated {
                recording_id,
                source,
            },
            Command::AddClip {
                recording_id,
                clip_id,
                source_range,
            } => {
                source_range.validate()?;
                Event::ClipAdded {
                    recording_id,
                    clip_id,
                    source_range,
                }
            }
            Command::MoveClip {
                recording_id,
                clip_id,
                target_index,
            } => Event::ClipMoved {
                recording_id,
                clip_id,
                target_index,
            },
            Command::DeleteClip {
                recording_id,
                clip_id,
            } => Event::ClipDeleted {
                recording_id,
                clip_id,
            },
            Command::CommitTranscript {
                recording_id,
                clip_id,
                transcript_id,
                provenance,
                text,
            } => {
                if text.trim().is_empty() {
                    return Err(DomainError::EmptyTranscript);
                }
                Event::TranscriptCommitted {
                    recording_id,
                    clip_id,
                    transcript_id,
                    provenance,
                    text,
                }
            }
        };

        let record = EventRecord {
            schema_version: EVENT_SCHEMA_VERSION,
            sequence: self.next_sequence,
            event,
        };
        self.apply_event(&record)?;
        Ok(record)
    }

    /// Replay one event record and enforce its sequence contract.
    ///
    /// # Errors
    ///
    /// Returns an error when the schema or sequence is not the expected next value.
    pub fn apply_event(&mut self, record: &EventRecord) -> Result<(), DomainError> {
        if record.schema_version != EVENT_SCHEMA_VERSION {
            return Err(DomainError::UnsupportedSchema {
                expected: EVENT_SCHEMA_VERSION,
                actual: record.schema_version,
            });
        }
        if record.sequence != self.next_sequence {
            return Err(DomainError::EventOutOfOrder {
                expected: self.next_sequence,
                actual: record.sequence,
            });
        }

        self.apply_event_payload(&record.event)?;
        self.next_sequence += 1;
        Ok(())
    }

    /// # Errors
    ///
    /// Returns a domain error when any record cannot be replayed.
    pub fn replay(records: impl IntoIterator<Item = EventRecord>) -> Result<Self, DomainError> {
        let mut state = Self::new();
        for record in records {
            state.apply_event(&record)?;
        }
        Ok(state)
    }

    #[must_use]
    pub fn recording(&self, recording_id: RecordingId) -> Option<&Recording> {
        self.recordings.get(&recording_id)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "the exhaustive event transition table stays adjacent to the state reducer"
    )]
    fn apply_event_payload(&mut self, event: &Event) -> Result<(), DomainError> {
        match event {
            Event::RecordingCreated {
                recording_id,
                source,
            } => {
                if self.recordings.contains_key(recording_id) {
                    return Err(DomainError::RecordingAlreadyExists(*recording_id));
                }
                self.recordings.insert(
                    *recording_id,
                    Recording {
                        id: *recording_id,
                        source: source.clone(),
                        status: RecordingStatus::Created,
                        clips: Vec::new(),
                        transcripts: Vec::new(),
                    },
                );
            }
            Event::ClipAdded {
                recording_id,
                clip_id,
                source_range,
            } => {
                source_range.validate()?;
                let recording = self.recording_mut(*recording_id)?;
                if recording.clips.iter().any(|clip| clip.id == *clip_id) {
                    return Err(DomainError::ClipAlreadyExists(*clip_id));
                }
                recording.clips.push(Clip {
                    id: *clip_id,
                    source_range: *source_range,
                    status: ClipStatus::Pending,
                });
            }
            Event::ClipMoved {
                recording_id,
                clip_id,
                target_index,
            } => {
                let recording = self.recording_mut(*recording_id)?;
                let current_index = recording
                    .clips
                    .iter()
                    .position(|clip| clip.id == *clip_id)
                    .ok_or(DomainError::ClipNotFound(*clip_id))?;
                let clip = recording.clips.remove(current_index);
                if *target_index > recording.clips.len() {
                    recording.clips.insert(current_index, clip);
                    return Err(DomainError::InvalidClipIndex {
                        index: *target_index,
                        length: recording.clips.len(),
                    });
                }
                recording.clips.insert(*target_index, clip);
            }
            Event::ClipDeleted {
                recording_id,
                clip_id,
            } => {
                let recording = self.recording_mut(*recording_id)?;
                let clip = recording
                    .clips
                    .iter_mut()
                    .find(|clip| clip.id == *clip_id)
                    .ok_or(DomainError::ClipNotFound(*clip_id))?;
                clip.status = ClipStatus::Deleted;
            }
            Event::TranscriptCommitted {
                recording_id,
                clip_id,
                transcript_id,
                provenance,
                text,
            } => {
                if text.trim().is_empty() {
                    return Err(DomainError::EmptyTranscript);
                }
                let recording = self.recording_mut(*recording_id)?;
                let clip = recording
                    .clips
                    .iter_mut()
                    .find(|clip| clip.id == *clip_id)
                    .ok_or(DomainError::ClipNotFound(*clip_id))?;
                if clip.status == ClipStatus::Deleted {
                    return Err(DomainError::ClipDeleted(*clip_id));
                }
                if recording
                    .transcripts
                    .iter()
                    .any(|transcript| transcript.id == *transcript_id)
                {
                    return Err(DomainError::TranscriptAlreadyExists(*transcript_id));
                }
                recording.transcripts.push(TranscriptVersion {
                    id: *transcript_id,
                    clip_id: *clip_id,
                    provenance: *provenance,
                    text: text.clone(),
                });
                clip.status = ClipStatus::Transcribed;
            }
        }
        Ok(())
    }

    fn recording_mut(&mut self, recording_id: RecordingId) -> Result<&mut Recording, DomainError> {
        self.recordings
            .get_mut(&recording_id)
            .ok_or(DomainError::RecordingNotFound(recording_id))
    }
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum Command {
    CreateRecording {
        recording_id: RecordingId,
        source: SourceAsset,
    },
    AddClip {
        recording_id: RecordingId,
        clip_id: ClipId,
        source_range: TimeRange,
    },
    MoveClip {
        recording_id: RecordingId,
        clip_id: ClipId,
        target_index: usize,
    },
    DeleteClip {
        recording_id: RecordingId,
        clip_id: ClipId,
    },
    CommitTranscript {
        recording_id: RecordingId,
        clip_id: ClipId,
        transcript_id: TranscriptId,
        provenance: TranscriptProvenance,
        text: String,
    },
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum Event {
    RecordingCreated {
        recording_id: RecordingId,
        source: SourceAsset,
    },
    ClipAdded {
        recording_id: RecordingId,
        clip_id: ClipId,
        source_range: TimeRange,
    },
    ClipMoved {
        recording_id: RecordingId,
        clip_id: ClipId,
        target_index: usize,
    },
    ClipDeleted {
        recording_id: RecordingId,
        clip_id: ClipId,
    },
    TranscriptCommitted {
        recording_id: RecordingId,
        clip_id: ClipId,
        transcript_id: TranscriptId,
        provenance: TranscriptProvenance,
        text: String,
    },
}

impl Event {
    #[must_use]
    pub fn recording_id(&self) -> RecordingId {
        match self {
            Self::RecordingCreated { recording_id, .. }
            | Self::ClipAdded { recording_id, .. }
            | Self::ClipMoved { recording_id, .. }
            | Self::ClipDeleted { recording_id, .. }
            | Self::TranscriptCommitted { recording_id, .. } => *recording_id,
        }
    }
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
pub struct EventRecord {
    pub schema_version: u16,
    pub sequence: u64,
    pub event: Event,
}

#[derive(Debug, Error)]
pub enum DomainError {
    #[error("source path cannot be empty")]
    EmptySourcePath,
    #[error("invalid time range: start={start_us}, end={end_us}")]
    InvalidTimeRange { start_us: u64, end_us: u64 },
    #[error("recording {0} was not found")]
    RecordingNotFound(RecordingId),
    #[error("recording {0} already exists")]
    RecordingAlreadyExists(RecordingId),
    #[error("clip {0} was not found")]
    ClipNotFound(ClipId),
    #[error("clip {0} already exists")]
    ClipAlreadyExists(ClipId),
    #[error("clip {0} is deleted")]
    ClipDeleted(ClipId),
    #[error("transcript {0} already exists")]
    TranscriptAlreadyExists(TranscriptId),
    #[error("transcript text cannot be empty")]
    EmptyTranscript,
    #[error("clip index {index} is outside a list of length {length}")]
    InvalidClipIndex { index: usize, length: usize },
    #[error("event schema {actual} is not supported; expected {expected}")]
    UnsupportedSchema { expected: u16, actual: u16 },
    #[error("event sequence {actual} is not next; expected {expected}")]
    EventOutOfOrder { expected: u64, actual: u64 },
}
