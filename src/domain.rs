use facet::Facet;
use std::collections::BTreeMap;
use std::path::Path;
use thiserror::Error;
use uuid::Uuid;

pub const EVENT_SCHEMA_VERSION: u16 = 1;

#[derive(Clone, Copy, Debug, Eq, Facet, Hash, Ord, PartialEq, PartialOrd)]
#[facet(transparent)]
#[repr(transparent)]
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
#[facet(transparent)]
#[repr(transparent)]
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
#[facet(transparent)]
#[repr(transparent)]
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

    #[must_use]
    pub const fn overlaps(self, other: Self) -> bool {
        self.start_us < other.end_us && other.start_us < self.end_us
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
    Failed,
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
    pub failure: Option<String>,
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
pub struct TranscriptVersion {
    pub id: TranscriptId,
    pub clip_id: ClipId,
    pub provenance: TranscriptProvenance,
    pub text: String,
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum ClipPlanKind {
    FixedDuration,
    VoiceActivity { weights_sha256: String },
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
pub struct PlannedClip {
    pub id: ClipId,
    pub source_range: TimeRange,
}

/// The complete initial plan is one event, including a successful empty VAD plan.
#[derive(Clone, Debug, Eq, Facet, PartialEq)]
pub struct ClipPlan {
    pub kind: ClipPlanKind,
    pub duration_us: u64,
    pub clips: Vec<PlannedClip>,
}

impl ClipPlan {
    fn validate(&self) -> Result<(), DomainError> {
        if self.duration_us == 0 {
            return Err(DomainError::InvalidClipPlan);
        }
        if let ClipPlanKind::VoiceActivity { weights_sha256 } = &self.kind
            && (weights_sha256.len() != 64
                || !weights_sha256.bytes().all(|b| b.is_ascii_hexdigit()))
        {
            return Err(DomainError::InvalidClipPlan);
        }
        let mut ids = std::collections::BTreeSet::new();
        let mut end = 0;
        for clip in &self.clips {
            clip.source_range.validate()?;
            if !ids.insert(clip.id)
                || clip.source_range.start_us < end
                || clip.source_range.end_us > self.duration_us
                || (self.kind == ClipPlanKind::FixedDuration && clip.source_range.start_us != end)
            {
                return Err(DomainError::InvalidClipPlan);
            }
            end = clip.source_range.end_us;
        }
        if self.kind == ClipPlanKind::FixedDuration && end != self.duration_us {
            return Err(DomainError::InvalidClipPlan);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
pub struct Recording {
    pub id: RecordingId,
    pub source: SourceAsset,
    pub status: RecordingStatus,
    pub failure: Option<String>,
    pub clips: Vec<Clip>,
    pub transcripts: Vec<TranscriptVersion>,
    #[facet(default)]
    pub clip_plan: Option<ClipPlan>,
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
    #[expect(
        clippy::too_many_lines,
        reason = "keep the command-to-event mapping together for lifecycle review"
    )]
    pub fn execute(&mut self, command: Command) -> Result<EventRecord, DomainError> {
        let event = match command {
            Command::CreateRecording {
                recording_id,
                source,
            } => Event::RecordingCreated {
                recording_id,
                source,
            },
            Command::StartRecording { recording_id } => Event::RecordingStarted { recording_id },
            Command::CompleteRecording { recording_id } => Event::RecordingSaved { recording_id },
            Command::FailRecording {
                recording_id,
                reason,
            } => {
                if reason.trim().is_empty() {
                    return Err(DomainError::EmptyRecordingFailure);
                }
                Event::RecordingFailed {
                    recording_id,
                    reason,
                }
            }
            Command::BeginTranscription {
                recording_id,
                clip_id,
            } => Event::TranscriptionStarted {
                recording_id,
                clip_id,
            },
            Command::PlanClips { recording_id, plan } => Event::ClipsPlanned { recording_id, plan },
            Command::CancelTranscription {
                recording_id,
                clip_id,
            } => Event::TranscriptionCancelled {
                recording_id,
                clip_id,
            },
            Command::FailTranscription {
                recording_id,
                clip_id,
                reason,
            } => {
                if reason.trim().is_empty() {
                    return Err(DomainError::EmptyTranscriptionFailure);
                }
                Event::TranscriptionFailed {
                    recording_id,
                    clip_id,
                    reason,
                }
            }
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
                        failure: None,
                        clips: Vec::new(),
                        transcripts: Vec::new(),
                        clip_plan: None,
                    },
                );
            }
            Event::ClipsPlanned { recording_id, plan } => {
                plan.validate()?;
                let recording = self.recording_mut(*recording_id)?;
                if !recording.clips.is_empty() {
                    return Err(DomainError::ClipPlanAlreadyExists);
                }
                recording.clips = plan
                    .clips
                    .iter()
                    .map(|clip| Clip {
                        id: clip.id,
                        source_range: clip.source_range,
                        status: ClipStatus::Pending,
                        failure: None,
                    })
                    .collect();
                recording.clip_plan = Some(plan.clone());
            }
            Event::RecordingStarted { recording_id } => {
                let recording = self.recording_mut(*recording_id)?;
                if recording.status != RecordingStatus::Created {
                    return Err(DomainError::InvalidRecordingTransition {
                        recording_id: *recording_id,
                        actual: recording.status,
                        action: "start",
                    });
                }
                recording.status = RecordingStatus::Recording;
                recording.failure = None;
            }
            Event::RecordingSaved { recording_id } => {
                let recording = self.recording_mut(*recording_id)?;
                if recording.status != RecordingStatus::Recording {
                    return Err(DomainError::InvalidRecordingTransition {
                        recording_id: *recording_id,
                        actual: recording.status,
                        action: "save",
                    });
                }
                recording.status = RecordingStatus::Saved;
            }
            Event::RecordingFailed {
                recording_id,
                reason,
            } => {
                if reason.trim().is_empty() {
                    return Err(DomainError::EmptyRecordingFailure);
                }
                let recording = self.recording_mut(*recording_id)?;
                if !matches!(
                    recording.status,
                    RecordingStatus::Created | RecordingStatus::Recording
                ) {
                    return Err(DomainError::InvalidRecordingTransition {
                        recording_id: *recording_id,
                        actual: recording.status,
                        action: "fail",
                    });
                }
                recording.status = RecordingStatus::Failed;
                recording.failure = Some(reason.clone());
            }
            Event::TranscriptionStarted {
                recording_id,
                clip_id,
            } => {
                let recording = self.recording_mut(*recording_id)?;
                let clip = recording
                    .clips
                    .iter_mut()
                    .find(|clip| clip.id == *clip_id)
                    .ok_or(DomainError::ClipNotFound(*clip_id))?;
                if !matches!(
                    clip.status,
                    ClipStatus::Pending
                        | ClipStatus::Ready
                        | ClipStatus::Failed
                        | ClipStatus::Transcribed
                        | ClipStatus::Edited
                ) {
                    return Err(DomainError::InvalidClipTranscriptionTransition {
                        clip_id: *clip_id,
                        actual: clip.status,
                        action: "start",
                    });
                }
                clip.status = ClipStatus::Processing;
                clip.failure = None;
            }
            Event::TranscriptionCancelled {
                recording_id,
                clip_id,
            } => {
                let recording = self.recording_mut(*recording_id)?;
                let latest = recording
                    .transcripts
                    .iter()
                    .rev()
                    .find(|t| t.clip_id == *clip_id);
                let status = latest.map_or(ClipStatus::Pending, |transcript| {
                    if transcript.provenance == TranscriptProvenance::UserEdit {
                        ClipStatus::Edited
                    } else {
                        ClipStatus::Transcribed
                    }
                });
                let clip = recording
                    .clips
                    .iter_mut()
                    .find(|clip| clip.id == *clip_id)
                    .ok_or(DomainError::ClipNotFound(*clip_id))?;
                if clip.status != ClipStatus::Processing {
                    return Err(DomainError::InvalidClipTranscriptionTransition {
                        clip_id: *clip_id,
                        actual: clip.status,
                        action: "cancel",
                    });
                }
                clip.status = status;
                clip.failure = None;
            }
            Event::TranscriptionFailed {
                recording_id,
                clip_id,
                reason,
            } => {
                if reason.trim().is_empty() {
                    return Err(DomainError::EmptyTranscriptionFailure);
                }
                let recording = self.recording_mut(*recording_id)?;
                let clip = recording
                    .clips
                    .iter_mut()
                    .find(|clip| clip.id == *clip_id)
                    .ok_or(DomainError::ClipNotFound(*clip_id))?;
                if !matches!(
                    clip.status,
                    ClipStatus::Processing | ClipStatus::Transcribed | ClipStatus::Edited
                ) {
                    return Err(DomainError::InvalidClipTranscriptionTransition {
                        clip_id: *clip_id,
                        actual: clip.status,
                        action: "fail",
                    });
                }
                clip.status = ClipStatus::Failed;
                clip.failure = Some(reason.clone());
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
                if let Some(existing) = recording.clips.iter().find(|clip| {
                    clip.status != ClipStatus::Deleted && clip.source_range.overlaps(*source_range)
                }) {
                    return Err(DomainError::ClipOverlaps {
                        existing: existing.id,
                        requested_start_us: source_range.start_us,
                        requested_end_us: source_range.end_us,
                    });
                }
                recording.clips.push(Clip {
                    id: *clip_id,
                    source_range: *source_range,
                    status: ClipStatus::Pending,
                    failure: None,
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
                if !matches!(
                    clip.status,
                    ClipStatus::Processing | ClipStatus::Transcribed | ClipStatus::Edited
                ) {
                    return Err(DomainError::InvalidClipTranscriptionTransition {
                        clip_id: *clip_id,
                        actual: clip.status,
                        action: "commit",
                    });
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
                clip.status = if *provenance == TranscriptProvenance::UserEdit {
                    ClipStatus::Edited
                } else {
                    ClipStatus::Transcribed
                };
                clip.failure = None;
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
    PlanClips {
        recording_id: RecordingId,
        plan: ClipPlan,
    },
    CreateRecording {
        recording_id: RecordingId,
        source: SourceAsset,
    },
    StartRecording {
        recording_id: RecordingId,
    },
    CompleteRecording {
        recording_id: RecordingId,
    },
    FailRecording {
        recording_id: RecordingId,
        reason: String,
    },
    BeginTranscription {
        recording_id: RecordingId,
        clip_id: ClipId,
    },
    CancelTranscription {
        recording_id: RecordingId,
        clip_id: ClipId,
    },
    FailTranscription {
        recording_id: RecordingId,
        clip_id: ClipId,
        reason: String,
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
    ClipsPlanned {
        recording_id: RecordingId,
        plan: ClipPlan,
    },
    RecordingCreated {
        recording_id: RecordingId,
        source: SourceAsset,
    },
    RecordingStarted {
        recording_id: RecordingId,
    },
    RecordingSaved {
        recording_id: RecordingId,
    },
    RecordingFailed {
        recording_id: RecordingId,
        reason: String,
    },
    TranscriptionStarted {
        recording_id: RecordingId,
        clip_id: ClipId,
    },
    TranscriptionCancelled {
        recording_id: RecordingId,
        clip_id: ClipId,
    },
    TranscriptionFailed {
        recording_id: RecordingId,
        clip_id: ClipId,
        reason: String,
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
            Self::ClipsPlanned { recording_id, .. }
            | Self::RecordingCreated { recording_id, .. }
            | Self::RecordingStarted { recording_id }
            | Self::RecordingSaved { recording_id }
            | Self::RecordingFailed { recording_id, .. }
            | Self::TranscriptionStarted { recording_id, .. }
            | Self::TranscriptionCancelled { recording_id, .. }
            | Self::TranscriptionFailed { recording_id, .. }
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
    #[error("clip plan has invalid ranges, identifiers or model hash")]
    InvalidClipPlan,
    #[error("cannot replace a recording's existing clip history with a new plan")]
    ClipPlanAlreadyExists,
    #[error("source path cannot be empty")]
    EmptySourcePath,
    #[error("invalid time range: start={start_us}, end={end_us}")]
    InvalidTimeRange { start_us: u64, end_us: u64 },
    #[error("recording {0} was not found")]
    RecordingNotFound(RecordingId),
    #[error("recording {0} already exists")]
    RecordingAlreadyExists(RecordingId),
    #[error("recording {recording_id} cannot {action} from status {actual:?}")]
    InvalidRecordingTransition {
        recording_id: RecordingId,
        actual: RecordingStatus,
        action: &'static str,
    },
    #[error("recording failure reason cannot be empty")]
    EmptyRecordingFailure,
    #[error("transcription failure reason cannot be empty")]
    EmptyTranscriptionFailure,
    #[error("clip {clip_id} cannot {action} from status {actual:?}")]
    InvalidClipTranscriptionTransition {
        clip_id: ClipId,
        actual: ClipStatus,
        action: &'static str,
    },
    #[error("clip {0} was not found")]
    ClipNotFound(ClipId),
    #[error("clip {0} already exists")]
    ClipAlreadyExists(ClipId),
    #[error("clip range {requested_start_us}..{requested_end_us} overlaps active clip {existing}")]
    ClipOverlaps {
        existing: ClipId,
        requested_start_us: u64,
        requested_end_us: u64,
    },
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
