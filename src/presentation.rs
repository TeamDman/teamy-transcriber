use crate::domain::ClipId;
use crate::domain::ClipStatus;
use crate::domain::Recording;
use crate::domain::RecordingId;
use crate::domain::RecordingStatus;
use crate::domain::TimeRange;
use crate::domain::TranscriptProvenance;
use facet::Facet;

/// Stable semantic targets used by every future renderer and input surface.
#[derive(Clone, Copy, Debug, Eq, Facet, Hash, Ord, PartialEq, PartialOrd)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum UiId {
    Root,
    RecordingControl,
    ClipTimeline,
    Transcript,
    Diagnostics,
    ExportAction,
}

/// Stable actions; pointer, keyboard, palette, and tray inputs resolve here
/// before any renderer-specific code is involved.
#[derive(Clone, Copy, Debug, Eq, Facet, Hash, Ord, PartialEq, PartialOrd)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum ActionId {
    ToggleRecording,
    StopRecording,
    PrepareRecording,
    TranscribeSelectedClip,
    CommitTranscriptEdit,
    ExportTranscript,
    CancelOperation,
}

#[derive(Clone, Copy, Debug, Default, Eq, Facet, Hash, Ord, PartialEq, PartialOrd)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum FocusContext {
    #[default]
    Global,
    RecordingControl,
    ClipTimeline,
    Transcript,
    Diagnostics,
}

#[derive(Clone, Copy, Debug, Eq, Facet, Hash, Ord, PartialEq, PartialOrd)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum InputSource {
    Pointer,
    Keyboard,
    Palette,
    Tray,
}

#[derive(Clone, Copy, Debug, Eq, Facet, Hash, Ord, PartialEq, PartialOrd)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum UiKey {
    Character(char),
    Enter,
    Escape,
    Space,
}

#[derive(Clone, Copy, Debug, Eq, Facet, Hash, Ord, PartialEq, PartialOrd)]
pub struct KeyChord {
    pub key: UiKey,
    pub control: bool,
    pub shift: bool,
    pub alt: bool,
}

impl KeyChord {
    #[must_use]
    pub const fn plain(key: UiKey) -> Self {
        Self {
            key,
            control: false,
            shift: false,
            alt: false,
        }
    }

    #[must_use]
    pub const fn control(key: UiKey) -> Self {
        Self {
            key,
            control: true,
            shift: false,
            alt: false,
        }
    }
}

/// Resolve one key with deterministic contextual precedence.
#[must_use]
pub const fn action_for_key(focus: FocusContext, chord: KeyChord) -> Option<ActionId> {
    if matches!(chord.key, UiKey::Escape) && !chord.control && !chord.shift && !chord.alt {
        return Some(ActionId::CancelOperation);
    }
    match (focus, chord.key, chord.control, chord.shift, chord.alt) {
        (FocusContext::RecordingControl, UiKey::Space, false, false, false) => {
            Some(ActionId::ToggleRecording)
        }
        (FocusContext::RecordingControl, UiKey::Enter, false, false, false) => {
            Some(ActionId::PrepareRecording)
        }
        (FocusContext::ClipTimeline, UiKey::Enter, true, false, false) => {
            Some(ActionId::TranscribeSelectedClip)
        }
        (FocusContext::Transcript, UiKey::Enter, true, false, false) => {
            Some(ActionId::CommitTranscriptEdit)
        }
        (FocusContext::Global, UiKey::Character('e'), true, false, false) => {
            Some(ActionId::ExportTranscript)
        }
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, Facet, Hash, Ord, PartialEq, PartialOrd)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
pub struct Diagnostic {
    pub severity: DiagnosticSeverity,
    pub message: String,
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
pub struct ClipProjection {
    pub id: ClipId,
    pub source_range: TimeRange,
    pub status: ClipStatus,
    pub failure: Option<String>,
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
pub struct TranscriptProjection {
    pub id: String,
    pub clip_id: ClipId,
    pub provenance: TranscriptProvenance,
    pub text: String,
    pub committed: bool,
}

#[derive(Clone, Debug, Default, Eq, Facet, PartialEq)]
pub struct PresentationState {
    pub recording_id: Option<RecordingId>,
    pub recording_status: Option<RecordingStatus>,
    pub selected_clip_id: Option<ClipId>,
    pub focus: FocusContext,
    pub clips: Vec<ClipProjection>,
    pub transcript: Option<TranscriptProjection>,
    pub diagnostics: Vec<Diagnostic>,
}

impl PresentationState {
    /// Project domain state into renderer-neutral presentation data.
    #[must_use]
    pub fn from_recording(recording: &Recording, selected_clip_id: Option<ClipId>) -> Self {
        let selected_clip_id = selected_clip_id
            .filter(|clip_id| {
                recording
                    .clips
                    .iter()
                    .any(|clip| clip.id == *clip_id && clip.status != ClipStatus::Deleted)
            })
            .or_else(|| {
                recording
                    .clips
                    .iter()
                    .find(|clip| clip.status != ClipStatus::Deleted)
                    .map(|clip| clip.id)
            });
        let clips = recording
            .clips
            .iter()
            .filter(|clip| clip.status != ClipStatus::Deleted)
            .map(|clip| ClipProjection {
                id: clip.id,
                source_range: clip.source_range,
                status: clip.status,
                failure: clip.failure.clone(),
            })
            .collect::<Vec<_>>();
        let transcript = selected_clip_id.and_then(|clip_id| {
            recording
                .transcripts
                .iter()
                .rev()
                .find(|transcript| transcript.clip_id == clip_id)
                .map(|transcript| TranscriptProjection {
                    id: transcript.id.to_string(),
                    clip_id: transcript.clip_id,
                    provenance: transcript.provenance,
                    text: transcript.text.clone(),
                    committed: true,
                })
        });
        let mut diagnostics = Vec::new();
        if let Some(failure) = recording.failure.as_deref() {
            diagnostics.push(Diagnostic {
                severity: DiagnosticSeverity::Error,
                message: format!("recording failed: {failure}"),
            });
        }
        for clip in &clips {
            if let Some(failure) = clip.failure.as_deref() {
                diagnostics.push(Diagnostic {
                    severity: DiagnosticSeverity::Error,
                    message: format!("clip {} failed: {failure}", clip.id),
                });
            }
        }
        Self {
            recording_id: Some(recording.id),
            recording_status: Some(recording.status),
            selected_clip_id,
            focus: FocusContext::Global,
            clips,
            transcript,
            diagnostics,
        }
    }
}
