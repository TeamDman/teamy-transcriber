use crate::cli::output::CliOutput;
use crate::domain::AppState;
use crate::domain::AssetKind;
use crate::domain::Command;
use crate::domain::RecordingId;
use crate::domain::SourceAsset;
use crate::storage::RecordingStore;
use arbitrary::Arbitrary;
use eyre::Context;
use eyre::Result;
use facet::Facet;
use figue as args;
use std::path::PathBuf;

#[derive(Clone, Copy, Facet, Arbitrary, Debug, PartialEq)]
#[facet(rename_all = "kebab-case")]
#[repr(u8)]
pub enum RecordingKind {
    Audio,
    Video,
    Microphone,
}

impl RecordingKind {
    const fn asset_kind(self) -> AssetKind {
        match self {
            Self::Audio => AssetKind::AudioFile,
            Self::Video => AssetKind::VideoFile,
            Self::Microphone => AssetKind::MicrophoneRecording,
        }
    }
}

#[derive(Facet, Debug)]
struct RecordingCreateReport {
    recording_id: String,
    manifest_path: String,
    events_path: String,
}

/// Create a durable recording manifest without decoding or transcribing the source yet.
#[derive(Facet, Arbitrary, Debug, PartialEq)]
pub struct RecordingCreateArgs {
    /// Source audio/video path or the intended microphone output path.
    #[facet(args::positional)]
    pub source: String,
    /// Source kind; defaults to audio.
    #[facet(args::named, default)]
    #[arbitrary(default)]
    pub kind: Option<RecordingKind>,
}

impl RecordingCreateArgs {
    /// # Errors
    ///
    /// Returns an error when the application home cannot be resolved or the
    /// recording manifest cannot be persisted.
    #[expect(
        clippy::unused_async,
        reason = "command invoke methods share the async CLI dispatch shape"
    )]
    pub async fn invoke(self) -> Result<CliOutput> {
        let app_home = crate::paths::AppHome::resolve()?;
        let store = RecordingStore::new(app_home.0.clone());
        let recording_id = RecordingId::new();
        let source = SourceAsset::new(
            self.kind.unwrap_or(RecordingKind::Audio).asset_kind(),
            PathBuf::from(&self.source),
        )?;
        let mut state = AppState::new();
        store
            .apply_command(
                &mut state,
                Command::CreateRecording {
                    recording_id,
                    source,
                },
            )
            .wrap_err("failed to persist recording manifest")?;

        Ok(CliOutput::facet(RecordingCreateReport {
            recording_id: recording_id.to_string(),
            manifest_path: store.manifest_path(recording_id).display().to_string(),
            events_path: store.events_path(recording_id).display().to_string(),
        }))
    }
}
