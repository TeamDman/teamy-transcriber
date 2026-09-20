//! Local model selection and offline discovery through the user's hf CLI.
use super::model::WhisperModelArtifacts;
use super::model::inspect_model_dir;
use eyre::Context;
use eyre::Result;
use eyre::ensure;
use facet::Facet;
use std::path::Path;
use std::path::PathBuf;

pub const DEFAULT_REPOSITORY: &str = "TeamDman/teamy-transcriber-whisper-large-v3";
pub const DEFAULT_REVISION: &str = "ca74fe211e39bcb869ae19204c39b7b4a2967107";
const SELECTION_FILE: &str = "model-selection.json";

#[derive(Debug, Facet)]
struct Selection {
    schema_version: u16,
    model_dir: String,
}

/// Read durable model selection. An environment override is handled by `ModelHome`.
pub fn read_selection(app_home: &Path) -> Result<Option<PathBuf>> {
    let path = app_home.join(SELECTION_FILE);
    if !path.exists() {
        return Ok(None);
    }
    let config: Selection = facet_json::from_slice(&std::fs::read(&path)?)
        .wrap_err_with(|| format!("invalid model selection at {}", path.display()))?;
    ensure!(
        config.schema_version == 1 && Path::new(&config.model_dir).is_absolute(),
        "invalid model selection schema/path"
    );
    Ok(Some(PathBuf::from(config.model_dir)))
}

/// Validate the complete prepared model without GPU initialization or conversion.
pub fn validate_prepared(path: &Path) -> Result<WhisperModelArtifacts> {
    let artifacts = inspect_model_dir(path)?;
    let vad = path.join(super::speech::DIRECTORY);
    if vad.exists() {
        super::speech::validate(&vad)?;
    }
    Ok(artifacts)
}

/// Validate first, then atomically remember an existing package in application config.
pub fn select_model(app_home: &Path, model_dir: &Path) -> Result<WhisperModelArtifacts> {
    let model_dir = model_dir
        .canonicalize()
        .wrap_err("prepared model directory is unavailable")?;
    let artifacts = validate_prepared(&model_dir)?;
    let config = Selection {
        schema_version: 1,
        model_dir: model_dir
            .to_str()
            .ok_or_else(|| eyre::eyre!("model path must be valid UTF-8"))?
            .to_owned(),
    };
    std::fs::create_dir_all(app_home)?;
    let temporary = app_home.join(format!(".model-selection-{}.tmp", uuid::Uuid::new_v4()));
    let result: Result<()> = (|| {
        std::fs::write(&temporary, facet_json::to_string_pretty(&config)?)?;
        std::fs::rename(&temporary, app_home.join(SELECTION_FILE))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result.wrap_err("saving model selection")?;
    Ok(artifacts)
}

#[derive(Debug, Facet)]
struct CachedRevision {
    repo_id: String,
    repo_type: String,
    revision: String,
    snapshot_path: PathBuf,
    refs: Vec<String>,
}

fn cached_snapshot(json: &str, repository: &str, revision: &str) -> Result<PathBuf> {
    let cached: Vec<CachedRevision> = facet_json::from_str(json)
        .wrap_err("invalid hf cache JSON; update hf or pass --source-dir")?;
    let mut matches = cached.into_iter().filter(|item| {
        item.repo_type == "model"
            && item.repo_id == repository
            && (item.revision == revision
                || item.refs.iter().any(|reference| reference == revision))
    });
    let entry = matches.next().ok_or_else(|| eyre::eyre!("Prepared model is not cached. Download it explicitly: hf download {repository} --revision {revision} --quiet\nThen run: teamy-transcriber model prepare"))?;
    ensure!(
        matches.next().is_none(),
        "multiple cached model revisions match; pass --source-dir explicitly"
    );
    Ok(entry.snapshot_path)
}

/// Discover only local HF cache contents; never download or refresh Hub metadata.
pub fn find_cached_model() -> Result<PathBuf> {
    let output = std::process::Command::new("hf")
        .args(["cache", "list", "--revisions", "--format", "json"])
        .env("HF_HUB_OFFLINE", "1")
        .env("HF_HUB_DISABLE_UPDATE_CHECK", "1")
        .env("HF_HUB_DISABLE_TELEMETRY", "1")
        .output()
        .wrap_err("Could not run hf cache list. Install the Hugging Face CLI or pass model prepare --source-dir <prepared-folder>")?;
    ensure!(
        output.status.success(),
        "hf cache list failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    cached_snapshot(
        std::str::from_utf8(&output.stdout)?,
        DEFAULT_REPOSITORY,
        DEFAULT_REVISION,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_round_trips_a_path_through_json() {
        let root = std::env::temp_dir().join(format!("model-selection-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&root).unwrap();
        let model = root.join("speaker's [model]");
        let config = Selection {
            schema_version: 1,
            model_dir: model.to_str().unwrap().to_owned(),
        };
        std::fs::write(
            root.join(SELECTION_FILE),
            facet_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();
        assert_eq!(read_selection(&root).unwrap(), Some(model));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn matches_repository_and_revision_without_needing_files_or_network() {
        let input = r#"[{"repo_id":"owner/model","repo_type":"model","revision":"abc","snapshot_path":"snapshot","refs":["v1"],"size":"3G"}]"#;
        assert_eq!(
            cached_snapshot(input, "owner/model", "v1").unwrap(),
            PathBuf::from("snapshot")
        );
        let _ = cached_snapshot(input, "owner/model", "other").unwrap_err();
        let _ = cached_snapshot(input, "owner/other", "v1").unwrap_err();
        assert!(
            cached_snapshot("[]", "owner/model", "v1")
                .unwrap_err()
                .to_string()
                .contains("hf download")
        );
    }
}
