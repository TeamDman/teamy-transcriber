use super::AppHome;
use super::CacheHome;
use crate::native_whisper::selection::read_selection;
use eyre::bail;
use std::env;
use std::ops::Deref;
use std::path::Path;
use std::path::PathBuf;

pub const MODEL_DIR_ENV_VAR: &str = "TEAMY_TRANSCRIBER_MODEL_DIR";

#[derive(Clone, Debug, Eq, PartialEq)]
/// Directory containing a Whisper model package and optional voice activity detection assets.
pub struct WhisperModelDir(PathBuf);

impl WhisperModelDir {
    /// Resolve the local model directory without downloading or modifying it.
    ///
    /// # Errors
    ///
    /// Returns an error when the platform cache directory cannot be resolved.
    pub fn resolve(app_home: &AppHome, cache_home: &CacheHome) -> eyre::Result<Self> {
        if let Ok(override_dir) = env::var(MODEL_DIR_ENV_VAR) {
            if override_dir.trim().is_empty() {
                bail!("{MODEL_DIR_ENV_VAR} cannot be empty");
            }
            return Ok(Self(PathBuf::from(override_dir)));
        }

        if let Some(selected) = read_selection(app_home)? {
            return Ok(Self(selected));
        }
        Ok(Self(cache_home.join("models")))
    }
}

impl From<PathBuf> for WhisperModelDir {
    fn from(value: PathBuf) -> Self {
        Self(value)
    }
}

impl From<WhisperModelDir> for PathBuf {
    fn from(value: WhisperModelDir) -> Self {
        value.0
    }
}

impl AsRef<Path> for WhisperModelDir {
    fn as_ref(&self) -> &Path {
        self.0.as_path()
    }
}

impl Deref for WhisperModelDir {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        self.0.as_path()
    }
}
