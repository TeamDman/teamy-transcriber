use super::CacheHome;
use eyre::bail;
use std::env;
use std::ops::Deref;
use std::path::Path;
use std::path::PathBuf;

pub const MODEL_DIR_ENV_VAR: &str = "TEAMY_TRANSCRIBER_MODEL_DIR";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelHome(pub PathBuf);

impl ModelHome {
    /// Resolve the local model directory without downloading or modifying it.
    ///
    /// # Errors
    ///
    /// Returns an error when the platform cache directory cannot be resolved.
    pub fn resolve() -> eyre::Result<Self> {
        if let Ok(override_dir) = env::var(MODEL_DIR_ENV_VAR) {
            if override_dir.trim().is_empty() {
                bail!("{MODEL_DIR_ENV_VAR} cannot be empty");
            }
            return Ok(Self(PathBuf::from(override_dir)));
        }

        Ok(Self(CacheHome::resolve()?.0.join("models")))
    }
}

impl Deref for ModelHome {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        self.0.as_path()
    }
}
