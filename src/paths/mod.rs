mod app_home;
mod cache;
mod phone_model_dir;
mod phone_model_selection_codec;
mod whisper_model_dir;

pub use app_home::*;
pub use cache::*;
pub(crate) use phone_model_dir::PhoneModelDir;
pub(crate) use phone_model_dir::PhoneModelDirSelection;
pub use whisper_model_dir::MODEL_DIR_ENV_VAR;
pub use whisper_model_dir::WhisperModelDir;

pub const APP_HOME_ENV_VAR: &str = "TEAMY_TRANSCRIBER_HOME_DIR";
pub const APP_HOME_DIR_NAME: &str = "teamy-transcriber";

pub const APP_CACHE_ENV_VAR: &str = "TEAMY_TRANSCRIBER_CACHE_DIR";
pub const APP_CACHE_DIR_NAME: &str = "teamy-transcriber";
