use super::AppHome;
use super::phone_model_selection_codec::PhoneModelSelectionCodec;
use crate::native_whisper::selection::find_cached_repository;
use eyre::Context;
use eyre::Result;
use std::path::PathBuf;
use teamy_whisper_native::CudaDeviceIndex;
pub(crate) use teamy_whisper_native::phones::PhoneModelDir;

const REPOSITORY: &str = "changelinglab/PhoneticXeus";
const REVISION: &str = "3a8d860fa68f8936ceb4196651221215bab9dae4";

pub(crate) trait PhoneModelDirSelection: Sized {
    fn resolve(app_home: &AppHome, explicit: Option<&str>) -> Result<Self>;
    fn select(&self, app_home: &AppHome) -> Result<Self>;
}

impl PhoneModelDirSelection for PhoneModelDir {
    fn resolve(app_home: &AppHome, explicit: Option<&str>) -> Result<Self> {
        if let Some(path) = explicit {
            return Ok(Self::from(PathBuf::from(path)));
        }
        if let Some(path) = std::env::var_os("TEAMY_TRANSCRIBER_PHONE_MODEL_DIR") {
            return Ok(Self::from(PathBuf::from(path)));
        }
        if let Some(model_dir) = PhoneModelSelectionCodec::read(app_home)? {
            return Ok(model_dir);
        }
        find_cached_repository(REPOSITORY, REVISION).map(Self::from)
            .wrap_err("Phone model missing. Use hf download changelinglab/PhoneticXeus model.safetensors config.json ipa_vocab.json --revision 3a8d860fa68f8936ceb4196651221215bab9dae4; or pass --model-dir to phones / --phone-model-dir to transcribe or microphone transcribe")
    }
    fn select(&self, app_home: &AppHome) -> Result<Self> {
        let selected = Self::from(self.canonicalize()?);
        selected
            .load(CudaDeviceIndex::FIRST)
            .map_err(|error| eyre::eyre!("{error:#}"))?;
        PhoneModelSelectionCodec::write(app_home, &selected)?;
        Ok(selected)
    }
}
