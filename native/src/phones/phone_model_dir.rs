use super::PhoneModel;
use crate::CudaDeviceIndex;
use anyhow::Result;
use std::ops::Deref;
use std::path::Path;
use std::path::PathBuf;

/// Directory containing a PhoneticXeus checkpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhoneModelDir(PathBuf);

impl From<PathBuf> for PhoneModelDir {
    fn from(path: PathBuf) -> Self {
        Self(path)
    }
}

impl Deref for PhoneModelDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl PhoneModelDir {
    /// Load validated PhoneticXeus weights onto the requested CUDA device.
    pub fn load(&self, device: CudaDeviceIndex) -> Result<PhoneModel> {
        PhoneModel::load(self, device)
    }
}
