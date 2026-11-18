use anyhow::Result;
use anyhow::ensure;
use std::str::FromStr;

/// A nonnegative CUDA device ordinal.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaDeviceIndex(i32);

impl CudaDeviceIndex {
    /// The first device visible to CUDA.
    pub const FIRST: Self = Self(0);

    pub(crate) const fn get(self) -> i32 {
        self.0
    }
}

impl TryFrom<i32> for CudaDeviceIndex {
    type Error = anyhow::Error;

    fn try_from(value: i32) -> Result<Self> {
        ensure!(value >= 0, "CUDA device index must be nonnegative");
        Ok(Self(value))
    }
}

impl FromStr for CudaDeviceIndex {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Self::try_from(s.parse::<i32>()?)
    }
}

impl std::fmt::Display for CudaDeviceIndex {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}
