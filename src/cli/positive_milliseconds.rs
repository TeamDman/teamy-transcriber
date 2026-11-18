use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use facet::Facet;
use std::num::NonZeroU64;
use std::time::Duration;

/// A strictly positive duration expressed in milliseconds.
#[derive(Clone, Copy, Debug, Facet, PartialEq)]
#[facet(opaque, proxy = u64)]
pub struct PositiveMilliseconds(NonZeroU64);

impl PositiveMilliseconds {
    pub(super) fn duration(self) -> Duration {
        Duration::from_millis(self.0.get())
    }
}

impl TryFrom<u64> for PositiveMilliseconds {
    type Error = &'static str;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        NonZeroU64::new(value)
            .map(Self)
            .ok_or("duration must be greater than zero")
    }
}

impl From<&PositiveMilliseconds> for u64 {
    fn from(value: &PositiveMilliseconds) -> Self {
        value.0.get()
    }
}

impl<'a> Arbitrary<'a> for PositiveMilliseconds {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        NonZeroU64::arbitrary(u).map(Self)
    }
}
