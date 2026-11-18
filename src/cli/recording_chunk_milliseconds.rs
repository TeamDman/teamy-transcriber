use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use facet::Facet;

/// A fixed recording transcription window in milliseconds.
#[derive(Clone, Copy, Debug, Facet, PartialEq)]
#[facet(opaque, proxy = u64)]
pub struct RecordingChunkMilliseconds(u64);

impl RecordingChunkMilliseconds {
    const MAX: u64 = 30_000;

    pub(super) const fn microseconds(self) -> u64 {
        self.0 * 1_000
    }
}

impl TryFrom<u64> for RecordingChunkMilliseconds {
    type Error = &'static str;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        if (1..=Self::MAX).contains(&value) {
            Ok(Self(value))
        } else {
            Err("recording chunk duration must be between 1 and 30000 milliseconds")
        }
    }
}

impl From<&RecordingChunkMilliseconds> for u64 {
    fn from(value: &RecordingChunkMilliseconds) -> Self {
        value.0
    }
}

impl<'a> Arbitrary<'a> for RecordingChunkMilliseconds {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        u.int_in_range(1..=Self::MAX).map(Self)
    }
}
