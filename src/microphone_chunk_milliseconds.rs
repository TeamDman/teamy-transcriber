use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use facet::Facet;

/// An audio window supported by live microphone transcription.
#[derive(Clone, Copy, Debug, Facet, PartialEq)]
#[facet(opaque, proxy = u64)]
pub struct MicrophoneChunkMilliseconds(u64);

impl MicrophoneChunkMilliseconds {
    const MIN: u64 = 500;
    const MAX: u64 = 30_000;

    pub(crate) const fn get(self) -> u64 {
        self.0
    }
}

impl Default for MicrophoneChunkMilliseconds {
    fn default() -> Self {
        Self(5_000)
    }
}

impl TryFrom<u64> for MicrophoneChunkMilliseconds {
    type Error = &'static str;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        if (Self::MIN..=Self::MAX).contains(&value) {
            Ok(Self(value))
        } else {
            Err("microphone chunk duration must be between 500 and 30000 milliseconds")
        }
    }
}

impl From<&MicrophoneChunkMilliseconds> for u64 {
    fn from(value: &MicrophoneChunkMilliseconds) -> Self {
        value.get()
    }
}

impl<'a> Arbitrary<'a> for MicrophoneChunkMilliseconds {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        u.int_in_range(Self::MIN..=Self::MAX).map(Self)
    }
}
