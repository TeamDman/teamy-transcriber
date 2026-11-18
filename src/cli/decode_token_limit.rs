use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use facet::Facet;
use std::num::NonZeroUsize;

/// The maximum number of tokens to decode, greater than zero.
#[derive(Clone, Copy, Debug, Facet, PartialEq)]
#[facet(opaque, proxy = usize)]
pub struct DecodeTokenLimit(NonZeroUsize);

impl DecodeTokenLimit {
    pub(super) const fn get(self) -> usize {
        self.0.get()
    }
}

impl TryFrom<usize> for DecodeTokenLimit {
    type Error = &'static str;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        NonZeroUsize::new(value)
            .map(Self)
            .ok_or("maximum decode tokens must be greater than zero")
    }
}

impl From<&DecodeTokenLimit> for usize {
    fn from(value: &DecodeTokenLimit) -> Self {
        value.get()
    }
}

impl<'a> Arbitrary<'a> for DecodeTokenLimit {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        NonZeroUsize::arbitrary(u).map(Self)
    }
}
