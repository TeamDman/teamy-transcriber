#[derive(Clone, Copy)]
pub(super) enum TranscriptStream {
    Unnumbered,
    Phonemes,
    Words,
}

impl TranscriptStream {
    pub(super) const fn prefix(self) -> &'static str {
        match self {
            Self::Unnumbered => "",
            Self::Phonemes => "1| ",
            Self::Words => "2| ",
        }
    }
}
