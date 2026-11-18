#[derive(Clone, Copy)]
pub(super) struct TranscriptText<'a>(&'a str);

impl<'a> TranscriptText<'a> {
    pub(super) const fn new(text: &'a str) -> Self {
        Self(text)
    }

    pub(super) const fn as_str(&self) -> &'a str {
        self.0
    }
}
