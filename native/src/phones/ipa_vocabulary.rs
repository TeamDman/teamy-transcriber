use super::VOCAB;
use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use std::collections::HashMap;
use std::path::Path;

const CTC_BLANK: &str = "<blank>";

#[derive(Debug)]
pub(super) struct IpaVocabulary(HashMap<String, usize>);

impl IpaVocabulary {
    fn try_from_map(tokens: HashMap<String, usize>) -> Result<Self> {
        ensure!(tokens.len() == VOCAB, "unsupported phone vocabulary");
        ensure!(tokens.contains_key(CTC_BLANK), "missing CTC blank token");
        let mut seen = [false; VOCAB];
        for &id in tokens.values() {
            ensure!(id < VOCAB && !seen[id], "invalid phone token ID {id}");
            seen[id] = true;
        }
        Ok(Self(tokens))
    }

    pub(super) fn into_decoder_tokens(self) -> (Vec<String>, usize) {
        let blank = self.0[CTC_BLANK];
        let mut tokens = vec![String::new(); VOCAB];
        for (token, id) in self.0 {
            tokens[id] = token;
        }
        (tokens, blank)
    }
}

pub(super) struct IpaVocabularyCodec;

impl IpaVocabularyCodec {
    pub(super) fn read(path: &Path) -> Result<IpaVocabulary> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading IPA vocabulary {}", path.display()))?;
        Self::decode(&bytes).with_context(|| format!("decoding IPA vocabulary {}", path.display()))
    }

    fn decode(bytes: &[u8]) -> Result<IpaVocabulary> {
        IpaVocabulary::try_from_map(facet_json::from_slice(bytes)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> HashMap<String, usize> {
        (1..VOCAB)
            .map(|id| (format!("token-{id}"), id))
            .chain([(CTC_BLANK.to_owned(), 0)])
            .collect()
    }

    #[test]
    fn decodes_checkpoint_map_into_token_id_order() -> Result<()> {
        let json = facet_json::to_string(&fixture())?;
        let (tokens, blank) = IpaVocabularyCodec::decode(json.as_bytes())?.into_decoder_tokens();
        assert_eq!(blank, 0);
        assert_eq!(tokens[blank], CTC_BLANK);
        assert_eq!(tokens[1], "token-1");
        assert_eq!(tokens.len(), VOCAB);
        Ok(())
    }

    #[test]
    fn rejects_duplicate_and_out_of_range_ids() -> Result<()> {
        for id in [0, VOCAB] {
            let mut tokens = fixture();
            tokens.insert("token-1".to_owned(), id);
            let json = facet_json::to_string(&tokens)?;
            let error = IpaVocabularyCodec::decode(json.as_bytes()).unwrap_err();
            assert!(error.to_string().contains("invalid phone token ID"));
        }
        Ok(())
    }
}
