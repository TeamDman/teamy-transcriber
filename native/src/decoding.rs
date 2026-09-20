//! Token selection data for fixed English greedy decoding.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

/// The suppression fields of a canonical Whisper generation configuration.
/// Other generation fields (beam search, language, prompts, sampling, etc.)
/// are not interpreted here. Callers must select those features explicitly.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GreedySuppression {
    pub suppress_tokens: Vec<usize>,
    pub begin_suppress_tokens: Vec<usize>,
}

impl GreedySuppression {
    pub(crate) fn masks(
        &self,
        vocab: usize,
        no_timestamps: usize,
        first_timestamp: usize,
        last_timestamp: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        ensure!(
            no_timestamps < vocab && first_timestamp <= last_timestamp && last_timestamp < vocab,
            "invalid tokenizer timestamp boundaries"
        );
        ensure!(
            self.suppress_tokens
                .iter()
                .chain(&self.begin_suppress_tokens)
                .all(|&t| t < vocab),
            "suppressed token exceeds model vocabulary"
        );
        let mut allowed = vec![1.; vocab];
        for &token in &self.suppress_tokens {
            allowed[token] = 0.;
        }
        allowed[no_timestamps] = 0.;
        allowed[first_timestamp..=last_timestamp].fill(0.);
        let mut begin = allowed.clone();
        for &token in &self.begin_suppress_tokens {
            begin[token] = 0.;
        }
        ensure!(
            begin.contains(&1.),
            "suppression leaves no valid first token"
        );
        Ok((allowed, begin))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blank_and_end_suppression_only_apply_to_first_token() -> Result<()> {
        // Text 0..3, EOT 4, control 5, no-timestamps 6, timestamps 7..10.
        let policy = GreedySuppression {
            suppress_tokens: vec![5],
            begin_suppress_tokens: vec![0, 4],
        };
        let (normal, first) = policy.masks(11, 6, 7, 10)?;
        assert_eq!(normal, vec![1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.]);
        assert_eq!(first, vec![0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.]);
        let invalid = GreedySuppression {
            suppress_tokens: vec![11],
            begin_suppress_tokens: vec![],
        };
        assert!(invalid.masks(11, 6, 7, 10).is_err());
        assert!(policy.masks(11, 6, 10, 7).is_err());
        let empty = GreedySuppression {
            suppress_tokens: vec![0, 1, 2, 3, 4, 5],
            begin_suppress_tokens: vec![],
        };
        assert!(empty.masks(11, 6, 7, 10).is_err());
        Ok(())
    }
}
