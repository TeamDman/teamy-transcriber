//! Metadata and tokenizer validation for source-defined CUDA Whisper.
use super::model::WhisperModelArtifacts;
use serde::Deserialize;
use serde::Serialize;
use std::path::Path;

/// Maximum generated tokens for one 30-second Whisper window.
pub const DEFAULT_MAX_DECODE_TOKENS: usize = 448;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WhisperDims {
    pub audio: AudioEncoderDims,
    pub text: TextDecoderDims,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioEncoderDims {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextDecoderDims {
    pub n_vocab: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
}

impl WhisperDims {
    #[must_use]
    pub fn render_lines(&self) -> Vec<String> {
        vec![
            format!("Audio encoder mel bins: {}", self.audio.n_mels),
            format!("Audio encoder context: {}", self.audio.n_audio_ctx),
            format!("Audio encoder state: {}", self.audio.n_audio_state),
            format!("Audio encoder heads: {}", self.audio.n_audio_head),
            format!("Audio encoder layers: {}", self.audio.n_audio_layer),
            format!("Text decoder vocab: {}", self.text.n_vocab),
            format!("Text decoder context: {}", self.text.n_text_ctx),
            format!("Text decoder state: {}", self.text.n_text_state),
            format!("Text decoder heads: {}", self.text.n_text_head),
            format!("Text decoder layers: {}", self.text.n_text_layer),
        ]
    }
}

pub fn default_decoder_prompt_token_ids(
    artifacts: &WhisperModelArtifacts,
) -> eyre::Result<Vec<usize>> {
    let tokenizer = load_tokenizer(artifacts)?;

    let mut tokens = Vec::new();
    for token in [
        "<|startoftranscript|>",
        "<|en|>",
        "<|transcribe|>",
        "<|notimestamps|>",
    ] {
        tokens.push(required_token_id(
            &tokenizer,
            &artifacts.tokenizer.path,
            token,
        )?);
    }

    Ok(tokens)
}

fn load_tokenizer(artifacts: &WhisperModelArtifacts) -> eyre::Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(&artifacts.tokenizer.path).map_err(|error| {
        eyre::eyre!(
            "Failed to reload tokenizer from {}: {}",
            artifacts.tokenizer.path.display(),
            error
        )
    })
}

fn required_token_id(
    tokenizer: &tokenizers::Tokenizer,
    tokenizer_path: &Path,
    token: &str,
) -> eyre::Result<usize> {
    tokenizer
        .token_to_id(token)
        .map(|token_id| token_id as usize)
        .ok_or_else(|| {
            eyre::eyre!(
                "Tokenizer {} did not contain required token {}",
                tokenizer_path.display(),
                token
            )
        })
}
