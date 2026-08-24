//! `TorchScript` Whisper execution through the pinned tch/LibTorch runtime.
//!
//! Sharing the runtime with `teamy-tts` does not make `GLaDOS` weights a
//! transcription model. This runner expects an exported Whisper graph with
//! `encoder` and `decoder` methods, plus the tokenizer and dimensions
//! sidecars in the project model package.

use super::frontend::WhisperLogMelSpectrogram;
use super::model::WhisperModelArtifacts;
use super::whisper::decode_token_ids;
use super::whisper::default_decoder_prompt_token_ids;
use super::whisper::default_suppressed_token_ids;
use crate::paths::TORCH_DEVICE_ENV_VAR;
use eyre::Context;
use eyre::bail;
use tch::CModule;
use tch::Device;
use tch::Kind;
use tch::Tensor;

#[derive(Debug)]
pub struct TchWhisperRuntime {
    model: CModule,
    device: Device,
}

impl TchWhisperRuntime {
    /// Load a `TorchScript` Whisper graph on the configured device.
    pub fn from_artifacts(artifacts: &WhisperModelArtifacts) -> eyre::Result<Self> {
        let model_path = artifacts
            .torchscript_path
            .as_deref()
            .ok_or_else(|| eyre::eyre!("TorchScript Whisper artifact is missing model.pt"))?;
        let device = configured_device()?;
        let mut model = CModule::load_on_device(model_path, device).wrap_err_with(|| {
            format!(
                "failed to load TorchScript Whisper model {} on {device:?}",
                model_path.display()
            )
        })?;
        model
            .f_set_eval()
            .wrap_err("failed to put TorchScript Whisper model into evaluation mode")?;
        Ok(Self { model, device })
    }

    /// Run the exported encoder/decoder graph using the existing Rust frontend.
    pub fn greedy_decode(
        &self,
        artifacts: &WhisperModelArtifacts,
        features: &WhisperLogMelSpectrogram,
        max_decode_tokens: usize,
    ) -> eyre::Result<String> {
        tch::no_grad(|| self.greedy_decode_without_grad(artifacts, features, max_decode_tokens))
    }

    fn greedy_decode_without_grad(
        &self,
        artifacts: &WhisperModelArtifacts,
        features: &WhisperLogMelSpectrogram,
        max_decode_tokens: usize,
    ) -> eyre::Result<String> {
        if max_decode_tokens == 0 {
            bail!("max decode tokens must be greater than zero");
        }
        let prompt = default_decoder_prompt_token_ids(artifacts)?;
        let end_of_text = super::whisper::special_token_id(artifacts, "<|endoftext|>")?;
        let suppressed = default_suppressed_token_ids(artifacts, end_of_text)?;
        let mel = Tensor::from_slice(&features.values)
            .reshape([1, features.n_mels as i64, features.n_frames as i64])
            .to_device(self.device);
        let encoded = self
            .model
            .method_ts("encoder", &[mel])
            .wrap_err("TorchScript Whisper graph must expose an encoder method")?;
        let n_text_ctx = artifacts
            .dims
            .as_ref()
            .map_or(max_decode_tokens + prompt.len(), |dims| {
                dims.text.n_text_ctx
            });
        let decode_limit = max_decode_tokens.min(n_text_ctx.saturating_sub(prompt.len()));
        let mut all_tokens = prompt;
        let mut generated = Vec::new();
        for _ in 0..decode_limit {
            let token_values = all_tokens
                .iter()
                .map(|token| i64::try_from(*token))
                .collect::<Result<Vec<_>, _>>()
                .wrap_err("Whisper token ID exceeded i64")?;
            let tokens = Tensor::from_slice(&token_values)
                .reshape([1, token_values.len() as i64])
                .to_kind(Kind::Int64)
                .to_device(self.device);
            let logits = self
                .model
                .method_ts("decoder", &[tokens, encoded.shallow_clone()])
                .wrap_err("TorchScript Whisper graph must expose a decoder method")?;
            let (batch_size, seq_len, vocab_size) = logits
                .size3()
                .wrap_err("TorchScript Whisper decoder must return [batch, sequence, vocab]")?;
            if batch_size != 1 || seq_len == 0 || vocab_size == 0 {
                bail!(
                    "TorchScript Whisper decoder returned invalid logits shape [{batch_size}, {seq_len}, {vocab_size}]"
                );
            }
            let final_logits = logits
                .select(1, seq_len - 1)
                .to_device(Device::Cpu)
                .to_kind(Kind::Float)
                .contiguous();
            let values = Vec::<f32>::try_from(final_logits)
                .wrap_err("failed to copy TorchScript Whisper logits to the host")?;
            let next_token = greedy_next_token_id_from_values(&values, &suppressed)?;
            if next_token == end_of_text {
                break;
            }
            all_tokens.push(next_token);
            generated.push(next_token);
        }
        decode_token_ids(artifacts, &generated, true)
    }
}

fn configured_device() -> eyre::Result<Device> {
    let device_index = std::env::var(TORCH_DEVICE_ENV_VAR)
        .ok()
        .map(|value| {
            value.parse::<i32>().wrap_err_with(|| {
                format!("{TORCH_DEVICE_ENV_VAR} must be an integer, found {value:?}")
            })
        })
        .transpose()?
        .unwrap_or(0);
    if device_index < 0 {
        return Ok(Device::Cpu);
    }
    let device_index = usize::try_from(device_index).wrap_err("invalid CUDA device index")?;
    if !tch::Cuda::is_available() {
        bail!("CUDA is unavailable to tch/LibTorch; use {TORCH_DEVICE_ENV_VAR}=-1 for CPU");
    }
    let count = tch::Cuda::device_count();
    if i64::try_from(device_index).is_ok_and(|index| index >= count) {
        bail!("requested CUDA device {device_index}, but LibTorch reports {count} device(s)");
    }
    Ok(Device::Cuda(device_index))
}

fn greedy_next_token_id_from_values(
    values: &[f32],
    suppressed_token_ids: &[usize],
) -> eyre::Result<usize> {
    if values.is_empty() {
        bail!("TorchScript Whisper decoder returned no vocabulary logits");
    }
    let (token_id, _) = values
        .iter()
        .copied()
        .enumerate()
        .filter(|(token_id, _)| !suppressed_token_ids.contains(token_id))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or_else(|| eyre::eyre!("all TorchScript Whisper vocabulary tokens were suppressed"))?;
    Ok(token_id)
}

#[cfg(test)]
mod tests {
    use super::greedy_next_token_id_from_values;

    #[test]
    fn greedy_logits_honor_suppression() {
        assert_eq!(
            greedy_next_token_id_from_values(&[10.0, 20.0, 30.0], &[2]).unwrap(),
            1
        );
    }
}
