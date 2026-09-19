//! Direct Whisper inference from canonical Hugging Face safetensors.
//!
//! This is deliberately separate from the `TorchScript` runner.  `TorchScript` is
//! still a useful optimized artifact when one is available, but a canonical
//! Whisper checkpoint should not require a Python export step to become usable
//! by the Rust application.  The loader converts the safetensor views into
//! resident `LibTorch` tensors and executes the Whisper graph with `tch`.

use super::frontend::WhisperLogMelSpectrogram;
use super::model::WhisperModelArtifacts;
use super::whisper::WhisperDims;
use super::whisper::decode_token_ids;
use super::whisper::default_decoder_prompt_token_ids;
use super::whisper::default_suppressed_token_ids;
use eyre::Context;
use eyre::bail;
use half::bf16;
use half::f16;
use memmap2::Mmap;
use safetensors::tensor::Dtype;
use safetensors::tensor::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use tch::Device;
use tch::Kind;
use tch::Tensor;

const LAYER_NORM_EPS: f64 = 1e-5;

/// A resident direct-tch Whisper model loaded from `model.safetensors`.
#[derive(Debug)]
pub struct TchSafetensorsWhisperRuntime {
    weights: HashMap<String, Tensor>,
    dims: WhisperDims,
    device: Device,
    model_kind: Kind,
}

#[derive(Debug)]
struct DecoderLayerCache {
    self_key: Option<Tensor>,
    self_value: Option<Tensor>,
    cross_key: Tensor,
    cross_value: Tensor,
}

#[derive(Debug)]
struct DecoderCache {
    layers: Vec<DecoderLayerCache>,
    position: i64,
}

impl TchSafetensorsWhisperRuntime {
    /// Load and validate a canonical safetensors Whisper package.
    pub fn from_artifacts(artifacts: &WhisperModelArtifacts) -> eyre::Result<Self> {
        let model_paths = if artifacts.safetensors_paths.is_empty() {
            artifacts
                .safetensors_path
                .clone()
                .map_or_else(Vec::new, |path| vec![path])
        } else {
            artifacts.safetensors_paths.clone()
        };
        if model_paths.is_empty() {
            bail!("safetensors Whisper artifact has no weight files");
        }
        let dims = artifacts
            .dims
            .clone()
            .ok_or_else(|| eyre::eyre!("safetensors Whisper artifact is missing dims.json"))?;
        let device = super::tch::configured_device_for_runtime()?;
        validate_safetensors_files(&model_paths, &dims)?;
        let model_kind = model_kind_from_files(&model_paths)?;
        let mut weights = HashMap::new();
        for model_path in &model_paths {
            let file = File::open(model_path)
                .wrap_err_with(|| format!("failed to open {}", model_path.display()))?;
            // SAFETY: the file handle remains open for the lifetime of the mapping, and the
            // mapping is only used as an immutable byte slice while SafeTensors validates it.
            let mapped = unsafe { Mmap::map(&file) }
                .wrap_err_with(|| format!("failed to memory-map {}", model_path.display()))?;
            let tensors = SafeTensors::deserialize(&mapped)
                .wrap_err_with(|| format!("failed to parse {}", model_path.display()))?;
            for (name, view) in tensors.tensors() {
                let tensor = tensor_from_view(&view, model_kind, device)
                    .wrap_err_with(|| format!("failed to load safetensor {name}"))?;
                if weights.insert(name.clone(), tensor).is_some() {
                    bail!("safetensors weight {name} appears in more than one shard");
                }
            }
        }

        Ok(Self {
            weights,
            dims,
            device,
            model_kind,
        })
    }

    /// Execute the direct Rust/LibTorch graph and greedily decode one window.
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
        if features.n_mels != self.dims.audio.n_mels {
            bail!(
                "Whisper model expects {} mel bins, got {}",
                self.dims.audio.n_mels,
                features.n_mels
            );
        }
        let prompt = default_decoder_prompt_token_ids(artifacts)?;
        let end_of_text = super::whisper::special_token_id(artifacts, "<|endoftext|>")?;
        let suppressed = default_suppressed_token_ids(artifacts, end_of_text)?;
        let mel = Tensor::from_slice(&features.values)
            .reshape([1, features.n_mels as i64, features.n_frames as i64])
            .to_kind(self.model_kind)
            .to_device(self.device);
        let encoded = self.encode(&mel)?;
        let decode_limit =
            max_decode_tokens.min(self.dims.text.n_text_ctx.saturating_sub(prompt.len()));
        let mut generated = Vec::new();
        let prompt_values = prompt
            .iter()
            .map(|token| i64::try_from(*token))
            .collect::<Result<Vec<_>, _>>()
            .wrap_err("Whisper token ID exceeded i64")?;
        let prompt_tokens = Tensor::from_slice(&prompt_values)
            .reshape([1, prompt_values.len() as i64])
            .to_kind(Kind::Int64)
            .to_device(self.device);
        let mut cache = self.new_decoder_cache(&encoded)?;
        let mut logits = self.decode_cached(&prompt_tokens, &mut cache)?;
        for _ in 0..decode_limit {
            let (_batch_size, seq_len, vocab_size) = logits
                .size3()
                .wrap_err("direct tch Whisper decoder must return [batch, sequence, vocabulary]")?;
            if seq_len == 0 || vocab_size == 0 {
                bail!("direct tch Whisper decoder returned empty logits");
            }
            let final_logits = logits
                .select(1, seq_len - 1)
                .to_device(Device::Cpu)
                .to_kind(Kind::Float)
                .view([-1])
                .contiguous();
            let values = Vec::<f32>::try_from(final_logits)
                .wrap_err("failed to copy direct tch Whisper logits to the host")?;
            let next_token = greedy_next_token_id(&values, &suppressed)?;
            if next_token == end_of_text {
                break;
            }
            generated.push(next_token);
            let token = i64::try_from(next_token).wrap_err("Whisper token ID exceeded i64")?;
            let token = Tensor::from_slice(&[token])
                .reshape([1, 1])
                .to_kind(Kind::Int64)
                .to_device(self.device);
            logits = self.decode_cached(&token, &mut cache)?;
        }
        decode_token_ids(artifacts, &generated, true)
    }

    fn encode(&self, input: &Tensor) -> eyre::Result<Tensor> {
        let x = input
            .conv1d(
                self.required("model.encoder.conv1.weight")?,
                Some(self.required("model.encoder.conv1.bias")?),
                [1],
                [1],
                [1],
                1,
            )
            .gelu("none");
        let mut x = x
            .conv1d(
                self.required("model.encoder.conv2.weight")?,
                Some(self.required("model.encoder.conv2.bias")?),
                [2],
                [1],
                [1],
                1,
            )
            .gelu("none")
            .transpose(1, 2);
        let encoded_len = x.size()[1];
        let positions = self.required("model.encoder.embed_positions.weight")?;
        if encoded_len > positions.size()[0] {
            bail!(
                "audio encoder produced {encoded_len} positions but the model has only {}",
                positions.size()[0]
            );
        }
        x += positions.narrow(0, 0, encoded_len).unsqueeze(0);

        for index in 0..self.dims.audio.n_audio_layer {
            let prefix = format!("model.encoder.layers.{index}");
            let normalized = self.layer_norm(&x, &format!("{prefix}.self_attn_layer_norm"))?;
            let attended = self.attention(
                &normalized,
                &prefix,
                "self_attn",
                self.dims.audio.n_audio_head,
                None,
                false,
            )?;
            x += attended;

            let normalized = self.layer_norm(&x, &format!("{prefix}.final_layer_norm"))?;
            let feed_forward = self.feed_forward(&normalized, &prefix)?;
            x += feed_forward;
        }

        self.layer_norm(&x, "model.encoder.layer_norm")
    }

    #[cfg(test)]
    fn decode(&self, tokens: &Tensor, encoder_output: &Tensor) -> eyre::Result<Tensor> {
        let token_ids = tokens.reshape([-1]);
        let embedding = self
            .required("model.decoder.embed_tokens.weight")?
            .index_select(0, &token_ids)
            .reshape([1, tokens.size()[1], self.dims.text.n_text_state as i64]);
        let seq_len = tokens.size()[1];
        let positions = self.required("model.decoder.embed_positions.weight")?;
        if seq_len > positions.size()[0] {
            bail!(
                "decoder sequence has {seq_len} tokens but the model has only {} positions",
                positions.size()[0]
            );
        }
        let mut x = embedding + positions.narrow(0, 0, seq_len).unsqueeze(0);

        for index in 0..self.dims.text.n_text_layer {
            let prefix = format!("model.decoder.layers.{index}");
            let normalized = self.layer_norm(&x, &format!("{prefix}.self_attn_layer_norm"))?;
            let attended = self.attention(
                &normalized,
                &prefix,
                "self_attn",
                self.dims.text.n_text_head,
                None,
                true,
            )?;
            x += attended;

            let normalized = self.layer_norm(&x, &format!("{prefix}.encoder_attn_layer_norm"))?;
            let attended = self.attention(
                &normalized,
                &prefix,
                "encoder_attn",
                self.dims.text.n_text_head,
                Some(encoder_output),
                false,
            )?;
            x += attended;

            let normalized = self.layer_norm(&x, &format!("{prefix}.final_layer_norm"))?;
            let feed_forward = self.feed_forward(&normalized, &prefix)?;
            x += feed_forward;
        }

        let x = self.layer_norm(&x, "model.decoder.layer_norm")?;
        let output_weight = self
            .weights
            .get("proj_out.weight")
            .or_else(|| self.weights.get("model.decoder.embed_tokens.weight"))
            .ok_or_else(|| {
                eyre::eyre!("Whisper model is missing proj_out.weight and tied token embeddings")
            })?;
        Ok(x.linear(output_weight, Option::<&Tensor>::None))
    }

    fn new_decoder_cache(&self, encoder_output: &Tensor) -> eyre::Result<DecoderCache> {
        let layers = (0..self.dims.text.n_text_layer)
            .map(|index| {
                let prefix = format!("model.decoder.layers.{index}.encoder_attn");
                Ok(DecoderLayerCache {
                    self_key: None,
                    self_value: None,
                    cross_key: self.linear(encoder_output, &format!("{prefix}.k_proj"))?,
                    cross_value: self.linear(encoder_output, &format!("{prefix}.v_proj"))?,
                })
            })
            .collect::<eyre::Result<Vec<_>>>()?;
        Ok(DecoderCache {
            layers,
            position: 0,
        })
    }

    fn decode_cached(&self, tokens: &Tensor, cache: &mut DecoderCache) -> eyre::Result<Tensor> {
        let token_ids = tokens.reshape([-1]);
        let seq_len = tokens.size()[1];
        let positions = self.required("model.decoder.embed_positions.weight")?;
        if cache.position + seq_len > positions.size()[0] {
            bail!(
                "decoder sequence position {} exceeded the model's {} positions",
                cache.position + seq_len,
                positions.size()[0]
            );
        }
        let embedding = self
            .required("model.decoder.embed_tokens.weight")?
            .index_select(0, &token_ids)
            .reshape([1, seq_len, self.dims.text.n_text_state as i64]);
        let mut x = embedding + positions.narrow(0, cache.position, seq_len).unsqueeze(0);
        let position = cache.position;

        for index in 0..self.dims.text.n_text_layer {
            let prefix = format!("model.decoder.layers.{index}");
            let normalized = self.layer_norm(&x, &format!("{prefix}.self_attn_layer_norm"))?;
            let attended = self.cached_self_attention(
                &normalized,
                &prefix,
                self.dims.text.n_text_head,
                &mut cache.layers[index],
                position,
            )?;
            x += attended;

            let normalized = self.layer_norm(&x, &format!("{prefix}.encoder_attn_layer_norm"))?;
            let attended = self.cached_cross_attention(
                &normalized,
                &prefix,
                self.dims.text.n_text_head,
                &cache.layers[index],
            )?;
            x += attended;

            let normalized = self.layer_norm(&x, &format!("{prefix}.final_layer_norm"))?;
            x += self.feed_forward(&normalized, &prefix)?;
        }
        cache.position += seq_len;

        let x = self.layer_norm(&x, "model.decoder.layer_norm")?;
        let output_weight = self
            .weights
            .get("proj_out.weight")
            .or_else(|| self.weights.get("model.decoder.embed_tokens.weight"))
            .ok_or_else(|| {
                eyre::eyre!("Whisper model is missing proj_out.weight and tied token embeddings")
            })?;
        Ok(x.linear(output_weight, Option::<&Tensor>::None))
    }

    fn cached_self_attention(
        &self,
        input: &Tensor,
        block_prefix: &str,
        n_heads: usize,
        cache: &mut DecoderLayerCache,
        position: i64,
    ) -> eyre::Result<Tensor> {
        let prefix = format!("{block_prefix}.self_attn");
        let query = self.linear(input, &format!("{prefix}.q_proj"))?;
        let new_key = self.linear(input, &format!("{prefix}.k_proj"))?;
        let new_value = self.linear(input, &format!("{prefix}.v_proj"))?;
        let key = match cache.self_key.as_ref() {
            Some(previous) => Tensor::cat(&[previous, &new_key], 1),
            None => new_key,
        };
        let value = match cache.self_value.as_ref() {
            Some(previous) => Tensor::cat(&[previous, &new_value], 1),
            None => new_value,
        };
        cache.self_key = Some(key.shallow_clone());
        cache.self_value = Some(value.shallow_clone());
        let context = self.attention_from_qkv(&query, &key, &value, n_heads, true, position)?;
        Ok(context.linear(
            self.required(&format!("{prefix}.out_proj.weight"))?,
            Some(self.required(&format!("{prefix}.out_proj.bias"))?),
        ))
    }

    fn cached_cross_attention(
        &self,
        input: &Tensor,
        block_prefix: &str,
        n_heads: usize,
        cache: &DecoderLayerCache,
    ) -> eyre::Result<Tensor> {
        let prefix = format!("{block_prefix}.encoder_attn");
        let query = self.linear(input, &format!("{prefix}.q_proj"))?;
        let context = self.attention_from_qkv(
            &query,
            &cache.cross_key,
            &cache.cross_value,
            n_heads,
            false,
            0,
        )?;
        Ok(context.linear(
            self.required(&format!("{prefix}.out_proj.weight"))?,
            Some(self.required(&format!("{prefix}.out_proj.bias"))?),
        ))
    }

    fn attention_from_qkv(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        n_heads: usize,
        causal: bool,
        past_length: i64,
    ) -> eyre::Result<Tensor> {
        let query_len = query.size()[1];
        let source_len = key.size()[1];
        let state = query.size()[2];
        let n_heads = i64::try_from(n_heads).wrap_err("Whisper head count exceeded i64")?;
        if n_heads == 0 || state % n_heads != 0 {
            bail!("Whisper attention state {state} is not divisible by head count {n_heads}");
        }
        let head_state = state / n_heads;
        let scale = (head_state as f64).powf(-0.25);
        let query = query
            .reshape([1, query_len, n_heads, head_state])
            .transpose(1, 2)
            * scale;
        let key = key
            .reshape([1, source_len, n_heads, head_state])
            .transpose(1, 2)
            .transpose(2, 3)
            * scale;
        let value = value
            .reshape([1, source_len, n_heads, head_state])
            .transpose(1, 2);
        let mut scores = query.matmul(&key);
        if causal {
            let mask = Tensor::ones([query_len, source_len], (Kind::Bool, self.device))
                .triu(past_length + 1);
            scores = scores.masked_fill(&mask, f64::NEG_INFINITY);
        }
        Ok(scores
            .softmax(-1, None)
            .matmul(&value)
            .transpose(1, 2)
            .reshape([1, query_len, state]))
    }

    fn attention(
        &self,
        input: &Tensor,
        block_prefix: &str,
        attention_name: &str,
        n_heads: usize,
        key_value_source: Option<&Tensor>,
        causal: bool,
    ) -> eyre::Result<Tensor> {
        let prefix = format!("{block_prefix}.{attention_name}");
        let source = key_value_source.unwrap_or(input);
        let query = self.linear(input, &format!("{prefix}.q_proj"))?;
        let key = self.linear(source, &format!("{prefix}.k_proj"))?;
        let value = self.linear(source, &format!("{prefix}.v_proj"))?;
        let query_len = query.size()[1];
        let source_len = key.size()[1];
        let state = query.size()[2];
        let n_heads = i64::try_from(n_heads).wrap_err("Whisper head count exceeded i64")?;
        if n_heads == 0 || state % n_heads != 0 {
            bail!("Whisper attention state {state} is not divisible by head count {n_heads}");
        }
        let head_state = state / n_heads;
        let scale = (head_state as f64).powf(-0.25);
        let query = query
            .reshape([1, query_len, n_heads, head_state])
            .transpose(1, 2)
            * scale;
        let key = key
            .reshape([1, source_len, n_heads, head_state])
            .transpose(1, 2)
            .transpose(2, 3)
            * scale;
        let value = value
            .reshape([1, source_len, n_heads, head_state])
            .transpose(1, 2);
        let mut scores = query.matmul(&key);
        if causal {
            let mask = Tensor::ones([query_len, source_len], (Kind::Bool, self.device)).triu(1);
            scores = scores.masked_fill(&mask, f64::NEG_INFINITY);
        }
        let context = scores
            .softmax(-1, None)
            .matmul(&value)
            .transpose(1, 2)
            .reshape([1, query_len, state]);
        Ok(context.linear(
            self.required(&format!("{prefix}.out_proj.weight"))?,
            Some(self.required(&format!("{prefix}.out_proj.bias"))?),
        ))
    }

    fn feed_forward(&self, input: &Tensor, block_prefix: &str) -> eyre::Result<Tensor> {
        let first = input.linear(
            self.required(&format!("{block_prefix}.fc1.weight"))?,
            Some(self.required(&format!("{block_prefix}.fc1.bias"))?),
        );
        let first = first.gelu("none");
        Ok(first.linear(
            self.required(&format!("{block_prefix}.fc2.weight"))?,
            Some(self.required(&format!("{block_prefix}.fc2.bias"))?),
        ))
    }

    fn layer_norm(&self, input: &Tensor, prefix: &str) -> eyre::Result<Tensor> {
        let input_size = input.size();
        Ok(input.layer_norm(
            [input_size[input_size.len() - 1]],
            Some(self.required(&format!("{prefix}.weight"))?),
            Some(self.required(&format!("{prefix}.bias"))?),
            LAYER_NORM_EPS,
            true,
        ))
    }

    fn linear(&self, input: &Tensor, prefix: &str) -> eyre::Result<Tensor> {
        let bias = self.weights.get(&format!("{prefix}.bias"));
        Ok(input.linear(self.required(&format!("{prefix}.weight"))?, bias))
    }

    fn required(&self, name: &str) -> eyre::Result<&Tensor> {
        self.weights
            .get(name)
            .ok_or_else(|| eyre::eyre!("Whisper safetensors model is missing {name}"))
    }
}

pub use super::safetensors_manifest::validate_safetensors_file;
pub use super::safetensors_manifest::validate_safetensors_files;
fn model_kind_from_files(paths: &[PathBuf]) -> eyre::Result<Kind> {
    for path in paths {
        let file =
            File::open(path).wrap_err_with(|| format!("failed to open {}", path.display()))?;
        // SAFETY: the file handle remains open for the lifetime of the mapping, and the mapping
        // is accessed only immutably by SafeTensors.
        let mapped = unsafe { Mmap::map(&file) }
            .wrap_err_with(|| format!("failed to memory-map {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&mapped)
            .wrap_err_with(|| format!("failed to parse {}", path.display()))?;
        if let Ok(view) = tensors.tensor("model.encoder.conv1.weight") {
            return kind_for_dtype(view.dtype());
        }
    }
    bail!("safetensors model is missing model.encoder.conv1.weight")
}

fn kind_for_dtype(dtype: Dtype) -> eyre::Result<Kind> {
    match dtype {
        Dtype::F32 => Ok(Kind::Float),
        Dtype::F16 => Ok(Kind::Half),
        Dtype::BF16 => Ok(Kind::BFloat16),
        other => bail!(
            "direct tch Whisper currently accepts only F32, F16, or BF16 safetensors; found {other}"
        ),
    }
}

fn tensor_from_view(
    view: &safetensors::tensor::TensorView<'_>,
    kind: Kind,
    device: Device,
) -> eyre::Result<Tensor> {
    let values = match view.dtype() {
        Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|bytes| {
                f32::from_le_bytes(bytes.try_into().expect("chunks_exact gives four bytes"))
            })
            .collect::<Vec<_>>(),
        Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|bytes| {
                f16::from_le_bytes(bytes.try_into().expect("chunks_exact gives two bytes")).to_f32()
            })
            .collect::<Vec<_>>(),
        Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|bytes| {
                bf16::from_le_bytes(bytes.try_into().expect("chunks_exact gives two bytes"))
                    .to_f32()
            })
            .collect::<Vec<_>>(),
        other => bail!("unsupported safetensor dtype {other}"),
    };
    let shape = view
        .shape()
        .iter()
        .map(|dimension| i64::try_from(*dimension))
        .collect::<Result<Vec<_>, _>>()
        .wrap_err("safetensor dimension exceeded i64")?;
    Ok(Tensor::from_slice(&values)
        .reshape(shape)
        .to_kind(kind)
        .to_device(device))
}

fn greedy_next_token_id(values: &[f32], suppressed_token_ids: &[usize]) -> eyre::Result<usize> {
    if values.is_empty() {
        bail!("direct tch Whisper decoder returned no vocabulary logits");
    }
    values
        .iter()
        .copied()
        .enumerate()
        .filter(|(token_id, _)| !suppressed_token_ids.contains(token_id))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(token_id, _)| token_id)
        .ok_or_else(|| eyre::eyre!("all Whisper vocabulary tokens were suppressed"))
}

#[cfg(test)]
mod tests {
    use super::TchSafetensorsWhisperRuntime;
    use super::greedy_next_token_id;
    use crate::native_whisper::whisper::AudioEncoderDims;
    use crate::native_whisper::whisper::TextDecoderDims;
    use crate::native_whisper::whisper::WhisperDims;
    use std::collections::HashMap;
    use tch::Device;
    use tch::Kind;
    use tch::Tensor;

    fn synthetic_runtime() -> TchSafetensorsWhisperRuntime {
        let dims = WhisperDims {
            audio: AudioEncoderDims {
                n_mels: 2,
                n_audio_ctx: 2,
                n_audio_state: 4,
                n_audio_head: 2,
                n_audio_layer: 1,
            },
            text: TextDecoderDims {
                n_vocab: 7,
                n_text_ctx: 4,
                n_text_state: 4,
                n_text_head: 2,
                n_text_layer: 1,
            },
        };
        let mut weights = HashMap::new();
        let mut add = |name: &str, shape: &[i64]| {
            let element_count: i64 = shape.iter().product();
            weights.insert(
                name.to_string(),
                Tensor::arange(element_count, (Kind::Float, Device::Cpu)).reshape(shape) / 100.0,
            );
        };
        add("model.encoder.conv1.weight", &[4, 2, 3]);
        add("model.encoder.conv1.bias", &[4]);
        add("model.encoder.conv2.weight", &[4, 4, 3]);
        add("model.encoder.conv2.bias", &[4]);
        add("model.encoder.embed_positions.weight", &[2, 4]);
        add("model.encoder.layer_norm.weight", &[4]);
        add("model.encoder.layer_norm.bias", &[4]);
        add("model.decoder.embed_tokens.weight", &[7, 4]);
        add("model.decoder.embed_positions.weight", &[4, 4]);
        add("model.decoder.layer_norm.weight", &[4]);
        add("model.decoder.layer_norm.bias", &[4]);
        for prefix in ["model.encoder.layers.0", "model.decoder.layers.0"] {
            for attention in ["self_attn"] {
                add(&format!("{prefix}.{attention}.q_proj.weight"), &[4, 4]);
                add(&format!("{prefix}.{attention}.k_proj.weight"), &[4, 4]);
                add(&format!("{prefix}.{attention}.v_proj.weight"), &[4, 4]);
                add(&format!("{prefix}.{attention}.out_proj.weight"), &[4, 4]);
                add(&format!("{prefix}.{attention}.out_proj.bias"), &[4]);
            }
            add(&format!("{prefix}.self_attn_layer_norm.weight"), &[4]);
            add(&format!("{prefix}.self_attn_layer_norm.bias"), &[4]);
            add(&format!("{prefix}.fc1.weight"), &[16, 4]);
            add(&format!("{prefix}.fc1.bias"), &[16]);
            add(&format!("{prefix}.fc2.weight"), &[4, 16]);
            add(&format!("{prefix}.fc2.bias"), &[4]);
            add(&format!("{prefix}.final_layer_norm.weight"), &[4]);
            add(&format!("{prefix}.final_layer_norm.bias"), &[4]);
        }
        for name in [
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "out_proj.weight",
        ] {
            let name = format!("model.decoder.layers.0.encoder_attn.{name}");
            add(&name, &[4, 4]);
        }
        add("model.decoder.layers.0.encoder_attn.out_proj.bias", &[4]);
        add(
            "model.decoder.layers.0.encoder_attn_layer_norm.weight",
            &[4],
        );
        add("model.decoder.layers.0.encoder_attn_layer_norm.bias", &[4]);
        TchSafetensorsWhisperRuntime {
            weights,
            dims,
            device: Device::Cpu,
            model_kind: Kind::Float,
        }
    }

    #[test]
    fn greedy_logits_honor_suppression() {
        assert_eq!(greedy_next_token_id(&[10.0, 20.0, 30.0], &[2]).unwrap(), 1);
    }

    #[test]
    fn synthetic_graph_has_whisper_encoder_and_decoder_shapes() {
        let runtime = synthetic_runtime();
        let input = Tensor::zeros([1, 2, 4], (Kind::Float, Device::Cpu));
        let encoded = runtime.encode(&input).unwrap();
        assert_eq!(encoded.size(), [1, 2, 4]);
        let tokens = Tensor::from_slice(&[1_i64, 2]).reshape([1, 2]);
        let logits = runtime.decode(&tokens, &encoded).unwrap();
        assert_eq!(logits.size(), [1, 2, 7]);
    }

    #[test]
    fn cached_decoder_matches_full_prefix_decoder() {
        let runtime = synthetic_runtime();
        let input = Tensor::zeros([1, 2, 4], (Kind::Float, Device::Cpu));
        let encoded = runtime.encode(&input).unwrap();
        let prompt = Tensor::from_slice(&[1_i64, 2]).reshape([1, 2]);
        let full_prompt = runtime.decode(&prompt, &encoded).unwrap();
        let mut cache = runtime.new_decoder_cache(&encoded).unwrap();
        let cached_prompt = runtime.decode_cached(&prompt, &mut cache).unwrap();
        assert!(full_prompt.allclose(&cached_prompt, 1e-5, 1e-5, false));

        let full_prefix = Tensor::from_slice(&[1_i64, 2, 3]).reshape([1, 3]);
        let full_next = runtime.decode(&full_prefix, &encoded).unwrap().select(1, 2);
        let cached_next = runtime
            .decode_cached(&Tensor::from_slice(&[3_i64]).reshape([1, 1]), &mut cache)
            .unwrap()
            .select(1, 0);
        assert!(full_next.allclose(&cached_next, 1e-5, 1e-5, false));
    }
}
