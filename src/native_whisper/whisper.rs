use super::frontend::WhisperLogMelSpectrogram;
use super::model::WhisperModelArtifacts;
use burn::backend::NdArray;
use burn::config::Config;
use burn::module::Ignored;
use burn::module::Module;
use burn::module::Param;
use burn::nn::Embedding;
use burn::nn::EmbeddingConfig;
use burn::nn::PaddingConfig1d;
use burn::nn::conv::Conv1d;
use burn::nn::conv::Conv1dConfig;
use burn::nn::{self};
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::tensor::TensorData;
use burn::tensor::activation::softmax;
use burn::tensor::backend::Backend;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;
use eyre::WrapErr;
use eyre::bail;
use npy::NpyData;
use serde::Deserialize;
use serde::Serialize;
use std::io::Read;
use std::path::Path;

pub type WhisperCpuBackend = NdArray<f32>;
pub const DEFAULT_MAX_DECODE_TOKENS: usize = 64;
const RUST_ONLY_REPEAT_TOKEN_LIMIT: usize = 4;

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

#[derive(Config, Debug)]
pub struct WhisperAudioEncoderConfig {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
}

impl WhisperAudioEncoderConfig {
    #[must_use]
    pub fn from_dims(dims: &AudioEncoderDims) -> Self {
        Self::new(
            dims.n_mels,
            dims.n_audio_ctx,
            dims.n_audio_state,
            dims.n_audio_head,
            dims.n_audio_layer,
        )
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> WhisperAudioEncoder<B> {
        let conv1 = Conv1dConfig::new(self.n_mels, self.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .init(device);
        let conv2 = Conv1dConfig::new(self.n_audio_state, self.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_stride(2)
            .init(device);
        let blocks = (0..self.n_audio_layer)
            .map(|_| {
                ResidualEncoderAttentionBlockConfig::new(self.n_audio_state, self.n_audio_head)
                    .init(device)
            })
            .collect();
        let ln_post = nn::LayerNormConfig::new(self.n_audio_state).init(device);
        let positional_embedding = Param::from_data(
            TensorData::zeros::<f32, _>([self.n_audio_ctx, self.n_audio_state]),
            device,
        );

        WhisperAudioEncoder {
            conv1,
            gelu1: nn::Gelu::new(),
            conv2,
            gelu2: nn::Gelu::new(),
            blocks,
            ln_post,
            positional_embedding,
            n_mels: self.n_mels,
            n_audio_ctx: self.n_audio_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct WhisperAudioEncoder<B: Backend> {
    pub conv1: Conv1d<B>,
    pub gelu1: nn::Gelu,
    pub conv2: Conv1d<B>,
    pub gelu2: nn::Gelu,
    pub blocks: Vec<ResidualEncoderAttentionBlock<B>>,
    pub ln_post: nn::LayerNorm<B>,
    pub positional_embedding: Param<Tensor<B, 2>>,
    pub n_mels: usize,
    pub n_audio_ctx: usize,
}

impl<B: Backend> WhisperAudioEncoder<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, n_mels, n_ctx] = input.dims();
        assert_eq!(
            n_mels, self.n_mels,
            "audio encoder expected {} mel bins",
            self.n_mels
        );
        assert!(
            n_ctx <= self.n_audio_ctx * 2,
            "audio encoder input context {} exceeded supported range for {} positional embeddings",
            n_ctx,
            self.n_audio_ctx
        );

        let x = self.gelu1.forward(self.conv1.forward(input));
        let x = self.gelu2.forward(self.conv2.forward(x));
        let x = x.swap_dims(1, 2);
        let encoded_ctx = x.dims()[1];
        let mut x = x + self
            .positional_embedding
            .val()
            .slice([0..encoded_ctx])
            .unsqueeze::<3>();

        for block in &self.blocks {
            x = block.forward(x);
        }

        self.ln_post.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct WhisperTextDecoderConfig {
    pub n_vocab: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
}

impl WhisperTextDecoderConfig {
    #[must_use]
    pub fn from_dims(dims: &TextDecoderDims) -> Self {
        Self::new(
            dims.n_vocab,
            dims.n_text_ctx,
            dims.n_text_state,
            dims.n_text_head,
            dims.n_text_layer,
        )
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> WhisperTextDecoder<B> {
        let token_embedding = EmbeddingConfig::new(self.n_vocab, self.n_text_state).init(device);
        let positional_embedding = Param::from_data(
            TensorData::zeros::<f32, _>([self.n_text_ctx, self.n_text_state]),
            device,
        );
        let blocks = (0..self.n_text_layer)
            .map(|_| {
                ResidualDecoderAttentionBlockConfig::new(self.n_text_state, self.n_text_head)
                    .init(device)
            })
            .collect();
        let ln = nn::LayerNormConfig::new(self.n_text_state).init(device);
        let mask = Param::from_tensor(attn_decoder_mask::<B>(self.n_text_ctx, device));

        WhisperTextDecoder {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
            n_vocab: self.n_vocab,
            n_text_ctx: self.n_text_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct WhisperTextDecoder<B: Backend> {
    pub token_embedding: Embedding<B>,
    pub positional_embedding: Param<Tensor<B, 2>>,
    pub blocks: Vec<ResidualDecoderAttentionBlock<B>>,
    pub ln: nn::LayerNorm<B>,
    pub mask: Param<Tensor<B, 2>>,
    pub n_vocab: usize,
    pub n_text_ctx: usize,
}

impl<B: Backend> WhisperTextDecoder<B> {
    pub fn forward(&self, tokens: Tensor<B, 2, Int>, encoder_output: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_batch, seq_len] = tokens.dims();
        assert!(
            seq_len <= self.n_text_ctx,
            "token sequence length {} exceeded decoder context {}",
            seq_len,
            self.n_text_ctx
        );

        let mut x = self.token_embedding.forward(tokens)
            + self
                .positional_embedding
                .val()
                .slice([0..seq_len])
                .unsqueeze::<3>();

        for block in &self.blocks {
            x = block.forward(x, encoder_output.clone(), self.mask.val());
        }

        let x = self.ln.forward(x);
        x.matmul(
            self.token_embedding
                .weight
                .val()
                .transpose()
                .unsqueeze::<3>(),
        )
    }
}

#[derive(Config, Debug)]
pub struct WhisperModelConfig {
    pub audio: WhisperAudioEncoderConfig,
    pub text: WhisperTextDecoderConfig,
}

impl WhisperModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> WhisperModel<B> {
        WhisperModel {
            encoder: self.audio.init(device),
            decoder: self.text.init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct WhisperModel<B: Backend> {
    pub encoder: WhisperAudioEncoder<B>,
    pub decoder: WhisperTextDecoder<B>,
}

impl<B: Backend> WhisperModel<B> {
    pub fn forward(&self, features: Tensor<B, 3>, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let encoder_output = self.encoder.forward(features);
        self.decoder.forward(tokens, encoder_output)
    }
}

#[derive(Config, Debug)]
pub struct ResidualEncoderAttentionBlockConfig {
    pub n_state: usize,
    pub n_head: usize,
}

impl ResidualEncoderAttentionBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualEncoderAttentionBlock<B> {
        ResidualEncoderAttentionBlock {
            attn: MultiHeadSelfAttentionConfig::new(self.n_state, self.n_head).init(device),
            attn_ln: nn::LayerNormConfig::new(self.n_state).init(device),
            mlp: MlpConfig::new(self.n_state).init(device),
            mlp_ln: nn::LayerNormConfig::new(self.n_state).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualEncoderAttentionBlock<B: Backend> {
    pub attn: MultiHeadSelfAttention<B>,
    pub attn_ln: nn::LayerNorm<B>,
    pub mlp: Mlp<B>,
    pub mlp_ln: nn::LayerNorm<B>,
}

impl<B: Backend> ResidualEncoderAttentionBlock<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let input = input.clone() + self.attn.forward(self.attn_ln.forward(input), None);
        input.clone() + self.mlp.forward(self.mlp_ln.forward(input))
    }
}

#[derive(Config, Debug)]
pub struct ResidualDecoderAttentionBlockConfig {
    pub n_state: usize,
    pub n_head: usize,
}

impl ResidualDecoderAttentionBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualDecoderAttentionBlock<B> {
        ResidualDecoderAttentionBlock {
            attn: MultiHeadSelfAttentionConfig::new(self.n_state, self.n_head).init(device),
            attn_ln: nn::LayerNormConfig::new(self.n_state).init(device),
            cross_attn: MultiHeadCrossAttentionConfig::new(self.n_state, self.n_head).init(device),
            cross_attn_ln: nn::LayerNormConfig::new(self.n_state).init(device),
            mlp: MlpConfig::new(self.n_state).init(device),
            mlp_ln: nn::LayerNormConfig::new(self.n_state).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualDecoderAttentionBlock<B: Backend> {
    pub attn: MultiHeadSelfAttention<B>,
    pub attn_ln: nn::LayerNorm<B>,
    pub cross_attn: MultiHeadCrossAttention<B>,
    pub cross_attn_ln: nn::LayerNorm<B>,
    pub mlp: Mlp<B>,
    pub mlp_ln: nn::LayerNorm<B>,
}

impl<B: Backend> ResidualDecoderAttentionBlock<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        encoder_output: Tensor<B, 3>,
        mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let input = input.clone() + self.attn.forward(self.attn_ln.forward(input), Some(mask));
        let input = input.clone()
            + self
                .cross_attn
                .forward(self.cross_attn_ln.forward(input), encoder_output);
        input.clone() + self.mlp.forward(self.mlp_ln.forward(input))
    }
}

#[derive(Config, Debug)]
pub struct MlpConfig {
    pub n_state: usize,
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        Mlp {
            lin1: nn::LinearConfig::new(self.n_state, 4 * self.n_state).init(device),
            gelu: nn::Gelu::new(),
            lin2: nn::LinearConfig::new(4 * self.n_state, self.n_state).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    pub lin1: nn::Linear<B>,
    pub gelu: nn::Gelu,
    pub lin2: nn::Linear<B>,
}

impl<B: Backend> Mlp<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let input = self.lin1.forward(input);
        let input = self.gelu.forward(input);
        self.lin2.forward(input)
    }
}

#[derive(Config, Debug)]
pub struct MultiHeadSelfAttentionConfig {
    pub n_state: usize,
    pub n_head: usize,
}

impl MultiHeadSelfAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadSelfAttention<B> {
        assert_eq!(
            self.n_state % self.n_head,
            0,
            "state size must be divisible by head count"
        );

        MultiHeadSelfAttention {
            n_head: self.n_head,
            query: nn::LinearConfig::new(self.n_state, self.n_state).init(device),
            key: nn::LinearConfig::new(self.n_state, self.n_state)
                .with_bias(false)
                .init(device),
            value: nn::LinearConfig::new(self.n_state, self.n_state).init(device),
            out: nn::LinearConfig::new(self.n_state, self.n_state).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadSelfAttention<B: Backend> {
    pub n_head: usize,
    pub query: nn::Linear<B>,
    pub key: nn::Linear<B>,
    pub value: nn::Linear<B>,
    pub out: nn::Linear<B>,
}

impl<B: Backend> MultiHeadSelfAttention<B> {
    pub fn forward(&self, input: Tensor<B, 3>, mask: Option<Tensor<B, 2>>) -> Tensor<B, 3> {
        let q = self.query.forward(input.clone());
        let k = self.key.forward(input.clone());
        let v = self.value.forward(input);
        self.out.forward(qkv_attention(q, k, v, mask, self.n_head))
    }
}

#[derive(Config, Debug)]
pub struct MultiHeadCrossAttentionConfig {
    pub n_state: usize,
    pub n_head: usize,
}

impl MultiHeadCrossAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadCrossAttention<B> {
        assert_eq!(
            self.n_state % self.n_head,
            0,
            "state size must be divisible by head count"
        );

        MultiHeadCrossAttention {
            n_head: self.n_head,
            query: nn::LinearConfig::new(self.n_state, self.n_state).init(device),
            key: nn::LinearConfig::new(self.n_state, self.n_state)
                .with_bias(false)
                .init(device),
            value: nn::LinearConfig::new(self.n_state, self.n_state).init(device),
            out: nn::LinearConfig::new(self.n_state, self.n_state).init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadCrossAttention<B: Backend> {
    pub n_head: usize,
    pub query: nn::Linear<B>,
    pub key: nn::Linear<B>,
    pub value: nn::Linear<B>,
    pub out: nn::Linear<B>,
}

impl<B: Backend> MultiHeadCrossAttention<B> {
    pub fn forward(&self, input: Tensor<B, 3>, encoder_output: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = self.query.forward(input);
        let k = self.key.forward(encoder_output.clone());
        let v = self.value.forward(encoder_output);
        self.out.forward(qkv_attention(q, k, v, None, self.n_head))
    }
}

pub fn qkv_attention<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    mask: Option<Tensor<B, 2>>,
    n_head: usize,
) -> Tensor<B, 3> {
    let [n_batch, n_qctx, n_state] = q.dims();
    let [_batch, n_ctx, _state] = k.dims();
    let n_hstate = n_state / n_head;
    let scale = (n_state as f64 / n_head as f64).powf(-0.25);

    let q = q
        .reshape([n_batch, n_qctx, n_head, n_hstate])
        .swap_dims(1, 2)
        * scale;
    let k = k
        .reshape([n_batch, n_ctx, n_head, n_hstate])
        .swap_dims(1, 2)
        .swap_dims(2, 3)
        * scale;
    let v = v
        .reshape([n_batch, n_ctx, n_head, n_hstate])
        .swap_dims(1, 2);

    let scores = q.matmul(k);
    let scores = if let Some(mask) = mask {
        scores + mask.slice([0..n_qctx, 0..n_ctx]).unsqueeze::<4>()
    } else {
        scores
    };
    let weights = softmax(scores, 3);

    weights.matmul(v).swap_dims(1, 2).flatten(2, 3)
}

/// Infer Whisper dimensions from a whisper-burn-style model artifact directory.
///
/// # Errors
///
/// This function will return an error if required artifact files are missing or malformed.
pub fn infer_dims_from_artifacts(artifacts: &WhisperModelArtifacts) -> eyre::Result<WhisperDims> {
    let encoder_dir = artifacts.encoder_dir.as_ref().ok_or_else(|| {
        eyre::eyre!(
            "Legacy packed-artifact dim inference requires an encoder directory in {}",
            artifacts.root.display()
        )
    })?;
    let decoder_dir = artifacts.decoder_dir.as_ref().ok_or_else(|| {
        eyre::eyre!(
            "Legacy packed-artifact dim inference requires a decoder directory in {}",
            artifacts.root.display()
        )
    })?;

    let n_mels = read_packed_scalar_usize(&encoder_dir.join("n_mels.npy"))?;
    let n_audio_state = read_packed_scalar_usize(&encoder_dir.join("n_audio_state.npy"))?;
    let n_audio_head = read_packed_scalar_usize(&encoder_dir.join("block_0/attn/n_head.npy"))?;
    let n_audio_layer = count_block_dirs(encoder_dir)?;
    let [n_audio_ctx, positional_state] =
        read_packed_tensor_shape::<2>(&encoder_dir.join("positional_embedding.npy"))?;

    let [conv1_out, conv1_in, _conv1_kernel] =
        read_packed_tensor_shape::<3>(&encoder_dir.join("conv1/weight.npy"))?;
    let [conv2_out, conv2_in, _conv2_kernel] =
        read_packed_tensor_shape::<3>(&encoder_dir.join("conv2/weight.npy"))?;

    if conv1_in != n_mels {
        bail!(
            "Encoder conv1 weight expected {} mel bins but found {} in {}",
            n_mels,
            conv1_in,
            encoder_dir.join("conv1/weight.npy").display()
        );
    }
    if conv1_out != n_audio_state || conv2_in != n_audio_state || conv2_out != n_audio_state {
        bail!(
            "Encoder convolution state mismatch in {}",
            encoder_dir.display()
        );
    }
    if positional_state != n_audio_state {
        bail!(
            "Encoder positional embedding state {} did not match {} in {}",
            positional_state,
            n_audio_state,
            encoder_dir.join("positional_embedding.npy").display()
        );
    }

    let [n_vocab, token_state] =
        read_packed_tensor_shape::<2>(&decoder_dir.join("token_embedding/weight.npy"))?;
    let [n_text_ctx, n_text_state] =
        read_packed_tensor_shape::<2>(&decoder_dir.join("positional_embedding.npy"))?;
    let n_text_head = read_packed_scalar_usize(&decoder_dir.join("block_0/attn/n_head.npy"))?;
    let n_text_layer = count_block_dirs(decoder_dir)?;

    if token_state != n_text_state {
        bail!(
            "Decoder token embedding state {} did not match positional embedding state {} in {}",
            token_state,
            n_text_state,
            decoder_dir.display()
        );
    }

    Ok(WhisperDims {
        audio: AudioEncoderDims {
            n_mels,
            n_audio_ctx,
            n_audio_state,
            n_audio_head,
            n_audio_layer,
        },
        text: TextDecoderDims {
            n_vocab,
            n_text_ctx,
            n_text_state,
            n_text_head,
            n_text_layer,
        },
    })
}

pub fn load_audio_encoder_from_artifacts(
    artifacts: &WhisperModelArtifacts,
) -> eyre::Result<WhisperAudioEncoder<WhisperCpuBackend>> {
    if matches!(artifacts.layout, super::model::WhisperModelLayout::BurnPack) {
        return Ok(load_whisper_model_from_artifacts(artifacts)?.encoder);
    }

    let dims = artifacts
        .dims
        .as_ref()
        .ok_or_else(|| eyre::eyre!("Model artifacts did not include inferred Whisper dims"))?;
    let device = Default::default();
    let config = WhisperAudioEncoderConfig::from_dims(&dims.audio);
    let mut encoder = config.init::<WhisperCpuBackend>(&device);
    let root = artifacts.encoder_dir.as_ref().ok_or_else(|| {
        eyre::eyre!("Legacy packed-artifact encoder dir missing from model artifacts")
    })?;

    encoder.conv1 = load_conv1d::<WhisperCpuBackend>(
        &root.join("conv1"),
        Conv1dConfig::new(dims.audio.n_mels, dims.audio.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1)),
        &device,
    )?;
    encoder.conv2 = load_conv1d::<WhisperCpuBackend>(
        &root.join("conv2"),
        Conv1dConfig::new(dims.audio.n_audio_state, dims.audio.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_stride(2),
        &device,
    )?;
    encoder.ln_post = load_layer_norm::<WhisperCpuBackend>(&root.join("ln_post"), &device)?;
    encoder.positional_embedding =
        Param::from_tensor(load_packed_float_tensor::<WhisperCpuBackend, 2>(
            &root.join("positional_embedding.npy"),
            &device,
        )?);
    encoder.blocks = (0..dims.audio.n_audio_layer)
        .map(|index| {
            load_encoder_block::<WhisperCpuBackend>(&root.join(format!("block_{index}")), &device)
        })
        .collect::<eyre::Result<Vec<_>>>()?;

    Ok(encoder)
}

pub fn load_text_decoder_from_artifacts(
    artifacts: &WhisperModelArtifacts,
) -> eyre::Result<WhisperTextDecoder<WhisperCpuBackend>> {
    if matches!(artifacts.layout, super::model::WhisperModelLayout::BurnPack) {
        return Ok(load_whisper_model_from_artifacts(artifacts)?.decoder);
    }

    let dims = artifacts
        .dims
        .as_ref()
        .ok_or_else(|| eyre::eyre!("Model artifacts did not include inferred Whisper dims"))?;
    let device = Default::default();
    let config = WhisperTextDecoderConfig::from_dims(&dims.text);
    let mut decoder = config.init::<WhisperCpuBackend>(&device);
    let root = artifacts.decoder_dir.as_ref().ok_or_else(|| {
        eyre::eyre!("Legacy packed-artifact decoder dir missing from model artifacts")
    })?;

    decoder.token_embedding = Embedding {
        weight: Param::from_tensor(load_packed_float_tensor::<WhisperCpuBackend, 2>(
            &root.join("token_embedding/weight.npy"),
            &device,
        )?),
    };
    decoder.positional_embedding =
        Param::from_tensor(load_packed_float_tensor::<WhisperCpuBackend, 2>(
            &root.join("positional_embedding.npy"),
            &device,
        )?);
    decoder.ln = load_layer_norm::<WhisperCpuBackend>(&root.join("ln"), &device)?;
    decoder.blocks = (0..dims.text.n_text_layer)
        .map(|index| {
            load_decoder_block::<WhisperCpuBackend>(&root.join(format!("block_{index}")), &device)
        })
        .collect::<eyre::Result<Vec<_>>>()?;

    Ok(decoder)
}

pub fn load_whisper_model_from_artifacts(
    artifacts: &WhisperModelArtifacts,
) -> eyre::Result<WhisperModel<WhisperCpuBackend>> {
    if matches!(artifacts.layout, super::model::WhisperModelLayout::BurnPack) {
        let dims = artifacts
            .dims
            .as_ref()
            .ok_or_else(|| eyre::eyre!("Burnpack model artifacts did not include Whisper dims"))?;
        let config = WhisperModelConfig {
            audio: WhisperAudioEncoderConfig::from_dims(&dims.audio),
            text: WhisperTextDecoderConfig::from_dims(&dims.text),
        };
        let device = Default::default();
        let mut model = config.init::<WhisperCpuBackend>(&device);
        let burnpack_path = artifacts.burnpack_path.as_ref().ok_or_else(|| {
            eyre::eyre!("Burnpack model artifacts did not include a weights file path")
        })?;
        let mut store = BurnpackStore::from_file(burnpack_path).allow_partial(true);
        let result = model.load_from(&mut store).map_err(|error| {
            eyre::eyre!(
                "Failed to load Burnpack model weights from {}: {}",
                burnpack_path.display(),
                error
            )
        })?;
        if !result.errors.is_empty() {
            bail!(
                "Burnpack model load reported tensor errors for {}: {:?}",
                burnpack_path.display(),
                result.errors
            );
        }
        return Ok(model);
    }

    Ok(WhisperModel {
        encoder: load_audio_encoder_from_artifacts(artifacts)?,
        decoder: load_text_decoder_from_artifacts(artifacts)?,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PromptForwardPassSummary {
    pub prompt_token_ids: Vec<usize>,
    pub encoder_output_dims: [usize; 3],
    pub decoder_logits_dims: [usize; 3],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GreedyDecodeSummary {
    pub prompt_token_ids: Vec<usize>,
    pub generated_token_ids: Vec<usize>,
    pub encoder_output_dims: [usize; 3],
    pub last_decoder_logits_dims: [usize; 3],
    pub terminated_on_end_of_text: bool,
    pub stop_reason: DecodeStopReason,
    pub text: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeStopReason {
    EndOfText,
    MaxDecodeTokens,
    RepeatedTokenCollapse,
}

pub fn run_prompt_forward_pass(
    artifacts: &WhisperModelArtifacts,
    features: &WhisperLogMelSpectrogram,
) -> eyre::Result<PromptForwardPassSummary> {
    let model = load_whisper_model_from_artifacts(artifacts)?;
    let prompt_token_ids = default_decoder_prompt_token_ids(artifacts)?;
    let device = Default::default();
    let feature_tensor = features_to_tensor(features, &device);
    let token_tensor = token_ids_to_tensor(&prompt_token_ids, &device);

    let encoder_output = model.encoder.forward(feature_tensor);
    let encoder_output_dims = encoder_output.dims();
    let decoder_logits_dims = model.decoder.forward(token_tensor, encoder_output).dims();

    Ok(PromptForwardPassSummary {
        prompt_token_ids,
        encoder_output_dims,
        decoder_logits_dims,
    })
}

pub fn greedy_decode_with_model(
    artifacts: &WhisperModelArtifacts,
    features: &WhisperLogMelSpectrogram,
    max_decode_tokens: usize,
) -> eyre::Result<GreedyDecodeSummary> {
    let model = load_whisper_model_from_artifacts(artifacts)?;
    let prompt_token_ids = default_decoder_prompt_token_ids(artifacts)?;
    let end_of_text = special_token_id(artifacts, "<|endoftext|>")?;
    // WHISPERX-PARITY GUARD:
    // WhisperX forwards `suppress_tokens` into faster-whisper generation.
    // This Rust greedy loop mirrors that intent by masking obvious control tokens.
    let suppressed_token_ids = default_suppressed_token_ids(artifacts, end_of_text)?;
    let device = Default::default();
    let encoder_output = model.encoder.forward(features_to_tensor(features, &device));
    let encoder_output_dims = encoder_output.dims();

    let remaining_ctx = model
        .decoder
        .n_text_ctx
        .saturating_sub(prompt_token_ids.len());
    let decode_limit = max_decode_tokens.min(remaining_ctx);
    let mut all_token_ids = prompt_token_ids.clone();
    let mut generated_token_ids = Vec::new();
    let mut last_decoder_logits_dims = [1, prompt_token_ids.len(), model.decoder.n_vocab];
    let mut terminated_on_end_of_text = false;
    let mut stop_reason = DecodeStopReason::MaxDecodeTokens;

    for _ in 0..decode_limit {
        let decoder_logits = model.decoder.forward(
            token_ids_to_tensor(&all_token_ids, &device),
            encoder_output.clone(),
        );
        last_decoder_logits_dims = decoder_logits.dims();
        let next_token_id = greedy_next_token_id(&decoder_logits, &suppressed_token_ids)?;
        if next_token_id == end_of_text {
            terminated_on_end_of_text = true;
            stop_reason = DecodeStopReason::EndOfText;
            break;
        }

        all_token_ids.push(next_token_id);
        generated_token_ids.push(next_token_id);

        // RUST-ONLY GUARD:
        // WhisperX does not expose an equivalent explicit repeated-token cut-off in its
        // Python batching loop. This is a local safeguard for the current naive greedy path.
        if has_repeated_token_collapse(&generated_token_ids, RUST_ONLY_REPEAT_TOKEN_LIMIT) {
            stop_reason = DecodeStopReason::RepeatedTokenCollapse;
            break;
        }
    }

    let text = decode_token_ids(artifacts, &generated_token_ids, true)?;

    Ok(GreedyDecodeSummary {
        prompt_token_ids,
        generated_token_ids,
        encoder_output_dims,
        last_decoder_logits_dims,
        terminated_on_end_of_text,
        stop_reason,
        text,
    })
}

pub fn encode_features_with_model(
    artifacts: &WhisperModelArtifacts,
    features: &WhisperLogMelSpectrogram,
) -> eyre::Result<[usize; 3]> {
    let encoder = load_audio_encoder_from_artifacts(artifacts)?;
    let device = Default::default();
    let input = Tensor::<WhisperCpuBackend, 3>::from_data(
        TensorData::new(
            features.values.clone(),
            [1, features.n_mels, features.n_frames],
        ),
        &device,
    );
    let output = encoder.forward(input);
    Ok(output.dims())
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

pub fn decode_token_ids(
    artifacts: &WhisperModelArtifacts,
    token_ids: &[usize],
    skip_special_tokens: bool,
) -> eyre::Result<String> {
    let tokenizer = load_tokenizer(artifacts)?;
    let token_ids = token_ids
        .iter()
        .map(|token_id| u32::try_from(*token_id))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| eyre::eyre!("Token id exceeded u32 range during decode"))?;
    tokenizer
        .decode(&token_ids, skip_special_tokens)
        .map_err(|error| eyre::eyre!("Failed to decode token ids with tokenizer: {}", error))
}

pub fn special_token_id(artifacts: &WhisperModelArtifacts, token: &str) -> eyre::Result<usize> {
    let tokenizer = load_tokenizer(artifacts)?;
    required_token_id(&tokenizer, &artifacts.tokenizer.path, token)
}

fn default_suppressed_token_ids(
    artifacts: &WhisperModelArtifacts,
    end_of_text: usize,
) -> eyre::Result<Vec<usize>> {
    let tokenizer = load_tokenizer(artifacts)?;
    let tokenizer_path = &artifacts.tokenizer.path;
    let mut token_ids = Vec::new();

    // WHISPERX-PARITY GUARD:
    // These are control-style tokens that should not be re-emitted during plain text decoding.
    // WhisperX relies on faster-whisper suppression settings rather than a handwritten greedy loop.
    for token in [
        "<|startoftranscript|>",
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        "<|en|>",
    ] {
        if let Some(token_id) = tokenizer.token_to_id(token) {
            let token_id = token_id as usize;
            if token_id != end_of_text {
                token_ids.push(token_id);
            }
        } else {
            tracing::debug!(token = token, path = %tokenizer_path.display(), "Tokenizer missing optional suppressed token");
        }
    }

    token_ids.sort_unstable();
    token_ids.dedup();
    Ok(token_ids)
}

pub fn attn_decoder_mask<B: Backend>(seq_length: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut mask = Tensor::<B, 2>::zeros([seq_length, seq_length], device);

    for row in 0..seq_length.saturating_sub(1) {
        let values = Tensor::<B, 2>::zeros([1, seq_length - (row + 1)], device)
            .add_scalar(f64::NEG_INFINITY);
        mask = mask.slice_assign([row..row + 1, row + 1..seq_length], values);
    }

    mask
}

fn load_encoder_block<B: Backend>(
    root: &Path,
    device: &B::Device,
) -> eyre::Result<ResidualEncoderAttentionBlock<B>> {
    Ok(ResidualEncoderAttentionBlock {
        attn: load_multi_head_self_attention::<B>(&root.join("attn"), device)?,
        attn_ln: load_layer_norm::<B>(&root.join("attn_ln"), device)?,
        mlp: load_mlp::<B>(&root.join("mlp"), device)?,
        mlp_ln: load_layer_norm::<B>(&root.join("mlp_ln"), device)?,
    })
}

fn load_decoder_block<B: Backend>(
    root: &Path,
    device: &B::Device,
) -> eyre::Result<ResidualDecoderAttentionBlock<B>> {
    Ok(ResidualDecoderAttentionBlock {
        attn: load_multi_head_self_attention::<B>(&root.join("attn"), device)?,
        attn_ln: load_layer_norm::<B>(&root.join("attn_ln"), device)?,
        cross_attn: load_multi_head_cross_attention::<B>(&root.join("cross_attn"), device)?,
        cross_attn_ln: load_layer_norm::<B>(&root.join("cross_attn_ln"), device)?,
        mlp: load_mlp::<B>(&root.join("mlp"), device)?,
        mlp_ln: load_layer_norm::<B>(&root.join("mlp_ln"), device)?,
    })
}

fn load_multi_head_self_attention<B: Backend>(
    root: &Path,
    device: &B::Device,
) -> eyre::Result<MultiHeadSelfAttention<B>> {
    Ok(MultiHeadSelfAttention {
        n_head: read_packed_scalar_usize(&root.join("n_head.npy"))?,
        query: load_linear::<B>(&root.join("query"), device)?,
        key: load_linear_no_bias::<B>(&root.join("key"), device)?,
        value: load_linear::<B>(&root.join("value"), device)?,
        out: load_linear::<B>(&root.join("out"), device)?,
    })
}

fn load_mlp<B: Backend>(root: &Path, device: &B::Device) -> eyre::Result<Mlp<B>> {
    Ok(Mlp {
        lin1: load_linear::<B>(&root.join("mlp1"), device)?,
        gelu: nn::Gelu::new(),
        lin2: load_linear::<B>(&root.join("mlp2"), device)?,
    })
}

fn load_multi_head_cross_attention<B: Backend>(
    root: &Path,
    device: &B::Device,
) -> eyre::Result<MultiHeadCrossAttention<B>> {
    Ok(MultiHeadCrossAttention {
        n_head: read_packed_scalar_usize(&root.join("n_head.npy"))?,
        query: load_linear::<B>(&root.join("query"), device)?,
        key: load_linear_no_bias::<B>(&root.join("key"), device)?,
        value: load_linear::<B>(&root.join("value"), device)?,
        out: load_linear::<B>(&root.join("out"), device)?,
    })
}

fn load_linear<B: Backend>(root: &Path, device: &B::Device) -> eyre::Result<nn::Linear<B>> {
    Ok(nn::Linear {
        weight: Param::from_tensor(load_packed_float_tensor::<B, 2>(
            &root.join("weight.npy"),
            device,
        )?),
        bias: Some(Param::from_tensor(load_packed_float_tensor::<B, 1>(
            &root.join("bias.npy"),
            device,
        )?)),
    })
}

fn load_linear_no_bias<B: Backend>(root: &Path, device: &B::Device) -> eyre::Result<nn::Linear<B>> {
    Ok(nn::Linear {
        weight: Param::from_tensor(load_packed_float_tensor::<B, 2>(
            &root.join("weight.npy"),
            device,
        )?),
        bias: None,
    })
}

fn load_layer_norm<B: Backend>(root: &Path, device: &B::Device) -> eyre::Result<nn::LayerNorm<B>> {
    let gamma = load_packed_float_tensor::<B, 1>(&root.join("weight.npy"), device)?;
    let beta = load_packed_float_tensor::<B, 1>(&root.join("bias.npy"), device)?;
    let [d_model] = gamma.dims();
    let mut layer_norm = nn::LayerNormConfig::new(d_model)
        .with_epsilon(read_packed_scalar_f64(&root.join("eps.npy"))?)
        .init(device);
    layer_norm.gamma = Param::from_tensor(gamma);
    layer_norm.beta = Param::from_tensor(beta);
    Ok(layer_norm)
}

fn load_conv1d<B: Backend>(
    root: &Path,
    config: Conv1dConfig,
    device: &B::Device,
) -> eyre::Result<Conv1d<B>> {
    Ok(Conv1d {
        weight: Param::from_tensor(load_packed_float_tensor::<B, 3>(
            &root.join("weight.npy"),
            device,
        )?),
        bias: Some(Param::from_tensor(load_packed_float_tensor::<B, 1>(
            &root.join("bias.npy"),
            device,
        )?)),
        stride: config.stride,
        kernel_size: config.kernel_size,
        dilation: config.dilation,
        groups: config.groups,
        padding: Ignored(config.padding),
    })
}

fn count_block_dirs(root: &Path) -> eyre::Result<usize> {
    let count = std::fs::read_dir(root)
        .wrap_err_with(|| format!("Failed to read block directories under {}", root.display()))?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_ok_and(|file_type| file_type.is_dir()))
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("block_"))
        .count();

    if count == 0 {
        bail!("No block_* directories found under {}", root.display());
    }

    Ok(count)
}

fn read_packed_scalar_usize(path: &Path) -> eyre::Result<usize> {
    let [len] = read_packed_tensor_shape::<1>(path)?;
    if len != 1 {
        bail!(
            "Expected scalar-packed tensor in {} but found length {len}",
            path.display()
        );
    }

    let data = read_packed_tensor_data(path)?;
    Ok(data[1] as usize)
}

fn read_packed_scalar_f64(path: &Path) -> eyre::Result<f64> {
    let [len] = read_packed_tensor_shape::<1>(path)?;
    if len != 1 {
        bail!(
            "Expected scalar-packed tensor in {} but found length {len}",
            path.display()
        );
    }

    let data = read_packed_tensor_data(path)?;
    Ok(f64::from(data[1]))
}

fn read_packed_tensor_shape<const D: usize>(path: &Path) -> eyre::Result<[usize; D]> {
    let data = read_packed_tensor_data(path)?;
    if data.len() < D {
        bail!(
            "Tensor file {} did not contain enough prefix entries for rank {}",
            path.display(),
            D
        );
    }

    let mut shape = [0_usize; D];
    for (index, value) in data.iter().take(D).enumerate() {
        shape[index] = *value as usize;
    }
    Ok(shape)
}

fn read_packed_tensor_data(path: &Path) -> eyre::Result<Vec<f32>> {
    let mut buffer = Vec::new();
    std::fs::File::open(path)
        .wrap_err_with(|| format!("Failed to open tensor file {}", path.display()))?
        .read_to_end(&mut buffer)
        .wrap_err_with(|| format!("Failed to read tensor file {}", path.display()))?;

    let array = NpyData::<f32>::from_bytes(&buffer)
        .wrap_err_with(|| format!("Failed to parse numpy tensor file {}", path.display()))?;
    Ok(array.to_vec())
}

fn load_packed_float_tensor<B: Backend, const D: usize>(
    path: &Path,
    device: &B::Device,
) -> eyre::Result<Tensor<B, D>> {
    let data = read_packed_tensor_data(path)?;
    if data.len() < D {
        bail!(
            "Tensor file {} did not contain enough prefix entries for rank {}",
            path.display(),
            D
        );
    }

    let mut shape = [0_usize; D];
    for (index, value) in data.iter().take(D).enumerate() {
        shape[index] = *value as usize;
    }
    let values = data.into_iter().skip(D).collect::<Vec<_>>();

    Ok(Tensor::from_data(TensorData::new(values, shape), device))
}

#[cfg(test)]
mod tests {
    use super::AudioEncoderDims;
    use super::TextDecoderDims;
    use super::WhisperAudioEncoderConfig;
    use super::WhisperCpuBackend;
    use super::WhisperDims;
    use super::WhisperModelConfig;
    use super::WhisperTextDecoderConfig;
    use super::attn_decoder_mask;
    use super::greedy_next_token_id;
    use super::has_repeated_token_collapse;
    use super::run_prompt_forward_pass;
    use crate::native_whisper::frontend::WhisperLogMelSpectrogram;
    use burn::tensor::Int;
    use burn::tensor::Tensor;
    use burn::tensor::TensorData;
    use burn_store::BurnpackStore;
    use burn_store::ModuleSnapshot;
    use tokenizers::AddedToken;

    #[test]
    fn render_lines_is_stable() {
        let dims = WhisperDims {
            audio: AudioEncoderDims {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 384,
                n_audio_head: 6,
                n_audio_layer: 4,
            },
            text: TextDecoderDims {
                n_vocab: 51_865,
                n_text_ctx: 448,
                n_text_state: 384,
                n_text_head: 6,
                n_text_layer: 4,
            },
        };

        let lines = dims.render_lines();
        assert!(
            lines
                .iter()
                .any(|line| line.contains("Audio encoder mel bins: 80"))
        );
        assert!(
            lines
                .iter()
                .any(|line| line.contains("Text decoder vocab: 51865"))
        );
    }

    #[test]
    fn audio_encoder_forward_has_expected_shape() {
        let device = Default::default();
        let encoder =
            WhisperAudioEncoderConfig::new(80, 6, 4, 2, 1).init::<WhisperCpuBackend>(&device);
        let input = Tensor::<WhisperCpuBackend, 3>::from_data(
            TensorData::zeros::<f32, _>([1, 80, 12]),
            &device,
        );

        let output = encoder.forward(input);
        assert_eq!(output.dims(), [1, 6, 4]);
    }

    #[test]
    fn text_decoder_forward_has_expected_shape() {
        let device = Default::default();
        let decoder =
            WhisperTextDecoderConfig::new(32, 8, 4, 2, 1).init::<WhisperCpuBackend>(&device);
        let tokens = Tensor::<WhisperCpuBackend, 2, Int>::from_data(
            TensorData::new(vec![0_i64, 1, 2], [1, 3]),
            &device,
        );
        let encoder_output = Tensor::<WhisperCpuBackend, 3>::from_data(
            TensorData::zeros::<f32, _>([1, 6, 4]),
            &device,
        );

        let logits = decoder.forward(tokens, encoder_output);
        assert_eq!(logits.dims(), [1, 3, 32]);
    }

    #[test]
    fn decoder_mask_blocks_future_positions() {
        let device = Default::default();
        let mask = attn_decoder_mask::<WhisperCpuBackend>(4, &device);
        let values = mask.to_data().to_vec::<f32>().expect("mask should be f32");

        assert_eq!(values[0], 0.0);
        assert!(values[1].is_infinite() && values[1].is_sign_negative());
        assert!(values[2].is_infinite() && values[2].is_sign_negative());
        assert!(values[3].is_infinite() && values[3].is_sign_negative());
        assert_eq!(values[5], 0.0);
        assert_eq!(values[10], 0.0);
        assert_eq!(values[15], 0.0);
    }

    #[test]
    fn greedy_next_token_reads_last_decoder_step() {
        let device = Default::default();
        let logits = Tensor::<WhisperCpuBackend, 3>::from_data(
            TensorData::new(
                vec![
                    0.0_f32, 3.0, 1.0, // step 0
                    2.0, 1.0, 9.0, // step 1
                ],
                [1, 2, 3],
            ),
            &device,
        );

        let token_id = greedy_next_token_id(&logits, &[]).expect("argmax should succeed");
        assert_eq!(token_id, 2);
    }

    #[test]
    fn greedy_next_token_skips_suppressed_ids() {
        let device = Default::default();
        let logits = Tensor::<WhisperCpuBackend, 3>::from_data(
            TensorData::new(vec![0.0_f32, 9.0, 8.0], [1, 1, 3]),
            &device,
        );

        let token_id =
            greedy_next_token_id(&logits, &[1]).expect("suppressed argmax should succeed");
        assert_eq!(token_id, 2);
    }

    #[test]
    fn repeated_token_collapse_detects_flatline_tail() {
        assert!(has_repeated_token_collapse(&[5, 7, 7, 7, 7], 4));
        assert!(!has_repeated_token_collapse(&[5, 7, 7, 7], 4));
        assert!(!has_repeated_token_collapse(&[5, 7, 7, 8, 7], 4));
    }

    #[test]
    fn burnpack_model_round_trip_reaches_prompt_forward_pass() {
        let root = std::env::temp_dir().join(format!(
            "teamy-transcriber-native-whisper-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("model fixture directory should be creatable");

        let dims = WhisperDims {
            audio: AudioEncoderDims {
                n_mels: 80,
                n_audio_ctx: 1500,
                n_audio_state: 4,
                n_audio_head: 1,
                n_audio_layer: 0,
            },
            text: TextDecoderDims {
                n_vocab: 5,
                n_text_ctx: 8,
                n_text_state: 4,
                n_text_head: 1,
                n_text_layer: 0,
            },
        };
        std::fs::write(
            root.join(crate::native_whisper::model::MODEL_DIMS_FILE_NAME),
            serde_json::to_string(&dims).expect("dims should serialize"),
        )
        .expect("dims should write");

        let mut tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        tokenizer.add_special_tokens(&[
            AddedToken::from("<|startoftranscript|>", true),
            AddedToken::from("<|en|>", true),
            AddedToken::from("<|transcribe|>", true),
            AddedToken::from("<|notimestamps|>", true),
            AddedToken::from("<|endoftext|>", true),
        ]);
        tokenizer
            .save(
                root.join(crate::native_whisper::model::TOKENIZER_FILE_NAME),
                false,
            )
            .expect("tokenizer should write");

        let device = Default::default();
        let config = WhisperModelConfig {
            audio: WhisperAudioEncoderConfig::from_dims(&dims.audio),
            text: WhisperTextDecoderConfig::from_dims(&dims.text),
        };
        let model = config.init::<WhisperCpuBackend>(&device);
        let burnpack_path = root.join(crate::native_whisper::model::MODEL_BURNPACK_FILE_NAME);
        let mut store = BurnpackStore::from_file(&burnpack_path).overwrite(true);
        model
            .save_into(&mut store)
            .expect("tiny Burnpack model should save");

        let artifacts = crate::native_whisper::model::inspect_model_dir(&root)
            .expect("generated Burnpack model should inspect");
        let features = WhisperLogMelSpectrogram {
            values: vec![0.0; 80 * 3000],
            n_mels: 80,
            n_frames: 3000,
        };
        let summary = run_prompt_forward_pass(&artifacts, &features)
            .expect("generated Burnpack model should reach the decoder");
        assert_eq!(summary.prompt_token_ids.len(), 4);
        assert_eq!(summary.encoder_output_dims, [1, 1500, 4]);
        assert_eq!(summary.decoder_logits_dims, [1, 4, 5]);

        std::fs::remove_dir_all(root).expect("model fixture directory should be removable");
    }
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

fn features_to_tensor(
    features: &WhisperLogMelSpectrogram,
    device: &<WhisperCpuBackend as Backend>::Device,
) -> Tensor<WhisperCpuBackend, 3> {
    Tensor::<WhisperCpuBackend, 3>::from_data(
        TensorData::new(
            features.values.clone(),
            [1, features.n_mels, features.n_frames],
        ),
        device,
    )
}

fn token_ids_to_tensor(
    token_ids: &[usize],
    device: &<WhisperCpuBackend as Backend>::Device,
) -> Tensor<WhisperCpuBackend, 2, Int> {
    Tensor::<WhisperCpuBackend, 2, Int>::from_data(
        TensorData::new(
            token_ids
                .iter()
                .map(|token_id| i64::try_from(*token_id).unwrap_or(i64::MAX))
                .collect::<Vec<_>>(),
            [1, token_ids.len()],
        ),
        device,
    )
}

fn greedy_next_token_id(
    logits: &Tensor<WhisperCpuBackend, 3>,
    suppressed_token_ids: &[usize],
) -> eyre::Result<usize> {
    let [batch_size, seq_len, vocab_size] = logits.dims();
    if batch_size != 1 {
        bail!("Greedy decode currently expects batch size 1 but found {batch_size}");
    }
    if seq_len == 0 || vocab_size == 0 {
        bail!(
            "Greedy decode expected non-empty logits but found sequence length {} and vocab size {}",
            seq_len,
            vocab_size
        );
    }

    let values = logits
        .to_data()
        .to_vec::<f32>()
        .map_err(|error| eyre::eyre!("Failed to read decoder logits: {:?}", error))?;
    let last_step_offset = (seq_len - 1) * vocab_size;
    let last_step = &values[last_step_offset..last_step_offset + vocab_size];

    let (token_id, _value) = last_step
        .iter()
        .copied()
        .enumerate()
        .filter(|(token_id, _)| !suppressed_token_ids.contains(token_id))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or_else(|| eyre::eyre!("Decoder logits did not contain a final timestep"))?;
    Ok(token_id)
}

fn has_repeated_token_collapse(token_ids: &[usize], repeat_limit: usize) -> bool {
    if token_ids.len() < repeat_limit || repeat_limit < 2 {
        return false;
    }

    let tail = &token_ids[token_ids.len() - repeat_limit..];
    tail.windows(2).all(|window| window[0] == window[1])
}
