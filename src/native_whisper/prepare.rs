use super::model::MODEL_BURNPACK_FILE_NAME;
#[cfg(any(feature = "tch-native", feature = "cuda-native"))]
use super::model::MODEL_CONFIG_FILE_NAME;
use super::model::MODEL_DIMS_FILE_NAME;
#[cfg(any(feature = "tch-native", feature = "cuda-native"))]
use super::model::MODEL_SAFETENSORS_FILE_NAME;
#[cfg(any(feature = "tch-native", feature = "cuda-native"))]
use super::model::MODEL_SAFETENSORS_INDEX_FILE_NAME;
use super::model::TOKENIZER_FILE_NAME;
use super::model::WhisperModelArtifacts;
use super::model::inspect_model_dir;
#[cfg(any(feature = "tch-native", feature = "cuda-native"))]
use super::model::resolve_safetensor_paths;
use super::whisper::AudioEncoderDims;
use super::whisper::TextDecoderDims;
use super::whisper::WhisperAudioEncoderConfig;
use super::whisper::WhisperCpuBackend;
use super::whisper::WhisperDims;
use super::whisper::WhisperModelConfig;
use super::whisper::WhisperTextDecoderConfig;
use super::whisper::load_whisper_model_from_artifacts;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;
use burn_store::PytorchStore;
use burn_store::pytorch::PytorchReader;
use eyre::WrapErr;
use eyre::bail;
use serde::Deserialize;
use std::path::Path;
use std::path::PathBuf;

#[derive(Clone, Debug, Deserialize)]
struct CheckpointDims {
    n_mels: usize,
    n_vocab: usize,
    n_audio_ctx: usize,
    n_audio_state: usize,
    n_audio_head: usize,
    n_audio_layer: usize,
    n_text_ctx: usize,
    n_text_state: usize,
    n_text_head: usize,
    n_text_layer: usize,
}

#[cfg(any(feature = "tch-native", feature = "cuda-native"))]
#[derive(Clone, Debug, Deserialize)]
struct HuggingFaceWhisperConfig {
    num_mel_bins: usize,
    max_source_positions: usize,
    d_model: usize,
    encoder_attention_heads: usize,
    encoder_layers: usize,
    decoder_attention_heads: usize,
    decoder_layers: usize,
    vocab_size: usize,
    max_target_positions: usize,
}

#[cfg(any(feature = "tch-native", feature = "cuda-native"))]
impl HuggingFaceWhisperConfig {
    fn into_whisper_dims(self) -> WhisperDims {
        WhisperDims {
            audio: AudioEncoderDims {
                n_mels: self.num_mel_bins,
                n_audio_ctx: self.max_source_positions,
                n_audio_state: self.d_model,
                n_audio_head: self.encoder_attention_heads,
                n_audio_layer: self.encoder_layers,
            },
            text: TextDecoderDims {
                n_vocab: self.vocab_size,
                n_text_ctx: self.max_target_positions,
                n_text_state: self.d_model,
                n_text_head: self.decoder_attention_heads,
                n_text_layer: self.decoder_layers,
            },
        }
    }
}

pub(crate) struct PartialModelDirectory {
    path: PathBuf,
    committed: bool,
}

impl PartialModelDirectory {
    pub(crate) fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            committed: false,
        }
    }

    pub(crate) fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for PartialModelDirectory {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

impl CheckpointDims {
    fn into_whisper_dims(self) -> WhisperDims {
        WhisperDims {
            audio: AudioEncoderDims {
                n_mels: self.n_mels,
                n_audio_ctx: self.n_audio_ctx,
                n_audio_state: self.n_audio_state,
                n_audio_head: self.n_audio_head,
                n_audio_layer: self.n_audio_layer,
            },
            text: TextDecoderDims {
                n_vocab: self.n_vocab,
                n_text_ctx: self.n_text_ctx,
                n_text_state: self.n_text_state,
                n_text_head: self.n_text_head,
                n_text_layer: self.n_text_layer,
            },
        }
    }
}

/// Convert a local `PyTorch` Whisper checkpoint into the native model directory.
///
/// The checkpoint must contain the Whisper dimensions under `dims` and weights
/// under either `model_state_dict` or `state_dict`, compatible with the Burn
/// Whisper module. The tokenizer is copied explicitly so this operation never
/// downloads model assets.
///
/// # Errors
///
/// Returns an error when the checkpoint, tokenizer, weights, or output layout
/// cannot be read or written.
#[expect(
    clippy::too_many_lines,
    reason = "Checkpoint conversion keeps the validated import and package layout together"
)]
pub fn convert_pytorch_checkpoint(
    checkpoint: &Path,
    tokenizer: &Path,
    output_dir: &Path,
) -> eyre::Result<WhisperModelArtifacts> {
    if !checkpoint.is_file() {
        bail!(
            "PyTorch Whisper checkpoint is missing: {}",
            checkpoint.display()
        );
    }
    if !tokenizer.is_file() {
        bail!("Whisper tokenizer is missing: {}", tokenizer.display());
    }
    tokenizers::Tokenizer::from_file(tokenizer).map_err(|error| {
        eyre::eyre!(
            "failed to read Whisper tokenizer {}: {}",
            tokenizer.display(),
            error
        )
    })?;
    if output_dir.exists() {
        bail!(
            "refusing to overwrite an existing model directory: {}",
            output_dir.display()
        );
    }
    let dims = PytorchReader::load_config::<CheckpointDims, _>(checkpoint, Some("dims"))
        .wrap_err_with(|| {
            format!(
                "failed to read Whisper dimensions from {}",
                checkpoint.display()
            )
        })?
        .into_whisper_dims();
    let config = WhisperModelConfig {
        audio: WhisperAudioEncoderConfig::from_dims(&dims.audio),
        text: WhisperTextDecoderConfig::from_dims(&dims.text),
    };
    let mut import_errors = Vec::new();
    let mut model = None;
    for top_level_key in ["model_state_dict", "state_dict"] {
        let device = Default::default();
        let mut candidate = config.init::<WhisperCpuBackend>(&device);
        let mut store = checkpoint_store(checkpoint, top_level_key);
        match candidate.load_from(&mut store) {
            Ok(load_result) => {
                let allowed_missing = ["decoder.mask"];
                let unexpected_missing = load_result
                    .missing
                    .iter()
                    .filter(|path| !allowed_missing.iter().any(|allowed| path == allowed))
                    .cloned()
                    .collect::<Vec<_>>();
                if !unexpected_missing.is_empty() {
                    import_errors.push(format!(
                        "{top_level_key}: unexpected missing tensors {unexpected_missing:?}"
                    ));
                    continue;
                }
                if !load_result.unused.is_empty() {
                    import_errors.push(format!(
                        "{top_level_key}: unused tensors {:?}",
                        load_result.unused
                    ));
                    continue;
                }
                model = Some(candidate);
                break;
            }
            Err(error) => import_errors.push(format!("{top_level_key}: {error}")),
        }
    }
    let Some(model) = model else {
        bail!(
            "failed to import Whisper checkpoint {} using model_state_dict or state_dict: {:?}",
            checkpoint.display(),
            import_errors
        );
    };

    std::fs::create_dir(output_dir)
        .wrap_err_with(|| format!("failed to create model directory {}", output_dir.display()))?;
    let mut partial_output = PartialModelDirectory::new(output_dir);
    let dims_path = output_dir.join(MODEL_DIMS_FILE_NAME);
    std::fs::write(
        &dims_path,
        serde_json::to_string_pretty(&dims).wrap_err("failed to serialize Whisper dimensions")?,
    )
    .wrap_err_with(|| format!("failed to write {}", dims_path.display()))?;
    let tokenizer_path = output_dir.join(TOKENIZER_FILE_NAME);
    std::fs::copy(tokenizer, &tokenizer_path).wrap_err_with(|| {
        format!(
            "failed to copy tokenizer {} to {}",
            tokenizer.display(),
            tokenizer_path.display()
        )
    })?;
    let burnpack_path = output_dir.join(MODEL_BURNPACK_FILE_NAME);
    let mut burnpack = BurnpackStore::from_file(&burnpack_path)
        .overwrite(true)
        .metadata("whisper.audio.n_mels", dims.audio.n_mels.to_string())
        .metadata(
            "whisper.audio.n_audio_ctx",
            dims.audio.n_audio_ctx.to_string(),
        )
        .metadata(
            "whisper.audio.n_audio_state",
            dims.audio.n_audio_state.to_string(),
        )
        .metadata(
            "whisper.audio.n_audio_head",
            dims.audio.n_audio_head.to_string(),
        )
        .metadata(
            "whisper.audio.n_audio_layer",
            dims.audio.n_audio_layer.to_string(),
        )
        .metadata("whisper.text.n_vocab", dims.text.n_vocab.to_string())
        .metadata("whisper.text.n_text_ctx", dims.text.n_text_ctx.to_string())
        .metadata(
            "whisper.text.n_text_state",
            dims.text.n_text_state.to_string(),
        )
        .metadata(
            "whisper.text.n_text_head",
            dims.text.n_text_head.to_string(),
        )
        .metadata(
            "whisper.text.n_text_layer",
            dims.text.n_text_layer.to_string(),
        );
    model
        .save_into(&mut burnpack)
        .wrap_err_with(|| format!("failed to write {}", burnpack_path.display()))?;
    let artifacts = inspect_model_dir(output_dir)?;
    let _ = load_whisper_model_from_artifacts(&artifacts)
        .wrap_err("native Burnpack validation failed after model preparation")?;
    partial_output.commit();
    Ok(artifacts)
}

/// Prepare a canonical Hugging Face Whisper directory for direct Rust/tch use.
///
/// This operation is local-only. It copies the existing safetensors file(s),
/// tokenizer, and config into the application package and writes the compact
/// dims.json sidecar consumed by the runtime. It intentionally does not
/// convert `CTranslate2` `model.bin` files: those are a different runtime format
/// and cannot be losslessly loaded by `LibTorch`.
#[cfg(any(feature = "tch-native", feature = "cuda-native"))]
pub fn prepare_safetensors_model(
    source_dir: &Path,
    output_dir: &Path,
) -> eyre::Result<WhisperModelArtifacts> {
    if !source_dir.is_dir() {
        bail!(
            "canonical Whisper source directory is missing: {}",
            source_dir.display()
        );
    }
    let config_path = source_dir.join(MODEL_CONFIG_FILE_NAME);
    let tokenizer_path = source_dir.join(TOKENIZER_FILE_NAME);
    let has_single_weights = source_dir.join(MODEL_SAFETENSORS_FILE_NAME).is_file();
    let has_shard_index = source_dir.join(MODEL_SAFETENSORS_INDEX_FILE_NAME).is_file();
    if !has_single_weights && !has_shard_index && source_dir.join("model.bin").is_file() {
        bail!(
            "{} is a CTranslate2/faster-whisper model.bin directory; it cannot be loaded or losslessly converted by the tch/LibTorch Whisper backend. Supply canonical model.safetensors weights instead",
            source_dir.display()
        );
    }
    let model_paths = resolve_safetensor_paths(source_dir)?;
    for path in [&config_path, &tokenizer_path] {
        if !path.is_file() {
            bail!("canonical Whisper source is missing {}", path.display());
        }
    }
    let config: HuggingFaceWhisperConfig =
        serde_json::from_str(&std::fs::read_to_string(&config_path).wrap_err_with(|| {
            format!(
                "failed to read Hugging Face Whisper config {}",
                config_path.display()
            )
        })?)
        .wrap_err_with(|| {
            format!(
                "failed to parse Hugging Face Whisper config {}",
                config_path.display()
            )
        })?;
    let dims = config.into_whisper_dims();
    if dims.audio.n_audio_state == 0
        || dims.audio.n_audio_head == 0
        || dims.text.n_text_state == 0
        || dims.text.n_text_head == 0
    {
        bail!("Hugging Face Whisper config contains zero-sized model dimensions");
    }
    if output_dir.exists() {
        bail!(
            "refusing to overwrite an existing model directory: {}",
            output_dir.display()
        );
    }

    // Parse the safetensor header and verify the graph manifest before making
    // the package visible. The weights themselves are memory-mapped rather
    // than copied into a temporary buffer.
    super::safetensors_manifest::validate_safetensors_files(&model_paths, &dims)?;

    std::fs::create_dir(output_dir)
        .wrap_err_with(|| format!("failed to create model directory {}", output_dir.display()))?;
    let mut partial_output = PartialModelDirectory::new(output_dir);
    for model_path in &model_paths {
        let relative_path = model_path.strip_prefix(source_dir).wrap_err_with(|| {
            format!(
                "safetensor shard {} is outside source directory {}",
                model_path.display(),
                source_dir.display()
            )
        })?;
        copy_model_file(model_path, &output_dir.join(relative_path))?;
    }
    let shard_index = source_dir.join(MODEL_SAFETENSORS_INDEX_FILE_NAME);
    if shard_index.is_file() {
        copy_model_file(
            &shard_index,
            &output_dir.join(MODEL_SAFETENSORS_INDEX_FILE_NAME),
        )?;
    }
    copy_model_file(&config_path, &output_dir.join(MODEL_CONFIG_FILE_NAME))?;
    copy_model_file(&tokenizer_path, &output_dir.join(TOKENIZER_FILE_NAME))?;
    let generation = source_dir.join("generation_config.json");
    if generation.is_file() {
        copy_model_file(&generation, &output_dir.join("generation_config.json"))?;
    }
    let dims_path = output_dir.join(MODEL_DIMS_FILE_NAME);
    std::fs::write(
        &dims_path,
        serde_json::to_string_pretty(&dims).wrap_err("failed to serialize Whisper dimensions")?,
    )
    .wrap_err_with(|| format!("failed to write {}", dims_path.display()))?;

    let artifacts = inspect_model_dir(output_dir)?;
    partial_output.commit();
    Ok(artifacts)
}

#[cfg(any(feature = "tch-native", feature = "cuda-native"))]
fn copy_model_file(source: &Path, destination: &Path) -> eyre::Result<()> {
    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent).wrap_err_with(|| {
            format!(
                "failed to create model asset directory {}",
                parent.display()
            )
        })?;
    }
    std::fs::copy(source, destination).wrap_err_with(|| {
        format!(
            "failed to copy model asset {} to {}",
            source.display(),
            destination.display()
        )
    })?;
    Ok(())
}

fn checkpoint_store(checkpoint: &Path, top_level_key: &str) -> PytorchStore {
    PytorchStore::from_file(checkpoint)
        .with_top_level_key(top_level_key)
        .with_key_remapping(
            r"^encoder\.blocks\.(\d+)\.mlp\.0\.",
            "encoder.blocks.$1.mlp.lin1.",
        )
        .with_key_remapping(
            r"^encoder\.blocks\.(\d+)\.mlp\.2\.",
            "encoder.blocks.$1.mlp.lin2.",
        )
        .with_key_remapping(
            r"^decoder\.blocks\.(\d+)\.mlp\.0\.",
            "decoder.blocks.$1.mlp.lin1.",
        )
        .with_key_remapping(
            r"^decoder\.blocks\.(\d+)\.mlp\.2\.",
            "decoder.blocks.$1.mlp.lin2.",
        )
        .with_key_remapping(r"^(.*\.attn_ln)\.weight$", "$1.gamma")
        .with_key_remapping(r"^(.*\.attn_ln)\.bias$", "$1.beta")
        .with_key_remapping(r"^(.*\.cross_attn_ln)\.weight$", "$1.gamma")
        .with_key_remapping(r"^(.*\.cross_attn_ln)\.bias$", "$1.beta")
        .with_key_remapping(r"^(.*\.mlp_ln)\.weight$", "$1.gamma")
        .with_key_remapping(r"^(.*\.mlp_ln)\.bias$", "$1.beta")
        .with_key_remapping(r"^encoder\.ln_post\.weight$", "encoder.ln_post.gamma")
        .with_key_remapping(r"^encoder\.ln_post\.bias$", "encoder.ln_post.beta")
        .with_key_remapping(r"^decoder\.ln\.weight$", "decoder.ln.gamma")
        .with_key_remapping(r"^decoder\.ln\.bias$", "decoder.ln.beta")
        .allow_partial(true)
}

#[cfg(test)]
mod tests {
    use super::PartialModelDirectory;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn test_directory(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("teamy-transcriber-{name}-{}", Uuid::new_v4()))
    }

    #[test]
    fn uncommitted_model_directory_is_removed() {
        let path = test_directory("partial-model");
        std::fs::create_dir_all(&path).expect("test model directory should be creatable");
        {
            let _partial = PartialModelDirectory::new(&path);
            std::fs::write(path.join("partial"), b"not a complete model")
                .expect("partial marker should be writable");
        };
        assert!(!path.exists());
    }

    #[test]
    fn committed_model_directory_is_retained() {
        let path = test_directory("committed-model");
        std::fs::create_dir_all(&path).expect("test model directory should be creatable");
        {
            let mut partial = PartialModelDirectory::new(&path);
            partial.commit();
        };
        assert!(path.is_dir());
        std::fs::remove_dir_all(&path).expect("test model directory should be removable");
    }
}
