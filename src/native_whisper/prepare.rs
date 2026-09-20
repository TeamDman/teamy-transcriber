use super::model::MODEL_CONFIG_FILE_NAME;
use super::model::MODEL_DIMS_FILE_NAME;
use super::model::MODEL_SAFETENSORS_FILE_NAME;
use super::model::MODEL_SAFETENSORS_INDEX_FILE_NAME;
use super::model::TOKENIZER_FILE_NAME;
use super::model::WhisperModelArtifacts;
use super::model::inspect_model_dir;
use super::model::resolve_safetensor_paths;
use super::whisper::AudioEncoderDims;
use super::whisper::TextDecoderDims;
use super::whisper::WhisperDims;
use eyre::WrapErr;
use eyre::bail;
use serde::Deserialize;
use std::path::Path;
use std::path::PathBuf;

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

/// Prepare a canonical Hugging Face Whisper directory for source-defined CUDA inference.
///
/// This operation is local-only. It copies the existing safetensors file(s),
/// tokenizer, and config into the application package and writes the compact
/// dims.json sidecar consumed by the runtime. It intentionally does not
/// convert `CTranslate2` `model.bin` files: those are a different runtime format
/// and are not supported by this runtime.
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
            "{} is a CTranslate2/faster-whisper model.bin directory; it cannot be loaded or losslessly converted by the native CUDA Whisper backend. Supply canonical model.safetensors weights instead",
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
    if source_dir.join(super::speech::DIRECTORY).exists() {
        super::speech::validate(&source_dir.join(super::speech::DIRECTORY))?;
        super::speech::prepare(
            &source_dir
                .join(super::speech::DIRECTORY)
                .join("silero.safetensors"),
            output_dir,
        )?;
    }
    partial_output.commit();
    Ok(artifacts)
}

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
