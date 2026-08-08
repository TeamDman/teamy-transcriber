use super::model::MODEL_BURNPACK_FILE_NAME;
use super::model::MODEL_DIMS_FILE_NAME;
use super::model::TOKENIZER_FILE_NAME;
use super::model::WhisperModelArtifacts;
use super::model::inspect_model_dir;
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

struct PartialModelDirectory {
    path: PathBuf,
    committed: bool,
}

impl PartialModelDirectory {
    fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            committed: false,
        }
    }

    fn commit(&mut self) {
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
