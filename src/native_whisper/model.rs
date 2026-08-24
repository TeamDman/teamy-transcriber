use super::whisper::WhisperDims;
use eyre::WrapErr;
use eyre::bail;
use std::path::Path;
use std::path::PathBuf;

pub const MODEL_BURNPACK_FILE_NAME: &str = "model.bpk";
pub const MODEL_TORCHSCRIPT_FILE_NAME: &str = "model.pt";
pub const MODEL_SAFETENSORS_FILE_NAME: &str = "model.safetensors";
pub const MODEL_SAFETENSORS_INDEX_FILE_NAME: &str = "model.safetensors.index.json";
pub const MODEL_CONFIG_FILE_NAME: &str = "config.json";
pub const MODEL_DIMS_FILE_NAME: &str = "dims.json";
pub const TOKENIZER_FILE_NAME: &str = "tokenizer.json";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WhisperModelLayout {
    TorchScript,
    TchSafetensors,
    WhisperBurnNpy,
    BurnPack,
}

impl WhisperModelLayout {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::TorchScript => "whisper-torchscript",
            Self::TchSafetensors => "whisper-tch-safetensors",
            Self::WhisperBurnNpy => "whisper-burn-npy",
            Self::BurnPack => "burnpack",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenizerMetadata {
    pub path: PathBuf,
    pub vocab_size: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhisperModelArtifacts {
    pub root: PathBuf,
    pub layout: WhisperModelLayout,
    pub tokenizer: TokenizerMetadata,
    pub encoder_dir: Option<PathBuf>,
    pub decoder_dir: Option<PathBuf>,
    pub burnpack_path: Option<PathBuf>,
    pub torchscript_path: Option<PathBuf>,
    pub safetensors_path: Option<PathBuf>,
    pub safetensors_paths: Vec<PathBuf>,
    pub config_path: Option<PathBuf>,
    pub dims_path: Option<PathBuf>,
    pub dims: Option<WhisperDims>,
}

/// Inspect a locally supplied native Whisper model directory.
///
/// The preferred tch layouts are a `TorchScript` `model.pt` or canonical
/// Hugging Face `model.safetensors`, each with `dims.json` and `tokenizer.json`.
/// Both are loaded through the same tch/LibTorch family as `teamy-tts`.
/// Burnpack and packed-NPY remain compatibility paths.
///
/// # Errors
///
/// Returns an error when required model files are absent or malformed.
#[expect(
    clippy::too_many_lines,
    reason = "The inspector keeps all native model layout detection and validation in one boundary"
)]
pub fn inspect_model_dir(root: &Path) -> eyre::Result<WhisperModelArtifacts> {
    if !root.is_dir() {
        bail!(
            "native Whisper model directory is missing: {}",
            root.display()
        );
    }
    let tokenizer_path = root.join(TOKENIZER_FILE_NAME);
    if !tokenizer_path.is_file() {
        bail!(
            "native Whisper model is missing {}",
            tokenizer_path.display()
        );
    }
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| eyre::eyre!("failed to load tokenizer: {error}"))?;
    let tokenizer = TokenizerMetadata {
        path: tokenizer_path,
        vocab_size: tokenizer.get_vocab_size(true),
    };

    let burnpack_path = root.join(MODEL_BURNPACK_FILE_NAME);
    let torchscript_path = root.join(MODEL_TORCHSCRIPT_FILE_NAME);
    let safetensors_path = root.join(MODEL_SAFETENSORS_FILE_NAME);
    let safetensors_index_path = root.join(MODEL_SAFETENSORS_INDEX_FILE_NAME);
    let config_path = root.join(MODEL_CONFIG_FILE_NAME);
    let dims_path = root.join(MODEL_DIMS_FILE_NAME);
    if torchscript_path.is_file() && dims_path.is_file() {
        let dims = read_dims_file(&dims_path)?;
        let artifacts = WhisperModelArtifacts {
            root: root.to_path_buf(),
            layout: WhisperModelLayout::TorchScript,
            tokenizer,
            encoder_dir: None,
            decoder_dir: None,
            burnpack_path: None,
            torchscript_path: Some(torchscript_path),
            safetensors_path: None,
            safetensors_paths: Vec::new(),
            config_path: config_path.is_file().then_some(config_path.clone()),
            dims_path: Some(dims_path),
            dims: Some(dims),
        };
        validate_model_artifacts(&artifacts)?;
        return Ok(artifacts);
    }
    if dims_path.is_file() && (safetensors_path.is_file() || safetensors_index_path.is_file()) {
        let dims = read_dims_file(&dims_path)?;
        let safetensors_paths = resolve_safetensor_paths(root)?;
        let artifacts = WhisperModelArtifacts {
            root: root.to_path_buf(),
            layout: WhisperModelLayout::TchSafetensors,
            tokenizer,
            encoder_dir: None,
            decoder_dir: None,
            burnpack_path: None,
            torchscript_path: None,
            safetensors_path: safetensors_path.is_file().then_some(safetensors_path),
            safetensors_paths,
            config_path: config_path.is_file().then_some(config_path),
            dims_path: Some(dims_path),
            dims: Some(dims),
        };
        validate_model_artifacts(&artifacts)?;
        return Ok(artifacts);
    }
    if burnpack_path.is_file() && dims_path.is_file() {
        let dims = read_dims_file(&dims_path)?;
        let artifacts = WhisperModelArtifacts {
            root: root.to_path_buf(),
            layout: WhisperModelLayout::BurnPack,
            tokenizer,
            encoder_dir: None,
            decoder_dir: None,
            burnpack_path: Some(burnpack_path),
            torchscript_path: None,
            safetensors_path: None,
            safetensors_paths: Vec::new(),
            config_path: None,
            dims_path: Some(dims_path),
            dims: Some(dims),
        };
        validate_model_artifacts(&artifacts)?;
        return Ok(artifacts);
    }

    let encoder_dir = root.join("encoder");
    let decoder_dir = root.join("decoder");
    if encoder_dir.is_dir() && decoder_dir.is_dir() {
        let mut artifacts = WhisperModelArtifacts {
            root: root.to_path_buf(),
            layout: WhisperModelLayout::WhisperBurnNpy,
            tokenizer,
            encoder_dir: Some(encoder_dir),
            decoder_dir: Some(decoder_dir),
            burnpack_path: None,
            torchscript_path: None,
            safetensors_path: None,
            safetensors_paths: Vec::new(),
            config_path: None,
            dims_path: None,
            dims: None,
        };
        artifacts.dims = super::whisper::infer_dims_from_artifacts(&artifacts).ok();
        validate_model_artifacts(&artifacts)?;
        return Ok(artifacts);
    }

    bail!(
        "native Whisper model {} is incomplete; expected {} or {} + {} + {} for tch/LibTorch, or {} + {} + {} for the legacy Burn path or encoder/decoder packed-NPY directories",
        root.display(),
        MODEL_TORCHSCRIPT_FILE_NAME,
        MODEL_SAFETENSORS_FILE_NAME,
        MODEL_DIMS_FILE_NAME,
        TOKENIZER_FILE_NAME,
        MODEL_BURNPACK_FILE_NAME,
        MODEL_DIMS_FILE_NAME,
        TOKENIZER_FILE_NAME,
    )
}

#[derive(Debug, serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: std::collections::BTreeMap<String, String>,
}

/// Resolve a canonical single-file or indexed-shard safetensors package.
///
/// Index entries are restricted to relative paths below `root`; this prevents
/// a model manifest from causing preparation or inference to read outside the
/// selected model directory.
pub fn resolve_safetensor_paths(root: &Path) -> eyre::Result<Vec<PathBuf>> {
    let single_path = root.join(MODEL_SAFETENSORS_FILE_NAME);
    if single_path.is_file() {
        return Ok(vec![single_path]);
    }
    let index_path = root.join(MODEL_SAFETENSORS_INDEX_FILE_NAME);
    if !index_path.is_file() {
        bail!(
            "canonical Whisper model is missing {} or {}",
            single_path.display(),
            index_path.display()
        );
    }
    let index: SafetensorsIndex = serde_json::from_str(
        &std::fs::read_to_string(&index_path)
            .wrap_err_with(|| format!("failed to read {}", index_path.display()))?,
    )
    .wrap_err_with(|| format!("failed to parse {}", index_path.display()))?;
    let mut paths = std::collections::BTreeSet::new();
    for relative_name in index.weight_map.values() {
        let relative = Path::new(relative_name);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| component == std::path::Component::ParentDir)
        {
            bail!("safetensors index contains an unsafe shard path: {relative_name}");
        }
        let path = root.join(relative);
        if !path.is_file() {
            bail!(
                "safetensors index references missing shard {}",
                path.display()
            );
        }
        paths.insert(path);
    }
    if paths.is_empty() {
        bail!("safetensors index contains no weight_map entries");
    }
    Ok(paths.into_iter().collect())
}

fn validate_model_artifacts(artifacts: &WhisperModelArtifacts) -> eyre::Result<()> {
    if let Some(dims) = &artifacts.dims
        && artifacts.tokenizer.vocab_size != dims.text.n_vocab
    {
        bail!(
            "native Whisper tokenizer vocabulary size {} does not match model vocabulary size {}",
            artifacts.tokenizer.vocab_size,
            dims.text.n_vocab
        );
    }
    super::whisper::default_decoder_prompt_token_ids(artifacts)
        .map(|_| ())
        .wrap_err("native Whisper tokenizer is missing required transcription tokens")
}

fn read_dims_file(path: &Path) -> eyre::Result<WhisperDims> {
    let contents = std::fs::read_to_string(path).wrap_err_with(|| {
        format!(
            "failed to read native Whisper dimensions {}",
            path.display()
        )
    })?;
    serde_json::from_str(&contents).wrap_err_with(|| {
        format!(
            "failed to parse native Whisper dimensions {}",
            path.display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::MODEL_SAFETENSORS_INDEX_FILE_NAME;
    use super::resolve_safetensor_paths;
    use std::path::PathBuf;
    use uuid::Uuid;

    fn test_directory(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("teamy-transcriber-{name}-{}", Uuid::new_v4()))
    }

    #[test]
    fn indexed_safetensors_resolve_all_unique_shards() {
        let root = test_directory("indexed-safetensors");
        std::fs::create_dir_all(&root).expect("test directory should be creatable");
        std::fs::write(root.join("model-00001-of-00002.safetensors"), b"one")
            .expect("first shard should be writable");
        std::fs::write(root.join("model-00002-of-00002.safetensors"), b"two")
            .expect("second shard should be writable");
        std::fs::write(
            root.join(MODEL_SAFETENSORS_INDEX_FILE_NAME),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {
                    "encoder.weight": "model-00001-of-00002.safetensors",
                    "decoder.weight": "model-00002-of-00002.safetensors",
                    "decoder.bias": "model-00002-of-00002.safetensors"
                }
            }))
            .expect("index should serialize"),
        )
        .expect("index should be writable");
        let paths = resolve_safetensor_paths(&root).expect("shards should resolve");
        assert_eq!(paths.len(), 2);
        assert!(paths.iter().all(|path| path.starts_with(&root)));
        std::fs::remove_dir_all(root).expect("test directory should be removable");
    }

    #[test]
    fn indexed_safetensors_reject_parent_paths() {
        let root = test_directory("unsafe-indexed-safetensors");
        std::fs::create_dir_all(&root).expect("test directory should be creatable");
        std::fs::write(
            root.join(MODEL_SAFETENSORS_INDEX_FILE_NAME),
            br#"{"weight_map":{"encoder.weight":"../outside.safetensors"}}"#,
        )
        .expect("index should be writable");
        let error = resolve_safetensor_paths(&root).expect_err("parent path should be rejected");
        assert!(error.to_string().contains("unsafe shard path"));
        std::fs::remove_dir_all(root).expect("test directory should be removable");
    }
}
