use super::whisper::WhisperDims;
use eyre::WrapErr;
use eyre::bail;
use std::path::Path;
use std::path::PathBuf;

pub const MODEL_BURNPACK_FILE_NAME: &str = "model.bpk";
pub const MODEL_DIMS_FILE_NAME: &str = "dims.json";
pub const TOKENIZER_FILE_NAME: &str = "tokenizer.json";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WhisperModelLayout {
    WhisperBurnNpy,
    BurnPack,
}

impl WhisperModelLayout {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
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
    pub dims_path: Option<PathBuf>,
    pub dims: Option<WhisperDims>,
}

/// Inspect a locally supplied native Whisper model directory.
///
/// The preferred layout is Burnpack weights plus `dims.json` and
/// `tokenizer.json`. The legacy whisper-burn packed-NPY layout remains
/// discoverable so existing local model artifacts can be migrated gradually.
///
/// # Errors
///
/// Returns an error when required model files are absent or malformed.
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
    let dims_path = root.join(MODEL_DIMS_FILE_NAME);
    if burnpack_path.is_file() && dims_path.is_file() {
        let dims = read_dims_file(&dims_path)?;
        let artifacts = WhisperModelArtifacts {
            root: root.to_path_buf(),
            layout: WhisperModelLayout::BurnPack,
            tokenizer,
            encoder_dir: None,
            decoder_dir: None,
            burnpack_path: Some(burnpack_path),
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
            dims_path: None,
            dims: None,
        };
        artifacts.dims = super::whisper::infer_dims_from_artifacts(&artifacts).ok();
        validate_model_artifacts(&artifacts)?;
        return Ok(artifacts);
    }

    bail!(
        "native Whisper model {} is incomplete; expected {} + {} + {} or encoder/decoder packed-NPY directories",
        root.display(),
        MODEL_BURNPACK_FILE_NAME,
        MODEL_DIMS_FILE_NAME,
        TOKENIZER_FILE_NAME,
    )
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
