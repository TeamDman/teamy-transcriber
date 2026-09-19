//! Weight manifest validation shared by native CUDA and retained `LibTorch` builds.
use super::whisper::WhisperDims;
use eyre::Context;
use eyre::bail;
use memmap2::Mmap;
use safetensors::tensor::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
/// Validate the names required by the direct tch graph without materializing
/// the model tensors.  This is used by model preparation for early diagnostics.
pub fn validate_safetensors_file(path: &Path, dims: &WhisperDims) -> eyre::Result<()> {
    validate_safetensors_files(&[path.to_path_buf()], dims)
}

/// Validate a canonical single-file or sharded safetensors package.
pub fn validate_safetensors_files(paths: &[PathBuf], dims: &WhisperDims) -> eyre::Result<()> {
    if paths.is_empty() {
        bail!("safetensors model has no weight files");
    }
    let mut shapes = HashMap::new();
    for path in paths {
        let file =
            File::open(path).wrap_err_with(|| format!("failed to open {}", path.display()))?;
        // SAFETY: the file handle remains open for the lifetime of the mapping, and the mapping
        // is accessed only immutably by SafeTensors.
        let mapped = unsafe { Mmap::map(&file) }
            .wrap_err_with(|| format!("failed to memory-map {}", path.display()))?;
        let tensors = SafeTensors::deserialize(&mapped)
            .wrap_err_with(|| format!("failed to parse {}", path.display()))?;
        for (name, view) in tensors.tensors() {
            if shapes.insert(name.clone(), view.shape().to_vec()).is_some() {
                bail!("safetensors weight {name} appears in more than one shard");
            }
        }
    }
    validate_tensor_manifest(&shapes, dims)
}

trait TensorShapeLookup {
    fn has(&self, name: &str) -> bool;
    fn shape(&self, name: &str) -> Option<Vec<usize>>;
}

impl TensorShapeLookup for SafeTensors<'_> {
    fn has(&self, name: &str) -> bool {
        self.tensor(name).is_ok()
    }

    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.tensor(name).ok().map(|view| view.shape().to_vec())
    }
}

impl TensorShapeLookup for HashMap<String, Vec<usize>> {
    fn has(&self, name: &str) -> bool {
        self.contains_key(name)
    }

    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.get(name).cloned()
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "Manifest validation keeps the canonical Whisper graph contract explicit"
)]
fn validate_tensor_manifest<T: TensorShapeLookup>(
    tensors: &T,
    dims: &WhisperDims,
) -> eyre::Result<()> {
    let required = [
        "model.encoder.conv1.weight",
        "model.encoder.conv1.bias",
        "model.encoder.conv2.weight",
        "model.encoder.conv2.bias",
        "model.encoder.embed_positions.weight",
        "model.encoder.layer_norm.weight",
        "model.encoder.layer_norm.bias",
        "model.decoder.embed_tokens.weight",
        "model.decoder.embed_positions.weight",
        "model.decoder.layer_norm.weight",
        "model.decoder.layer_norm.bias",
    ];
    for name in required {
        if !tensors.has(name) {
            bail!("Whisper safetensors model is missing {name}");
        }
    }
    let audio_state = dims.audio.n_audio_state;
    let text_state = dims.text.n_text_state;
    ensure_shape(
        tensors,
        "model.encoder.conv1.weight",
        &[audio_state, dims.audio.n_mels, 3],
    )?;
    ensure_shape(
        tensors,
        "model.encoder.conv2.weight",
        &[audio_state, audio_state, 3],
    )?;
    ensure_shape(
        tensors,
        "model.encoder.embed_positions.weight",
        &[dims.audio.n_audio_ctx, audio_state],
    )?;
    ensure_shape(
        tensors,
        "model.decoder.embed_tokens.weight",
        &[dims.text.n_vocab, text_state],
    )?;
    ensure_shape(
        tensors,
        "model.decoder.embed_positions.weight",
        &[dims.text.n_text_ctx, text_state],
    )?;
    if tensors.has("proj_out.weight") {
        ensure_shape(tensors, "proj_out.weight", &[dims.text.n_vocab, text_state])?;
    }
    for prefix in ["model.encoder.layers", "model.decoder.layers"] {
        let layers = if prefix.ends_with("encoder.layers") {
            dims.audio.n_audio_layer
        } else {
            dims.text.n_text_layer
        };
        let state = if prefix.ends_with("encoder.layers") {
            audio_state
        } else {
            text_state
        };
        for index in 0..layers {
            let block = format!("{prefix}.{index}");
            let names = [
                format!("{block}.self_attn.q_proj.weight"),
                format!("{block}.self_attn.k_proj.weight"),
                format!("{block}.self_attn.v_proj.weight"),
                format!("{block}.self_attn.out_proj.weight"),
                format!("{block}.self_attn.out_proj.bias"),
                format!("{block}.self_attn_layer_norm.weight"),
                format!("{block}.self_attn_layer_norm.bias"),
                format!("{block}.fc1.weight"),
                format!("{block}.fc1.bias"),
                format!("{block}.fc2.weight"),
                format!("{block}.fc2.bias"),
                format!("{block}.final_layer_norm.weight"),
                format!("{block}.final_layer_norm.bias"),
            ];
            for name in names {
                if !tensors.has(&name) {
                    bail!("Whisper safetensors model is missing {name}");
                }
            }
            for name in [
                format!("{block}.self_attn.q_proj.weight"),
                format!("{block}.self_attn.k_proj.weight"),
                format!("{block}.self_attn.v_proj.weight"),
                format!("{block}.self_attn.out_proj.weight"),
                format!("{block}.fc1.weight"),
                format!("{block}.fc2.weight"),
            ] {
                let expected = if name.ends_with("fc1.weight") {
                    vec![4 * state, state]
                } else if name.ends_with("fc2.weight") {
                    vec![state, 4 * state]
                } else {
                    vec![state, state]
                };
                ensure_shape(tensors, &name, &expected)?;
            }
            for name in [
                format!("{block}.self_attn.q_proj.bias"),
                format!("{block}.self_attn.v_proj.bias"),
                format!("{block}.self_attn.out_proj.bias"),
                format!("{block}.self_attn_layer_norm.weight"),
                format!("{block}.self_attn_layer_norm.bias"),
                format!("{block}.fc1.bias"),
                format!("{block}.fc2.bias"),
                format!("{block}.final_layer_norm.weight"),
                format!("{block}.final_layer_norm.bias"),
            ] {
                if tensors.has(&name) {
                    let expected = if name.ends_with("fc1.bias") {
                        vec![4 * state]
                    } else {
                        vec![state]
                    };
                    ensure_shape(tensors, &name, &expected)?;
                }
            }
            if prefix.ends_with("decoder.layers") {
                for name in [
                    format!("{block}.encoder_attn.q_proj.weight"),
                    format!("{block}.encoder_attn.k_proj.weight"),
                    format!("{block}.encoder_attn.v_proj.weight"),
                    format!("{block}.encoder_attn.out_proj.weight"),
                    format!("{block}.encoder_attn.out_proj.bias"),
                    format!("{block}.encoder_attn_layer_norm.weight"),
                    format!("{block}.encoder_attn_layer_norm.bias"),
                ] {
                    if !tensors.has(&name) {
                        bail!("Whisper safetensors model is missing {name}");
                    }
                }
                for name in [
                    format!("{block}.encoder_attn.q_proj.weight"),
                    format!("{block}.encoder_attn.k_proj.weight"),
                    format!("{block}.encoder_attn.v_proj.weight"),
                    format!("{block}.encoder_attn.out_proj.weight"),
                ] {
                    ensure_shape(tensors, &name, &[state, state])?;
                }
            }
        }
    }
    Ok(())
}

fn ensure_shape<T: TensorShapeLookup>(
    tensors: &T,
    name: &str,
    expected: &[usize],
) -> eyre::Result<()> {
    let shape = tensors
        .shape(name)
        .ok_or_else(|| eyre::eyre!("Whisper safetensors model is missing {name}"))?;
    if shape != expected {
        bail!(
            "Whisper safetensor {name} has shape {:?}, expected {expected:?}",
            shape
        );
    }
    Ok(())
}
