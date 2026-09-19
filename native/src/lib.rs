//! Whisper model topology in Rust, numerical primitives in CUDA, weights in safetensors.
//! No serialized computation graph, LibTorch, or Python is used by this runtime.
mod cuda;
pub mod frontend;
#[cfg(test)]
mod loader_tests;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cuda::Buffer;
use cuda::Device;
use half::bf16;
use half::f16;
use half::slice::HalfFloatSliceExt;
use memmap2::Mmap;
use safetensors::Dtype;
use safetensors::SafeTensors;
use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs::File;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

#[derive(Clone, Debug, Deserialize)]
pub struct Dims {
    pub audio: AudioDims,
    pub text: TextDims,
}
#[derive(Clone, Debug, Deserialize)]
pub struct AudioDims {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize,
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
}
#[derive(Clone, Debug, Deserialize)]
pub struct TextDims {
    pub n_vocab: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize,
    pub n_text_head: usize,
    pub n_text_layer: usize,
}
impl Dims {
    fn validate(&self) -> Result<()> {
        let a = &self.audio;
        let t = &self.text;
        ensure!(
            (a.n_mels == 80 || a.n_mels == 128) && a.n_audio_ctx == 1500,
            "expected Whisper 80/128-mel, 1500-position encoder"
        );
        ensure!(
            (1..=1280).contains(&a.n_audio_state) && t.n_text_state == a.n_audio_state,
            "invalid Whisper hidden dimensions"
        );
        ensure!(
            (1..=32).contains(&a.n_audio_layer) && (1..=32).contains(&t.n_text_layer),
            "invalid Whisper layer count"
        );
        ensure!(
            a.n_audio_head > 0
                && t.n_text_head > 0
                && a.n_audio_state / a.n_audio_head == 64
                && t.n_text_state / t.n_text_head == 64
                && a.n_audio_state % a.n_audio_head == 0
                && t.n_text_state % t.n_text_head == 0,
            "expected 64-wide Whisper heads"
        );
        ensure!(
            (1..=448).contains(&t.n_text_ctx) && (1..=60000).contains(&t.n_vocab),
            "invalid Whisper vocabulary/context"
        );
        Ok(())
    }
}

struct Weight {
    shape: Vec<usize>,
    data: Buffer,
}
struct Weights(HashMap<String, Weight>);
fn extend_f16(values: &mut Vec<f32>, bytes: &[u8]) {
    if cfg!(target_endian = "little") {
        // SAFETY: every 16-bit pattern is a valid f16. align_to checks the
        // alignment/extent; only a completely aligned slice uses this path.
        let (prefix, source, suffix) = unsafe { bytes.align_to::<f16>() };
        if prefix.is_empty() && suffix.is_empty() {
            let offset = values.len();
            values.resize(offset + source.len(), 0.);
            source.convert_to_f32_slice(&mut values[offset..]);
            return;
        }
    }
    values.extend(
        bytes
            .chunks_exact(2)
            .map(|b| f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()),
    );
}
impl Weights {
    fn upload_group(
        weights: &mut HashMap<String, Weight>,
        device: &Rc<Device>,
        values: &mut Vec<f32>,
        entries: &mut Vec<(String, Vec<usize>, usize, usize)>,
    ) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        let arena = device.upload(values)?;
        for (name, shape, offset, len) in entries.drain(..) {
            ensure!(!weights.contains_key(&name), "duplicate weight {name}");
            weights.insert(
                name,
                Weight {
                    shape,
                    data: arena.slice(offset, len)?,
                },
            );
        }
        values.clear();
        Ok(())
    }
    fn open(root: &Path, device: &Rc<Device>) -> Result<Self> {
        let single = root.join("model.safetensors");
        let index_path = root.join("model.safetensors.index.json");
        ensure!(
            !(single.is_file() && index_path.is_file()),
            "ambiguous single-file and indexed model"
        );
        let mut expected: Option<HashMap<String, String>> = None;
        let paths: Vec<PathBuf> = if single.is_file() {
            vec![single]
        } else {
            let index: serde_json::Value = serde_json::from_reader(File::open(index_path)?)?;
            let map: HashMap<String, String> = serde_json::from_value(index["weight_map"].clone())
                .context("invalid weight_map")?;
            let mut names = HashSet::new();
            for value in map.values() {
                let name = value.as_str();
                let p = Path::new(name);
                ensure!(
                    p.components().count() == 1
                        && matches!(p.components().next(), Some(Component::Normal(_))),
                    "unsafe shard filename"
                );
                names.insert(name.to_owned());
            }
            let mut names: Vec<_> = names.into_iter().collect();
            names.sort();
            expected = Some(map);
            names.into_iter().map(|n| root.join(n)).collect()
        };
        ensure!(!paths.is_empty(), "model has no shards");
        let mut weights = HashMap::new();
        // Bounded staging groups amortize allocation/transfer/synchronization
        // overhead. Large individual tensors get their own group. Subranges
        // retain the arena until its last model weight is released.
        const GROUP_FLOATS: usize = 16 * 1024 * 1024;
        let mut values = Vec::new();
        let mut entries = Vec::new();
        for path in paths {
            let file = File::open(&path)?;
            // SAFETY: immutable mapping is consumed before the file/mapping drops.
            let mapped = unsafe { Mmap::map(&file) }?;
            let tensors = SafeTensors::deserialize(&mapped)?;
            let mut ordered = tensors.tensors();
            // Traverse mapped bytes in file order rather than hash-map order.
            ordered.sort_unstable_by_key(|(_, view)| view.data().as_ptr().addr());
            for (name, view) in ordered {
                if let Some(expected) = &mut expected {
                    let shard = expected
                        .remove(&name)
                        .ok_or_else(|| anyhow!("unindexed or duplicate weight {name}"))?;
                    ensure!(
                        path.file_name().is_some_and(|n| n == shard.as_str()),
                        "weight {name} is in the wrong shard"
                    );
                }
                ensure!(
                    matches!(view.dtype(), Dtype::F32 | Dtype::F16 | Dtype::BF16),
                    "unsupported dtype {:?} for {name}",
                    view.dtype()
                );
                let len = view.data().len() / (view.dtype().bitsize() / 8);
                if values.len() + len > GROUP_FLOATS {
                    Self::upload_group(&mut weights, device, &mut values, &mut entries)?;
                }
                let offset = values.len();
                match view.dtype() {
                    Dtype::F32 => values.extend(
                        view.data()
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
                    ),
                    Dtype::F16 => extend_f16(&mut values, view.data()),
                    Dtype::BF16 => values.extend(view.data().chunks_exact(2).map(|b| {
                        bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32()
                    })),
                    _ => unreachable!(),
                }
                ensure!(
                    values[offset..].iter().all(|x| x.is_finite()),
                    "non-finite weight {name}"
                );
                entries.push((name, view.shape().to_vec(), offset, len));
            }
        }
        ensure!(
            expected.as_ref().is_none_or(|m| m.is_empty()),
            "index references missing tensors"
        );
        Self::upload_group(&mut weights, device, &mut values, &mut entries)?;
        Ok(Self(weights))
    }
    fn take(&mut self, name: &str, shape: &[usize]) -> Result<Buffer> {
        let w = self
            .0
            .remove(name)
            .ok_or_else(|| anyhow!("missing weight {name}"))?;
        ensure!(
            w.shape == shape,
            "{name}: shape {:?}, expected {shape:?}",
            w.shape
        );
        Ok(w.data)
    }
    fn norm(&mut self, name: &str, width: usize) -> Result<Norm> {
        Ok(Norm {
            w: self.take(&format!("{name}.weight"), &[width])?,
            b: self.take(&format!("{name}.bias"), &[width])?,
        })
    }
    fn linear(&mut self, name: &str, input: usize, output: usize, bias: bool) -> Result<Linear> {
        Ok(Linear {
            w: self.take(&format!("{name}.weight"), &[output, input])?,
            b: if bias {
                Some(self.take(&format!("{name}.bias"), &[output])?)
            } else {
                None
            },
            input,
            output,
        })
    }
    fn attention(&mut self, name: &str, width: usize) -> Result<Attention> {
        Ok(Attention {
            q: self.linear(&format!("{name}.q_proj"), width, width, true)?,
            k: self.linear(&format!("{name}.k_proj"), width, width, false)?,
            v: self.linear(&format!("{name}.v_proj"), width, width, true)?,
            out: self.linear(&format!("{name}.out_proj"), width, width, true)?,
        })
    }
    fn layer(&mut self, name: &str, width: usize, decoder: bool) -> Result<Layer> {
        Ok(Layer {
            norm: self.norm(&format!("{name}.self_attn_layer_norm"), width)?,
            attn: self.attention(&format!("{name}.self_attn"), width)?,
            cross: if decoder {
                Some((
                    self.norm(&format!("{name}.encoder_attn_layer_norm"), width)?,
                    self.attention(&format!("{name}.encoder_attn"), width)?,
                ))
            } else {
                None
            },
            ff_norm: self.norm(&format!("{name}.final_layer_norm"), width)?,
            fc1: self.linear(&format!("{name}.fc1"), width, 4 * width, true)?,
            fc2: self.linear(&format!("{name}.fc2"), 4 * width, width, true)?,
        })
    }
}
struct Linear {
    w: Buffer,
    b: Option<Buffer>,
    input: usize,
    output: usize,
}
impl Linear {
    fn run(
        &self,
        x: &Buffer,
        out: &Buffer,
        rows: usize,
        residual: Option<&Buffer>,
        gelu: bool,
    ) -> Result<()> {
        x.linear(
            &self.w,
            self.b.as_ref(),
            residual,
            out,
            rows,
            self.input,
            self.output,
            gelu,
        )
    }
}
struct Norm {
    w: Buffer,
    b: Buffer,
}
impl Norm {
    fn run(&self, x: &Buffer, y: &Buffer, rows: usize, width: usize) -> Result<()> {
        x.norm(&self.w, &self.b, y, rows, width)
    }
}
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}
struct Layer {
    norm: Norm,
    attn: Attention,
    cross: Option<(Norm, Attention)>,
    ff_norm: Norm,
    fc1: Linear,
    fc2: Linear,
}
struct Cache {
    key: Buffer,
    value: Buffer,
    cross_key: Buffer,
    cross_value: Buffer,
}
struct Workspace {
    mel: Buffer,
    col: Buffer,
    conv: Buffer,
    x: Buffer,
    norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    context: Buffer,
    temp: Buffer,
    ff: Buffer,
    scores: Buffer,
    encoded: Buffer,
    logits: Buffer,
    result: Buffer,
}
impl Workspace {
    fn new(d: &Rc<Device>, dims: &Dims) -> Result<Self> {
        let a = &dims.audio;
        let width = a.n_audio_state;
        let rows = a.n_audio_ctx;
        Ok(Self {
            mel: d.alloc(a.n_mels * rows * 2)?,
            col: d.alloc((rows * width * 3).max(rows * 2 * a.n_mels * 3))?,
            conv: d.alloc(rows * 2 * width)?,
            x: d.alloc(rows * width)?,
            norm: d.alloc(rows * width)?,
            q: d.alloc(rows * width)?,
            k: d.alloc(rows * width)?,
            v: d.alloc(rows * width)?,
            context: d.alloc(rows * width)?,
            temp: d.alloc(rows * width)?,
            ff: d.alloc(rows * 4 * width)?,
            scores: d.alloc(a.n_audio_head.max(dims.text.n_text_head) * rows * rows)?,
            encoded: d.alloc(rows * width)?,
            logits: d.alloc(dims.text.n_vocab)?,
            result: d.alloc(1)?,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct DecodeResult {
    pub text: String,
    pub tokens: Vec<usize>,
    pub ended: bool,
    pub encoder_ms: f64,
    pub decoder_ms: f64,
}

/// Single-thread-owned resident engine. Use on its creating inference thread.
pub struct Engine {
    device: Rc<Device>,
    dims: Dims,
    tokenizer: tokenizers::Tokenizer,
    prompt: Vec<usize>,
    eot: usize,
    allowed: Buffer,
    conv1: (Buffer, Buffer),
    conv2: (Buffer, Buffer),
    audio_pos: Buffer,
    text_pos: Buffer,
    embed: Buffer,
    output: Option<Buffer>,
    encoder: Vec<Layer>,
    decoder: Vec<Layer>,
    encoder_norm: Norm,
    decoder_norm: Norm,
    cache: Vec<Cache>,
    work: Workspace,
}
impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeWhisperEngine")
            .field("dims", &self.dims)
            .finish_non_exhaustive()
    }
}
impl Engine {
    pub fn load(root: &Path, device_index: i32, tf32: bool) -> Result<Self> {
        let dims: Dims = serde_json::from_reader(File::open(root.join("dims.json"))?)?;
        dims.validate()?;
        let device = Device::new(device_index, tf32)?;
        let mut weights = Weights::open(root, &device)?;
        let d = dims.audio.n_audio_state;
        let tokenizer = tokenizers::Tokenizer::from_file(root.join("tokenizer.json"))
            .map_err(|e| anyhow!(e.to_string()))?;
        let id = |s: &str| {
            tokenizer
                .token_to_id(s)
                .map(|n| n as usize)
                .ok_or_else(|| anyhow!("tokenizer missing {s}"))
        };
        let prompt = vec![
            id("<|startoftranscript|>")?,
            id("<|en|>")?,
            id("<|transcribe|>")?,
            id("<|notimestamps|>")?,
        ];
        let eot = id("<|endoftext|>")?;
        ensure!(
            prompt.len() < dims.text.n_text_ctx
                && prompt
                    .iter()
                    .chain(std::iter::once(&eot))
                    .all(|&t| t < dims.text.n_vocab),
            "tokenizer/model mismatch"
        );
        let mut allowed = vec![1.; dims.text.n_vocab];
        for s in [
            "<|startoftranscript|>",
            "<|translate|>",
            "<|transcribe|>",
            "<|startoflm|>",
            "<|startofprev|>",
            "<|nospeech|>",
            "<|notimestamps|>",
            "<|en|>",
        ] {
            if let Some(t) = tokenizer.token_to_id(s) {
                ensure!((t as usize) < allowed.len(), "tokenizer/model mismatch");
                if t as usize != eot {
                    allowed[t as usize] = 0.;
                }
            }
        }
        let allowed = device.upload(&allowed)?;
        let conv1 = (
            weights.take("model.encoder.conv1.weight", &[d, dims.audio.n_mels, 3])?,
            weights.take("model.encoder.conv1.bias", &[d])?,
        );
        let conv2 = (
            weights.take("model.encoder.conv2.weight", &[d, d, 3])?,
            weights.take("model.encoder.conv2.bias", &[d])?,
        );
        let audio_pos = weights.take(
            "model.encoder.embed_positions.weight",
            &[dims.audio.n_audio_ctx, d],
        )?;
        let text_pos = weights.take(
            "model.decoder.embed_positions.weight",
            &[dims.text.n_text_ctx, d],
        )?;
        let embed = weights.take("model.decoder.embed_tokens.weight", &[dims.text.n_vocab, d])?;
        let output = if weights.0.contains_key("proj_out.weight") {
            Some(weights.take("proj_out.weight", &[dims.text.n_vocab, d])?)
        } else {
            None
        };
        let encoder = (0..dims.audio.n_audio_layer)
            .map(|i| weights.layer(&format!("model.encoder.layers.{i}"), d, false))
            .collect::<Result<Vec<_>>>()?;
        let decoder = (0..dims.text.n_text_layer)
            .map(|i| weights.layer(&format!("model.decoder.layers.{i}"), d, true))
            .collect::<Result<Vec<_>>>()?;
        let encoder_norm = weights.norm("model.encoder.layer_norm", d)?;
        let decoder_norm = weights.norm("model.decoder.layer_norm", d)?;
        let cache = (0..dims.text.n_text_layer)
            .map(|_| {
                Ok(Cache {
                    key: device.alloc(d * dims.text.n_text_ctx)?,
                    value: device.alloc(d * dims.text.n_text_ctx)?,
                    cross_key: device.alloc(d * dims.audio.n_audio_ctx)?,
                    cross_value: device.alloc(d * dims.audio.n_audio_ctx)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let work = Workspace::new(&device, &dims)?;
        device.sync()?;
        Ok(Self {
            device,
            dims,
            tokenizer,
            prompt,
            eot,
            allowed,
            conv1,
            conv2,
            audio_pos,
            text_pos,
            embed,
            output,
            encoder,
            decoder,
            encoder_norm,
            decoder_norm,
            cache,
            work,
        })
    }
    pub fn dims(&self) -> &Dims {
        &self.dims
    }
    /// Diagnostic boundary for numerical comparison with an independent model.
    pub fn prompt_logits(&mut self, mel: &[f32]) -> Result<Vec<f32>> {
        self.encode(mel)?;
        self.decoder_tokens(&self.prompt, 0)?;
        let width = self.dims.text.n_text_state;
        self.work
            .norm
            .slice((self.prompt.len() - 1) * width, width)?
            .linear(
                self.output.as_ref().unwrap_or(&self.embed),
                None,
                None,
                &self.work.logits,
                1,
                self.dims.text.n_text_state,
                self.dims.text.n_vocab,
                false,
            )?;
        self.work.logits.read(self.dims.text.n_vocab)
    }
    fn encode(&self, mel: &[f32]) -> Result<()> {
        let a = &self.dims.audio;
        let d = a.n_audio_state;
        let n = a.n_audio_ctx;
        let w = &self.work;
        ensure!(
            mel.len() == a.n_mels * n * 2 && mel.iter().all(|n| n.is_finite()),
            "expected finite mel matrix [{},{}]",
            a.n_mels,
            n * 2
        );
        w.mel.write(mel)?;
        w.mel.conv(
            &self.conv1.0,
            &self.conv1.1,
            &w.col,
            &w.conv,
            n * 2,
            a.n_mels,
            d,
            1,
            true,
        )?;
        w.conv.conv(
            &self.conv2.0,
            &self.conv2.1,
            &w.col,
            &w.x,
            n * 2,
            d,
            d,
            2,
            false,
        )?;
        w.x.position(&self.audio_pos, n * d)?;
        for layer in &self.encoder {
            layer.norm.run(&w.x, &w.norm, n, d)?;
            layer.attn.q.run(&w.norm, &w.q, n, None, false)?;
            layer.attn.k.run(&w.norm, &w.k, n, None, false)?;
            layer.attn.v.run(&w.norm, &w.v, n, None, false)?;
            w.q.attention(
                &w.k,
                &w.v,
                &w.scores,
                &w.context,
                n,
                n,
                d,
                a.n_audio_head,
                false,
                0,
            )?;
            layer
                .attn
                .out
                .run(&w.context, &w.temp, n, Some(&w.x), false)?;
            layer.ff_norm.run(&w.temp, &w.norm, n, d)?;
            layer.fc1.run(&w.norm, &w.ff, n, None, true)?;
            layer.fc2.run(&w.ff, &w.x, n, Some(&w.temp), false)?;
        }
        self.encoder_norm.run(&w.x, &w.encoded, n, d)?;
        for (layer, cache) in self.decoder.iter().zip(&self.cache) {
            let (_, attn) = layer.cross.as_ref().unwrap();
            attn.k.run(&w.encoded, &cache.cross_key, n, None, false)?;
            attn.v.run(&w.encoded, &cache.cross_value, n, None, false)?;
        }
        Ok(())
    }
    fn decoder_tokens(&self, tokens: &[usize], position: usize) -> Result<()> {
        let t = &self.dims.text;
        let d = t.n_text_state;
        let n = self.dims.audio.n_audio_ctx;
        let w = &self.work;
        let rows = tokens.len();
        ensure!(
            rows > 0
                && position < t.n_text_ctx
                && rows <= t.n_text_ctx - position
                && tokens.iter().all(|&token| token < t.n_vocab),
            "decoder token/position out of bounds"
        );
        for (row, &token) in tokens.iter().enumerate() {
            self.embed.embedding(
                &self.text_pos,
                &w.x.slice(row * d, d)?,
                token,
                position + row,
                d,
            )?;
        }
        for (layer, cache) in self.decoder.iter().zip(&self.cache) {
            layer.norm.run(&w.x, &w.norm, rows, d)?;
            layer.attn.q.run(&w.norm, &w.q, rows, None, false)?;
            layer.attn.k.run(&w.norm, &w.k, rows, None, false)?;
            layer.attn.v.run(&w.norm, &w.v, rows, None, false)?;
            cache.key.copy_from(&w.k, position * d, rows * d)?;
            cache.value.copy_from(&w.v, position * d, rows * d)?;
            w.q.attention(
                &cache.key,
                &cache.value,
                &w.scores,
                &w.context,
                rows,
                position + rows,
                d,
                t.n_text_head,
                rows > 1,
                position,
            )?;
            layer
                .attn
                .out
                .run(&w.context, &w.temp, rows, Some(&w.x), false)?;
            let (norm, attn) = layer.cross.as_ref().unwrap();
            norm.run(&w.temp, &w.norm, rows, d)?;
            attn.q.run(&w.norm, &w.q, rows, None, false)?;
            w.q.attention(
                &cache.cross_key,
                &cache.cross_value,
                &w.scores,
                &w.context,
                rows,
                n,
                d,
                t.n_text_head,
                false,
                0,
            )?;
            attn.out.run(&w.context, &w.x, rows, Some(&w.temp), false)?;
            layer.ff_norm.run(&w.x, &w.norm, rows, d)?;
            layer.fc1.run(&w.norm, &w.ff, rows, None, true)?;
            layer.fc2.run(&w.ff, &w.temp, rows, Some(&w.x), false)?;
            w.x.copy_from(&w.temp, 0, rows * d)?;
        }
        self.decoder_norm.run(&w.x, &w.norm, rows, d)?;
        Ok(())
    }
    pub fn transcribe_mel(&mut self, mel: &[f32], max_tokens: usize) -> Result<DecodeResult> {
        ensure!(max_tokens > 0, "max tokens must be positive");
        let started = Instant::now();
        self.encode(mel)?;
        self.device.sync()?;
        let encoder_ms = started.elapsed().as_secs_f64() * 1000.;
        let started = Instant::now();
        self.decoder_tokens(&self.prompt, 0)?;
        let mut tokens = Vec::new();
        let mut ended = false;
        let limit = max_tokens.min(self.dims.text.n_text_ctx - self.prompt.len());
        for step in 0..limit {
            let weight = self.output.as_ref().unwrap_or(&self.embed);
            let width = self.dims.text.n_text_state;
            let row = if step == 0 { self.prompt.len() - 1 } else { 0 };
            self.work.norm.slice(row * width, width)?.linear(
                weight,
                None,
                None,
                &self.work.logits,
                1,
                self.dims.text.n_text_state,
                self.dims.text.n_vocab,
                false,
            )?;
            let next = self.work.logits.argmax(
                &self.allowed,
                &self.work.result,
                self.dims.text.n_vocab,
            )?;
            if next == self.eot {
                ended = true;
                break;
            }
            tokens.push(next);
            if step + 1 < limit {
                self.decoder_tokens(&[next], self.prompt.len() + step)?;
            }
        }
        self.device.sync()?;
        let decoder_ms = started.elapsed().as_secs_f64() * 1000.;
        let ids: Vec<_> = tokens.iter().map(|&t| t as u32).collect();
        let text = self
            .tokenizer
            .decode(&ids, true)
            .map_err(|e| anyhow!(e.to_string()))
            .context("token decoding")?;
        Ok(DecodeResult {
            text,
            tokens,
            ended,
            encoder_ms,
            decoder_ms,
        })
    }
}
