//! Source-defined PhoneticXeus E-Branchformer/CTC inference.
//! Architecture follows changelinglab/PhoneticXeus (Apache-2.0), including
//! self-conditioning after layers 4, 8 and 12. No executable model graph is loaded.
use super::Buffer;
use super::Device;
use super::Linear;
use super::Norm;
use super::Weights;
use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use serde::Deserialize;
use serde::Serialize;
use std::path::Path;
use std::rc::Rc;

const D: usize = 1024;
const VOCAB: usize = 428;
#[derive(Deserialize)]
struct Config {
    model_type: String,
    sampling_rate: usize,
    interctc_layer_idx: Vec<usize>,
    interctc_use_conditioning: bool,
}
struct Conv {
    w: Buffer,
    b: Buffer,
    input: usize,
    output: usize,
    k: usize,
    stride: usize,
    pad: usize,
    groups: usize,
}
impl Conv {
    #[expect(
        clippy::too_many_arguments,
        reason = "Convolution shape is explicit when binding named checkpoint tensors"
    )]
    fn load(
        w: &mut Weights,
        p: &str,
        input: usize,
        output: usize,
        k: usize,
        stride: usize,
        pad: usize,
        groups: usize,
    ) -> Result<Self> {
        Ok(Self {
            w: w.take(&format!("{p}.weight"), &[output, input / groups, k])?,
            b: w.take(&format!("{p}.bias"), &[output])?,
            input,
            output,
            k,
            stride,
            pad,
            groups,
        })
    }
    fn run(&self, d: &Rc<Device>, x: &Buffer, time: usize, rows: usize) -> Result<Buffer> {
        let y = d.alloc(rows * self.output)?;
        let col = d.alloc(rows * (self.input / self.groups) * self.k)?;
        x.phone_conv(
            &self.w,
            &self.b,
            &col,
            &y,
            time,
            self.input,
            self.output,
            self.k,
            self.stride,
            self.pad,
            self.groups,
            rows,
        )?;
        Ok(y)
    }
}
struct Block {
    mac_norm: Norm,
    mac1: Linear,
    mac2: Linear,
    mha: Norm,
    q: Linear,
    k: Linear,
    v: Linear,
    attout: Linear,
    mlp: Norm,
    proj1: Linear,
    gate_norm: Norm,
    gate: Conv,
    proj2: Linear,
    fusion: Conv,
    merge: Linear,
    ff_norm: Norm,
    ff1: Linear,
    ff2: Linear,
    final_norm: Norm,
}
impl Block {
    fn load(w: &mut Weights, p: &str) -> Result<Self> {
        Ok(Self {
            mac_norm: w.norm(&format!("{p}.norm_ff_macaron"), D)?,
            mac1: w.linear(&format!("{p}.feed_forward_macaron.w_1"), D, 4096, true)?,
            mac2: w.linear(&format!("{p}.feed_forward_macaron.w_2"), 4096, D, true)?,
            mha: w.norm(&format!("{p}.norm_mha"), D)?,
            q: w.linear(&format!("{p}.attn.linear_q"), D, D, true)?,
            k: w.linear(&format!("{p}.attn.linear_k"), D, D, true)?,
            v: w.linear(&format!("{p}.attn.linear_v"), D, D, true)?,
            attout: w.linear(&format!("{p}.attn.linear_out"), D, D, true)?,
            mlp: w.norm(&format!("{p}.norm_mlp"), D)?,
            proj1: w.linear(&format!("{p}.cgmlp.channel_proj1.0"), D, 4096, true)?,
            gate_norm: w.norm(&format!("{p}.cgmlp.csgu.norm"), 2048)?,
            gate: Conv::load(
                w,
                &format!("{p}.cgmlp.csgu.conv"),
                2048,
                2048,
                31,
                1,
                15,
                2048,
            )?,
            proj2: w.linear(&format!("{p}.cgmlp.channel_proj2"), 2048, D, true)?,
            fusion: Conv::load(
                w,
                &format!("{p}.depthwise_conv_fusion"),
                2048,
                2048,
                31,
                1,
                15,
                2048,
            )?,
            merge: w.linear(&format!("{p}.merge_proj"), 2048, D, true)?,
            ff_norm: w.norm(&format!("{p}.norm_ff"), D)?,
            ff1: w.linear(&format!("{p}.feed_forward.w_1"), D, 4096, true)?,
            ff2: w.linear(&format!("{p}.feed_forward.w_2"), 4096, D, true)?,
            final_norm: w.norm(&format!("{p}.norm_final"), D)?,
        })
    }
    fn run(&self, d: &Rc<Device>, x: &Buffer, t: usize) -> Result<Buffer> {
        let x = ff(d, x, t, &self.mac_norm, &self.mac1, &self.mac2)?;
        let a = norm(d, &self.mha, &x, t, D, 1e-12)?;
        let q = linear(d, &self.q, &a, t, false)?;
        let k = linear(d, &self.k, &a, t, false)?;
        let v = linear(d, &self.v, &a, t, false)?;
        let scores = d.alloc(8 * t * t)?;
        let att = d.alloc(t * D)?;
        q.attention(&k, &v, &scores, &att, t, t, D, 8, false, 0)?;
        let att = linear(d, &self.attout, &att, t, false)?;
        let m = norm(d, &self.mlp, &x, t, D, 1e-12)?;
        let m = linear(d, &self.proj1, &m, t, true)?;
        let left = d.alloc(t * 2048)?;
        let right = d.alloc(t * 2048)?;
        m.rows(&left, t, 2048, 4096, 2048, 0, 0)?;
        m.rows(&right, t, 2048, 4096, 2048, 2048, 0)?;
        let right = norm(d, &self.gate_norm, &right, t, 2048, 1e-12)?;
        let gate = self.gate.run(d, &right, t, t)?;
        left.pointwise(Some(&gate), &left, t * 2048, 3, 1.)?;
        let m = linear(d, &self.proj2, &left, t, false)?;
        let cat = d.alloc(t * 2048)?;
        att.rows(&cat, t, D, D, 2048, 0, 0)?;
        m.rows(&cat, t, D, D, 2048, 0, D)?;
        let merged = self.fusion.run(d, &cat, t, t)?;
        merged.pointwise(Some(&cat), &merged, t * 2048, 2, 1.)?;
        let merged = linear(d, &self.merge, &merged, t, false)?;
        merged.pointwise(Some(&x), &merged, t * D, 2, 1.)?;
        let out = ff(d, &merged, t, &self.ff_norm, &self.ff1, &self.ff2)?;
        norm(d, &self.final_norm, &out, t, D, 1e-12)
    }
}
fn linear(d: &Rc<Device>, l: &Linear, x: &Buffer, t: usize, gelu: bool) -> Result<Buffer> {
    let out = d.alloc(t * l.output)?;
    l.run(x, &out, t, None, gelu)?;
    Ok(out)
}
fn norm(d: &Rc<Device>, n: &Norm, x: &Buffer, t: usize, width: usize, eps: f32) -> Result<Buffer> {
    let out = d.alloc(t * width)?;
    x.phone_norm(&n.w, &n.b, &out, t, width, eps)?;
    Ok(out)
}
fn ff(d: &Rc<Device>, x: &Buffer, t: usize, n: &Norm, a: &Linear, b: &Linear) -> Result<Buffer> {
    let y = norm(d, n, x, t, D, 1e-12)?;
    let y = linear(d, a, &y, t, false)?;
    y.pointwise(None, &y, t * 4096, 0, 1.)?;
    let y = linear(d, b, &y, t, false)?;
    y.pointwise(Some(x), &y, t * D, 2, 0.5)?;
    Ok(y)
}

#[derive(Debug, Serialize)]
pub struct PhoneResult {
    pub phones: Vec<String>,
    pub ipa: String,
    pub frames: usize,
}
/// Resident, thread-confined phone recognizer. Audio must be mono16kHz.
pub struct PhoneModel {
    d: Rc<Device>,
    frontend: Vec<(Conv, Norm)>,
    pre: Linear,
    pos: Conv,
    blocks: Vec<Block>,
    after: Norm,
    ctc: Linear,
    condition: Linear,
    vocab: Vec<String>,
    blank: usize,
}
impl PhoneModel {
    pub fn load(root: &Path, device: i32) -> Result<Self> {
        let c: Config = serde_json::from_reader(std::fs::File::open(root.join("config.json"))?)?;
        ensure!(
            c.model_type == "phoneticxeus"
                && c.sampling_rate == 16000
                && c.interctc_layer_idx == [4, 8, 12]
                && c.interctc_use_conditioning,
            "unsupported PhoneticXeus architecture/conditioning configuration"
        );
        let vocab: std::collections::HashMap<String, usize> =
            serde_json::from_reader(std::fs::File::open(root.join("ipa_vocab.json"))?)?;
        ensure!(vocab.len() == VOCAB, "unsupported phone vocabulary");
        let blank = *vocab.get("<blank>").context("missing CTC blank token")?;
        let mut tokens = vec![String::new(); VOCAB];
        for (token, id) in vocab {
            ensure!(
                id < VOCAB && tokens[id].is_empty(),
                "invalid phone token ID"
            );
            tokens[id] = token;
        }
        let d = Device::new(device, false)?;
        let mut w = Weights::open(root, &d)?;
        let mut frontend = Vec::new();
        let mut input = 1;
        for (i, (kernel, stride)) in [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]
            .into_iter()
            .enumerate()
        {
            let p = format!("model.frontend.layers.{i}");
            frontend.push((
                Conv::load(
                    &mut w,
                    &format!("{p}.conv"),
                    input,
                    512,
                    kernel,
                    stride,
                    0,
                    1,
                )?,
                w.norm(&format!("{p}.layer_norm"), 512)?,
            ));
            input = 512;
        }
        let pre = w.linear("model.preencoder.linear_out", 512, D, true)?;
        let p = "model.encoder.embed.0.convs.0";
        let g = w
            .take(
                &format!("{p}.parametrizations.weight.original0"),
                &[1, 1, 128],
            )?
            .read(128)?;
        let mut v = w
            .take(
                &format!("{p}.parametrizations.weight.original1"),
                &[D, 64, 128],
            )?
            .read(D * 64 * 128)?;
        for (k, &gain) in g.iter().enumerate() {
            let sq: f64 = v
                .iter()
                .skip(k)
                .step_by(128)
                .map(|&x| f64::from(x).powi(2))
                .sum();
            ensure!(sq > 0., "invalid position weight norm");
            let scale = f64::from(gain) / sq.sqrt();
            for x in v.iter_mut().skip(k).step_by(128) {
                *x = (f64::from(*x) * scale) as f32;
            }
        }
        let pos = Conv {
            w: d.upload(&v)?,
            b: w.take(&format!("{p}.bias"), &[D])?,
            input: D,
            output: D,
            k: 128,
            stride: 1,
            pad: 64,
            groups: 16,
        };
        let mut blocks = Vec::new();
        for i in 0..19 {
            blocks.push(Block::load(&mut w, &format!("model.encoder.encoders.{i}"))?);
        }
        let after = w.norm("model.encoder.after_norm", D)?;
        let ctc = w.linear("model.ctc.ctc_lo", D, VOCAB, true)?;
        let condition = w.linear("model.encoder.conditioning_layer", VOCAB, D, true)?;
        ensure!(w.0.is_empty(), "unrecognized phone model tensors");
        Ok(Self {
            d,
            frontend,
            pre,
            pos,
            blocks,
            after,
            ctc,
            condition,
            vocab: tokens,
            blank,
        })
    }
    /// Full utterance recognition. Explicit30s bound caps quadratic attention memory.
    pub fn logits(&self, audio: &[f32]) -> Result<(Vec<f32>, usize)> {
        ensure!(
            (400..=480_000).contains(&audio.len()),
            "phone recognition requires25ms..30s of mono16kHz audio"
        );
        ensure!(
            audio.iter().all(|x| x.is_finite()),
            "non-finite phone input"
        );
        let mean = audio.iter().map(|&x| f64::from(x)).sum::<f64>() / audio.len() as f64;
        let var = audio
            .iter()
            .map(|&x| (f64::from(x) - mean).powi(2))
            .sum::<f64>()
            / audio.len() as f64;
        let data: Vec<f32> = audio
            .iter()
            .map(|&x| ((f64::from(x) - mean) / (var + 1e-5).sqrt()) as f32)
            .collect();
        let d = &self.d;
        let mut x = d.upload(&data)?;
        let mut t = audio.len();
        for (conv, n) in &self.frontend {
            let out = (t - conv.k) / conv.stride + 1;
            x = conv.run(d, &x, t, out)?;
            t = out;
            x = norm(d, n, &x, t, 512, 1e-5)?;
            x.pointwise(None, &x, t * 512, 1, 1.)?;
        }
        x = linear(d, &self.pre, &x, t, false)?;
        x = self.pos.run(d, &x, t, t)?;
        x.pointwise(None, &x, t * D, 1, 1.)?;
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.run(d, &x, t)?;
            if [4, 8, 12].contains(&(i + 1)) {
                let c = linear(d, &self.ctc, &x, t, false)?;
                c.phone_softmax(&c, t, VOCAB)?;
                let cond = linear(d, &self.condition, &c, t, false)?;
                cond.pointwise(Some(&x), &x, t * D, 2, 1.)?;
            }
        }
        x = norm(d, &self.after, &x, t, D, 1e-12)?;
        let logits = linear(d, &self.ctc, &x, t, false)?;
        Ok((logits.read(t * VOCAB)?, t))
    }
    pub fn recognize(&self, audio: &[f32]) -> Result<PhoneResult> {
        let (logits, frames) = self.logits(audio)?;
        let ids: Vec<usize> = logits
            .chunks_exact(VOCAB)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                    .unwrap()
                    .0
            })
            .collect();
        let collapsed = collapse(&ids, self.blank);
        let phones: Vec<String> = collapsed
            .into_iter()
            .map(|i| self.vocab[i].clone())
            .filter(|t| !(t.starts_with('<') && t.ends_with('>')))
            .collect();
        let ipa = phones.concat();
        Ok(PhoneResult {
            phones,
            ipa,
            frames,
        })
    }
}
fn collapse(ids: &[usize], blank: usize) -> Vec<usize> {
    ids.iter()
        .enumerate()
        .filter_map(|(i, &v)| (v != blank && (i == 0 || ids[i - 1] != v)).then_some(v))
        .collect()
}
#[cfg(test)]
mod tests {
    #[test]
    fn ctc_preserves_repetition_separated_by_blank() {
        assert_eq!(super::collapse(&[0, 1, 1, 0, 1, 2, 2, 0], 0), [1, 1, 2]);
    }
}
