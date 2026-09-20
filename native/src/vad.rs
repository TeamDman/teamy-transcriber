//! Source-defined Silero 16 kHz inference. Weights are ordinary FP32 safetensors.
//! The 512-sample step, context, convolutions and LSTM are defined here; no graph
//! interpreter, Torch, ONNX runtime or GPU is used. See licenses/silero.txt.
use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use safetensors::Dtype;
use safetensors::SafeTensors;
use std::path::Path;

pub const WINDOW: usize = 512;
const HIDDEN: usize = 128;

fn tensor(tensors: &SafeTensors<'_>, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let view = tensors
        .tensor(name)
        .with_context(|| format!("missing VAD tensor {name}"))?;
    ensure!(
        view.dtype() == Dtype::F32 && view.shape() == shape,
        "VAD tensor {name}: expected F32 {shape:?}"
    );
    let values: Vec<_> = view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "non-finite VAD tensor {name}"
    );
    Ok(values)
}

fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx_dot(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    // Caller checks CPU support. All loads are unaligned and bounded by both slices.
    unsafe {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = s0;
        let mut i = 0;
        while i + 16 <= a.len() {
            s0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
                s0,
            );
            s1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 8)),
                _mm256_loadu_ps(b.as_ptr().add(i + 8)),
                s1,
            );
            i += 16;
        }
        let mut lanes = [0.; 8];
        _mm256_storeu_ps(lanes.as_mut_ptr(), _mm256_add_ps(s0, s1));
        let mut sum: f32 = lanes.iter().sum();
        while i < a.len() {
            sum += a[i] * b[i];
            i += 1;
        }
        sum
    }
}

#[derive(Clone, Copy)]
struct Dot(unsafe fn(&[f32], &[f32]) -> f32);
impl Dot {
    fn new() -> Self {
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return Self(avx_dot);
        }
        Self(scalar_dot)
    }
    fn run(self, a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        // The only constructor selects a supported kernel; lengths match.
        unsafe { (self.0)(a, b) }
    }
}

struct Conv {
    input: usize,
    output: usize,
    stride: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}
impl Conv {
    fn load(
        tensors: &SafeTensors<'_>,
        index: usize,
        input: usize,
        output: usize,
        stride: usize,
    ) -> Result<Self> {
        let prefix = format!("encoder.{index}.reparam_conv");
        Ok(Self {
            input,
            output,
            stride,
            weight: tensor(tensors, &format!("{prefix}.weight"), &[output, input, 3])?,
            bias: tensor(tensors, &format!("{prefix}.bias"), &[output])?,
        })
    }
    fn run(&self, input: &[f32], frames: usize, output: &mut [f32], dot: Dot) {
        let output_frames = frames.div_ceil(self.stride);
        let width = self.input * 3;
        let mut column = [0.; 129 * 3];
        for t in 0..output_frames {
            for c in 0..self.input {
                for k in 0..3 {
                    let p = (t * self.stride + k) as isize - 1;
                    column[c * 3 + k] = if p >= 0 && (p as usize) < frames {
                        input[c * frames + p as usize]
                    } else {
                        0.
                    };
                }
            }
            for c in 0..self.output {
                output[c * output_frames + t] = (dot
                    .run(&column[..width], &self.weight[c * width..(c + 1) * width])
                    + self.bias[c])
                    .max(0.);
            }
        }
    }
}

/// One reusable 16 kHz voice-activity model. `reset` separates recordings.
pub struct Silero {
    basis: Vec<f32>,
    conv: [Conv; 4],
    weight_ih: Vec<f32>,
    weight_hh: Vec<f32>,
    bias_ih: Vec<f32>,
    bias_hh: Vec<f32>,
    final_weight: Vec<f32>,
    final_bias: f32,
    context: [f32; 64],
    hidden: [f32; HIDDEN],
    cell: [f32; HIDDEN],
    dot: Dot,
}

fn sigmoid(x: f32) -> f32 {
    1. / (1. + (-x).exp())
}

/// Half-open sample range in the original 16 kHz recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SpeechSegment {
    pub start: usize,
    pub end: usize,
}

/// Silero's default timestamp policy as used by WhisperX (16 kHz only).
/// Keeps >250 ms speech, waits 100 ms for silence, pads by 30 ms, and splits
/// long speech at its longest eligible (>98 ms) internal silence.
pub fn speech_segments(
    probabilities: &[f32],
    samples: usize,
    threshold: f64,
    max_seconds: f64,
) -> Result<Vec<SpeechSegment>> {
    ensure!(
        probabilities.len() == samples.div_ceil(WINDOW),
        "VAD probability count does not match audio"
    );
    ensure!(
        threshold.is_finite() && threshold > 0. && threshold < 1.,
        "invalid VAD threshold"
    );
    ensure!(
        max_seconds.is_finite() && max_seconds > 0.342,
        "VAD maximum duration too short"
    );
    ensure!(
        probabilities
            .iter()
            .all(|p| p.is_finite() && (0. ..=1.).contains(p)),
        "invalid VAD probabilities"
    );
    let negative = (threshold - 0.15).max(0.01);
    let maximum = max_seconds * 16000. - WINDOW as f64 - 960.;
    let mut start = None;
    let mut silence = None;
    let mut longest: Option<(usize, usize)> = None;
    let mut result = Vec::new();
    // Port of get_speech_timestamps's default longest-silence policy. See the
    // retained Silero MIT notice; alternative legacy policy is not exposed.
    for (i, &p) in probabilities.iter().enumerate() {
        let position = i * WINDOW;
        if f64::from(p) >= threshold {
            if let Some(end) = silence.take() {
                let duration = position - end;
                if duration > 1568 && longest.is_none_or(|(_, old)| duration > old) {
                    longest = Some((end, duration));
                }
            }
            if start.is_none() {
                start = Some(position);
                continue;
            }
        }
        if let Some(begin) = start
            && (position - begin) as f64 > maximum
        {
            if let Some((end, duration)) = longest.take() {
                result.push(SpeechSegment { start: begin, end });
                // Equivalent to upstream next_start < prev_end + cur_sample.
                start = (duration < position).then_some(end + duration);
                silence = None;
            } else {
                result.push(SpeechSegment {
                    start: begin,
                    end: position,
                });
                start = None;
                silence = None;
                continue;
            }
        }
        if f64::from(p) < negative
            && let Some(begin) = start
        {
            let end = *silence.get_or_insert(position);
            if position - end >= 1600 {
                if end - begin > 4000 {
                    result.push(SpeechSegment { start: begin, end });
                }
                start = None;
                silence = None;
                longest = None;
            }
        }
    }
    if let Some(begin) = start
        && samples - begin > 4000
    {
        result.push(SpeechSegment {
            start: begin,
            end: samples,
        });
    }
    if let Some(first) = result.first_mut() {
        first.start = first.start.saturating_sub(480);
    }
    for i in 0..result.len() {
        if i + 1 == result.len() {
            result[i].end = result[i].end.saturating_add(480).min(samples);
        } else {
            let gap = result[i + 1].start - result[i].end;
            if gap < 960 {
                result[i].end += gap / 2;
                result[i + 1].start -= gap / 2;
            } else {
                result[i].end = result[i].end.saturating_add(480).min(samples);
                result[i + 1].start = result[i + 1].start.saturating_sub(480);
            }
        }
    }
    Ok(result)
}

/// Merge adjacent speech ranges using WhisperX's maximum-span rule. Silence
/// inside a merged span is retained; timestamps still refer to original audio.
pub fn merge_segments(
    segments: &[SpeechSegment],
    max_samples: usize,
) -> Result<Vec<SpeechSegment>> {
    ensure!(max_samples > 0, "zero VAD merge duration");
    let mut merged: Vec<SpeechSegment> = Vec::new();
    let mut previous_end = 0;
    for &segment in segments {
        ensure!(
            segment.start >= previous_end && segment.end > segment.start,
            "VAD segments must be nonempty, sorted and non-overlapping"
        );
        ensure!(
            segment.end - segment.start <= max_samples,
            "VAD segment exceeds merge duration"
        );
        if let Some(last) = merged.last_mut()
            && segment.end - last.start <= max_samples
        {
            last.end = segment.end;
        } else {
            merged.push(segment);
        }
        previous_end = segment.end;
    }
    Ok(merged)
}

impl Silero {
    pub fn load(path: &Path) -> Result<Self> {
        Self::from_bytes(
            &std::fs::read(path).with_context(|| format!("reading {}", path.display()))?,
        )
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let tensors = SafeTensors::deserialize(bytes)?;
        ensure!(
            tensors.len() == 15,
            "expected exactly 15 Silero 16 kHz tensors"
        );
        Ok(Self {
            basis: tensor(&tensors, "stft.forward_basis_buffer", &[258, 1, 256])?,
            conv: [
                Conv::load(&tensors, 0, 129, 128, 1)?,
                Conv::load(&tensors, 1, 128, 64, 2)?,
                Conv::load(&tensors, 2, 64, 64, 2)?,
                Conv::load(&tensors, 3, 64, 128, 1)?,
            ],
            weight_ih: tensor(&tensors, "decoder.rnn.weight_ih", &[512, 128])?,
            weight_hh: tensor(&tensors, "decoder.rnn.weight_hh", &[512, 128])?,
            bias_ih: tensor(&tensors, "decoder.rnn.bias_ih", &[512])?,
            bias_hh: tensor(&tensors, "decoder.rnn.bias_hh", &[512])?,
            final_weight: tensor(&tensors, "decoder.decoder.2.weight", &[1, 128, 1])?,
            final_bias: tensor(&tensors, "decoder.decoder.2.bias", &[1])?[0],
            context: [0.; 64],
            hidden: [0.; HIDDEN],
            cell: [0.; HIDDEN],
            dot: Dot::new(),
        })
    }

    pub fn reset(&mut self) {
        self.context.fill(0.);
        self.hidden.fill(0.);
        self.cell.fill(0.);
    }

    /// Advance by exactly 32 ms. The caller zero-pads a final partial window.
    /// Validation happens before state mutation, so invalid input is retryable.
    pub fn step(&mut self, audio: &[f32; WINDOW]) -> Result<f32> {
        ensure!(
            audio.iter().all(|x| x.is_finite() && x.abs() <= 1.),
            "VAD requires finite normalized PCM in [-1, 1]"
        );
        Ok(self.step_validated(audio))
    }

    fn step_validated(&mut self, audio: &[f32; WINDOW]) -> f32 {
        let mut padded = [0.; 640];
        padded[..64].copy_from_slice(&self.context);
        padded[64..576].copy_from_slice(audio);
        for i in 0..64 {
            padded[576 + i] = padded[574 - i];
        }
        self.context.copy_from_slice(&audio[448..]);
        let mut spectrum = [0.; 129 * 4];
        for bin in 0..129 {
            for frame in 0..4 {
                let window = &padded[frame * 128..frame * 128 + 256];
                let real = self
                    .dot
                    .run(window, &self.basis[bin * 256..(bin + 1) * 256]);
                let imag = self
                    .dot
                    .run(window, &self.basis[(bin + 129) * 256..(bin + 130) * 256]);
                spectrum[bin * 4 + frame] = (real * real + imag * imag).sqrt();
            }
        }
        let mut x0 = [0.; 128 * 4];
        let mut x1 = [0.; 64 * 2];
        let mut x2 = [0.; 64];
        let mut x3 = [0.; 128];
        self.conv[0].run(&spectrum, 4, &mut x0, self.dot);
        self.conv[1].run(&x0, 4, &mut x1, self.dot);
        self.conv[2].run(&x1, 2, &mut x2, self.dot);
        self.conv[3].run(&x2, 1, &mut x3, self.dot);
        let mut gates = [0.; 512];
        for (i, gate) in gates.iter_mut().enumerate() {
            let range = i * HIDDEN..(i + 1) * HIDDEN;
            *gate = (self.dot.run(&x3, &self.weight_ih[range.clone()]) + self.bias_ih[i])
                + (self.dot.run(&self.hidden, &self.weight_hh[range]) + self.bias_hh[i]);
        }
        for i in 0..HIDDEN {
            self.cell[i] =
                sigmoid(gates[i + 128]) * self.cell[i] + sigmoid(gates[i]) * gates[i + 256].tanh();
            self.hidden[i] = sigmoid(gates[i + 384]) * self.cell[i].tanh();
            x3[i] = self.hidden[i].max(0.);
        }
        sigmoid(self.dot.run(&x3, &self.final_weight) + self.final_bias)
    }

    /// Independent complete recording, retaining one probability per 512 samples.
    pub fn probabilities(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            audio.iter().all(|x| x.is_finite() && x.abs() <= 1.),
            "VAD requires finite normalized PCM in [-1, 1]"
        );
        self.reset();
        let mut result = Vec::with_capacity(audio.len().div_ceil(WINDOW));
        for chunk in audio.chunks(WINDOW) {
            let mut window = [0.; WINDOW];
            window[..chunk.len()].copy_from_slice(chunk);
            result.push(self.step_validated(&window));
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_independent_f64_sum() {
        let dot = Dot::new();
        for len in [0, 1, 7, 16, 37, 128, 256, 387] {
            let a: Vec<f32> = (0..len + 1).map(|i| (i as f32 * 0.7).sin()).collect();
            let b: Vec<f32> = (0..len + 1).map(|i| (i as f32 * 0.3).cos()).collect();
            let a = &a[1..];
            let b = &b[1..];
            let expected: f64 = a
                .iter()
                .zip(b)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            assert!((f64::from(dot.run(a, b)) - expected).abs() < 0.00001);
        }
    }

    #[test]
    fn segmentation_edges_and_long_speech() -> Result<()> {
        assert!(speech_segments(&[], 0, 0.5, 30.)?.is_empty());
        assert!(speech_segments(&[0.; 20], 20 * WINDOW, 0.5, 30.)?.is_empty());
        assert!(speech_segments(&[1.; 7], 7 * WINDOW, 0.5, 30.)?.is_empty());
        let mut p = vec![0.; 4];
        p.extend([0.9; 10]);
        p.extend([0.; 7]);
        p.extend([0.9; 10]);
        let segments = speech_segments(&p, p.len() * WINDOW - 17, 0.5, 30.)?;
        assert_eq!(
            segments,
            vec![
                SpeechSegment {
                    start: 1568,
                    end: 7648
                },
                SpeechSegment {
                    start: 10272,
                    end: p.len() * WINDOW - 17
                }
            ]
        );
        assert_eq!(
            merge_segments(&segments, 480000)?,
            vec![SpeechSegment {
                start: 1568,
                end: p.len() * WINDOW - 17
            }]
        );
        let long = speech_segments(&[1.; 1000], 512000, 0.5, 30.)?;
        assert_eq!(
            long,
            vec![
                SpeechSegment {
                    start: 0,
                    end: 478976
                },
                SpeechSegment {
                    start: 478976,
                    end: 512000
                }
            ]
        );
        assert!(speech_segments(&[f32::NAN], 1, 0.5, 30.).is_err());
        assert!(speech_segments(&[0.], 0, 0.5, 30.).is_err());
        assert!(speech_segments(&[], 0, f64::NAN, 30.).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires SILERO_TEST_WEIGHTS exported from the local reference"]
    fn model_reset_padding_and_rejected_input_preserve_state() -> Result<()> {
        let path = std::env::var("SILERO_TEST_WEIGHTS")?;
        let mut model = Silero::load(Path::new(&path))?;
        let audio: Vec<_> = (0..WINDOW * 4 + 17)
            .map(|i| (i as f32 * 0.037).sin() * 0.2)
            .collect();
        let first = model.probabilities(&audio)?;
        assert_eq!(first.len(), 5);
        assert!(
            first
                .iter()
                .all(|p| p.is_finite() && (0. ..=1.).contains(p))
        );
        assert_eq!(first, model.probabilities(&audio)?);
        model.reset();
        for (index, chunk) in audio.chunks(WINDOW).enumerate() {
            let mut padded = [0.; WINDOW];
            padded[..chunk.len()].copy_from_slice(chunk);
            assert_eq!(model.step(&padded)?, first[index]);
        }
        let next = model.step(&[0.; WINDOW])?;
        model.probabilities(&audio)?;
        assert!(model.step(&[f32::NAN; WINDOW]).is_err());
        assert!(model.probabilities(&[2.]).is_err());
        assert_eq!(model.step(&[0.; WINDOW])?, next);
        assert!(model.probabilities(&[])?.is_empty());
        // Scalar fallback is a real independent path, not an AVX alias.
        model.dot = Dot(scalar_dot);
        let scalar = model.probabilities(&audio)?;
        assert!(
            first
                .iter()
                .zip(scalar)
                .all(|(a, b)| (a - b).abs() < 0.0001)
        );
        Ok(())
    }

    #[test]
    fn loader_rejects_shape_dtype_missing_and_nonfinite() -> Result<()> {
        use safetensors::tensor::TensorView;
        use safetensors::tensor::serialize;
        assert!(Silero::from_bytes(&[]).is_err());
        let bytes = [0u8; 8];
        let data = serialize([("x", TensorView::new(Dtype::F32, vec![2], &bytes)?)], None)?;
        let tensors = SafeTensors::deserialize(&data)?;
        assert_eq!(tensor(&tensors, "x", &[2])?, [0., 0.]);
        assert!(tensor(&tensors, "x", &[1, 2]).is_err());
        assert!(tensor(&tensors, "missing", &[2]).is_err());
        assert!(Silero::from_bytes(&data).is_err());
        let data = serialize([("x", TensorView::new(Dtype::I32, vec![2], &bytes)?)], None)?;
        assert!(tensor(&SafeTensors::deserialize(&data)?, "x", &[2]).is_err());
        let nan = f32::NAN.to_le_bytes();
        let data = serialize([("x", TensorView::new(Dtype::F32, vec![1], &nan)?)], None)?;
        assert!(tensor(&SafeTensors::deserialize(&data)?, "x", &[1]).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires SILERO_POLICY_CASES from tools/export_silero_policy_cases.py"]
    fn segmentation_matches_upstream_oracle() -> Result<()> {
        #[derive(serde::Deserialize)]
        struct Case {
            pattern: Vec<(f32, usize)>,
            samples: usize,
            threshold: f64,
            max_seconds: f64,
            segments: Vec<SpeechSegment>,
        }
        #[derive(serde::Deserialize)]
        struct Receipt {
            cases: Vec<Case>,
        }
        let bytes = std::fs::read(std::env::var("SILERO_POLICY_CASES")?)?;
        let receipt: Receipt = serde_json::from_slice(&bytes)?;
        assert!(receipt.cases.len() >= 100);
        for (index, case) in receipt.cases.iter().enumerate() {
            let probabilities: Vec<_> = case
                .pattern
                .iter()
                .flat_map(|&(p, n)| std::iter::repeat_n(p, n))
                .collect();
            assert_eq!(
                speech_segments(
                    &probabilities,
                    case.samples,
                    case.threshold,
                    case.max_seconds
                )?,
                case.segments,
                "oracle case {index}"
            );
        }
        Ok(())
    }
}
