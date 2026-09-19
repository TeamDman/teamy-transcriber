//! Whisper's Slaney log-mel frontend with a resident FFT plan and sparse filters.
//! Skipping zero filter coefficients and wholly zero padding frames preserves
//! the existing FP32 calculation while avoiding work proportional to silence.
use anyhow::Result;
use anyhow::ensure;
use rustfft::Fft;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;
use std::f32::consts::PI;
use std::sync::Arc;

pub const SAMPLES: usize = 480_000;
pub const FRAMES: usize = 3_000;
const FFT: usize = 400;
const HOP: usize = 160;
pub struct Frontend {
    bins: usize,
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    filters: Vec<Vec<(usize, f32)>>,
    padded: Vec<f32>,
    frame: Vec<Complex32>,
    scratch: Vec<Complex32>,
}
impl Frontend {
    pub fn new(bins: usize) -> Result<Self> {
        ensure!(
            bins == 80 || bins == 128,
            "Whisper frontend requires 80 or 128 mel bins"
        );
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT);
        let scratch = vec![Complex32::default(); fft.get_inplace_scratch_len()];
        let window = (0..FFT)
            .map(|i| 0.5 - 0.5 * (2. * PI * i as f32 / FFT as f32).cos())
            .collect();
        let max_mel = hz_to_mel(8_000.);
        let edges: Vec<_> = (0..bins + 2)
            .map(|i| mel_to_hz(max_mel * i as f32 / (bins + 1) as f32))
            .collect();
        let filters = (0..bins)
            .map(|m| {
                let (left, center, right) = (edges[m], edges[m + 1], edges[m + 2]);
                (0..=FFT / 2)
                    .filter_map(|f| {
                        let hz = f as f32 * 16_000. / FFT as f32;
                        let w = ((hz - left) / (center - left))
                            .min((right - hz) / (right - center))
                            .max(0.)
                            * (2. / (right - left));
                        (w > 0.).then_some((f, w))
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            bins,
            fft,
            window,
            filters,
            padded: vec![0.; SAMPLES],
            frame: vec![Complex32::default(); FFT],
            scratch,
        })
    }
    pub fn compute(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            samples.len() <= SAMPLES,
            "Whisper window exceeds 30 seconds; chunk it explicitly"
        );
        ensure!(
            samples.iter().all(|s| s.is_finite()),
            "audio contains non-finite samples"
        );
        self.padded.fill(0.);
        self.padded[..samples.len()].copy_from_slice(samples);
        let mut mel = vec![-10.; self.bins * FRAMES];
        let active = (samples.len() + FFT / 2).div_ceil(HOP).min(FRAMES);
        for time in 0..active {
            for i in 0..FFT {
                let index = (time * HOP + i) as isize - (FFT / 2) as isize;
                let reflected = if index < 0 {
                    (-index) as usize
                } else if index >= SAMPLES as isize {
                    2 * SAMPLES - 2 - index as usize
                } else {
                    index as usize
                };
                self.frame[i] = Complex32::new(self.padded[reflected] * self.window[i], 0.);
            }
            self.fft
                .process_with_scratch(&mut self.frame, &mut self.scratch);
            let mut power = [0.; FFT / 2 + 1];
            for (p, z) in power.iter_mut().zip(&self.frame) {
                *p = z.norm_sqr();
            }
            for (m, filter) in self.filters.iter().enumerate() {
                let mut energy = 0.;
                for &(f, w) in filter {
                    energy += w * power[f];
                }
                mel[m * FRAMES + time] = energy.max(1e-10).log10();
            }
        }
        let max = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in &mut mel {
            *v = (v.max(max - 8.) + 4.) / 4.;
        }
        Ok(mel)
    }
}
fn hz_to_mel(hz: f32) -> f32 {
    let sp = 200. / 3.;
    if hz >= 1000. {
        1000. / sp + (hz / 1000.).ln() / ((6.4_f32).ln() / 27.)
    } else {
        hz / sp
    }
}
fn mel_to_hz(m: f32) -> f32 {
    let sp = 200. / 3.;
    if m >= 1000. / sp {
        1000. * (((6.4_f32).ln() / 27.) * (m - 1000. / sp)).exp()
    } else {
        sp * m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tone_matches_existing_frontend_fixture() {
        let samples: Vec<_> = (0..16_000)
            .map(|i| 0.25 * (2. * PI * 440. * i as f32 / 16_000.).sin())
            .collect();
        let mel = Frontend::new(80).unwrap().compute(&samples).unwrap();
        for (m, t, v) in [
            (0, 0, 0.832764),
            (0, 1, 0.320756),
            (10, 10, 1.1982229),
            (40, 100, 0.11720395),
            (79, 2999, -0.7123103),
        ] {
            assert!(
                (mel[m * FRAMES + t] - v).abs() < 1e-3,
                "{m}/{t}: {} != {v}",
                mel[m * FRAMES + t]
            );
        }
    }
    #[test]
    fn silence_and_reuse_do_not_leak_previous_audio() {
        for bins in [80, 128] {
            let mut f = Frontend::new(bins).unwrap();
            f.compute(&vec![0.25; 16_000]).unwrap();
            assert!(f.compute(&[]).unwrap().iter().all(|&x| x == -1.5));
            assert!(f.compute(&[f32::NAN]).is_err());
            assert!(f.compute(&vec![0.; SAMPLES + 1]).is_err());
        }
    }
}
