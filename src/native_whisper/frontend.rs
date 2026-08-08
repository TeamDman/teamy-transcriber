use crate::media::WHISPER_SAMPLE_RATE_HZ as TRANSCRIPTION_SAMPLE_RATE;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;
use std::f32::consts::PI;

pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 80;
pub const CHUNK_LENGTH_SECONDS: usize = 30;
pub const N_SAMPLES: usize = CHUNK_LENGTH_SECONDS * TRANSCRIPTION_SAMPLE_RATE as usize;
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH;

#[derive(Clone, Debug, PartialEq)]
pub struct WhisperLogMelSpectrogram {
    pub n_mels: usize,
    pub n_frames: usize,
    pub values: Vec<f32>,
}

impl WhisperLogMelSpectrogram {
    #[must_use]
    pub fn at(&self, mel_bin: usize, frame: usize) -> f32 {
        self.values[mel_bin * self.n_frames + frame]
    }

    #[must_use]
    pub fn min_value(&self) -> f32 {
        self.values.iter().copied().fold(f32::INFINITY, f32::min)
    }

    #[must_use]
    pub fn max_value(&self) -> f32 {
        self.values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
    }
}

#[must_use]
pub fn pad_or_trim(samples: &[f32], target_len: usize) -> Vec<f32> {
    match samples.len().cmp(&target_len) {
        std::cmp::Ordering::Greater => samples[..target_len].to_vec(),
        std::cmp::Ordering::Equal => samples.to_vec(),
        std::cmp::Ordering::Less => {
            let mut padded = Vec::with_capacity(target_len);
            padded.extend_from_slice(samples);
            padded.resize(target_len, 0.0);
            padded
        }
    }
}

#[must_use]
pub fn whisper_log_mel_spectrogram(samples: &[f32]) -> WhisperLogMelSpectrogram {
    let padded = pad_or_trim(samples, N_SAMPLES);
    let power = stft_power_spectrogram(&padded);
    let filters = mel_filter_bank();
    let mut mel_spec = vec![0.0_f32; N_MELS * N_FRAMES];

    for mel_bin in 0..N_MELS {
        for frame in 0..N_FRAMES {
            let mut energy = 0.0_f32;
            for freq_bin in 0..=N_FFT / 2 {
                let filter = filters[mel_bin * (N_FFT / 2 + 1) + freq_bin];
                let magnitude = power[frame * (N_FFT / 2 + 1) + freq_bin];
                energy += filter * magnitude;
            }
            mel_spec[mel_bin * N_FRAMES + frame] = energy;
        }
    }

    let mut log_spec = mel_spec
        .into_iter()
        .map(|value| value.max(1e-10).log10())
        .collect::<Vec<_>>();
    let max_value = log_spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let floor = max_value - 8.0;

    for value in &mut log_spec {
        *value = value.max(floor);
        *value = (*value + 4.0) / 4.0;
    }

    WhisperLogMelSpectrogram {
        n_mels: N_MELS,
        n_frames: N_FRAMES,
        values: log_spec,
    }
}

fn stft_power_spectrogram(samples: &[f32]) -> Vec<f32> {
    let window = hann_window(N_FFT);
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut frame_buffer = vec![Complex32::new(0.0, 0.0); N_FFT];
    let mut power = vec![0.0_f32; N_FRAMES * (N_FFT / 2 + 1)];
    let pad = (N_FFT / 2) as isize;

    for frame in 0..N_FRAMES {
        for sample_index in 0..N_FFT {
            let centered_index = frame * HOP_LENGTH + sample_index;
            let reflected_index = reflect_index(centered_index as isize - pad, samples.len());
            frame_buffer[sample_index] =
                Complex32::new(samples[reflected_index] * window[sample_index], 0.0);
        }

        fft.process(&mut frame_buffer);
        for freq_bin in 0..=N_FFT / 2 {
            power[frame * (N_FFT / 2 + 1) + freq_bin] = frame_buffer[freq_bin].norm_sqr();
        }
    }

    power
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|index| 0.5 - 0.5 * (2.0 * PI * index as f32 / size as f32).cos())
        .collect()
}

fn reflect_index(mut index: isize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }

    let len = len as isize;
    while index < 0 || index >= len {
        if index < 0 {
            index = -index;
        } else {
            index = 2 * len - 2 - index;
        }
    }

    index as usize
}

fn mel_filter_bank() -> Vec<f32> {
    let fft_frequencies = fft_frequencies();
    let mel_frequencies = mel_frequencies(N_MELS + 2, 0.0, TRANSCRIPTION_SAMPLE_RATE as f32 / 2.0);
    let mut filters = vec![0.0_f32; N_MELS * (N_FFT / 2 + 1)];

    for mel_bin in 0..N_MELS {
        let left = mel_frequencies[mel_bin];
        let center = mel_frequencies[mel_bin + 1];
        let right = mel_frequencies[mel_bin + 2];
        let norm = 2.0 / (right - left);

        for (freq_bin, fft_frequency) in fft_frequencies.iter().copied().enumerate() {
            let lower = (fft_frequency - left) / (center - left);
            let upper = (right - fft_frequency) / (right - center);
            let weight = lower.min(upper).max(0.0) * norm;
            filters[mel_bin * (N_FFT / 2 + 1) + freq_bin] = weight;
        }
    }

    filters
}

fn fft_frequencies() -> Vec<f32> {
    (0..=N_FFT / 2)
        .map(|bin| bin as f32 * TRANSCRIPTION_SAMPLE_RATE as f32 / N_FFT as f32)
        .collect()
}

fn mel_frequencies(count: usize, fmin: f32, fmax: f32) -> Vec<f32> {
    let min_mel = hz_to_mel(fmin);
    let max_mel = hz_to_mel(fmax);
    (0..count)
        .map(|index| {
            let mel = min_mel + (max_mel - min_mel) * index as f32 / (count - 1) as f32;
            mel_to_hz(mel)
        })
        .collect()
}

fn hz_to_mel(frequency: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1_000.0;
    let min_log_mel = min_log_hz / f_sp;
    let log_step = (6.4_f32).ln() / 27.0;

    if frequency >= min_log_hz {
        min_log_mel + (frequency / min_log_hz).ln() / log_step
    } else {
        frequency / f_sp
    }
}

fn mel_to_hz(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1_000.0;
    let min_log_mel = min_log_hz / f_sp;
    let log_step = (6.4_f32).ln() / 27.0;

    if mel >= min_log_mel {
        min_log_hz * (log_step * (mel - min_log_mel)).exp()
    } else {
        f_sp * mel
    }
}

#[cfg(test)]
mod tests {
    use super::N_FRAMES;
    use super::N_MELS;
    use super::N_SAMPLES;
    use super::pad_or_trim;
    use super::whisper_log_mel_spectrogram;

    fn assert_close(actual: f32, expected: f32, tolerance: f32) {
        let delta = (actual - expected).abs();
        assert!(
            delta <= tolerance,
            "expected {expected}, got {actual}, delta {delta}, tolerance {tolerance}"
        );
    }

    #[test]
    fn pad_or_trim_matches_whisper_contract_length() {
        assert_eq!(
            pad_or_trim(&[1.0, 2.0, 3.0], 5),
            vec![1.0, 2.0, 3.0, 0.0, 0.0]
        );
        assert_eq!(pad_or_trim(&[1.0, 2.0, 3.0], 2), vec![1.0, 2.0]);
        assert_eq!(pad_or_trim(&[1.0, 2.0, 3.0], 3), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn log_mel_spectrogram_matches_python_reference_for_sine_wave() {
        let waveform = (0..16_000)
            .map(|sample| {
                let angle = 2.0 * std::f32::consts::PI * 440.0 * sample as f32 / 16_000.0;
                0.25 * angle.sin()
            })
            .collect::<Vec<_>>();
        let spectrogram = whisper_log_mel_spectrogram(&waveform);

        assert_eq!(spectrogram.n_mels, N_MELS);
        assert_eq!(spectrogram.n_frames, N_FRAMES);
        assert_eq!(pad_or_trim(&waveform, N_SAMPLES).len(), N_SAMPLES);

        let reference_points = [
            ((0, 0), 0.832_764_f32),
            ((0, 1), 0.320_756_f32),
            ((0, 2), -0.712_310_3_f32),
            ((10, 0), 1.185_726_f32),
            ((10, 1), 1.199_681_f32),
            ((10, 10), 1.198_222_9_f32),
            ((40, 100), 0.117_203_95_f32),
            ((79, 2999), -0.712_310_3_f32),
        ];

        for ((mel_bin, frame), expected) in reference_points {
            assert_close(spectrogram.at(mel_bin, frame), expected, 1e-3);
        }

        assert_close(spectrogram.min_value(), -0.712_310_3_f32, 1e-3);
        assert_close(spectrogram.max_value(), 1.287_689_7_f32, 1e-3);
    }
}
