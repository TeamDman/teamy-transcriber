use crate::domain::ClipId;
use crate::domain::TimeRange;
use facet::Facet;
use std::path::Path;
use std::path::PathBuf;
use thiserror::Error;

pub const WHISPER_SAMPLE_RATE_HZ: u32 = 16_000;

#[derive(Clone, Debug, Facet, PartialEq, Eq)]
pub struct MediaMetadata {
    pub duration_us: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub frame_count: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedAudio {
    pub path: PathBuf,
    pub metadata: MediaMetadata,
}

pub trait MediaAdapter {
    /// Inspect a supported source without producing a derived file.
    ///
    /// # Errors
    ///
    /// Returns an error when the source format cannot be read.
    fn inspect(&self, source: &Path) -> Result<MediaMetadata, MediaError>;

    /// Normalize one source to mono 16 kHz floating-point WAV.
    ///
    /// # Errors
    ///
    /// Returns an error when the source cannot be read or the output cannot be written.
    fn prepare_audio(&self, source: &Path, output_dir: &Path) -> Result<PreparedAudio, MediaError>;

    /// Extract one source-time clip from normalized audio.
    ///
    /// The source is expected to be the adapter's normalized output. The
    /// returned path is a new immutable derived artifact.
    ///
    /// # Errors
    ///
    /// Returns an error when the source or range cannot be read or written.
    fn prepare_clip(
        &self,
        normalized_source: &Path,
        output_dir: &Path,
        source_range: TimeRange,
        clip_id: ClipId,
    ) -> Result<PreparedAudio, MediaError>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WavMediaAdapter;

impl MediaAdapter for WavMediaAdapter {
    fn inspect(&self, source: &Path) -> Result<MediaMetadata, MediaError> {
        let reader = hound::WavReader::open(source)?;
        let spec = reader.spec();
        if spec.channels == 0 {
            return Err(MediaError::InvalidChannels);
        }
        // hound reports WAV duration in frames per channel, not interleaved samples.
        let frame_count = u64::from(reader.duration());
        let duration_us = if spec.sample_rate == 0 {
            0
        } else {
            frame_count
                .saturating_mul(1_000_000)
                .checked_div(u64::from(spec.sample_rate))
                .unwrap_or(0)
        };
        Ok(MediaMetadata {
            duration_us,
            sample_rate_hz: spec.sample_rate,
            channels: spec.channels,
            frame_count,
        })
    }

    fn prepare_audio(&self, source: &Path, output_dir: &Path) -> Result<PreparedAudio, MediaError> {
        let reader = hound::WavReader::open(source)?;
        let spec = reader.spec();
        if spec.channels == 0 {
            return Err(MediaError::InvalidChannels);
        }
        let mono = decode_mono(reader)?;
        let normalized = resample_linear(&mono, spec.sample_rate, WHISPER_SAMPLE_RATE_HZ)?;
        std::fs::create_dir_all(output_dir)?;
        let output_path = output_dir.join("normalized-16khz-mono.wav");
        write_normalized_wav(&output_path, &normalized)?;
        Ok(PreparedAudio {
            path: output_path,
            metadata: normalized_metadata(normalized.len()),
        })
    }

    fn prepare_clip(
        &self,
        normalized_source: &Path,
        output_dir: &Path,
        source_range: TimeRange,
        clip_id: ClipId,
    ) -> Result<PreparedAudio, MediaError> {
        source_range
            .validate()
            .map_err(|error| MediaError::InvalidClipRange(error.to_string()))?;
        let reader = hound::WavReader::open(normalized_source)?;
        let spec = reader.spec();
        if spec.channels == 0 {
            return Err(MediaError::InvalidChannels);
        }
        let mono = decode_mono(reader)?;
        let normalized = resample_linear(&mono, spec.sample_rate, WHISPER_SAMPLE_RATE_HZ)?;
        let source_duration_us = normalized_metadata(normalized.len()).duration_us;
        if source_range.end_us > source_duration_us {
            return Err(MediaError::InvalidClipRange(format!(
                "end {} exceeds source duration {source_duration_us}",
                source_range.end_us
            )));
        }
        let start = sample_index_floor(source_range.start_us)?.min(normalized.len());
        let end = sample_index_ceil(source_range.end_us)?.min(normalized.len());
        if start >= end {
            return Err(MediaError::EmptyClip);
        }

        std::fs::create_dir_all(output_dir)?;
        let output_path = output_dir.join(format!("clip-{clip_id}.wav"));
        write_normalized_wav(&output_path, &normalized[start..end])?;
        Ok(PreparedAudio {
            path: output_path,
            metadata: normalized_metadata(end - start),
        })
    }
}

fn write_normalized_wav(path: &Path, samples: &[f32]) -> Result<(), MediaError> {
    let output_spec = hound::WavSpec {
        channels: 1,
        sample_rate: WHISPER_SAMPLE_RATE_HZ,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, output_spec)?;
    for sample in samples.iter().copied() {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

fn normalized_metadata(sample_count: usize) -> MediaMetadata {
    let frame_count = u64::try_from(sample_count).unwrap_or(u64::MAX);
    let duration_us = u64::try_from(sample_count)
        .unwrap_or(u64::MAX)
        .saturating_mul(1_000_000)
        .checked_div(u64::from(WHISPER_SAMPLE_RATE_HZ))
        .unwrap_or(0);
    MediaMetadata {
        duration_us,
        sample_rate_hz: WHISPER_SAMPLE_RATE_HZ,
        channels: 1,
        frame_count,
    }
}

fn sample_index_floor(time_us: u64) -> Result<usize, MediaError> {
    usize::try_from(
        u128::from(time_us) * u128::from(WHISPER_SAMPLE_RATE_HZ) / u128::from(1_000_000_u32),
    )
    .map_err(|error| MediaError::TooManySamples(error.to_string()))
}

fn sample_index_ceil(time_us: u64) -> Result<usize, MediaError> {
    let numerator = u128::from(time_us) * u128::from(WHISPER_SAMPLE_RATE_HZ);
    let denominator = u128::from(1_000_000_u32);
    usize::try_from(numerator.div_ceil(denominator))
        .map_err(|error| MediaError::TooManySamples(error.to_string()))
}

fn decode_mono(
    mut reader: hound::WavReader<std::io::BufReader<std::fs::File>>,
) -> Result<Vec<f32>, MediaError> {
    let spec = reader.spec();
    let channel_count = spec.channels;
    let mut samples = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Int => {
            if spec.bits_per_sample > 16 {
                return Err(MediaError::UnsupportedIntegerBits(spec.bits_per_sample));
            }
            let scale = 2_f32.powi(i32::from(spec.bits_per_sample.saturating_sub(1)));
            for sample in reader.samples::<i16>() {
                samples.push(f32::from(sample?) / scale);
            }
        }
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                samples.push(sample?);
            }
        }
    }
    let mut mono = Vec::with_capacity(samples.len() / usize::from(channel_count).max(1));
    for frame in samples.chunks(usize::from(channel_count)) {
        let sum: f32 = frame.iter().copied().sum();
        mono.push(sum / f32::from(channel_count));
    }
    Ok(mono)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the output format intentionally stores interpolated audio as f32 samples"
)]
fn resample_linear(
    samples: &[f32],
    source_rate_hz: u32,
    target_rate_hz: u32,
) -> Result<Vec<f32>, MediaError> {
    if source_rate_hz == 0 || target_rate_hz == 0 {
        return Err(MediaError::InvalidSampleRate);
    }
    if samples.is_empty() || source_rate_hz == target_rate_hz {
        return Ok(samples.to_vec());
    }
    let sample_count = u64::try_from(samples.len())
        .map_err(|error| MediaError::TooManySamples(error.to_string()))?;
    let output_len = ((u128::from(sample_count) * u128::from(target_rate_hz))
        / u128::from(source_rate_hz))
    .max(1);
    let output_len = usize::try_from(output_len)
        .map_err(|error| MediaError::TooManySamples(error.to_string()))?;
    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let index =
            u64::try_from(index).map_err(|error| MediaError::TooManySamples(error.to_string()))?;
        let numerator = u128::from(index) * u128::from(source_rate_hz);
        let lower = usize::try_from(numerator / u128::from(target_rate_hz))
            .map_err(|error| MediaError::TooManySamples(error.to_string()))?;
        let upper = (lower + 1).min(samples.len() - 1);
        let remainder = u32::try_from(numerator % u128::from(target_rate_hz))
            .map_err(|error| MediaError::TooManySamples(error.to_string()))?;
        let fraction = f64::from(remainder) / f64::from(target_rate_hz);
        let value =
            f64::from(samples[lower]) * (1.0 - fraction) + f64::from(samples[upper]) * fraction;
        output.push(value as f32);
    }
    Ok(output)
}

#[derive(Debug, Error)]
pub enum MediaError {
    #[error("WAV operation failed")]
    Wav(#[from] hound::Error),
    #[error("filesystem operation failed")]
    Io(#[from] std::io::Error),
    #[error("WAV source has no channels")]
    InvalidChannels,
    #[error("WAV source has unsupported integer depth: {0} bits")]
    UnsupportedIntegerBits(u16),
    #[error("source and target sample rates must be nonzero")]
    InvalidSampleRate,
    #[error("source contains too many samples to normalize: {0}")]
    TooManySamples(String),
    #[error("source clip range is invalid: {0}")]
    InvalidClipRange(String),
    #[error("source clip range contains no samples")]
    EmptyClip,
}
