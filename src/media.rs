use crate::domain::ClipId;
use crate::domain::TimeRange;
use facet::Facet;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;
use thiserror::Error;

pub const WHISPER_SAMPLE_RATE_HZ: u32 = 16_000;
pub const FFMPEG_ENV_VAR: &str = "TEAMY_TRANSCRIBER_FFMPEG";
pub const FFPROBE_ENV_VAR: &str = "TEAMY_TRANSCRIBER_FFPROBE";

#[derive(Clone, Copy, Debug, Default, Eq, Facet, PartialEq)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum AudioProfile {
    #[default]
    Original,
    Gain6Db,
    NoiseGate,
    VoiceEq,
}

impl AudioProfile {
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Original => Self::Gain6Db,
            Self::Gain6Db => Self::NoiseGate,
            Self::NoiseGate => Self::VoiceEq,
            Self::VoiceEq => Self::Original,
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Original => "ORIGINAL",
            Self::Gain6Db => "GAIN +6DB",
            Self::NoiseGate => "NOISE GATE",
            Self::VoiceEq => "VOICE EQ",
        }
    }

    #[must_use]
    pub const fn short_label(self) -> &'static str {
        match self {
            Self::Original => "RAW",
            Self::Gain6Db => "GAIN",
            Self::NoiseGate => "GATE",
            Self::VoiceEq => "EQ",
        }
    }

    #[must_use]
    pub const fn file_stem(self) -> Option<&'static str> {
        match self {
            Self::Original => None,
            Self::Gain6Db => Some("gain-6db"),
            Self::NoiseGate => Some("noise-gate"),
            Self::VoiceEq => Some("voice-eq"),
        }
    }
}

/// Plan contiguous, non-overlapping source-time chunks for one recording.
///
/// The final chunk is shorter when the duration is not an exact multiple of
/// `max_chunk_duration_us`. The returned ranges cover the recording exactly.
///
/// # Errors
///
/// Returns an error when either duration is zero or a range cannot be formed.
pub fn plan_time_chunks(
    duration_us: u64,
    max_chunk_duration_us: u64,
) -> Result<Vec<TimeRange>, MediaError> {
    if duration_us == 0 {
        return Err(MediaError::InvalidChunkDuration(
            "recording duration must be greater than zero".to_string(),
        ));
    }
    if max_chunk_duration_us == 0 {
        return Err(MediaError::InvalidChunkDuration(
            "maximum chunk duration must be greater than zero".to_string(),
        ));
    }

    let mut ranges = Vec::new();
    let mut start_us = 0;
    while start_us < duration_us {
        let end_us = start_us
            .saturating_add(max_chunk_duration_us)
            .min(duration_us);
        let range = TimeRange::new(start_us, end_us)
            .map_err(|error| MediaError::InvalidChunkDuration(error.to_string()))?;
        ranges.push(range);
        start_us = end_us;
    }
    Ok(ranges)
}

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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FfmpegMediaAdapter {
    pub ffmpeg_executable: PathBuf,
    pub ffprobe_executable: PathBuf,
}

impl Default for FfmpegMediaAdapter {
    fn default() -> Self {
        Self {
            ffmpeg_executable: PathBuf::from("ffmpeg"),
            ffprobe_executable: PathBuf::from("ffprobe"),
        }
    }
}

impl FfmpegMediaAdapter {
    #[must_use]
    pub fn from_environment() -> Self {
        Self {
            ffmpeg_executable: std::env::var(FFMPEG_ENV_VAR)
                .map_or_else(|_| PathBuf::from("ffmpeg"), PathBuf::from),
            ffprobe_executable: std::env::var(FFPROBE_ENV_VAR)
                .map_or_else(|_| PathBuf::from("ffprobe"), PathBuf::from),
        }
    }
}

impl MediaAdapter for FfmpegMediaAdapter {
    fn inspect(&self, source: &Path) -> Result<MediaMetadata, MediaError> {
        let output = Command::new(&self.ffprobe_executable)
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels,nb_frames,duration",
                "-of",
                "csv=p=0:s=,",
            ])
            .arg(source)
            .stdin(Stdio::null())
            .output()
            .map_err(|error| {
                MediaError::Probe(format!(
                    "{} could not be launched: {error}",
                    self.ffprobe_executable.display()
                ))
            })?;
        if !output.status.success() {
            return Err(MediaError::Probe(
                String::from_utf8_lossy(&output.stderr).trim().to_string(),
            ));
        }
        parse_probe_output(&String::from_utf8_lossy(&output.stdout))
    }

    fn prepare_audio(&self, source: &Path, output_dir: &Path) -> Result<PreparedAudio, MediaError> {
        std::fs::create_dir_all(output_dir)?;
        let output_path = output_dir.join("normalized-16khz-mono.wav");
        let output = Command::new(&self.ffmpeg_executable)
            .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i"])
            .arg(source)
            .args([
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_f32le",
            ])
            .arg(&output_path)
            .stdin(Stdio::null())
            .output()
            .map_err(|error| {
                MediaError::Ffmpeg(format!(
                    "{} could not be launched: {error}",
                    self.ffmpeg_executable.display()
                ))
            })?;
        if !output.status.success() {
            return Err(MediaError::Ffmpeg(
                String::from_utf8_lossy(&output.stderr).trim().to_string(),
            ));
        }
        let metadata = WavMediaAdapter.inspect(&output_path)?;
        Ok(PreparedAudio {
            path: output_path,
            metadata,
        })
    }

    fn prepare_clip(
        &self,
        normalized_source: &Path,
        output_dir: &Path,
        source_range: TimeRange,
        clip_id: ClipId,
    ) -> Result<PreparedAudio, MediaError> {
        WavMediaAdapter.prepare_clip(normalized_source, output_dir, source_range, clip_id)
    }
}

/// Apply one small, deterministic profile to already-normalized audio.
///
/// The original normalized WAV is never overwritten. Each non-original
/// profile produces a separate derived WAV under `profiles/`; the caller owns
/// the parameter receipt that accompanies it.
///
/// # Errors
///
/// Returns an error when the normalized source cannot be decoded or the
/// derived WAV cannot be written.
pub fn apply_audio_profile(
    normalized_source: &Path,
    output_dir: &Path,
    profile: AudioProfile,
) -> Result<PreparedAudio, MediaError> {
    let source_metadata = WavMediaAdapter.inspect(normalized_source)?;
    if source_metadata.sample_rate_hz != WHISPER_SAMPLE_RATE_HZ || source_metadata.channels != 1 {
        return Err(MediaError::InvalidProfile(
            "audio profiles require 16 kHz mono normalized audio".to_string(),
        ));
    }
    if profile == AudioProfile::Original {
        return Ok(PreparedAudio {
            path: normalized_source.to_path_buf(),
            metadata: source_metadata,
        });
    }
    let reader = hound::WavReader::open(normalized_source)?;
    let samples = decode_mono(reader)?;
    let processed = match profile {
        AudioProfile::Original => samples,
        AudioProfile::Gain6Db => samples
            .into_iter()
            .map(|sample| (sample * 1.995_262_3).clamp(-1.0, 1.0))
            .collect(),
        AudioProfile::NoiseGate => samples
            .into_iter()
            .map(|sample| {
                if sample.abs() < 0.02 {
                    sample * 0.08
                } else {
                    sample
                }
            })
            .collect(),
        AudioProfile::VoiceEq => apply_voice_eq(&samples),
    };
    std::fs::create_dir_all(output_dir)?;
    let stem = profile
        .file_stem()
        .ok_or_else(|| MediaError::InvalidProfile("profile has no derived file".to_string()))?;
    let output_path = output_dir.join(format!("{stem}.wav"));
    write_normalized_wav(&output_path, &processed)?;
    Ok(PreparedAudio {
        path: output_path,
        metadata: normalized_metadata(processed.len()),
    })
}

fn apply_voice_eq(samples: &[f32]) -> Vec<f32> {
    let mut previous = 0.0;
    samples
        .iter()
        .copied()
        .map(|sample| {
            let high_frequency = sample - previous * 0.995;
            previous = sample;
            (sample + high_frequency * 0.25).clamp(-1.0, 1.0)
        })
        .collect()
}

fn parse_probe_output(output: &str) -> Result<MediaMetadata, MediaError> {
    let fields: Vec<&str> = output.trim().split(',').map(str::trim).collect();
    if fields.len() < 4 {
        return Err(MediaError::InvalidProbe(format!(
            "expected four audio fields, got: {output:?}"
        )));
    }
    let sample_rate_hz = fields[0]
        .parse::<u32>()
        .map_err(|error| MediaError::InvalidProbe(format!("invalid sample rate: {error}")))?;
    let channels = fields[1]
        .parse::<u16>()
        .map_err(|error| MediaError::InvalidProbe(format!("invalid channel count: {error}")))?;
    let duration_us = parse_decimal_seconds_us(fields[3])?;
    let frame_count = if fields[2].is_empty() || fields[2] == "N/A" {
        u128::from(duration_us)
            .saturating_mul(u128::from(sample_rate_hz))
            .checked_div(u128::from(1_000_000_u32))
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| MediaError::InvalidProbe("frame count overflowed".to_string()))?
    } else {
        fields[2]
            .parse::<u64>()
            .map_err(|error| MediaError::InvalidProbe(format!("invalid frame count: {error}")))?
    };
    Ok(MediaMetadata {
        duration_us,
        sample_rate_hz,
        channels,
        frame_count,
    })
}

fn parse_decimal_seconds_us(value: &str) -> Result<u64, MediaError> {
    let (whole, fraction) = value
        .split_once('.')
        .ok_or_else(|| MediaError::InvalidProbe(format!("invalid duration: {value:?}")))?;
    let whole_us = whole
        .parse::<u64>()
        .ok()
        .and_then(|seconds| seconds.checked_mul(1_000_000))
        .ok_or_else(|| MediaError::InvalidProbe(format!("invalid duration: {value:?}")))?;
    let mut fractional_us = 0_u64;
    let mut digits = 0;
    for byte in fraction.bytes().take(6) {
        if !byte.is_ascii_digit() {
            return Err(MediaError::InvalidProbe(format!(
                "invalid duration: {value:?}"
            )));
        }
        fractional_us = fractional_us * 10 + u64::from(byte - b'0');
        digits += 1;
    }
    if fraction.len() > 6 && !fraction.bytes().skip(6).all(|byte| byte.is_ascii_digit()) {
        return Err(MediaError::InvalidProbe(format!(
            "invalid duration: {value:?}"
        )));
    }
    for _ in digits..6 {
        fractional_us *= 10;
    }
    whole_us
        .checked_add(fractional_us)
        .ok_or_else(|| MediaError::InvalidProbe(format!("duration overflowed: {value:?}")))
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
    #[error("source chunk duration is invalid: {0}")]
    InvalidChunkDuration(String),
    #[error("source clip range contains no samples")]
    EmptyClip,
    #[error("ffmpeg operation failed: {0}")]
    Ffmpeg(String),
    #[error("ffprobe operation failed: {0}")]
    Probe(String),
    #[error("ffprobe returned invalid metadata: {0}")]
    InvalidProbe(String),
    #[error("audio profile is invalid: {0}")]
    InvalidProfile(String),
}

#[cfg(test)]
mod tests {
    use super::parse_probe_output;

    #[test]
    fn parses_ffprobe_duration_and_missing_frame_count() {
        let metadata =
            parse_probe_output("48000,2,N/A,1.250000\n").expect("ffprobe output should parse");
        assert_eq!(metadata.sample_rate_hz, 48_000);
        assert_eq!(metadata.channels, 2);
        assert_eq!(metadata.duration_us, 1_250_000);
        assert_eq!(metadata.frame_count, 60_000);
    }

    #[test]
    fn rejects_malformed_ffprobe_output() {
        assert!(parse_probe_output("not,a,probe").is_err());
    }
}
