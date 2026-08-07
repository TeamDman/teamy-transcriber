use std::path::Path;
use std::path::PathBuf;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use teamy_transcriber::domain::ClipId;
use teamy_transcriber::domain::TimeRange;
use teamy_transcriber::media::FfmpegMediaAdapter;
use teamy_transcriber::media::MediaAdapter;
use teamy_transcriber::media::MediaError;
use teamy_transcriber::media::WHISPER_SAMPLE_RATE_HZ;
use teamy_transcriber::media::WavMediaAdapter;

#[test]
fn wav_adapter_inspects_and_normalizes_to_whisper_format() {
    let root = unique_temp_dir("teamy-transcriber-media");
    std::fs::create_dir_all(&root).expect("fixture directory should be creatable");
    let source = root.join("stereo-8khz.wav");
    write_fixture(&source);

    let adapter = WavMediaAdapter;
    let source_metadata = adapter
        .inspect(&source)
        .expect("source should be inspectable");
    assert_eq!(source_metadata.sample_rate_hz, 8_000);
    assert_eq!(source_metadata.channels, 2);
    assert_eq!(source_metadata.frame_count, 800);
    assert_eq!(source_metadata.duration_us, 100_000);

    let prepared = adapter
        .prepare_audio(&source, &root.join("audio"))
        .expect("source should normalize");
    assert_eq!(prepared.metadata.sample_rate_hz, WHISPER_SAMPLE_RATE_HZ);
    assert_eq!(prepared.metadata.channels, 1);
    assert_eq!(prepared.metadata.frame_count, 1_600);
    assert_eq!(prepared.metadata.duration_us, 100_000);
    assert!(prepared.path.is_file());

    let clip = adapter
        .prepare_clip(
            &prepared.path,
            &root.join("clips"),
            TimeRange::new(25_000, 75_000).expect("clip range should be valid"),
            ClipId::new(),
        )
        .expect("normalized clip should be writable");
    assert_eq!(clip.metadata.frame_count, 800);
    assert_eq!(clip.metadata.duration_us, 50_000);
    assert!(clip.path.is_file());

    let prepared_metadata = adapter
        .inspect(&prepared.path)
        .expect("normalized file should be inspectable");
    assert_eq!(prepared_metadata.sample_rate_hz, WHISPER_SAMPLE_RATE_HZ);
    assert_eq!(prepared_metadata.channels, 1);
    assert_eq!(prepared_metadata.frame_count, 1_600);

    std::fs::remove_dir_all(root).expect("fixture directory should be removable");
}

#[test]
fn ffprobe_adapter_reports_missing_tool_explicitly() {
    let adapter = FfmpegMediaAdapter {
        ffmpeg_executable: PathBuf::from("teamy-transcriber-missing-ffmpeg"),
        ffprobe_executable: PathBuf::from("teamy-transcriber-missing-ffprobe"),
    };
    let error = adapter
        .inspect(Path::new("missing-source.mp4"))
        .expect_err("missing ffprobe should be reported");
    assert!(matches!(error, MediaError::Probe(detail) if detail.contains("could not be launched")));
}

fn write_fixture(path: &Path) {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 8_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("fixture should be writable");
    for _ in 0..800 {
        writer
            .write_sample(10_000_i16)
            .expect("left sample should write");
        writer
            .write_sample(-10_000_i16)
            .expect("right sample should write");
    }
    writer.finalize().expect("fixture should finalize");
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be after the Unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{timestamp}", std::process::id()))
}
