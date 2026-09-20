//! Exercise the same import/prepare/transcribe/persist/export workflow as the
//! application, with a resident session or the previous per-recording lifetime.
use eyre::Result;
use eyre::ensure;
use facet::Facet;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;
use teamy_transcriber::domain::AssetKind;
use teamy_transcriber::domain::ClipPlanKind;
use teamy_transcriber::media::AudioProfile;
use teamy_transcriber::storage::RecordingStore;
use teamy_transcriber::workflow::TranscriptionOptions;
use teamy_transcriber::workflow::TranscriptionSession;
use teamy_transcriber::workflow::create_recording;
use teamy_transcriber::workflow::export_recording_with_timestamps;
use teamy_transcriber::workflow::prepare_recording;

#[derive(Facet)]
struct Chunk {
    start_us: u64,
    end_us: u64,
    wav: String,
    text: String,
}
#[derive(Facet)]
struct Run {
    source: String,
    index: usize,
    duration_us: u64,
    prepare_ms: f64,
    transcribe_ms: f64,
    release_ms: f64,
    total_ms: f64,
    backend: String,
    chunks: Vec<Chunk>,
    text: String,
    segmentation: Option<ClipPlanKind>,
    no_speech: bool,
    first_clip_ms: Option<f64>,
}
#[derive(Facet)]
struct Receipt {
    scope: String,
    session_mode: String,
    revision: String,
    worktree_status: String,
    runs: Vec<Run>,
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        (4..=5).contains(&args.len()),
        "usage: workflow_bench MODEL WAV_OR_JSON_LIST OUTPUT_DIR REPEATS [cold|resident]"
    );
    let model_dir = PathBuf::from(&args[0]);
    let paths: Vec<PathBuf> = if Path::new(&args[1])
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("json"))
    {
        facet_json::from_slice(&std::fs::read(&args[1])?)?
    } else {
        vec![PathBuf::from(&args[1])]
    };
    let root = PathBuf::from(&args[2]);
    ensure!(!root.exists(), "choose a new output directory");
    ensure!(!paths.is_empty(), "empty input list");
    let repeats: usize = args[3].parse()?;
    ensure!(repeats > 0, "empty repetitions");
    let mode = args.get(4).map_or("resident", String::as_str);
    ensure!(matches!(mode, "resident" | "cold"), "invalid session mode");
    let store = RecordingStore::new(root);
    let mut session = TranscriptionSession::default();
    let options = TranscriptionOptions {
        model_dir,
        max_decode_tokens: 448,
        chunk_duration_us: None,
        profile: AudioProfile::Original,
    };
    let mut runs = Vec::new();
    for path in paths {
        for index in 0..repeats {
            let mut per_recording_session = TranscriptionSession::default();
            let selected_session = if mode == "cold" {
                &mut per_recording_session
            } else {
                &mut session
            };
            let mut run = run_recording(&store, selected_session, &options, &path, index)?;
            if mode == "cold" {
                let release_start = Instant::now();
                drop(per_recording_session);
                run.release_ms = release_start.elapsed().as_secs_f64() * 1000.;
                run.total_ms += run.release_ms;
            }
            runs.push(run);
        }
    }
    println!("{}", facet_json::to_string_pretty(&Receipt {
        scope: "Application workflow: import, normalize, optional prepared VAD, chunk, ASR, persist and timestamped export; source clip ranges, no word alignment/diarization".into(),
        session_mode: mode.into(), revision: env!("GIT_REVISION").into(),
        worktree_status: env!("GIT_WORKTREE_STATUS").into(), runs,
    })?);
    Ok(())
}

fn run_recording(
    store: &RecordingStore,
    session: &mut TranscriptionSession,
    options: &TranscriptionOptions,
    path: &Path,
    index: usize,
) -> Result<Run> {
    let started = Instant::now();
    let id = create_recording(store, AssetKind::AudioFile, path)?;
    let prepared = prepare_recording(store, id)?;
    let prepare_ms = started.elapsed().as_secs_f64() * 1000.;
    let decode_start = Instant::now();
    let mut events = Vec::new();
    let mut first_clip_ms = None;
    let mut progress = |completed, total| {
        events.push((completed, total));
        if completed > 0 && first_clip_ms.is_none() {
            first_clip_ms = Some(started.elapsed().as_secs_f64() * 1000.);
        }
    };
    let report = session.transcribe(store, id, options.clone(), None, Some(&mut progress))?;
    let transcribe_ms = decode_start.elapsed().as_secs_f64() * 1000.;
    ensure!(
        !report.cancelled && (!report.chunks.is_empty() || report.no_speech),
        "incomplete transcription"
    );
    ensure!(
        events.first() == Some(&(0, report.chunks.len()))
            && events.last() == Some(&(report.chunks.len(), report.chunks.len())),
        "incomplete progress delivery"
    );
    let mut cursor = 0;
    let segmentation = store.load_recording(id)?.clip_plan.map(|plan| plan.kind);
    let speech = matches!(segmentation, Some(ClipPlanKind::VoiceActivity { .. }));
    for chunk in &report.chunks {
        ensure!(
            chunk.source_range.start_us >= cursor
                && chunk.source_range.end_us > chunk.source_range.start_us
                && chunk.source_range.end_us <= prepared.metadata.duration_us,
            "invalid or overlapping source coverage"
        );
        ensure!(
            speech || chunk.source_range.start_us == cursor,
            "non-contiguous fixed chunks"
        );
        cursor = chunk.source_range.end_us;
    }
    ensure!(
        speech || cursor == prepared.metadata.duration_us,
        "source duration was not fully covered"
    );
    if report.no_speech {
        ensure!(report.chunks.is_empty(), "silent recording has transcripts");
        // The application has no transcript to export for a silent recording.
        // Keep the completed VAD/import/persistence request in the receipt.
    } else {
        let exported = export_recording_with_timestamps(store, id, None)?;
        ensure!(
            exported.transcript_count == report.chunks.len(),
            "persisted transcript count mismatch"
        );
    }
    let text = report
        .chunks
        .iter()
        .map(|chunk| chunk.text.trim())
        .collect::<Vec<_>>()
        .join(" ");
    let total_ms = started.elapsed().as_secs_f64() * 1000.;
    Ok(Run {
        source: path.display().to_string(),
        index,
        duration_us: prepared.metadata.duration_us,
        prepare_ms,
        transcribe_ms,
        release_ms: 0.,
        total_ms,
        backend: report.backend_id,
        text,
        segmentation,
        no_speech: report.no_speech,
        first_clip_ms,
        chunks: report
            .chunks
            .into_iter()
            .map(|chunk| Chunk {
                start_us: chunk.source_range.start_us,
                end_us: chunk.source_range.end_us,
                wav: chunk.audio_path.display().to_string(),
                text: chunk.text,
            })
            .collect(),
    })
}
