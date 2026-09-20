//! Reproducible, user-owned speech verification fixtures.
//!
//! The VCTK canary is deliberately a thin integration check around the real
//! application workflow. It does not download data, substitute a fake model,
//! or turn a missing corpus/model into a passing result.

use crate::domain::AssetKind;
use crate::domain::RecordingId;
use crate::domain::TranscriptProvenance;
use crate::native_whisper::model::inspect_model_dir;
use crate::paths::AppHome;
use crate::paths::MODEL_DIR_ENV_VAR;
use crate::paths::ModelHome;
use crate::paths::TORCH_DEVICE_ENV_VAR;
use crate::storage::RecordingStore;
use crate::transcription::NativeWhisperBackend;
use crate::transcription::NativeWhisperConfig;
use crate::transcription::TranscriptionBackend;
use crate::workflow::MediaToolConfig;
use crate::workflow::create_recording;
use crate::workflow::prepare_recording_with_tools_and_profile;
use crate::workflow::transcribe_recording;
use facet::Facet;
use sha2::Digest;
use sha2::Sha256;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

const VCTK_DESCRIPTOR_JSON: &str = include_str!("../fixtures/vctk-p230-385.json");
const DEFAULT_RECEIPT_PATH: &str = "artifacts/verification/vctk-p230-385.json";
const RECEIPT_SCHEMA_VERSION: u16 = 3;

#[derive(Clone, Debug, Facet)]
struct VctkDescriptor {
    schema_version: u16,
    logical_id: String,
    relative_audio_path: String,
    expected_audio_sha256: String,
    reference_text: String,
    normalization_policy: NormalizationPolicy,
    attribution: Attribution,
}

#[derive(Clone, Debug, Facet)]
pub struct NormalizationPolicy {
    pub source_format: String,
    pub target_sample_rate_hz: u32,
    pub target_channels: u16,
    pub downmix: String,
    pub resampling: String,
    pub output_format: String,
}

#[derive(Clone, Debug, Facet)]
pub struct Attribution {
    pub corpus: String,
    pub version: String,
    pub license: String,
    pub source: String,
    pub notice: String,
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum VerificationStatus {
    Passed,
    Unavailable,
    Failed,
}

#[derive(Clone, Debug, Eq, Facet, PartialEq)]
#[facet(rename_all = "snake_case")]
#[repr(u8)]
pub enum FailureStage {
    Corpus,
    AppHome,
    Import,
    Normalize,
    Model,
    Transcription,
    Oracle,
    Replay,
    Receipt,
}

#[derive(Clone, Debug, Default, Facet)]
pub struct TimingReceipt {
    pub total_ms: u64,
    pub import_ms: u64,
    pub normalize_ms: u64,
    pub model_load_ms: u64,
    pub transcription_ms: u64,
    pub replay_ms: u64,
}

#[derive(Clone, Debug, Facet)]
pub struct BinaryReceipt {
    pub package: String,
    pub version: String,
    pub path: String,
    pub sha256: Option<String>,
    pub bytes: Option<u64>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Facet)]
pub struct GitReceipt {
    pub repository: String,
    pub branch: String,
    pub revision: String,
    pub worktree_status: String,
}

#[derive(Clone, Debug, Facet)]
pub struct ConfigurationReceipt {
    pub backend_id: Option<String>,
    pub vctk_root: String,
    pub vctk_root_source: String,
    pub model_dir: String,
    pub model_source: String,
    pub app_home: String,
    pub app_home_source: String,
    pub recording_store_root: Option<String>,
    pub max_decode_tokens: usize,
    pub torch_device: String,
    pub torch_cuda_available: bool,
    pub torch_device_count: i64,
    pub torch_cudart_version: String,
    pub torch_cudnn_version: String,
    pub media_adapter: String,
    pub network_access: String,
}

#[derive(Clone, Debug, Facet)]
pub struct ArtifactHash {
    pub relative_path: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, Facet)]
pub struct ArtifactReceipt {
    pub path: String,
    pub sha256: Option<String>,
    pub bytes: Option<u64>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Facet)]
pub struct ModelReceipt {
    pub directory: String,
    pub source: String,
    pub layout: Option<String>,
    pub aggregate_sha256: Option<String>,
    pub artifacts: Vec<ArtifactHash>,
    pub readiness: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Facet)]
pub struct ReplayEvidence {
    pub attempted: bool,
    pub ok: bool,
    pub event_count: usize,
    pub next_sequence: Option<u64>,
    pub recording_matches_manifest: bool,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Facet)]
pub struct TranscriptReceipt {
    pub transcript_id: String,
    pub provenance: String,
    pub exact_transcript_text: String,
    pub normalized_transcript_text: String,
    pub normalized_reference_text: String,
    pub sha256: String,
    pub character_errors: usize,
    pub reference_character_count: usize,
    pub cer: f64,
    pub word_errors: usize,
    pub reference_word_count: usize,
    pub wer: f64,
}

#[derive(Clone, Debug, Facet)]
pub struct CanaryReceipt {
    pub schema_version: u16,
    pub logical_id: String,
    pub status: VerificationStatus,
    pub passed: bool,
    pub failure_stage: Option<FailureStage>,
    pub failure: Option<String>,
    pub descriptor_schema_version: u16,
    pub reference_text: String,
    pub normalization_policy: NormalizationPolicy,
    pub attribution: Attribution,
    pub binary: BinaryReceipt,
    pub git: GitReceipt,
    pub configuration: ConfigurationReceipt,
    pub input: Option<ArtifactReceipt>,
    pub normalized_audio: Option<ArtifactReceipt>,
    pub manifest: Option<ArtifactReceipt>,
    pub events: Option<ArtifactReceipt>,
    pub model: ModelReceipt,
    pub transcript: Option<TranscriptReceipt>,
    pub replay: ReplayEvidence,
    pub timings: TimingReceipt,
    pub recording_id: Option<String>,
    pub receipt_path: String,
}

#[derive(Clone, Debug, Default)]
pub struct VctkCanaryOptions {
    pub vctk_root: Option<PathBuf>,
    pub receipt_path: Option<PathBuf>,
    pub model_dir: Option<PathBuf>,
    pub max_decode_tokens: Option<usize>,
}

#[derive(Clone, Debug)]
struct HashResult {
    sha256: String,
    bytes: u64,
}

/// Execute the VCTK canary and always write a typed receipt for expected
/// unavailable, failed, and passed verification outcomes.
///
/// # Errors
///
/// Returns an error when the committed descriptor is malformed or the
/// requested receipt cannot be written. Dataset/model/inference failures are
/// represented in the returned non-passing receipt instead.
#[expect(
    clippy::too_many_lines,
    reason = "the canary deliberately keeps its bounded receipt-producing workflow together"
)]
pub fn run_vctk_canary(options: VctkCanaryOptions) -> eyre::Result<CanaryReceipt> {
    let descriptor = facet_json::from_str::<VctkDescriptor>(VCTK_DESCRIPTOR_JSON)
        .map_err(|error| eyre::eyre!("committed VCTK descriptor is invalid: {error}"))?;
    let receipt_path = options
        .receipt_path
        .clone()
        .unwrap_or_else(|| PathBuf::from(DEFAULT_RECEIPT_PATH));
    let vctk_root = options.vctk_root.clone();
    let vctk_root_display = vctk_root.as_deref().map_or_else(
        || "<not supplied>".to_string(),
        |path| path.display().to_string(),
    );
    let (model_dir, model_source, model_resolution_error) = resolve_model_dir(options.model_dir);
    let (app_home, app_home_source, app_home_error) = resolve_app_home();
    let max_decode_tokens = options
        .max_decode_tokens
        .unwrap_or(crate::native_whisper::whisper::DEFAULT_MAX_DECODE_TOKENS);
    let model_dir_display = model_dir.as_deref().map_or_else(
        || "<unresolved>".to_string(),
        |path| path.display().to_string(),
    );
    let app_home_display = app_home.as_ref().map_or_else(
        || "<unresolved>".to_string(),
        |path| path.0.display().to_string(),
    );
    let binary = binary_receipt();
    let (
        torch_device,
        torch_cuda_available,
        torch_device_count,
        torch_cudart_version,
        torch_cudnn_version,
    ) = torch_runtime_receipt();
    let mut receipt = CanaryReceipt {
        schema_version: RECEIPT_SCHEMA_VERSION,
        logical_id: descriptor.logical_id.clone(),
        status: VerificationStatus::Failed,
        passed: false,
        failure_stage: None,
        failure: None,
        descriptor_schema_version: descriptor.schema_version,
        reference_text: descriptor.reference_text.clone(),
        normalization_policy: descriptor.normalization_policy.clone(),
        attribution: descriptor.attribution.clone(),
        binary,
        git: GitReceipt {
            repository: env!("GIT_REPOSITORY_URL").to_string(),
            branch: env!("GIT_BRANCH").to_string(),
            revision: env!("GIT_REVISION").to_string(),
            worktree_status: env!("GIT_WORKTREE_STATUS").to_string(),
        },
        configuration: ConfigurationReceipt {
            backend_id: None,
            vctk_root: vctk_root_display,
            vctk_root_source: if vctk_root.is_some() {
                "argument".to_string()
            } else {
                "not_supplied".to_string()
            },
            model_dir: model_dir_display,
            model_source,
            app_home: app_home_display,
            app_home_source,
            recording_store_root: None,
            max_decode_tokens,
            torch_device,
            torch_cuda_available,
            torch_device_count,
            torch_cudart_version,
            torch_cudnn_version,
            media_adapter: "native-wav".to_string(),
            network_access: "disabled-by-workflow-no-downloaders".to_string(),
        },
        input: None,
        normalized_audio: None,
        manifest: None,
        events: None,
        model: ModelReceipt {
            directory: model_dir
                .as_ref()
                .map_or_else(String::new, |path| path.display().to_string()),
            source: "local-only".to_string(),
            layout: None,
            aggregate_sha256: None,
            artifacts: Vec::new(),
            readiness: None,
            error: model_resolution_error,
        },
        transcript: None,
        replay: ReplayEvidence {
            attempted: false,
            ok: false,
            event_count: 0,
            next_sequence: None,
            recording_matches_manifest: false,
            error: None,
        },
        timings: TimingReceipt::default(),
        recording_id: None,
        receipt_path: receipt_path.display().to_string(),
    };
    let total_start = Instant::now();

    let Some(vctk_root) = vctk_root else {
        mark_unavailable(
            &mut receipt,
            "--vctk-root was not supplied; the user-owned VCTK corpus is unavailable",
        );
        finish_and_write(&mut receipt, total_start, &receipt_path)?;
        return Ok(receipt);
    };
    let input_path = vctk_root.join(&descriptor.relative_audio_path);
    if !input_path.is_file() {
        mark_unavailable(
            &mut receipt,
            format!("VCTK canary input is unavailable: {}", input_path.display()),
        );
        finish_and_write(&mut receipt, total_start, &receipt_path)?;
        return Ok(receipt);
    }
    let input_hash = match hash_file(&input_path) {
        Ok(hash) => hash,
        Err(error) => {
            fail(&mut receipt, FailureStage::Corpus, error.to_string());
            finish_and_write(&mut receipt, total_start, &receipt_path)?;
            return Ok(receipt);
        }
    };
    receipt.input = Some(ArtifactReceipt {
        path: input_path.display().to_string(),
        sha256: Some(input_hash.sha256.clone()),
        bytes: Some(input_hash.bytes),
        error: None,
    });
    if !input_hash
        .sha256
        .eq_ignore_ascii_case(&descriptor.expected_audio_sha256)
    {
        fail(
            &mut receipt,
            FailureStage::Corpus,
            format!(
                "VCTK canary input hash mismatch: expected {}, found {}",
                descriptor.expected_audio_sha256, input_hash.sha256
            ),
        );
        finish_and_write(&mut receipt, total_start, &receipt_path)?;
        return Ok(receipt);
    }

    let Some(app_home) = app_home else {
        fail(
            &mut receipt,
            FailureStage::AppHome,
            app_home_error.unwrap_or_else(|| "application home could not be resolved".to_string()),
        );
        finish_and_write(&mut receipt, total_start, &receipt_path)?;
        return Ok(receipt);
    };
    if let Err(error) = app_home.ensure_dir() {
        fail(&mut receipt, FailureStage::AppHome, error.to_string());
        finish_and_write(&mut receipt, total_start, &receipt_path)?;
        return Ok(receipt);
    }
    let store_root = app_home.0.join("verification").join(&descriptor.logical_id);
    let store = RecordingStore::new(&store_root);
    receipt.configuration.recording_store_root = Some(store_root.display().to_string());

    let import_start = Instant::now();
    let recording_id = match create_recording(&store, AssetKind::AudioFile, &input_path) {
        Ok(recording_id) => recording_id,
        Err(error) => {
            receipt.timings.import_ms = elapsed_ms(import_start);
            fail(&mut receipt, FailureStage::Import, error.to_string());
            finish_and_write(&mut receipt, total_start, &receipt_path)?;
            return Ok(receipt);
        }
    };
    receipt.timings.import_ms = elapsed_ms(import_start);
    receipt.recording_id = Some(recording_id.to_string());

    let normalize_start = Instant::now();
    let preparation = prepare_recording_with_tools_and_profile(
        &store,
        recording_id,
        &MediaToolConfig::from_environment(),
        crate::media::AudioProfile::Original,
    );
    receipt.timings.normalize_ms = elapsed_ms(normalize_start);
    let preparation = match preparation {
        Ok(preparation) => preparation,
        Err(error) => {
            fail(&mut receipt, FailureStage::Normalize, error.to_string());
            attach_recording_artifacts(&mut receipt, &store, recording_id);
            finish_and_write(&mut receipt, total_start, &receipt_path)?;
            return Ok(receipt);
        }
    };
    if preparation.metadata.sample_rate_hz != descriptor.normalization_policy.target_sample_rate_hz
        || preparation.metadata.channels != descriptor.normalization_policy.target_channels
    {
        fail(
            &mut receipt,
            FailureStage::Normalize,
            format!(
                "normalized audio policy mismatch: found {} Hz / {} channels",
                preparation.metadata.sample_rate_hz, preparation.metadata.channels
            ),
        );
        attach_normalized_audio(&mut receipt, &preparation.normalized_path);
        attach_recording_artifacts(&mut receipt, &store, recording_id);
        finish_and_write(&mut receipt, total_start, &receipt_path)?;
        return Ok(receipt);
    }
    attach_normalized_audio(&mut receipt, &preparation.normalized_path);
    if receipt
        .normalized_audio
        .as_ref()
        .is_some_and(|audio| audio.sha256.is_none())
    {
        fail(
            &mut receipt,
            FailureStage::Normalize,
            "normalized audio hash could not be computed",
        );
        attach_recording_artifacts(&mut receipt, &store, recording_id);
        finish_and_write(&mut receipt, total_start, &receipt_path)?;
        return Ok(receipt);
    }

    let model_start = Instant::now();
    let Some(model_dir) = model_dir else {
        receipt.timings.model_load_ms = elapsed_ms(model_start);
        let model_error = receipt
            .model
            .error
            .clone()
            .unwrap_or_else(|| "native model directory could not be resolved".to_string());
        fail(&mut receipt, FailureStage::Model, model_error);
        attach_recording_artifacts(&mut receipt, &store, recording_id);
        finish_and_write(&mut receipt, total_start, &receipt_path)?;
        return Ok(receipt);
    };
    let backend = NativeWhisperBackend::new(NativeWhisperConfig {
        model_dir: model_dir.clone(),
        max_decode_tokens,
    });
    let readiness = backend.readiness();
    receipt.configuration.backend_id = Some(backend.capabilities().backend_id);
    receipt.model.readiness = Some(format!(
        "model_dir={} weights={} dims={} tokenizer={}",
        readiness.model_dir, readiness.weights, readiness.dims, readiness.tokenizer
    ));
    match hash_model_dir(&model_dir) {
        Ok((aggregate, artifacts)) => {
            receipt.model.aggregate_sha256 = Some(aggregate);
            receipt.model.artifacts = artifacts;
        }
        Err(error) => receipt.model.error = Some(error.to_string()),
    }
    let artifacts = match inspect_model_dir(&model_dir) {
        Ok(artifacts) => artifacts,
        Err(error) => {
            receipt.timings.model_load_ms = elapsed_ms(model_start);
            receipt.model.error = Some(error.to_string());
            fail(&mut receipt, FailureStage::Model, error.to_string());
            attach_recording_artifacts(&mut receipt, &store, recording_id);
            finish_and_write(&mut receipt, total_start, &receipt_path)?;
            return Ok(receipt);
        }
    };
    receipt.model.layout = Some(artifacts.layout.as_str().to_string());
    receipt.timings.model_load_ms = elapsed_ms(model_start);

    let transcription_start = Instant::now();
    let transcription =
        transcribe_recording(&store, recording_id, model_dir, max_decode_tokens, None);
    receipt.timings.transcription_ms = elapsed_ms(transcription_start);
    let transcription = match transcription {
        Ok(report) => report,
        Err(error) => {
            fail(&mut receipt, FailureStage::Transcription, error.to_string());
            attach_recording_artifacts(&mut receipt, &store, recording_id);
            finish_and_write(&mut receipt, total_start, &receipt_path)?;
            return Ok(receipt);
        }
    };
    let persisted = match store.load_recording(recording_id) {
        Ok(recording) => recording,
        Err(error) => {
            fail(&mut receipt, FailureStage::Transcription, error.to_string());
            attach_recording_artifacts(&mut receipt, &store, recording_id);
            finish_and_write(&mut receipt, total_start, &receipt_path)?;
            return Ok(receipt);
        }
    };
    let Some(chunk) = transcription.chunks.first() else {
        fail(
            &mut receipt,
            FailureStage::Transcription,
            "native transcription returned no committed chunks",
        );
        attach_recording_artifacts(&mut receipt, &store, recording_id);
        finish_and_write(&mut receipt, total_start, &receipt_path)?;
        return Ok(receipt);
    };
    let Some(transcript) = persisted
        .transcripts
        .iter()
        .find(|transcript| transcript.id == chunk.transcript_id)
    else {
        fail(
            &mut receipt,
            FailureStage::Transcription,
            "native transcription report did not resolve to a persisted transcript",
        );
        attach_recording_artifacts(&mut receipt, &store, recording_id);
        finish_and_write(&mut receipt, total_start, &receipt_path)?;
        return Ok(receipt);
    };
    if transcript.provenance != TranscriptProvenance::RawAsr {
        fail(
            &mut receipt,
            FailureStage::Transcription,
            format!(
                "expected raw_asr provenance, found {:?}",
                transcript.provenance
            ),
        );
        attach_recording_artifacts(&mut receipt, &store, recording_id);
        finish_and_write(&mut receipt, total_start, &receipt_path)?;
        return Ok(receipt);
    }
    let exact_text = transcript.text.clone();
    let normalized_reference = normalize_for_metrics(&descriptor.reference_text);
    let normalized_text = normalize_for_metrics(&exact_text);
    let reference_chars = normalized_reference.chars().collect::<Vec<_>>();
    let transcript_chars = normalized_text.chars().collect::<Vec<_>>();
    let character_errors = levenshtein(&reference_chars, &transcript_chars);
    let (word_errors, reference_word_count) = {
        let reference_words = normalized_reference.split_whitespace().collect::<Vec<_>>();
        let transcript_words = normalized_text.split_whitespace().collect::<Vec<_>>();
        (
            levenshtein(&reference_words, &transcript_words),
            reference_words.len(),
        )
    };
    let transcript_hash = sha256_bytes(exact_text.as_bytes());
    receipt.transcript = Some(TranscriptReceipt {
        transcript_id: transcript.id.to_string(),
        provenance: "raw_asr".to_string(),
        exact_transcript_text: exact_text,
        normalized_transcript_text: normalized_text,
        normalized_reference_text: normalized_reference,
        sha256: transcript_hash,
        character_errors,
        reference_character_count: reference_chars.len(),
        cer: ratio(character_errors, reference_chars.len()),
        word_errors,
        reference_word_count,
        wer: ratio(word_errors, reference_word_count),
    });
    attach_recording_artifacts(&mut receipt, &store, recording_id);
    if receipt
        .transcript
        .as_ref()
        .is_none_or(|transcript| transcript.character_errors != 0 || transcript.word_errors != 0)
    {
        fail(
            &mut receipt,
            FailureStage::Oracle,
            "native raw-ASR text did not match the normalized VCTK reference",
        );
    } else {
        receipt.status = VerificationStatus::Passed;
        receipt.passed = true;
    }

    let replay_start = Instant::now();
    receipt.replay.attempted = true;
    match replay_evidence(&store, recording_id, &persisted) {
        Ok(evidence) => {
            receipt.replay = evidence;
            if !receipt.replay.ok {
                let replay_error =
                    receipt.replay.error.clone().unwrap_or_else(|| {
                        "event replay did not match persisted state".to_string()
                    });
                fail(&mut receipt, FailureStage::Replay, replay_error);
            }
        }
        Err(error) => {
            receipt.replay.error = Some(error.to_string());
            fail(&mut receipt, FailureStage::Replay, error.to_string());
        }
    }
    receipt.timings.replay_ms = elapsed_ms(replay_start);
    finish_and_write(&mut receipt, total_start, &receipt_path)?;
    Ok(receipt)
}

fn resolve_model_dir(explicit: Option<PathBuf>) -> (Option<PathBuf>, String, Option<String>) {
    if let Some(path) = explicit {
        return (Some(path), "argument".to_string(), None);
    }
    if let Some(path) = std::env::var_os(MODEL_DIR_ENV_VAR) {
        return (
            Some(PathBuf::from(path)),
            MODEL_DIR_ENV_VAR.to_string(),
            None,
        );
    }
    match ModelHome::resolve() {
        Ok(home) => (Some(home.0), "resolved_default".to_string(), None),
        Err(error) => (None, "unresolved".to_string(), Some(error.to_string())),
    }
}

fn resolve_app_home() -> (Option<AppHome>, String, Option<String>) {
    let source = if std::env::var_os(crate::paths::APP_HOME_ENV_VAR).is_some() {
        crate::paths::APP_HOME_ENV_VAR.to_string()
    } else {
        "resolved_default".to_string()
    };
    match AppHome::resolve() {
        Ok(home) => (Some(home), source, None),
        Err(error) => (None, "unresolved".to_string(), Some(error.to_string())),
    }
}

fn torch_runtime_receipt() -> (String, bool, i64, String, String) {
    let requested = std::env::var(TORCH_DEVICE_ENV_VAR).unwrap_or_else(|_| "0".to_string());
    let selected_device = requested.parse::<i32>().map_or_else(
        |_| format!("invalid:{requested}"),
        |device| {
            if device < 0 {
                "cpu".to_string()
            } else {
                format!("cuda:{device}")
            }
        },
    );

    #[cfg(feature = "tch-native")]
    {
        let available = tch::Cuda::is_available();
        // CPU-only/unlinked CUDA builds can still produce a useful diagnostic.
        // LibTorch's version accessors panic when its CUDA hooks are absent.
        let (cudart, cudnn) = if available {
            (
                format!("{:?}", tch::utils::version_cudart()),
                format!("{:?}", tch::utils::version_cudnn()),
            )
        } else {
            (
                "CUDA unavailable to LibTorch".to_string(),
                "CUDA unavailable to LibTorch".to_string(),
            )
        };
        (
            selected_device,
            available,
            tch::Cuda::device_count(),
            cudart,
            cudnn,
        )
    }

    #[cfg(not(feature = "tch-native"))]
    {
        (
            selected_device,
            false,
            -1,
            "tch-native feature disabled".to_string(),
            "tch-native feature disabled".to_string(),
        )
    }
}

fn binary_receipt() -> BinaryReceipt {
    let path = std::env::current_exe().unwrap_or_default();
    match hash_file(&path) {
        Ok(hash) => BinaryReceipt {
            package: env!("CARGO_PKG_NAME").to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            path: path.display().to_string(),
            sha256: Some(hash.sha256),
            bytes: Some(hash.bytes),
            error: None,
        },
        Err(error) => BinaryReceipt {
            package: env!("CARGO_PKG_NAME").to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            path: path.display().to_string(),
            sha256: None,
            bytes: None,
            error: Some(error.to_string()),
        },
    }
}

fn attach_normalized_audio(receipt: &mut CanaryReceipt, path: &Path) {
    receipt.normalized_audio = Some(match hash_file(path) {
        Ok(hash) => ArtifactReceipt {
            path: path.display().to_string(),
            sha256: Some(hash.sha256),
            bytes: Some(hash.bytes),
            error: None,
        },
        Err(error) => ArtifactReceipt {
            path: path.display().to_string(),
            sha256: None,
            bytes: None,
            error: Some(error.to_string()),
        },
    });
}

fn attach_recording_artifacts(
    receipt: &mut CanaryReceipt,
    store: &RecordingStore,
    recording_id: RecordingId,
) {
    let manifest_path = store.manifest_path(recording_id);
    let events_path = store.events_path(recording_id);
    receipt.manifest = Some(artifact_receipt(&manifest_path));
    receipt.events = Some(artifact_receipt(&events_path));
    receipt.replay = match store.load_recording(recording_id) {
        Ok(persisted) => replay_evidence(store, recording_id, &persisted).unwrap_or_else(|error| {
            ReplayEvidence {
                attempted: true,
                ok: false,
                event_count: 0,
                next_sequence: None,
                recording_matches_manifest: false,
                error: Some(error.to_string()),
            }
        }),
        Err(error) => ReplayEvidence {
            attempted: true,
            ok: false,
            event_count: 0,
            next_sequence: None,
            recording_matches_manifest: false,
            error: Some(format!(
                "could not load persisted recording for replay: {error}"
            )),
        },
    };
}

fn artifact_receipt(path: &Path) -> ArtifactReceipt {
    match hash_file(path) {
        Ok(hash) => ArtifactReceipt {
            path: path.display().to_string(),
            sha256: Some(hash.sha256),
            bytes: Some(hash.bytes),
            error: None,
        },
        Err(error) => ArtifactReceipt {
            path: path.display().to_string(),
            sha256: None,
            bytes: None,
            error: Some(error.to_string()),
        },
    }
}

fn replay_evidence(
    store: &RecordingStore,
    recording_id: RecordingId,
    persisted: &crate::domain::Recording,
) -> eyre::Result<ReplayEvidence> {
    let events_path = store.events_path(recording_id);
    let event_count = std::fs::read_to_string(&events_path).map(|contents| {
        contents
            .lines()
            .filter(|line| !line.trim().is_empty())
            .count()
    })?;
    let state = store.load_state(recording_id)?;
    let recording_matches_manifest = state.recording(recording_id) == Some(persisted);
    Ok(ReplayEvidence {
        attempted: true,
        ok: recording_matches_manifest,
        event_count,
        next_sequence: Some(state.next_sequence),
        recording_matches_manifest,
        error: (!recording_matches_manifest)
            .then(|| "replayed state differs from the materialized recording".to_string()),
    })
}

fn hash_file(path: &Path) -> eyre::Result<HashResult> {
    let mut file = File::open(path)
        .map_err(|error| eyre::eyre!("failed to open {} for hashing: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut bytes = 0_u64;
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        bytes = bytes.saturating_add(count as u64);
    }
    Ok(HashResult {
        sha256: format!("{:x}", hasher.finalize()),
        bytes,
    })
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn hash_model_dir(root: &Path) -> eyre::Result<(String, Vec<ArtifactHash>)> {
    let mut paths = Vec::new();
    collect_files(root, root, &mut paths)?;
    paths.sort_by(|left, right| left.0.cmp(&right.0));
    let mut aggregate = Sha256::new();
    let mut artifacts = Vec::with_capacity(paths.len());
    for (relative_path, path) in paths {
        let hash = hash_file(&path)?;
        aggregate.update(relative_path.as_bytes());
        aggregate.update([0]);
        aggregate.update(hash.sha256.as_bytes());
        aggregate.update([0]);
        artifacts.push(ArtifactHash {
            relative_path,
            sha256: hash.sha256,
            bytes: hash.bytes,
        });
    }
    Ok((format!("{:x}", aggregate.finalize()), artifacts))
}

fn collect_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<(String, PathBuf)>,
) -> eyre::Result<()> {
    let mut entries = std::fs::read_dir(directory)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_files(root, &path, files)?;
        } else if entry.file_type()?.is_file() {
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            files.push((relative, path));
        }
    }
    Ok(())
}

fn normalize_for_metrics(text: &str) -> String {
    text.chars()
        .map(|character| {
            if character.is_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn levenshtein<T: Eq>(left: &[T], right: &[T]) -> usize {
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    for (left_index, left_value) in left.iter().enumerate() {
        let mut current = vec![left_index + 1; right.len() + 1];
        for (right_index, right_value) in right.iter().enumerate() {
            current[right_index + 1] = if left_value == right_value {
                previous[right_index]
            } else {
                1 + previous[right_index]
                    .min(previous[right_index + 1])
                    .min(current[right_index])
            };
        }
        previous = current;
    }
    previous[right.len()]
}

#[expect(
    clippy::cast_precision_loss,
    reason = "CER/WER are ratios over bounded fixture text lengths"
)]
fn ratio(errors: usize, reference: usize) -> f64 {
    if reference == 0 {
        f64::from(errors != 0)
    } else {
        errors as f64 / reference as f64
    }
}

fn elapsed_ms(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn mark_unavailable(receipt: &mut CanaryReceipt, message: impl Into<String>) {
    receipt.status = VerificationStatus::Unavailable;
    receipt.passed = false;
    receipt.failure_stage = Some(FailureStage::Corpus);
    receipt.failure = Some(message.into());
}

fn fail(receipt: &mut CanaryReceipt, stage: FailureStage, message: impl Into<String>) {
    receipt.status = VerificationStatus::Failed;
    receipt.passed = false;
    receipt.failure_stage = Some(stage);
    receipt.failure = Some(message.into());
}

fn finish_and_write(
    receipt: &mut CanaryReceipt,
    total_start: Instant,
    receipt_path: &Path,
) -> eyre::Result<()> {
    receipt.timings.total_ms = elapsed_ms(total_start);
    if let Some(parent) = receipt_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let contents = facet_json::to_string_pretty(receipt)?;
    std::fs::write(receipt_path, format!("{contents}\n"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::VctkCanaryOptions;
    use super::VerificationStatus;
    use super::levenshtein;
    use super::normalize_for_metrics;
    use super::ratio;
    use super::run_vctk_canary;
    use std::path::PathBuf;

    #[test]
    fn metric_normalization_is_punctuation_insensitive_and_deterministic() {
        assert_eq!(
            normalize_for_metrics("If you can get it."),
            "if you can get it"
        );
        assert_eq!(
            normalize_for_metrics(" IF  you\ncan get it! "),
            "if you can get it"
        );
    }

    #[test]
    fn metric_distances_and_ratios_are_bounded_by_reference() {
        let reference = "if you can get it".split_whitespace().collect::<Vec<_>>();
        let candidate = "if you can get".split_whitespace().collect::<Vec<_>>();
        assert_eq!(levenshtein(&reference, &candidate), 1);
        assert!((ratio(1, reference.len()) - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn missing_corpus_is_unavailable_and_never_passes() {
        let receipt = std::env::temp_dir().join(format!(
            "teamy-transcriber-vctk-test-{}.json",
            uuid::Uuid::new_v4()
        ));
        let report = run_vctk_canary(VctkCanaryOptions {
            vctk_root: Some(PathBuf::from("this-corpus-does-not-exist")),
            receipt_path: Some(receipt.clone()),
            model_dir: None,
            max_decode_tokens: None,
        })
        .expect("unavailable corpus should still produce a receipt");
        assert_eq!(report.status, VerificationStatus::Unavailable);
        assert!(!report.passed);
        assert!(receipt.is_file());
        let _ = std::fs::remove_file(receipt);
    }
}
