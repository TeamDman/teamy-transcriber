#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "The reference rasterizer converts bounded window coordinates between screen numeric types"
)]

use crate::capture::AudioInputDevice;
use crate::capture::list_audio_input_devices;
use crate::domain::AssetKind;
use crate::domain::ClipId;
use crate::domain::Recording;
use crate::domain::RecordingId;
use crate::domain::RecordingStatus;
use crate::native_whisper::model::inspect_model_dir;
use crate::paths::AppHome;
use crate::paths::ModelHome;
use crate::storage::RecordingStore;
use crate::transcription::NativeWhisperBackend;
use crate::transcription::NativeWhisperConfig;
use crate::transcription::NativeWhisperReadiness;
use crate::transcription::RuntimeAssetStatus;
use crate::workflow::ExportReport;
use crate::workflow::MicrophoneReport;
use crate::workflow::PrepareReport;
use crate::workflow::TranscriptionReport;
use crate::workflow::commit_transcript_edit;
use crate::workflow::create_recording;
use crate::workflow::export_recording;
use crate::workflow::prepare_recording;
use crate::workflow::record_microphone;
use crate::workflow::transcribe_recording;
use ash::Entry;
use ash::vk;
use eyre::Context;
use eyre::Result;
use eyre::bail;
use facet::Facet;
use raw_window_handle::HasDisplayHandle;
use raw_window_handle::HasWindowHandle;
use rfd::FileDialog;
use std::ffi::CString;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::sync::mpsc::Sender;
use std::sync::mpsc::channel;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::dpi::PhysicalSize;
use winit::event::ElementState;
use winit::event::Ime;
use winit::event::KeyEvent;
use winit::event::MouseButton;
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::ControlFlow;
use winit::event_loop::EventLoop;
use winit::keyboard::Key;
use winit::keyboard::ModifiersState;
use winit::keyboard::NamedKey;
use winit::window::Window;
use winit::window::WindowId;

const INITIAL_WIDTH: u32 = 1_200;
const INITIAL_HEIGHT: u32 = 760;
const BACKGROUND: Rgba = Rgba::new(0x00, 0x4d, 0x2a, 0xff);
const INK: Rgba = Rgba::new(0xdc, 0xe2, 0xdc, 0xff);
const ACTIVE: Rgba = Rgba::new(0xff, 0xb4, 0x5e, 0xff);
const INACTIVE: Rgba = Rgba::new(0x81, 0xa9, 0x91, 0xff);

/// Run the first native Teamy-Transcriber desktop surface.
///
/// The renderer is intentionally a small Ash/Vulkan transfer renderer. It
/// establishes the real window, surface, swapchain, resize, input, and redraw
/// lifecycle while the richer text and audio renderers remain replaceable.
///
/// # Errors
///
/// Returns an error when the event loop, Vulkan loader, window surface, or
/// presentation device cannot be initialized.
pub fn run() -> Result<()> {
    let event_loop = EventLoop::new().wrap_err("failed to create GUI event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut application = GuiApplication::new()?;
    event_loop
        .run_app(&mut application)
        .wrap_err("GUI event loop failed")
}

struct GuiApplication {
    window: Option<Window>,
    renderer: Option<VulkanRenderer>,
    state: GuiState,
    app_home: AppHome,
    store: RecordingStore,
    preferences: GuiPreferences,
    message_tx: Sender<GuiMessage>,
    message_rx: Receiver<GuiMessage>,
    stop_recording: Option<Arc<AtomicBool>>,
}

impl GuiApplication {
    fn new() -> Result<Self> {
        let app_home = AppHome::resolve()?;
        app_home.ensure_dir()?;
        let store = RecordingStore::new(app_home.0.clone());
        let preferences = load_preferences(&app_home);
        let model_dir = preferences
            .model_dir
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or(ModelHome::resolve()?.0);
        let save_dir = preferences
            .save_dir
            .as_deref()
            .map_or_else(|| default_save_dir(&app_home), PathBuf::from);
        let recordings = store.list_recordings()?;
        let preferred_recording = preferences
            .recording_id
            .as_deref()
            .and_then(|value| RecordingId::parse(value).ok())
            .and_then(|recording_id| {
                recordings
                    .iter()
                    .find(|recording| recording.id == recording_id)
                    .cloned()
            });
        let current_recording = preferred_recording.or_else(|| recordings.last().cloned());
        let (message_tx, message_rx) = channel();
        let mut state = GuiState::new(
            model_dir,
            save_dir,
            preferences.microphone_id.clone(),
            preferences.chunk_duration_ms,
        );
        state.set_recording(current_recording.as_ref(), &store);
        let mut application = Self {
            window: None,
            renderer: None,
            state,
            app_home,
            store,
            preferences,
            message_tx,
            message_rx,
            stop_recording: None,
        };
        application.inspect_model();
        application.refresh_devices();
        Ok(application)
    }

    fn inspect_model(&mut self) {
        let backend = NativeWhisperBackend::new(NativeWhisperConfig {
            model_dir: self.state.model_dir.clone(),
            max_decode_tokens: crate::native_whisper::whisper::DEFAULT_MAX_DECODE_TOKENS,
        });
        let readiness = backend.readiness();
        self.state.model_readiness = readiness.clone();
        self.state.model_status = model_status_text(&readiness);
        self.state.model_ready = model_is_ready(&readiness);
        if self.state.model_ready {
            match inspect_model_dir(&self.state.model_dir) {
                Ok(artifacts) => {
                    self.state.model_status = format!(
                        "MODEL READY {} W:{} D:{} T:{}",
                        artifacts.layout.as_str(),
                        readiness.weights,
                        readiness.dims,
                        readiness.tokenizer
                    );
                }
                Err(error) => {
                    self.state.model_ready = false;
                    self.state.model_status = "MODEL INVALID".to_string();
                    self.state.status_line = format!("ERROR: model validation failed: {error}");
                }
            }
        }
    }

    fn refresh_devices(&self) {
        let sender = self.message_tx.clone();
        std::thread::spawn(move || {
            let message = list_audio_input_devices().map_or_else(
                |error| GuiMessage::Failure {
                    recording_id: None,
                    operation: "microphone inventory".to_string(),
                    message: error.to_string(),
                },
                GuiMessage::Devices,
            );
            let _ = sender.send(message);
        });
    }

    fn persist_preferences(&mut self) {
        self.preferences.model_dir = Some(self.state.model_dir.to_string_lossy().into_owned());
        self.preferences.save_dir = Some(self.state.save_dir.to_string_lossy().into_owned());
        self.preferences.microphone_id = self.state.selected_microphone.clone();
        self.preferences.chunk_duration_ms = self.state.chunk_duration_ms;
        self.preferences.recording_id = self.state.recording_id.map(|id| id.to_string());
        if let Err(error) = save_preferences(&self.app_home, &self.preferences) {
            self.state.status_line = format!("ERROR: preferences not saved: {error}");
        }
    }

    fn reload_recording(&mut self, recording_id: RecordingId) -> Result<()> {
        let recording = self.store.load_recording(recording_id)?;
        self.state.set_recording(Some(&recording), &self.store);
        self.persist_preferences();
        Ok(())
    }

    fn handle_action(&mut self, action: GuiAction) {
        match action {
            GuiAction::ImportFile => self.import_file(),
            GuiAction::ChooseModel => self.choose_model(),
            GuiAction::ChooseSaveDirectory => self.choose_save_directory(),
            GuiAction::CycleMicrophone => self.cycle_microphone(),
            GuiAction::CycleRecording => self.cycle_recording(),
            GuiAction::PreviousClip => self.cycle_clip(-1),
            GuiAction::NextClip => self.cycle_clip(1),
            GuiAction::CycleChunkDuration => self.cycle_chunk_duration(),
            GuiAction::ToggleRecording => self.toggle_recording(),
            GuiAction::Prepare => self.start_prepare(),
            GuiAction::Transcribe => self.start_transcription(),
            GuiAction::Export => self.start_export(),
            GuiAction::CommitTranscriptEdit => self.commit_edit(),
            GuiAction::RefreshDevices => self.refresh_devices(),
            GuiAction::CancelEdit => {
                self.state.transcript_editing = false;
                self.state.transcript_draft = self.state.transcript.clone();
                self.state.status_line = "Transcript edit cancelled".to_string();
            }
        }
    }

    fn import_file(&mut self) {
        let Some(path) = FileDialog::new()
            .add_filter(
                "Audio and video",
                &[
                    "wav", "mp3", "m4a", "flac", "ogg", "aac", "opus", "aiff", "mp4", "mov", "mkv",
                    "webm", "avi",
                ],
            )
            .pick_file()
        else {
            return;
        };
        let kind = asset_kind_for_path(&path);
        match create_recording(&self.store, kind, &path) {
            Ok(recording_id) => {
                if let Err(error) = self.reload_recording(recording_id) {
                    self.state.status_line =
                        format!("ERROR: imported but could not reload: {error}");
                    return;
                }
                self.state.status_line =
                    format!("Imported {}; preparing audio...", display_path(&path));
                self.start_prepare();
            }
            Err(error) => {
                self.state.status_line = format!("ERROR: import failed: {error}");
            }
        }
    }

    fn choose_model(&mut self) {
        let Some(path) = FileDialog::new().pick_folder() else {
            return;
        };
        self.state.model_dir = path;
        self.inspect_model();
        self.persist_preferences();
        if self.state.model_ready {
            self.state.status_line = "Model ready for local transcription".to_string();
        } else if self.state.model_status == "MODEL INVALID" {
            self.state.status_line =
                "Model files were found but failed validation; choose another model folder"
                    .to_string();
        } else {
            self.state.status_line =
                "Model incomplete: select a folder containing model.bpk, dims.json, tokenizer.json"
                    .to_string();
        }
    }

    fn choose_save_directory(&mut self) {
        let Some(path) = FileDialog::new()
            .set_directory(&self.state.save_dir)
            .pick_folder()
        else {
            return;
        };
        self.state.save_dir = path;
        self.persist_preferences();
        self.state.status_line = format!("Save directory: {}", display_path(&self.state.save_dir));
    }

    fn cycle_microphone(&mut self) {
        if self.state.microphones.is_empty() {
            self.state.status_line = "No microphones found; refreshing devices...".to_string();
            self.refresh_devices();
            return;
        }
        let current_index = self
            .state
            .selected_microphone
            .as_ref()
            .and_then(|id| {
                self.state
                    .microphones
                    .iter()
                    .position(|device| &device.id == id)
            })
            .unwrap_or(usize::MAX);
        let next_index = if current_index == usize::MAX {
            0
        } else {
            (current_index + 1) % self.state.microphones.len()
        };
        self.state.selected_microphone = Some(self.state.microphones[next_index].id.clone());
        self.persist_preferences();
        self.state.status_line = format!("Microphone: {}", self.state.microphone_label());
    }

    fn cycle_recording(&mut self) {
        let Ok(recordings) = self.store.list_recordings() else {
            self.state.status_line = "ERROR: saved recordings could not be listed".to_string();
            return;
        };
        if recordings.is_empty() {
            self.state.status_line = "No saved recordings yet".to_string();
            return;
        }
        let current_index = self
            .state
            .recording_id
            .and_then(|recording_id| {
                recordings
                    .iter()
                    .position(|recording| recording.id == recording_id)
            })
            .unwrap_or(usize::MAX);
        let next_index = if current_index == usize::MAX {
            0
        } else {
            (current_index + 1) % recordings.len()
        };
        self.state.transcript_editing = false;
        let recording = &recordings[next_index];
        self.state.set_recording(Some(recording), &self.store);
        self.persist_preferences();
        self.state.status_line = format!(
            "Selected recording {}",
            display_path(Path::new(&recording.source.path))
        );
    }

    fn cycle_chunk_duration(&mut self) {
        const PRESETS_MS: [Option<u64>; 4] = [None, Some(10_000), Some(30_000), Some(60_000)];
        let current_index = PRESETS_MS
            .iter()
            .position(|preset| *preset == self.state.chunk_duration_ms)
            .unwrap_or(0);
        self.state.chunk_duration_ms = PRESETS_MS[(current_index + 1) % PRESETS_MS.len()];
        self.persist_preferences();
        self.state.status_line = format!(
            "Transcription chunks: {}; apply on the next transcription",
            self.state.chunk_duration_label()
        );
    }

    fn cycle_clip(&mut self, direction: isize) {
        let Some(recording_id) = self.state.recording_id else {
            self.state.status_line = "Import or record audio first".to_string();
            return;
        };
        let Ok(recording) = self.store.load_recording(recording_id) else {
            self.state.status_line = "ERROR: current recording could not be reloaded".to_string();
            return;
        };
        let Some(clip_id) = self.state.cycled_clip_id(direction) else {
            self.state.status_line =
                "No persisted clips yet; transcribe the recording first".to_string();
            return;
        };
        self.state.transcript_editing = false;
        self.state
            .set_recording_clip(&recording, &self.store, clip_id);
        self.state.status_line = format!("Selected {}", self.state.clip_label());
    }

    fn toggle_recording(&mut self) {
        if matches!(self.state.operation, GuiOperation::Recording) {
            self.request_recording_stop();
            return;
        }
        if !matches!(self.state.operation, GuiOperation::Idle) {
            self.state.status_line = "Finish the current operation first".to_string();
            return;
        }
        let stop_requested = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop_requested);
        let sender = self.message_tx.clone();
        let store = self.store.clone();
        let endpoint_id = self.state.selected_microphone.clone();
        self.stop_recording = Some(stop_requested);
        self.state.recording = true;
        self.state.operation = GuiOperation::Recording;
        self.state.status_line = "Recording microphone — click again to stop".to_string();
        std::thread::spawn(move || {
            let message = record_microphone(&store, endpoint_id.as_deref(), worker_stop)
                .map_or_else(
                    |error| GuiMessage::Failure {
                        recording_id: Some(error.recording_id),
                        operation: "microphone recording".to_string(),
                        message: error.to_string(),
                    },
                    GuiMessage::Recorded,
                );
            let _ = sender.send(message);
        });
    }

    fn request_recording_stop(&mut self) {
        if let Some(stop_requested) = &self.stop_recording {
            stop_requested.store(true, Ordering::Relaxed);
            self.state.operation = GuiOperation::Stopping;
            self.state.status_line = "Stopping microphone and saving recording...".to_string();
        }
    }

    fn start_prepare(&mut self) {
        let Some(recording_id) = self.state.recording_id else {
            self.state.status_line = "Import or record audio first".to_string();
            return;
        };
        if !matches!(self.state.operation, GuiOperation::Idle) {
            self.state.status_line = "Finish the current operation first".to_string();
            return;
        }
        let sender = self.message_tx.clone();
        let store = self.store.clone();
        self.state.operation = GuiOperation::Preparing;
        self.state.status_line = "Preparing audio to 16 kHz mono...".to_string();
        std::thread::spawn(move || {
            let message = prepare_recording(&store, recording_id).map_or_else(
                |error| GuiMessage::Failure {
                    recording_id: Some(recording_id),
                    operation: "audio preparation".to_string(),
                    message: error.to_string(),
                },
                |report| GuiMessage::Prepared {
                    recording_id,
                    report,
                },
            );
            let _ = sender.send(message);
        });
    }

    fn start_transcription(&mut self) {
        let Some(recording_id) = self.state.recording_id else {
            self.state.status_line = "Import or record audio first".to_string();
            return;
        };
        if !self.state.model_ready {
            self.state.status_line = "Choose a complete local model folder first".to_string();
            return;
        }
        if !self.state.prepared {
            self.state.status_line = "Prepare the recording before transcription".to_string();
            return;
        }
        if !matches!(self.state.operation, GuiOperation::Idle) {
            self.state.status_line = "Finish the current operation first".to_string();
            return;
        }
        let sender = self.message_tx.clone();
        let store = self.store.clone();
        let model_dir = self.state.model_dir.clone();
        let chunk_duration_us = self
            .state
            .chunk_duration_ms
            .map(|duration_ms| duration_ms.saturating_mul(1_000));
        self.state.operation = GuiOperation::Transcribing;
        self.state.status_line = "Transcribing locally with native Whisper...".to_string();
        std::thread::spawn(move || {
            let message = transcribe_recording(
                &store,
                recording_id,
                model_dir,
                crate::native_whisper::whisper::DEFAULT_MAX_DECODE_TOKENS,
                chunk_duration_us,
            )
            .map_or_else(
                |error| GuiMessage::Failure {
                    recording_id: Some(recording_id),
                    operation: "local transcription".to_string(),
                    message: error.to_string(),
                },
                |report| GuiMessage::Transcribed {
                    recording_id,
                    report,
                },
            );
            let _ = sender.send(message);
        });
    }

    fn start_export(&mut self) {
        let Some(recording_id) = self.state.recording_id else {
            self.state.status_line = "Import or record audio first".to_string();
            return;
        };
        if !matches!(self.state.operation, GuiOperation::Idle) {
            self.state.status_line = "Finish the current operation first".to_string();
            return;
        }
        let Some(path) = FileDialog::new()
            .set_directory(&self.state.save_dir)
            .set_file_name("transcript.txt")
            .save_file()
        else {
            return;
        };
        let sender = self.message_tx.clone();
        let store = self.store.clone();
        self.state.operation = GuiOperation::Exporting;
        self.state.status_line = "Exporting transcript...".to_string();
        std::thread::spawn(move || {
            let message = export_recording(&store, recording_id, Some(path)).map_or_else(
                |error| GuiMessage::Failure {
                    recording_id: Some(recording_id),
                    operation: "transcript export".to_string(),
                    message: error.to_string(),
                },
                GuiMessage::Exported,
            );
            let _ = sender.send(message);
        });
    }

    fn commit_edit(&mut self) {
        let Some(recording_id) = self.state.recording_id else {
            return;
        };
        let Some(clip_id) = self.state.selected_clip_id else {
            self.state.status_line = "Transcribe a clip before editing its text".to_string();
            return;
        };
        let text = self.state.transcript_draft.trim().to_string();
        if text.is_empty() {
            self.state.status_line = "Transcript cannot be empty".to_string();
            return;
        }
        let sender = self.message_tx.clone();
        let store = self.store.clone();
        self.state.operation = GuiOperation::Editing;
        self.state.status_line = "Saving transcript edit...".to_string();
        std::thread::spawn(move || {
            let message = commit_transcript_edit(&store, recording_id, clip_id, text).map_or_else(
                |error| GuiMessage::Failure {
                    recording_id: Some(recording_id),
                    operation: "transcript edit".to_string(),
                    message: error.to_string(),
                },
                |_| GuiMessage::Edited { recording_id },
            );
            let _ = sender.send(message);
        });
    }

    fn drain_messages(&mut self) {
        while let Ok(message) = self.message_rx.try_recv() {
            self.handle_message(message);
        }
    }

    #[expect(
        clippy::too_many_lines,
        reason = "The GUI message reducer keeps all async operation transitions together"
    )]
    fn handle_message(&mut self, message: GuiMessage) {
        match message {
            GuiMessage::Devices(devices) => {
                self.state.microphones = devices;
                let selected_is_valid =
                    self.state.selected_microphone.as_ref().is_some_and(|id| {
                        self.state.microphones.iter().any(|device| &device.id == id)
                    });
                if !selected_is_valid {
                    self.state.selected_microphone = self
                        .state
                        .microphones
                        .iter()
                        .find(|device| device.is_default)
                        .or_else(|| self.state.microphones.first())
                        .map(|device| device.id.clone());
                    self.persist_preferences();
                }
                self.state.status_line = if self.state.microphones.is_empty() {
                    "No active microphones found".to_string()
                } else {
                    format!("{} microphone(s) available", self.state.microphones.len())
                };
            }
            GuiMessage::Prepared {
                recording_id,
                report,
            } => {
                self.state.operation = GuiOperation::Idle;
                if let Err(error) = self.reload_recording(recording_id) {
                    self.state.status_line = format!("ERROR: prepared but reload failed: {error}");
                } else {
                    self.state.status_line = format!(
                        "Prepared {:.2}s at {} Hz — ready to transcribe",
                        report.metadata.duration_us as f64 / 1_000_000.0,
                        report.metadata.sample_rate_hz
                    );
                }
            }
            GuiMessage::Recorded(report) => {
                self.stop_recording = None;
                self.state.recording = false;
                self.state.operation = GuiOperation::Idle;
                if let Err(error) = self.reload_recording(report.recording_id) {
                    self.state.status_line =
                        format!("ERROR: recording saved but reload failed: {error}");
                } else {
                    self.state.status_line = "Recording saved; preparing audio...".to_string();
                    self.start_prepare();
                }
            }
            GuiMessage::Transcribed {
                recording_id,
                report,
            } => {
                self.state.operation = GuiOperation::Idle;
                if let Err(error) = self.reload_recording(recording_id) {
                    self.state.status_line =
                        format!("ERROR: transcribed but reload failed: {error}");
                } else {
                    self.state.status_line = format!(
                        "Transcription complete: {} chunk(s), {}",
                        report.chunks.len(),
                        report.backend_id
                    );
                }
            }
            GuiMessage::Exported(report) => {
                self.state.operation = GuiOperation::Idle;
                self.state.status_line = format!(
                    "Exported {} transcript(s) to {}",
                    report.transcript_count,
                    display_path(&report.output_path)
                );
            }
            GuiMessage::Edited { recording_id } => {
                self.state.operation = GuiOperation::Idle;
                self.state.transcript_editing = false;
                if let Err(error) = self.reload_recording(recording_id) {
                    self.state.status_line =
                        format!("ERROR: edit saved but reload failed: {error}");
                } else {
                    self.state.status_line =
                        "Transcript edit saved with user-edit provenance".to_string();
                }
            }
            GuiMessage::Failure {
                recording_id,
                operation,
                message,
            } => {
                self.stop_recording = None;
                self.state.recording = false;
                self.state.operation = GuiOperation::Idle;
                let reload_message = recording_id.and_then(|recording_id| {
                    self.reload_recording(recording_id)
                        .err()
                        .map(|error| format!("; recovery reload failed: {error}"))
                });
                self.state.status_line = format!(
                    "ERROR: {operation}: {message}{}",
                    reload_message.unwrap_or_default()
                );
            }
        }
    }

    fn handle_key(&mut self, event: &KeyEvent, modifiers: ModifiersState) {
        if event.state != ElementState::Pressed {
            return;
        }
        if self.state.transcript_editing {
            match &event.logical_key {
                Key::Named(NamedKey::Escape) => self.handle_action(GuiAction::CancelEdit),
                Key::Named(NamedKey::Backspace) => self.state.backspace(),
                Key::Named(NamedKey::Enter) => self.handle_action(GuiAction::CommitTranscriptEdit),
                _ => {
                    if let Some(text) = event.text.as_deref() {
                        self.state.commit_input(text);
                    }
                }
            }
            return;
        }
        if matches!(&event.logical_key, Key::Named(NamedKey::Escape)) {
            if matches!(self.state.operation, GuiOperation::Recording) {
                self.toggle_recording();
            }
        } else if matches!(&event.logical_key, Key::Named(NamedKey::Space)) {
            self.toggle_recording();
        } else if modifiers.control_key()
            && matches!(&event.logical_key, Key::Character(value) if value.eq_ignore_ascii_case("e"))
        {
            self.start_export();
        }
    }
}

impl ApplicationHandler for GuiApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("Teamy-Transcriber")
            .with_inner_size(PhysicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT))
            .with_min_inner_size(PhysicalSize::new(720, 520));
        let Ok(window) = event_loop.create_window(attributes) else {
            event_loop.exit();
            return;
        };

        match VulkanRenderer::new(&window) {
            Ok(renderer) => {
                self.window = Some(window);
                self.renderer = Some(renderer);
            }
            Err(error) => {
                eprintln!("failed to initialize Teamy-Transcriber GUI: {error:#}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                self.request_recording_stop();
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    return;
                }
                if let (Some(window), Some(renderer)) =
                    (self.window.as_ref(), self.renderer.as_mut())
                    && let Err(error) = renderer.recreate_swapchain(window)
                {
                    eprintln!("failed to resize Teamy-Transcriber GUI: {error:#}");
                    event_loop.exit();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.state.cursor = position;
            }
            WindowEvent::ModifiersChanged(modifiers) => self.state.modifiers = modifiers.state(),
            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_key(&event, self.state.modifiers);
            }
            WindowEvent::Ime(Ime::Commit(text)) => {
                if self.state.transcript_editing {
                    self.state.commit_input(&text);
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                if let Some(window) = self.window.as_ref()
                    && let Some(action) = self.state.click(window.inner_size())
                {
                    self.handle_action(action);
                }
            }
            WindowEvent::RedrawRequested => self.draw(event_loop),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.drain_messages();
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}

impl GuiApplication {
    fn draw(&mut self, event_loop: &ActiveEventLoop) {
        let (Some(window), Some(renderer)) = (self.window.as_ref(), self.renderer.as_mut()) else {
            return;
        };
        self.state.phase += 0.025;
        match renderer.draw(&self.state) {
            Ok(true) => {
                if let Err(error) = renderer.recreate_swapchain(window) {
                    eprintln!("failed to recreate GUI swapchain: {error:#}");
                    event_loop.exit();
                }
            }
            Ok(false) => {}
            Err(error) => {
                eprintln!("failed to draw Teamy-Transcriber GUI: {error:#}");
                event_loop.exit();
            }
        }
    }
}

#[derive(Clone, Debug, Default, Facet)]
struct GuiPreferences {
    model_dir: Option<String>,
    save_dir: Option<String>,
    microphone_id: Option<String>,
    chunk_duration_ms: Option<u64>,
    recording_id: Option<String>,
}

#[derive(Debug)]
enum GuiMessage {
    Devices(Vec<AudioInputDevice>),
    Prepared {
        recording_id: RecordingId,
        report: PrepareReport,
    },
    Recorded(MicrophoneReport),
    Transcribed {
        recording_id: RecordingId,
        report: TranscriptionReport,
    },
    Exported(ExportReport),
    Edited {
        recording_id: RecordingId,
    },
    Failure {
        recording_id: Option<RecordingId>,
        operation: String,
        message: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GuiAction {
    ImportFile,
    ChooseModel,
    ChooseSaveDirectory,
    CycleMicrophone,
    CycleRecording,
    PreviousClip,
    NextClip,
    CycleChunkDuration,
    ToggleRecording,
    Prepare,
    Transcribe,
    Export,
    CommitTranscriptEdit,
    RefreshDevices,
    CancelEdit,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum GuiOperation {
    #[default]
    Idle,
    Preparing,
    Recording,
    Stopping,
    Transcribing,
    Exporting,
    Editing,
}

#[derive(Clone, Copy, Debug)]
struct Rect {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
}

impl Rect {
    const fn new(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    fn contains(self, point: Point) -> bool {
        point.x >= self.left
            && point.x <= self.right
            && point.y >= self.top
            && point.y <= self.bottom
    }

    fn as_i32(self) -> (i32, i32, i32, i32) {
        (
            self.left.round() as i32,
            self.top.round() as i32,
            self.right.round() as i32,
            self.bottom.round() as i32,
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct GuiLayout {
    mic_center: Point,
    mic_radius: f32,
    import: Rect,
    model: Rect,
    recording: Rect,
    microphone: Rect,
    save_directory: Rect,
    prepare: Rect,
    transcribe: Rect,
    waveform: Rect,
    transcript: Rect,
    export: Rect,
    refresh_devices: Rect,
    previous_clip: Rect,
    next_clip: Rect,
    chunk_duration: Rect,
}

impl GuiLayout {
    fn new(size: PhysicalSize<u32>) -> Self {
        let width = size.width as f32;
        let height = size.height as f32;
        Self {
            mic_center: Point::new(width * 0.16, height * 0.38),
            mic_radius: (height * 0.115).max(58.0),
            import: Rect::new(width * 0.69, 16.0, width * 0.80, 94.0),
            model: Rect::new(width * 0.81, 16.0, width * 0.96, 94.0),
            recording: Rect::new(width * 0.56, 16.0, width * 0.68, 94.0),
            microphone: Rect::new(width * 0.27, height * 0.26, width * 0.63, height * 0.35),
            save_directory: Rect::new(width * 0.27, height * 0.37, width * 0.63, height * 0.46),
            prepare: Rect::new(width * 0.69, height * 0.26, width * 0.81, height * 0.35),
            transcribe: Rect::new(width * 0.84, height * 0.26, width * 0.96, height * 0.35),
            waveform: Rect::new(width * 0.05, height * 0.54, width * 0.67, height * 0.70),
            transcript: Rect::new(width * 0.05, height * 0.74, width * 0.67, height * 0.95),
            export: Rect::new(width * 0.69, height * 0.54, width * 0.81, height * 0.64),
            refresh_devices: Rect::new(width * 0.84, height * 0.37, width * 0.96, height * 0.46),
            previous_clip: Rect::new(width * 0.69, height * 0.66, width * 0.81, height * 0.75),
            next_clip: Rect::new(width * 0.84, height * 0.66, width * 0.96, height * 0.75),
            chunk_duration: Rect::new(width * 0.69, height * 0.78, width * 0.96, height * 0.87),
        }
    }
}

#[derive(Debug, Clone)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "These booleans are independent visible state facets in the small reference GUI"
)]
struct GuiState {
    cursor: PhysicalPosition<f64>,
    modifiers: ModifiersState,
    phase: f32,
    recording: bool,
    transcript: String,
    transcript_draft: String,
    transcript_editable: bool,
    transcript_editing: bool,
    status_line: String,
    model_dir: PathBuf,
    model_readiness: NativeWhisperReadiness,
    model_ready: bool,
    model_status: String,
    save_dir: PathBuf,
    microphones: Vec<AudioInputDevice>,
    selected_microphone: Option<String>,
    operation: GuiOperation,
    recording_id: Option<RecordingId>,
    recording_status: Option<RecordingStatus>,
    recording_source: String,
    selected_clip_id: Option<ClipId>,
    clip_ids: Vec<ClipId>,
    chunk_duration_ms: Option<u64>,
    prepared: bool,
}

impl GuiState {
    fn new(
        model_dir: PathBuf,
        save_dir: PathBuf,
        selected_microphone: Option<String>,
        chunk_duration_ms: Option<u64>,
    ) -> Self {
        Self {
            cursor: PhysicalPosition::new(0.0, 0.0),
            modifiers: ModifiersState::default(),
            phase: 0.0,
            recording: false,
            transcript: "No transcript yet. Import or record audio.".to_string(),
            transcript_draft: String::new(),
            transcript_editable: false,
            transcript_editing: false,
            status_line: "Choose a model, import media, or record from the microphone".to_string(),
            model_dir,
            model_readiness: missing_readiness(),
            model_ready: false,
            model_status: "MODEL MISSING".to_string(),
            save_dir,
            microphones: Vec::new(),
            selected_microphone,
            operation: GuiOperation::Idle,
            recording_id: None,
            recording_status: None,
            recording_source: "NO RECORDING".to_string(),
            selected_clip_id: None,
            clip_ids: Vec::new(),
            chunk_duration_ms,
            prepared: false,
        }
    }

    fn set_recording(&mut self, recording: Option<&Recording>, store: &RecordingStore) {
        let preferred_clip = self.selected_clip_id;
        self.set_recording_with_clip(recording, store, preferred_clip);
    }

    fn set_recording_clip(
        &mut self,
        recording: &Recording,
        store: &RecordingStore,
        clip_id: ClipId,
    ) {
        self.set_recording_with_clip(Some(recording), store, Some(clip_id));
    }

    fn set_recording_with_clip(
        &mut self,
        recording: Option<&Recording>,
        store: &RecordingStore,
        preferred_clip: Option<ClipId>,
    ) {
        let Some(recording) = recording else {
            self.recording = false;
            self.recording_id = None;
            self.recording_status = None;
            self.recording_source = "NO RECORDING".to_string();
            self.selected_clip_id = None;
            self.clip_ids.clear();
            self.prepared = false;
            self.transcript_editable = false;
            if !self.transcript_editing {
                self.transcript = "No transcript yet. Import or record audio.".to_string();
                self.transcript_draft = self.transcript.clone();
            }
            return;
        };
        self.recording_id = Some(recording.id);
        self.recording_status = Some(recording.status);
        self.recording_source = display_path(Path::new(&recording.source.path));
        self.recording = recording.status == RecordingStatus::Recording;
        self.clip_ids = recording
            .clips
            .iter()
            .find(|clip| clip.status != crate::domain::ClipStatus::Deleted)
            .into_iter()
            .chain(
                recording
                    .clips
                    .iter()
                    .filter(|clip| clip.status != crate::domain::ClipStatus::Deleted)
                    .skip(1),
            )
            .map(|clip| clip.id)
            .collect();
        self.selected_clip_id = preferred_clip
            .filter(|clip_id| self.clip_ids.contains(clip_id))
            .or_else(|| self.clip_ids.first().copied());
        self.prepared = store
            .recording_dir(recording.id)
            .join("audio")
            .join("normalized-16khz-mono.wav")
            .is_file();
        if !self.transcript_editing {
            let transcript = self
                .selected_clip_id
                .and_then(|clip_id| {
                    recording
                        .transcripts
                        .iter()
                        .rev()
                        .find(|transcript| transcript.clip_id == clip_id)
                })
                .map(|transcript| transcript.text.clone());
            if let Some(transcript) = transcript {
                self.transcript_editable = true;
                self.transcript = transcript;
            } else {
                self.transcript_editable = false;
                self.transcript = "No transcript yet. Prepare audio, then transcribe.".to_string();
            }
            self.transcript_draft = self.transcript.clone();
        }
    }

    fn cycled_clip_id(&self, direction: isize) -> Option<ClipId> {
        if self.clip_ids.is_empty() {
            return None;
        }
        let current = self
            .selected_clip_id
            .and_then(|clip_id| self.clip_ids.iter().position(|id| *id == clip_id))
            .unwrap_or(0);
        let length = isize::try_from(self.clip_ids.len()).ok()?;
        let next = (isize::try_from(current).ok()? + direction).rem_euclid(length);
        self.clip_ids.get(usize::try_from(next).ok()?).copied()
    }

    fn clip_label(&self) -> String {
        let index = self
            .selected_clip_id
            .and_then(|clip_id| self.clip_ids.iter().position(|id| *id == clip_id))
            .map_or(0, |index| index + 1);
        if self.clip_ids.is_empty() {
            "CLIP NONE".to_string()
        } else {
            format!("CLIP {index}/{}", self.clip_ids.len())
        }
    }

    fn chunk_duration_label(&self) -> String {
        self.chunk_duration_ms.map_or_else(
            || "FULL".to_string(),
            |duration_ms| format!("{}S", duration_ms / 1_000),
        )
    }

    fn microphone_label(&self) -> String {
        self.selected_microphone.as_ref().map_or_else(
            || "Select microphone".to_string(),
            |id| {
                self.microphones
                    .iter()
                    .find(|device| &device.id == id)
                    .map_or_else(
                        || "Select microphone".to_string(),
                        |device| device.name.clone(),
                    )
            },
        )
    }

    fn click(&mut self, size: PhysicalSize<u32>) -> Option<GuiAction> {
        let layout = GuiLayout::new(size);
        let cursor = Point::new(self.cursor.x as f32, self.cursor.y as f32);
        if cursor.distance_squared(layout.mic_center) <= layout.mic_radius.powi(2) {
            return Some(GuiAction::ToggleRecording);
        }
        if !matches!(self.operation, GuiOperation::Idle) {
            return None;
        }
        if layout.import.contains(cursor) {
            return Some(GuiAction::ImportFile);
        }
        if layout.model.contains(cursor) {
            return Some(GuiAction::ChooseModel);
        }
        if layout.recording.contains(cursor) {
            return Some(GuiAction::CycleRecording);
        }
        if layout.microphone.contains(cursor) {
            return Some(GuiAction::CycleMicrophone);
        }
        if layout.save_directory.contains(cursor) {
            return Some(GuiAction::ChooseSaveDirectory);
        }
        if layout.prepare.contains(cursor) {
            return Some(GuiAction::Prepare);
        }
        if layout.transcribe.contains(cursor) {
            return Some(GuiAction::Transcribe);
        }
        if layout.export.contains(cursor) {
            return Some(GuiAction::Export);
        }
        if layout.refresh_devices.contains(cursor) {
            return Some(GuiAction::RefreshDevices);
        }
        if layout.previous_clip.contains(cursor) {
            return Some(GuiAction::PreviousClip);
        }
        if layout.next_clip.contains(cursor) {
            return Some(GuiAction::NextClip);
        }
        if layout.chunk_duration.contains(cursor) {
            return Some(GuiAction::CycleChunkDuration);
        }
        if layout.transcript.contains(cursor) {
            if !self.transcript_editable {
                self.status_line = "Transcribe a clip before editing its text".to_string();
                return None;
            }
            self.transcript_editing = true;
            self.transcript_draft = self.transcript.clone();
            self.status_line = "Editing transcript — Enter saves, Escape cancels".to_string();
        }
        None
    }

    fn commit_input(&mut self, text: &str) {
        if self.transcript_draft.chars().count() >= 10_000 {
            return;
        }
        self.transcript_draft.push_str(text);
    }

    fn backspace(&mut self) {
        self.transcript_draft.pop();
    }

    fn transcript_display(&self) -> &str {
        if self.transcript_editing {
            &self.transcript_draft
        } else {
            &self.transcript
        }
    }
}

impl Default for GuiState {
    fn default() -> Self {
        Self::new(PathBuf::new(), PathBuf::new(), None, None)
    }
}

fn missing_readiness() -> NativeWhisperReadiness {
    NativeWhisperReadiness {
        model_dir: RuntimeAssetStatus::Missing,
        weights: RuntimeAssetStatus::Missing,
        dims: RuntimeAssetStatus::Missing,
        tokenizer: RuntimeAssetStatus::Missing,
    }
}

fn model_is_ready(readiness: &NativeWhisperReadiness) -> bool {
    readiness.model_dir == RuntimeAssetStatus::Present
        && readiness.weights == RuntimeAssetStatus::Present
        && readiness.dims == RuntimeAssetStatus::Present
        && readiness.tokenizer == RuntimeAssetStatus::Present
}

fn model_status_text(readiness: &NativeWhisperReadiness) -> String {
    format!(
        "MODEL {} W:{} D:{} T:{}",
        if model_is_ready(readiness) {
            "READY"
        } else {
            "MISSING"
        },
        readiness.weights,
        readiness.dims,
        readiness.tokenizer
    )
}

fn asset_kind_for_path(path: &Path) -> AssetKind {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("mp4" | "mov" | "mkv" | "webm" | "avi") => AssetKind::VideoFile,
        _ => AssetKind::AudioFile,
    }
}

fn default_save_dir(app_home: &AppHome) -> PathBuf {
    std::env::var_os("USERPROFILE").map_or_else(
        || app_home.0.join("exports"),
        |path| PathBuf::from(path).join("Downloads"),
    )
}

fn display_path(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.to_string_lossy().into_owned(),
        |name| name.to_string_lossy().into_owned(),
    )
}

fn compact_text(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

fn operation_label(operation: GuiOperation) -> &'static str {
    match operation {
        GuiOperation::Idle => "IDLE",
        GuiOperation::Preparing => "PREPARING AUDIO",
        GuiOperation::Recording => "RECORDING",
        GuiOperation::Stopping => "SAVING RECORDING",
        GuiOperation::Transcribing => "TRANSCRIBING LOCALLY",
        GuiOperation::Exporting => "EXPORTING TEXT",
        GuiOperation::Editing => "SAVING EDIT",
    }
}

fn load_preferences(app_home: &AppHome) -> GuiPreferences {
    std::fs::read_to_string(app_home.file_path("gui-settings.json"))
        .ok()
        .and_then(|contents| facet_json::from_str(&contents).ok())
        .unwrap_or_default()
}

fn save_preferences(app_home: &AppHome, preferences: &GuiPreferences) -> Result<()> {
    let path = app_home.file_path("gui-settings.json");
    let temporary_path = path.with_extension("json.tmp");
    let contents = facet_json::to_string_pretty(preferences)?;
    std::fs::write(&temporary_path, format!("{contents}\n"))?;
    if path.exists() {
        std::fs::remove_file(&path)?;
    }
    std::fs::rename(temporary_path, path)?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct Rgba {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

impl Rgba {
    const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }
}

#[derive(Debug, Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
}

impl Point {
    const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    fn distance_squared(self, other: Self) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }
}

#[derive(Debug)]
struct Canvas {
    width: u32,
    height: u32,
    bgra: bool,
    pixels: Vec<u8>,
}

impl Canvas {
    fn new(width: u32, height: u32, format: vk::Format) -> Result<Self> {
        let pixel_count = usize::try_from(width)
            .ok()
            .and_then(|width| {
                usize::try_from(height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| eyre::eyre!("GUI surface dimensions overflowed"))?;
        let bgra = matches!(
            format,
            vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_UNORM
        );
        let mut canvas = Self {
            width,
            height,
            bgra,
            pixels: vec![0; pixel_count],
        };
        canvas.clear(BACKGROUND);
        Ok(canvas)
    }

    fn clear(&mut self, color: Rgba) {
        for y in 0..self.height as i32 {
            for x in 0..self.width as i32 {
                self.set_pixel(x, y, color);
            }
        }
    }

    fn set_pixel(&mut self, x: i32, y: i32, color: Rgba) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }
        let index = (y as usize * self.width as usize + x as usize) * 4;
        if self.bgra {
            self.pixels[index..index + 4].copy_from_slice(&[color.b, color.g, color.r, color.a]);
        } else {
            self.pixels[index..index + 4].copy_from_slice(&[color.r, color.g, color.b, color.a]);
        }
    }

    fn line(&mut self, start: Point, end: Point, color: Rgba) {
        let mut x0 = start.x.round() as i32;
        let mut y0 = start.y.round() as i32;
        let x1 = end.x.round() as i32;
        let y1 = end.y.round() as i32;
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut error = dx + dy;
        loop {
            self.set_pixel(x0, y0, color);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let twice = 2 * error;
            if twice >= dy {
                error += dy;
                x0 += sx;
            }
            if twice <= dx {
                error += dx;
                y0 += sy;
            }
        }
    }

    fn rect(&mut self, left: i32, top: i32, right: i32, bottom: i32, color: Rgba) {
        self.line(
            Point::new(left as f32, top as f32),
            Point::new(right as f32, top as f32),
            color,
        );
        self.line(
            Point::new(right as f32, top as f32),
            Point::new(right as f32, bottom as f32),
            color,
        );
        self.line(
            Point::new(right as f32, bottom as f32),
            Point::new(left as f32, bottom as f32),
            color,
        );
        self.line(
            Point::new(left as f32, bottom as f32),
            Point::new(left as f32, top as f32),
            color,
        );
    }

    fn circle(&mut self, center: Point, radius: f32, color: Rgba) {
        let steps = 96;
        let mut previous = Point::new(center.x + radius, center.y);
        for step in 1..=steps {
            let angle = std::f32::consts::TAU * step as f32 / steps as f32;
            let current = Point::new(
                center.x + radius * angle.cos(),
                center.y + radius * angle.sin(),
            );
            self.line(previous, current, color);
            previous = current;
        }
    }

    fn draw_text(&mut self, origin: Point, text: &str, scale: i32, color: Rgba) {
        let mut x = origin.x as i32;
        let mut y = origin.y as i32;
        for character in text.chars() {
            if character == '\n' {
                x = origin.x as i32;
                y += 8 * scale;
                continue;
            }
            let glyph = glyph(character);
            for (row, bits) in glyph.iter().copied().enumerate() {
                for column in 0..5 {
                    if bits & (1 << (4 - column)) != 0 {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                self.set_pixel(
                                    x + column * scale + dx,
                                    y + row as i32 * scale + dy,
                                    color,
                                );
                            }
                        }
                    }
                }
            }
            x += 6 * scale;
        }
    }

    fn panel(&mut self, left: i32, top: i32, right: i32, bottom: i32, color: Rgba) {
        self.rect(left, top, right, bottom, color);
        self.rect(left + 2, top + 2, right - 2, bottom - 2, color);
    }

    fn button(&mut self, rect: Rect, label: &str, active: bool, enabled: bool) {
        let color = if !enabled {
            INACTIVE
        } else if active {
            ACTIVE
        } else {
            INK
        };
        let (left, top, right, bottom) = rect.as_i32();
        self.panel(left, top, right, bottom, color);
        self.draw_text(
            Point::new((left + 12) as f32, (top + (bottom - top - 14) / 2) as f32),
            label,
            2,
            color,
        );
    }

    fn draw_wrapped_text(
        &mut self,
        origin: Point,
        text: &str,
        max_width: i32,
        scale: i32,
        color: Rgba,
    ) {
        let max_chars = usize::try_from((max_width / (6 * scale).max(1)).max(1)).unwrap_or(1);
        let mut wrapped = String::new();
        for line in text.lines() {
            let characters = line.chars().collect::<Vec<_>>();
            for chunk in characters.chunks(max_chars) {
                if !wrapped.is_empty() {
                    wrapped.push('\n');
                }
                wrapped.extend(chunk);
            }
        }
        if wrapped.is_empty() {
            wrapped.push(' ');
        }
        self.draw_text(origin, &wrapped, scale, color);
    }

    #[expect(
        clippy::too_many_lines,
        reason = "The reference layout is kept together so the first GUI slice mirrors the supplied sketch"
    )]
    fn render_ui(&mut self, state: &GuiState) {
        self.clear(BACKGROUND);
        let width = self.width as f32;
        let height = self.height as f32;
        let margin = (width * 0.018).max(16.0) as i32;
        let layout = GuiLayout::new(PhysicalSize::new(self.width, self.height));
        let ink = if state.recording { ACTIVE } else { INK };
        let enabled = matches!(state.operation, GuiOperation::Idle);

        self.panel(margin, 16, self.width as i32 - margin, 94, ink);
        self.draw_text(
            Point::new((margin + 24) as f32, 35.0),
            "TEAMY-TRANSCRIBER",
            3,
            ink,
        );
        self.draw_text(
            Point::new((margin + 24) as f32, 70.0),
            &compact_text(&state.status_line, 54),
            2,
            if state.status_line.starts_with("ERROR") {
                ACTIVE
            } else {
                INACTIVE
            },
        );
        self.button(layout.recording, "RECENT", false, enabled);
        self.button(layout.import, "IMPORT", false, enabled);
        self.button(layout.model, "MODEL", state.model_ready, enabled);

        self.circle(layout.mic_center, layout.mic_radius, ink);
        self.circle(layout.mic_center, layout.mic_radius - 3.0, ink);
        self.circle(
            Point::new(layout.mic_center.x, layout.mic_center.y - 12.0),
            19.0,
            ink,
        );
        self.line(
            Point::new(layout.mic_center.x - 19.0, layout.mic_center.y - 12.0),
            Point::new(layout.mic_center.x - 19.0, layout.mic_center.y + 18.0),
            ink,
        );
        self.line(
            Point::new(layout.mic_center.x + 19.0, layout.mic_center.y - 12.0),
            Point::new(layout.mic_center.x + 19.0, layout.mic_center.y + 18.0),
            ink,
        );
        self.line(
            Point::new(layout.mic_center.x - 19.0, layout.mic_center.y + 18.0),
            Point::new(layout.mic_center.x, layout.mic_center.y + 30.0),
            ink,
        );
        self.line(
            Point::new(layout.mic_center.x + 19.0, layout.mic_center.y + 18.0),
            Point::new(layout.mic_center.x, layout.mic_center.y + 30.0),
            ink,
        );
        self.line(
            Point::new(layout.mic_center.x, layout.mic_center.y + 30.0),
            Point::new(layout.mic_center.x, layout.mic_center.y + 48.0),
            ink,
        );
        self.line(
            Point::new(layout.mic_center.x - 24.0, layout.mic_center.y + 49.0),
            Point::new(layout.mic_center.x + 24.0, layout.mic_center.y + 49.0),
            ink,
        );

        let (microphone_left, microphone_top, microphone_right, microphone_bottom) =
            layout.microphone.as_i32();
        let (save_left, save_top, save_right, save_bottom) = layout.save_directory.as_i32();
        self.panel(
            microphone_left,
            microphone_top,
            microphone_right,
            microphone_bottom,
            ink,
        );
        self.panel(save_left, save_top, save_right, save_bottom, ink);
        self.draw_text(
            Point::new((microphone_left + 16) as f32, (microphone_top + 20) as f32),
            &compact_text(&format!("MIC: {}", state.microphone_label()), 33),
            2,
            ink,
        );
        self.draw_text(
            Point::new((save_left + 16) as f32, (save_top + 20) as f32),
            &compact_text(&format!("SAVE: {}", display_path(&state.save_dir)), 33),
            2,
            ink,
        );
        self.button(layout.prepare, "PREPARE", state.prepared, enabled);
        self.button(
            layout.transcribe,
            "TRANSCRIBE",
            state.recording_id.is_some() && state.model_ready && state.prepared,
            enabled,
        );

        let (panel_left, waveform_top, panel_right, waveform_bottom) = layout.waveform.as_i32();
        let (_, transcript_top, transcript_right, transcript_bottom) = layout.transcript.as_i32();
        self.panel(panel_left, waveform_top, panel_right, waveform_bottom, ink);
        self.panel(
            panel_left,
            transcript_top,
            transcript_right,
            transcript_bottom,
            if state.transcript_editing {
                ACTIVE
            } else {
                ink
            },
        );

        let center_y = (waveform_top + waveform_bottom) as f32 * 0.5;
        let amplitude =
            (waveform_bottom - waveform_top) as f32 * if state.recording { 0.42 } else { 0.16 };
        let mut previous = Point::new(panel_left as f32 + 12.0, center_y);
        for index in 1..=180 {
            let fraction = index as f32 / 180.0;
            let x = panel_left as f32 + 12.0 + fraction * (panel_right - panel_left - 24) as f32;
            let y = center_y
                + (fraction * 32.0 + state.phase).sin()
                    * amplitude
                    * (0.45 + 0.55 * (fraction * 11.0).sin().abs());
            let current = Point::new(x, y);
            self.line(previous, current, ink);
            previous = current;
        }
        self.draw_wrapped_text(
            Point::new((panel_left + 18) as f32, (transcript_top + 20) as f32),
            state.transcript_display(),
            transcript_right - panel_left - 36,
            2,
            if state.transcript_editing {
                ACTIVE
            } else {
                ink
            },
        );
        self.button(
            layout.export,
            "EXPORT",
            false,
            state.recording_id.is_some() && enabled,
        );
        self.button(layout.refresh_devices, "MIC LIST", false, enabled);
        self.button(
            layout.previous_clip,
            "PREV",
            false,
            state.clip_ids.len() > 1 && enabled,
        );
        self.button(
            layout.next_clip,
            "NEXT",
            false,
            state.clip_ids.len() > 1 && enabled,
        );
        self.button(
            layout.chunk_duration,
            &format!("CHUNK {}", state.chunk_duration_label()),
            state.chunk_duration_ms.is_some(),
            enabled,
        );
        let status_top = (height * 0.49) as i32;
        self.draw_wrapped_text(
            Point::new(width * 0.70, status_top as f32),
            &format!(
                "{}\nMODEL PATH {}\n{}\n{}\nSOURCE {}",
                state.model_status,
                display_path(&state.model_dir),
                operation_label(state.operation),
                state.clip_label(),
                &state.recording_source,
            ),
            (width * 0.26) as i32,
            2,
            INACTIVE,
        );
    }
}

#[expect(
    clippy::too_many_lines,
    reason = "Keeping the small fixed bitmap alphabet together makes the renderer deterministic"
)]
fn glyph(character: char) -> [u8; 7] {
    match character.to_ascii_uppercase() {
        'A' => [
            0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'B' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
        ],
        'C' => [
            0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111,
        ],
        'D' => [
            0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
        ],
        'E' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
        ],
        'F' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'G' => [
            0b01111, 0b10000, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111,
        ],
        'H' => [
            0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
        ],
        'I' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b11111,
        ],
        'J' => [
            0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100,
        ],
        'K' => [
            0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
        ],
        'L' => [
            0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
        ],
        'M' => [
            0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001,
        ],
        'N' => [
            0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
        ],
        'O' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'P' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
        ],
        'Q' => [
            0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101,
        ],
        'R' => [
            0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
        ],
        'S' => [
            0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        'T' => [
            0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'U' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
        ],
        'V' => [
            0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
        ],
        'W' => [
            0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001,
        ],
        'X' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001,
        ],
        'Y' => [
            0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
        ],
        'Z' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
        ],
        '0' => [
            0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
        ],
        '1' => [
            0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
        ],
        '2' => [
            0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
        ],
        '3' => [
            0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110,
        ],
        '4' => [
            0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
        ],
        '5' => [
            0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110,
        ],
        '6' => [
            0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
        ],
        '7' => [
            0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
        ],
        '8' => [
            0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
        ],
        '9' => [
            0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110,
        ],
        '-' => [0, 0, 0, 0b11111, 0, 0, 0],
        ':' => [0, 0b00100, 0b00100, 0, 0b00100, 0b00100, 0],
        '/' => [0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0, 0],
        '~' => [0, 0, 0b01001, 0b10110, 0, 0, 0],
        ',' => [0, 0, 0, 0, 0, 0b00100, 0b01000],
        _ => [0; 7],
    }
}

struct VulkanRenderer {
    _entry: Entry,
    instance: ash::Instance,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    initialized_images: Vec<bool>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
}

impl VulkanRenderer {
    #[expect(
        clippy::too_many_lines,
        reason = "Vulkan initialization is kept as one ordered transaction so cleanup ownership is auditable"
    )]
    fn new(window: &Window) -> Result<Self> {
        // SAFETY: Loading the process Vulkan loader is the first step before any
        // instance/device handle is used.
        let entry = unsafe { Entry::load().wrap_err("failed to load Vulkan loader")? };
        let app_name = CString::new("teamy-transcriber").expect("static app name has no NUL");
        let engine_name = CString::new("teamy-transcriber").expect("static engine name has no NUL");
        let display_handle = window
            .display_handle()
            .wrap_err("failed to obtain display handle")?
            .as_raw();
        let extensions = ash_window::enumerate_required_extensions(display_handle)
            .wrap_err("failed to enumerate required window extensions")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&engine_name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_1);
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(extensions);
        // SAFETY: The instance create-info references live application names and
        // only the extensions reported by the windowing backend.
        let instance = unsafe {
            entry
                .create_instance(&instance_info, None)
                .wrap_err("failed to create Vulkan instance")?
        };
        let window_handle = window
            .window_handle()
            .wrap_err("failed to obtain window handle")?
            .as_raw();
        // SAFETY: The raw display/window handles belong to the live Winit window
        // and remain valid for the renderer lifetime.
        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
                .wrap_err("failed to create Vulkan window surface")?
        };
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let (physical_device, queue_family_index) =
            pick_physical_device(&instance, &surface_loader, surface)?;
        let queue_priorities = [1.0_f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities)];
        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extensions);
        // SAFETY: The selected queue family supports graphics and presentation,
        // and the swapchain extension is enabled for the logical device.
        let device = unsafe {
            instance
                .create_device(physical_device, &device_info, None)
                .wrap_err("failed to create Vulkan logical device")?
        };
        // SAFETY: Queue zero exists in the selected queue family created above.
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let (swapchain, format, extent, images) = create_swapchain(
            window,
            &surface_loader,
            surface,
            &swapchain_loader,
            physical_device,
            vk::SwapchainKHR::null(),
        )?;
        let image_count = images.len();
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: The command pool uses the selected live graphics queue family.
        let command_pool = unsafe {
            device
                .create_command_pool(&command_pool_info, None)
                .wrap_err("failed to create GUI command pool")?
        };
        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: The command buffer allocation references the live command pool.
        let command_buffer = unsafe {
            device
                .allocate_command_buffers(&command_buffer_info)
                .wrap_err("failed to allocate GUI command buffer")?
        }[0];
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        // SAFETY: The semaphore create-info is complete and the device is live.
        let image_available = unsafe {
            device
                .create_semaphore(&semaphore_info, None)
                .wrap_err("failed to create image-available semaphore")?
        };
        // SAFETY: The semaphore create-info is complete and the device is live.
        let render_finished = unsafe {
            device
                .create_semaphore(&semaphore_info, None)
                .wrap_err("failed to create render-finished semaphore")?
        };
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        // SAFETY: The fence create-info is complete and the device is live.
        let fence = unsafe {
            device
                .create_fence(&fence_info, None)
                .wrap_err("failed to create GUI fence")?
        };
        Ok(Self {
            _entry: entry,
            instance,
            surface_loader,
            surface,
            physical_device,
            device,
            queue,
            swapchain_loader,
            swapchain,
            format,
            extent,
            images,
            initialized_images: vec![false; image_count],
            command_pool,
            command_buffer,
            fence,
            image_available,
            render_finished,
        })
    }

    fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        // SAFETY: The device is live and waiting makes swapchain destruction safe.
        unsafe { self.device.device_wait_idle() }
            .wrap_err("failed waiting for GUI device during resize")?;
        let old_swapchain = self.swapchain;
        let (swapchain, format, extent, images) = create_swapchain(
            window,
            &self.surface_loader,
            self.surface,
            &self.swapchain_loader,
            self.physical_device,
            old_swapchain,
        )?;
        // SAFETY: The old swapchain is no longer in use after device_wait_idle.
        unsafe { self.swapchain_loader.destroy_swapchain(old_swapchain, None) };
        self.swapchain = swapchain;
        self.format = format;
        self.extent = extent;
        self.images = images;
        self.initialized_images = vec![false; self.images.len()];
        Ok(())
    }

    fn draw(&mut self, state: &GuiState) -> Result<bool> {
        // SAFETY: The fence belongs to this live device and is used for this frame.
        unsafe { self.device.wait_for_fences(&[self.fence], true, u64::MAX) }
            .wrap_err("failed waiting for GUI frame fence")?;
        // SAFETY: The swapchain and image-available semaphore belong to the live device.
        let (image_index, suboptimal) = match unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            )
        } {
            Ok(result) => result,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Ok(true),
            Err(error) => return Err(error).wrap_err("failed to acquire GUI swapchain image"),
        };
        let image_index = usize::try_from(image_index).wrap_err("invalid swapchain image index")?;
        // SAFETY: The previous frame was waited above, so these frame resources
        // are not in flight.
        unsafe { self.device.reset_fences(&[self.fence]) }
            .wrap_err("failed to reset GUI frame fence")?;
        // SAFETY: The command buffer is allocated from a resettable command pool
        // and the previous submission has completed.
        unsafe {
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
        }
        .wrap_err("failed to reset GUI command buffer")?;
        let mut canvas = Canvas::new(self.extent.width, self.extent.height, self.format)?;
        canvas.render_ui(state);
        let (staging_buffer, staging_memory) = self.create_staging_buffer(&canvas.pixels)?;
        let command_result =
            self.record_copy(image_index, staging_buffer, canvas.width, canvas.height);
        if let Err(error) = command_result {
            // SAFETY: Recording failed before submission, so the staging resources
            // are not referenced by the device.
            unsafe { self.device.destroy_buffer(staging_buffer, None) };
            // SAFETY: The staging allocation is no longer bound to a live buffer.
            unsafe { self.device.free_memory(staging_memory, None) };
            return Err(error);
        }
        let wait_stages = [vk::PipelineStageFlags::TRANSFER];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(std::slice::from_ref(&self.image_available))
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(std::slice::from_ref(&self.command_buffer))
            .signal_semaphores(std::slice::from_ref(&self.render_finished));
        // SAFETY: The command buffer, semaphores, queue, and fence all belong to
        // this live device and the wait/signal chain is fully specified.
        unsafe {
            self.device
                .queue_submit(self.queue, &[submit_info], self.fence)
        }
        .wrap_err("failed to submit GUI frame")?;
        let present_image_index = image_index as u32;
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(std::slice::from_ref(&self.render_finished))
            .swapchains(std::slice::from_ref(&self.swapchain))
            .image_indices(std::slice::from_ref(&present_image_index));
        // SAFETY: The swapchain image was acquired above and the render-finished
        // semaphore is signaled by the submitted command buffer.
        let present_suboptimal = match unsafe {
            self.swapchain_loader
                .queue_present(self.queue, &present_info)
        } {
            Ok(value) => value,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => true,
            Err(error) => return Err(error).wrap_err("failed to present GUI frame"),
        };
        // SAFETY: Waiting for idle completes the copy before its staging resources
        // are released.
        unsafe { self.device.device_wait_idle() }
            .wrap_err("failed waiting for GUI frame completion")?;
        // SAFETY: The device is idle and no command references the staging buffer.
        unsafe { self.device.destroy_buffer(staging_buffer, None) };
        // SAFETY: The device is idle and the staging buffer has been destroyed.
        unsafe { self.device.free_memory(staging_memory, None) };
        self.initialized_images[image_index] = true;
        Ok(suboptimal || present_suboptimal)
    }

    fn record_copy(
        &mut self,
        image_index: usize,
        staging_buffer: vk::Buffer,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let old_layout = if self.initialized_images[image_index] {
            vk::ImageLayout::PRESENT_SRC_KHR
        } else {
            vk::ImageLayout::UNDEFINED
        };
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let to_transfer = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .image(self.images[image_index])
            .subresource_range(subresource_range);
        let to_present = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::empty())
            .image(self.images[image_index])
            .subresource_range(subresource_range);
        let copy_region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        let begin_info = vk::CommandBufferBeginInfo::default();
        // SAFETY: The command buffer was reset above and all referenced handles
        // belong to this live device.
        unsafe {
            self.device
                .begin_command_buffer(self.command_buffer, &begin_info)
        }
        .wrap_err("failed to begin GUI command buffer")?;
        // SAFETY: The barrier transitions the acquired swapchain image from its
        // tracked layout to the transfer destination layout.
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_transfer],
            );
        };
        // SAFETY: The staging buffer contains exactly width*height RGBA/BGRA
        // bytes and the destination image has the matching swapchain extent.
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                self.command_buffer,
                staging_buffer,
                self.images[image_index],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
            );
        };
        // SAFETY: The final barrier makes the copied image available for present.
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_present],
            );
        };
        // SAFETY: All commands in this recording use live handles and valid
        // command-buffer state.
        unsafe { self.device.end_command_buffer(self.command_buffer) }
            .wrap_err("failed to record GUI command buffer")?;
        Ok(())
    }

    fn create_staging_buffer(&self, pixels: &[u8]) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let size =
            vk::DeviceSize::try_from(pixels.len()).wrap_err("GUI pixel buffer is too large")?;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: The buffer create-info is complete and the device is live.
        let buffer = unsafe {
            self.device
                .create_buffer(&buffer_info, None)
                .wrap_err("failed to create GUI staging buffer")?
        };
        // SAFETY: The buffer was created on this live device.
        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory_type = find_memory_type(
            &self.instance,
            self.physical_device,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
        // SAFETY: The allocation request uses requirements returned for the live
        // staging buffer and a compatible host-visible memory type.
        let memory = unsafe {
            self.device
                .allocate_memory(&allocate_info, None)
                .wrap_err("failed to allocate GUI staging memory")?
        };
        // SAFETY: The allocation was selected for this buffer's memory
        // requirements and has not been bound elsewhere.
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }
            .wrap_err("failed to bind GUI staging memory")?;
        // SAFETY: The host-visible allocation is large enough for the pixel slice
        // and remains mapped only for the copy below.
        let mapped = unsafe {
            self.device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
        }
        .wrap_err("failed to map GUI staging memory")?;
        // SAFETY: `mapped` points to at least pixels.len() writable bytes, and the
        // source and destination ranges do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), mapped.cast::<u8>(), pixels.len());
        };
        // SAFETY: The mapped range belongs to this live allocation and the copy is
        // complete before it is unmapped.
        unsafe { self.device.unmap_memory(memory) };
        Ok((buffer, memory))
    }
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        // SAFETY: Drop is the final owner of the live device resources, so idle
        // guarantees no submitted work references the objects being destroyed.
        unsafe {
            let _ = self.device.device_wait_idle();
        }
        // SAFETY: The device is idle and owns this semaphore.
        unsafe { self.device.destroy_semaphore(self.render_finished, None) };
        // SAFETY: The device is idle and owns this semaphore.
        unsafe { self.device.destroy_semaphore(self.image_available, None) };
        // SAFETY: The device is idle and owns this fence.
        unsafe { self.device.destroy_fence(self.fence, None) };
        // SAFETY: The device is idle and owns this command pool.
        unsafe { self.device.destroy_command_pool(self.command_pool, None) };
        // SAFETY: The device is idle and owns this swapchain.
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        };
        // SAFETY: All child device resources have been destroyed first.
        unsafe { self.device.destroy_device(None) };
        // SAFETY: The logical device is gone and the instance still owns this surface.
        unsafe { self.surface_loader.destroy_surface(self.surface, None) };
        // SAFETY: The instance is the final Vulkan owner and no child handles remain.
        unsafe { self.instance.destroy_instance(None) };
    }
}

fn pick_physical_device(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
    // SAFETY: The Vulkan instance is live for the duration of this query.
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .wrap_err("failed to enumerate Vulkan physical devices")?
    };
    for physical_device in physical_devices {
        // SAFETY: The physical device was returned by the live instance.
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        for (index, family) in queue_families.iter().enumerate() {
            let index = u32::try_from(index).wrap_err("queue family index overflowed")?;
            // SAFETY: The surface and physical device belong to the live instance.
            let present_support = unsafe {
                surface_loader
                    .get_physical_device_surface_support(physical_device, index, surface)
                    .wrap_err("failed to query surface support")?
            };
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) && present_support {
                return Ok((physical_device, index));
            }
        }
    }
    bail!("no Vulkan physical device supports graphics presentation")
}

fn create_swapchain(
    window: &Window,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    swapchain_loader: &ash::khr::swapchain::Device,
    physical_device: vk::PhysicalDevice,
    old_swapchain: vk::SwapchainKHR,
) -> Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D, Vec<vk::Image>)> {
    // SAFETY: The surface and physical device belong to the live instance.
    let capabilities = unsafe {
        surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface)
            .wrap_err("failed to query GUI surface capabilities")?
    };
    if !capabilities
        .supported_usage_flags
        .contains(vk::ImageUsageFlags::TRANSFER_DST)
    {
        bail!("GUI surface does not support transfer-to-present images");
    }
    // SAFETY: The surface and physical device belong to the live instance.
    let formats = unsafe {
        surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .wrap_err("failed to query GUI surface formats")?
    };
    let format = formats
        .iter()
        .copied()
        .find(|format| {
            matches!(
                format.format,
                vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB
            )
        })
        .or_else(|| formats.first().copied())
        .ok_or_else(|| eyre::eyre!("GUI surface reported no supported formats"))?;
    let window_size = window.inner_size();
    let extent = if capabilities.current_extent.width == u32::MAX {
        vk::Extent2D {
            width: window_size.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: window_size.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    } else {
        capabilities.current_extent
    };
    let mut image_count = capabilities.min_image_count.saturating_add(1);
    if capabilities.max_image_count != 0 {
        image_count = image_count.min(capabilities.max_image_count);
    }
    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::TRANSFER_DST)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(capabilities.current_transform)
        .composite_alpha(choose_composite_alpha(
            capabilities.supported_composite_alpha,
        ))
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true)
        .old_swapchain(old_swapchain);
    // SAFETY: The create-info references live surface/device state and valid
    // presentation capabilities.
    let swapchain = unsafe {
        swapchain_loader
            .create_swapchain(&create_info, None)
            .wrap_err("failed to create GUI swapchain")?
    };
    // SAFETY: The swapchain was created successfully on the live device.
    let images = unsafe {
        swapchain_loader
            .get_swapchain_images(swapchain)
            .wrap_err("failed to enumerate GUI swapchain images")?
    };
    Ok((swapchain, format.format, extent, images))
}

fn choose_composite_alpha(supported: vk::CompositeAlphaFlagsKHR) -> vk::CompositeAlphaFlagsKHR {
    [
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::INHERIT,
    ]
    .into_iter()
    .find(|candidate| supported.contains(*candidate))
    .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE)
}

fn find_memory_type(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    type_filter: u32,
    required: vk::MemoryPropertyFlags,
) -> Result<u32> {
    // SAFETY: The physical device was returned by the live instance.
    let properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    for index in 0..properties.memory_type_count {
        let supported = type_filter & (1 << index) != 0;
        let memory_type = properties.memory_types[index as usize];
        if supported && memory_type.property_flags.contains(required) {
            return Ok(index);
        }
    }
    bail!("no Vulkan memory type satisfies GUI staging requirements")
}

#[cfg(test)]
mod tests {
    use super::ClipId;
    use super::GuiAction;
    use super::GuiState;
    use super::INITIAL_HEIGHT;
    use super::INITIAL_WIDTH;
    use super::glyph;
    use winit::dpi::PhysicalPosition;
    use winit::dpi::PhysicalSize;

    #[test]
    fn reference_labels_have_bitmap_glyphs() {
        for character in "TEAMY-TRANSCRIBER MICROPHONE: WOER SAVE DIR: ~/DOWNLOADS TESTING, 1, 2"
            .chars()
            .filter(|character| !character.is_ascii_whitespace())
        {
            assert!(
                glyph(character).iter().any(|row| *row != 0),
                "missing glyph for {character:?}"
            );
        }
    }

    #[test]
    fn microphone_hit_toggles_recording_state() {
        let size = PhysicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT);
        let mut state = GuiState {
            cursor: PhysicalPosition::new(192.0, 288.8),
            ..GuiState::default()
        };

        assert_eq!(state.click(size), Some(GuiAction::ToggleRecording));
        assert_eq!(state.click(size), Some(GuiAction::ToggleRecording));
    }

    #[test]
    fn clip_navigation_wraps_and_chunk_label_is_explicit() {
        let first = ClipId::new();
        let second = ClipId::new();
        let third = ClipId::new();
        let mut state = GuiState {
            clip_ids: vec![first, second, third],
            selected_clip_id: Some(first),
            ..GuiState::default()
        };

        assert_eq!(state.cycled_clip_id(-1), Some(third));
        assert_eq!(state.cycled_clip_id(1), Some(second));
        assert_eq!(state.clip_label(), "CLIP 1/3");
        assert_eq!(state.chunk_duration_label(), "FULL");
        state.chunk_duration_ms = Some(30_000);
        assert_eq!(state.chunk_duration_label(), "30S");
    }
}
