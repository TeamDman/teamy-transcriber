//! Explicit, hidden native-window verification. Drives real GUI input/reducers,
//! rendering, worker messages, persistence and close-during-transcription.
use super::*;
use crate::media::{MediaAdapter, WavMediaAdapter};
use eyre::ensure;
use winit::platform::windows::EventLoopBuilderExtWindows;

#[derive(Facet)]
struct Run {
    total_ms: f64,
    chunks: usize,
    text: String,
}

#[derive(Facet)]
struct Receipt {
    scope: String,
    revision: String,
    worktree_status: String,
    rendered_frames: usize,
    resize_events: usize,
    idle_callbacks: usize,
    idle_ms: f64,
    runs: Vec<Run>,
    cancelled_chunks: usize,
    cancelled_total_chunks: usize,
    close_ms: f64,
    session_released: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Idle,
    Import,
    Preparing,
    Transcribing,
    CloseWhenPartial,
    Closing,
}

struct Harness {
    app: GuiApplication,
    wav: PathBuf,
    stage: Stage,
    start: Instant,
    request_start: Instant,
    close_start: Option<Instant>,
    runs: Vec<Run>,
    frames: usize,
    resize_events: usize,
    idle_callbacks: usize,
    idle_start: Option<Instant>,
    idle_ready: Option<Sender<()>>,
    idle_ms: f64,
    error: Option<String>,
}

impl Harness {
    fn begin_idle_probe(&mut self) {
        if self.stage == Stage::Idle && self.idle_start.is_none() {
            self.idle_start = Some(Instant::now());
            self.idle_callbacks = 0;
            if let Some(ready) = self.idle_ready.take() {
                let _ = ready.send(());
            }
        }
    }

    fn resize_window(&self, width: u32, height: u32) {
        if let Some(window) = self.app.window.as_ref() {
            let _ = window.request_inner_size(PhysicalSize::new(width, height));
        }
    }

    fn finish_idle_probe(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        ensure!(
            self.idle_callbacks < 100,
            "idle GUI is busy-polling: {} callbacks",
            self.idle_callbacks
        );
        self.idle_ms = self
            .idle_start
            .map_or(0., |start| start.elapsed().as_secs_f64() * 1000.);
        self.resize_window(INITIAL_WIDTH, INITIAL_HEIGHT);
        self.stage = Stage::Import;
        self.drive(event_loop)
    }

    fn finish_request(&mut self) -> Result<()> {
        let id = self
            .app
            .state
            .recording_id
            .ok_or_else(|| eyre::eyre!("recording missing"))?;
        let recording = self.app.store.load_recording(id)?;
        ensure!(
            recording.transcripts.len() == recording.clips.len() && !recording.clips.is_empty(),
            "GUI finished without complete persisted output"
        );
        let selected = recording
            .transcripts
            .iter()
            .find(|text| Some(text.clip_id) == self.app.state.selected_clip_id)
            .ok_or_else(|| eyre::eyre!("GUI selected transcript missing"))?;
        ensure!(
            self.app.state.transcript.trim() == selected.text.trim(),
            "GUI did not project the committed transcript"
        );
        let text = recording
            .transcripts
            .iter()
            .map(|t| t.text.trim())
            .collect::<Vec<_>>()
            .join(" ");
        ensure!(!text.is_empty(), "empty GUI transcript");
        if let Some(first) = self.runs.first() {
            ensure!(text == first.text, "resident GUI output changed");
        }
        self.runs.push(Run {
            total_ms: self.request_start.elapsed().as_secs_f64() * 1000.,
            chunks: recording.clips.len(),
            text,
        });
        Ok(())
    }

    fn drive(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        ensure!(
            self.start.elapsed() < Duration::from_mins(2),
            "GUI verification timed out"
        );
        ensure!(
            !self.app.state.status_line.starts_with("ERROR:"),
            "{}",
            self.app.state.status_line
        );
        if self.frames == 0 {
            return Ok(());
        }
        self.begin_idle_probe();
        let window_id = self
            .app
            .window
            .as_ref()
            .ok_or_else(|| eyre::eyre!("GUI window missing"))?
            .id();
        let idle = matches!(self.app.state.operation, GuiOperation::Idle);
        match self.stage {
            Stage::Idle
                if self
                    .idle_start
                    .is_some_and(|start| start.elapsed() >= Duration::from_millis(500)) =>
            {
                self.finish_idle_probe(event_loop)?;
            }
            Stage::Import => {
                self.request_start = Instant::now();
                self.app.window_event(
                    event_loop,
                    window_id,
                    WindowEvent::DroppedFile(self.wav.clone()),
                );
                self.stage = Stage::Preparing;
            }
            Stage::Preparing if idle => {
                ensure!(
                    self.app.state.prepared,
                    "GUI did not prepare the imported audio"
                );
                // Use the same hit-testing and mouse-release handler as TRANSCRIBE.
                let rect = GuiLayout::new(self.app.window_size()).transcribe;
                self.app.state.cursor = PhysicalPosition::new(
                    f64::from(rect.left.midpoint(rect.right)),
                    f64::from(rect.top.midpoint(rect.bottom)),
                );
                self.app.window_event(
                    event_loop,
                    window_id,
                    WindowEvent::MouseInput {
                        device_id: winit::event::DeviceId::dummy(),
                        state: ElementState::Released,
                        button: MouseButton::Left,
                    },
                );
                ensure!(
                    matches!(self.app.state.operation, GuiOperation::Transcribing),
                    "GUI transcribe action did not start: {}",
                    self.app.state.status_line
                );
                self.stage = if self.runs.len() < 2 {
                    Stage::Transcribing
                } else {
                    Stage::CloseWhenPartial
                };
            }
            Stage::Transcribing if idle => {
                self.finish_request()?;
                self.stage = Stage::Import;
                self.drive(event_loop)?;
            }
            Stage::CloseWhenPartial => {
                let id = self
                    .app
                    .state
                    .recording_id
                    .ok_or_else(|| eyre::eyre!("recording missing"))?;
                let recording = self.app.store.load_recording(id)?;
                ensure!(
                    !idle,
                    "recording finished before close-during-transcription could be exercised; use a longer WAV"
                );
                if !recording.transcripts.is_empty() {
                    self.close_start = Some(Instant::now());
                    self.app
                        .window_event(event_loop, window_id, WindowEvent::CloseRequested);
                    ensure!(
                        self.app.close_requested,
                        "GUI did not defer closing until inference completes"
                    );
                    self.stage = Stage::Closing;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

impl ApplicationHandler for Harness {
    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: ()) {
        // Assert before dispatch so the deadline wake cannot accidentally make
        // delayed shutdown look like a passing cancellation check.
        if self.start.elapsed() >= Duration::from_mins(2) {
            self.error = Some("GUI verification deadline reached before clean completion".into());
        }
        self.app.user_event(event_loop, event);
    }
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.app.resumed(event_loop);
        self.resize_window(1_180, 740);
        if self
            .app
            .window
            .as_ref()
            .is_some_and(|window| window.is_visible() != Some(false))
        {
            self.error = Some("verification window unexpectedly became visible".into());
            event_loop.exit();
        }
    }
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let redraw = matches!(event, WindowEvent::RedrawRequested);
        if matches!(event, WindowEvent::Resized(_)) {
            self.resize_events += 1;
        }
        self.app.window_event(event_loop, window_id, event);
        if redraw && !event_loop.exiting() {
            self.frames += 1;
        }
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.stage == Stage::Idle {
            self.idle_callbacks += 1;
        }
        self.app.about_to_wait(event_loop);
        // Windows does not deliver paint events for an invisible HWND. Render
        // only when the application requests it, through its real draw handler.
        if !event_loop.exiting()
            && self.app.redraw_pending
            && let Some(window) = self.app.window.as_ref()
        {
            let id = window.id();
            self.window_event(event_loop, id, WindowEvent::RedrawRequested);
        }
        if !event_loop.exiting()
            && let Err(error) = self.drive(event_loop)
        {
            self.error = Some(format!("{error:#}"));
            self.app.cancel_active_operation();
            self.app.close_requested = true;
            // Keep dispatching until the current worker reports completion.
            self.stage = Stage::Closing;
            if matches!(self.app.state.operation, GuiOperation::Idle) {
                event_loop.exit();
            }
        }
    }
}

#[test]
#[ignore = "requires Windows desktop/Vulkan/CUDA, TEST_MODEL, TEST_WAV and GUI_TEST_HOME"]
fn hidden_gui_reuses_model_and_closes_during_transcription() -> Result<()> {
    let model = PathBuf::from(std::env::var("TEAMY_TRANSCRIBER_TEST_MODEL")?);
    let wav = PathBuf::from(std::env::var("TEAMY_TRANSCRIBER_TEST_WAV")?);
    ensure!(
        WavMediaAdapter.inspect(&wav)?.duration_us > 60_000_000,
        "GUI cancellation verification needs a WAV longer than sixty seconds"
    );
    let root = PathBuf::from(std::env::var("TEAMY_TRANSCRIBER_GUI_TEST_HOME")?);
    ensure!(!root.exists(), "choose a fresh GUI test home");
    std::fs::create_dir_all(&root)?;
    let home = AppHome(root.clone());
    save_preferences(
        &home,
        &GuiPreferences {
            model_dir: Some(model.display().to_string()),
            global_hotkey_enabled: Some(false),
            ..GuiPreferences::default()
        },
    )?;
    let event_loop = EventLoop::builder().with_any_thread(true).build()?;
    let mut app = GuiApplication::from_home(home, false, event_loop.create_proxy())?;
    ensure!(app.state.model_ready, "{}", app.state.model_status);
    app.window_attributes = app
        .window_attributes
        .clone()
        .with_visible(false)
        .with_active(false);
    let weak_session = Arc::downgrade(&app.transcription_session);
    let mut harness = Harness {
        app,
        wav,
        stage: Stage::Idle,
        start: Instant::now(),
        request_start: Instant::now(),
        close_start: None,
        runs: Vec::new(),
        frames: 0,
        resize_events: 0,
        idle_callbacks: 0,
        idle_start: None,
        idle_ready: None,
        idle_ms: 0.,
        error: None,
    };
    event_loop.set_control_flow(ControlFlow::Wait);
    run_with_deadline(event_loop, &mut harness)?;
    ensure!(
        harness.error.is_none(),
        "GUI verification failed: {:?}",
        harness.error
    );
    ensure!(
        harness.frames > 0 && harness.runs.len() == 2 && harness.stage == Stage::Closing,
        "GUI exited before verification completed"
    );
    ensure!(
        harness.resize_events >= 2,
        "GUI did not process the requested native-window resizes"
    );
    ensure!(
        matches!(harness.app.state.operation, GuiOperation::Idle),
        "GUI exited with active inference"
    );
    let id = harness
        .app
        .state
        .recording_id
        .ok_or_else(|| eyre::eyre!("recording missing"))?;
    let cancelled = harness.app.store.load_recording(id)?;
    ensure!(
        !cancelled.transcripts.is_empty() && cancelled.transcripts.len() < cancelled.clips.len(),
        "closing did not preserve a partial transcript"
    );
    let close_start = harness
        .close_start
        .ok_or_else(|| eyre::eyre!("close request missing"))?;
    ensure!(
        close_start.elapsed() < Duration::from_secs(30),
        "GUI waited for an unrelated event to close"
    );
    let mut receipt = Receipt {
        scope: "Hidden native Winit/Vulkan GUI: explicit redraws for invisible HWND, two fresh imports/transcriptions, close during third; no tray/hotkey/capture/dialogs or visual appearance claim".into(),
        revision: env!("GIT_REVISION").into(), worktree_status: env!("GIT_WORKTREE_STATUS").into(),
        rendered_frames: harness.frames, resize_events: harness.resize_events, idle_callbacks: harness.idle_callbacks, idle_ms: harness.idle_ms, runs: std::mem::take(&mut harness.runs),
        cancelled_chunks: cancelled.transcripts.len(), cancelled_total_chunks: cancelled.clips.len(),
        close_ms: 0., session_released: false,
    };
    drop(harness);
    receipt.close_ms = close_start.elapsed().as_secs_f64() * 1000.;
    receipt.session_released = weak_session.upgrade().is_none();
    ensure!(
        receipt.session_released,
        "GUI shutdown retained an inference session owner"
    );
    std::fs::write(
        root.join("gui-receipt.json"),
        facet_json::to_string_pretty(&receipt)?,
    )?;
    Ok(())
}

fn run_with_deadline(event_loop: EventLoop<()>, harness: &mut Harness) -> Result<()> {
    // One wake ends the idle check. No periodic test polling is allowed to mask
    // a missing production worker wake. The second wake only enforces timeout.
    let proxy = event_loop.create_proxy();
    let (stop, stopped) = channel::<()>();
    let (idle_ready, ready) = channel::<()>();
    harness.idle_ready = Some(idle_ready);
    let deadline = std::thread::spawn(move || {
        match ready.recv_timeout(Duration::from_mins(2)) {
            Ok(()) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                let _ = proxy.send_event(());
                return;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
        }
        if matches!(
            stopped.recv_timeout(Duration::from_millis(500)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ) {
            let _ = proxy.send_event(());
            if matches!(
                stopped.recv_timeout(Duration::from_mins(2)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ) {
                let _ = proxy.send_event(());
            }
        }
    });
    let result = event_loop.run_app(harness);
    harness.idle_ready.take();
    let _ = stop.send(());
    deadline
        .join()
        .map_err(|_panic| eyre::eyre!("GUI deadline helper panicked"))?;
    result?;
    Ok(())
}
