//! Keep the CUDA engine on its creating thread, including destruction.
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

const BATCH_SIZE_ENV: &str = "TEAMY_TRANSCRIBER_CUDA_BATCH_SIZE";

#[derive(Debug)]
enum Response {
    Completed {
        index: usize,
        text: String,
        acknowledge: mpsc::SyncSender<bool>,
    },
    Finished(Result<bool, String>),
}

#[derive(Debug)]
struct Request {
    windows: Vec<Vec<f32>>,
    max_tokens: usize,
    stop: Arc<AtomicBool>,
    reply: mpsc::Sender<Response>,
}

#[derive(Debug)]
struct Worker {
    sender: Option<mpsc::SyncSender<Request>>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Closing the sole sender drains accepted work before destroying the
        // engine on its owner thread. No inference helper outlives its backend.
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[derive(Clone, Debug)]
pub struct CudaWhisperRuntime(Arc<Worker>);

impl CudaWhisperRuntime {
    /// Create the resident model on its dedicated inference thread.
    ///
    /// # Errors
    /// Returns an error for invalid model assets, unavailable CUDA, or a failed worker.
    pub fn load(path: PathBuf) -> Result<Self, String> {
        let (sender, receiver) = mpsc::sync_channel::<Request>(1);
        let (ready_tx, ready_rx) = mpsc::channel();
        let device = std::env::var("TEAMY_TRANSCRIBER_CUDA_DEVICE").map_or(Ok(0), |value| {
            value.parse::<i32>().map_err(|error| error.to_string())
        })?;
        let tf32 = match std::env::var("TEAMY_TRANSCRIBER_CUDA_MATH") {
            Err(std::env::VarError::NotPresent) => true,
            Ok(value) if value == "tf32" => true,
            Ok(value) if value == "fp32" => false,
            _ => return Err("TEAMY_TRANSCRIBER_CUDA_MATH must be tf32 or fp32".to_string()),
        };
        let batch_size = configured_batch_size()?;
        let generation = path.join("generation_config.json");
        let policy = if generation.exists() {
            Some(
                serde_json::from_slice::<teamy_whisper_native::decoding::GreedySuppression>(
                    &std::fs::read(&generation).map_err(|e| e.to_string())?,
                )
                .map_err(|e| format!("invalid generation_config.json: {e}"))?,
            )
        } else {
            None
        };
        let thread = std::thread::Builder::new()
            .name("whisper-cuda-inference".into())
            .spawn(move || {
                let initialized = (|| {
                    let mut engine = teamy_whisper_native::Engine::load(&path, device, tf32)
                        .map_err(|error| error.to_string())?;
                    if let Some(policy) = policy {
                        engine
                            .configure_greedy(&policy)
                            .map_err(|e| e.to_string())?;
                    }
                    let frontend =
                        teamy_whisper_native::frontend::Frontend::new(engine.dims().audio.n_mels)
                            .map_err(|error| error.to_string())?;
                    Ok::<_, String>((engine, frontend))
                })();
                let (mut engine, mut frontend) = match initialized {
                    Ok(runtime) => {
                        if ready_tx.send(Ok(())).is_err() {
                            return;
                        }
                        runtime
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                for request in receiver {
                    let result = run_request(&mut engine, &mut frontend, &request, batch_size);
                    let _ = request.reply.send(Response::Finished(result));
                }
            })
            .map_err(|error| error.to_string())?;
        let worker = Worker {
            sender: Some(sender),
            thread: Some(thread),
        };
        ready_rx.recv().map_err(|error| error.to_string())??;
        Ok(Self(Arc::new(worker)))
    }

    /// Transcribe one complete, normalized window on the resident worker.
    ///
    /// # Errors
    /// Returns an error for invalid audio or failed inference/worker communication.
    pub fn transcribe(&self, samples: Vec<f32>, max_tokens: usize) -> Result<String, String> {
        let mut text = None;
        self.transcribe_batch(
            vec![samples],
            max_tokens,
            &mut || false,
            &mut |_, result| {
                text = Some(result);
                Ok(())
            },
        )?;
        text.ok_or_else(|| "CUDA worker returned no transcript".to_string())
    }

    /// Transcribe bounded windows, acknowledging ordered completions on the
    /// calling thread before the worker may deliver another. Returns whether
    /// cancellation discarded work. The resident model survives cancellation.
    ///
    /// # Errors
    /// Returns inference, callback or worker communication errors. Callback
    /// failures cancel and drain the request before returning.
    pub fn transcribe_batch(
        &self,
        windows: Vec<Vec<f32>>,
        max_tokens: usize,
        should_stop: &mut dyn FnMut() -> bool,
        on_complete: &mut dyn FnMut(usize, String) -> Result<(), String>,
    ) -> Result<bool, String> {
        if !(1..=teamy_whisper_native::MAX_BATCH_SIZE).contains(&windows.len()) {
            return Err("CUDA requests must contain one to eight windows".into());
        }
        let (reply, result) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(should_stop()));
        let _cancel_on_drop = CancelOnDrop(Arc::clone(&stop));
        self.0
            .sender
            .as_ref()
            .ok_or_else(|| "CUDA worker stopped".to_string())?
            .send(Request {
                windows,
                max_tokens,
                stop: Arc::clone(&stop),
                reply,
            })
            .map_err(|error| error.to_string())?;
        let mut callback_error = None;
        loop {
            if should_stop() {
                stop.store(true, Ordering::Relaxed);
            }
            match result.recv_timeout(Duration::from_millis(10)) {
                Ok(Response::Completed {
                    index,
                    text,
                    acknowledge,
                }) => {
                    if !stop.load(Ordering::Relaxed)
                        && let Err(error) = on_complete(index, text)
                    {
                        callback_error = Some(error);
                        stop.store(true, Ordering::Relaxed);
                    }
                    if should_stop() {
                        stop.store(true, Ordering::Relaxed);
                    }
                    let _ = acknowledge.send(!stop.load(Ordering::Relaxed));
                }
                Ok(Response::Finished(outcome)) => {
                    if let Some(error) = callback_error {
                        return Err(error);
                    }
                    return outcome.map(|cancelled| cancelled || stop.load(Ordering::Relaxed));
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(error) => return Err(error.to_string()),
            }
        }
    }
}

struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

fn configured_batch_size() -> Result<usize, String> {
    let size = match std::env::var(BATCH_SIZE_ENV) {
        Err(std::env::VarError::NotPresent) => teamy_whisper_native::MAX_BATCH_SIZE,
        Ok(value) => value
            .parse()
            .map_err(|error| format!("{BATCH_SIZE_ENV} must be 1..=8: {error}"))?,
        Err(error) => return Err(error.to_string()),
    };
    if !(1..=teamy_whisper_native::MAX_BATCH_SIZE).contains(&size) {
        return Err(format!("{BATCH_SIZE_ENV} must be 1..=8"));
    }
    Ok(size)
}

fn run_request(
    engine: &mut teamy_whisper_native::Engine,
    frontend: &mut teamy_whisper_native::frontend::Frontend,
    request: &Request,
    batch_size: usize,
) -> Result<bool, String> {
    let mut should_stop = || request.stop.load(Ordering::Relaxed);
    let mut offset = 0;
    while offset < request.windows.len() {
        if should_stop() {
            return Ok(true);
        }
        let count = engine
            .fitting_batch_size(batch_size.min(request.windows.len() - offset))
            .map_err(|error| error.to_string())?;
        let mut mels = Vec::with_capacity(count);
        for window in &request.windows[offset..offset + count] {
            if should_stop() {
                return Ok(true);
            }
            mels.push(
                frontend
                    .compute(window)
                    .map_err(|error| error.to_string())?,
            );
        }
        let deliver = |index, text| {
            let (acknowledge, acknowledged) = mpsc::sync_channel(0);
            if request
                .reply
                .send(Response::Completed {
                    index,
                    text,
                    acknowledge,
                })
                .is_err()
                || acknowledged.recv() != Ok(true)
            {
                request.stop.store(true, Ordering::Relaxed);
            }
        };
        if count == 1 {
            let Some(result) = engine
                .transcribe_mel_interruptible(&mels[0], request.max_tokens, &mut should_stop)
                .map_err(|error| error.to_string())?
            else {
                return Ok(true);
            };
            deliver(offset, result.text);
        } else {
            let inputs: Vec<_> = mels.iter().map(Vec::as_slice).collect();
            engine
                .transcribe_mel_batch_with(
                    &inputs,
                    request.max_tokens,
                    &mut should_stop,
                    &mut |index, result| {
                        deliver(offset + index, result.text.clone());
                        Ok(())
                    },
                )
                .map_err(|error| error.to_string())?;
        }
        offset += count;
    }
    Ok(should_stop())
}
