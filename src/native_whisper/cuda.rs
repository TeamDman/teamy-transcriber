//! Keep the CUDA engine on its creating thread, including destruction.
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread::JoinHandle;

#[derive(Debug)]
struct Request {
    samples: Vec<f32>,
    max_tokens: usize,
    reply: mpsc::Sender<Result<String, String>>,
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
        let thread = std::thread::Builder::new()
            .name("whisper-cuda-inference".into())
            .spawn(move || {
                let initialized = (|| {
                    let engine = teamy_whisper_native::Engine::load(&path, device, tf32)
                        .map_err(|error| error.to_string())?;
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
                    let result = frontend
                        .compute(&request.samples)
                        .and_then(|mel| engine.transcribe_mel(&mel, request.max_tokens))
                        .map(|result| result.text)
                        .map_err(|error| error.to_string());
                    let _ = request.reply.send(result);
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
        let (reply, result) = mpsc::channel();
        self.0
            .sender
            .as_ref()
            .ok_or_else(|| "CUDA worker stopped".to_string())?
            .send(Request {
                samples,
                max_tokens,
                reply,
            })
            .map_err(|error| error.to_string())?;
        result.recv().map_err(|error| error.to_string())?
    }
}

/// Optional native builds select CUDA unless the caller explicitly requests
/// the retained tch backend (including its established negative-device CPU setting).
#[must_use]
pub fn selected() -> bool {
    std::env::var("TEAMY_TRANSCRIBER_BACKEND").is_ok_and(|value| value == "cuda")
        || (std::env::var("TEAMY_TRANSCRIBER_BACKEND").is_err()
            && std::env::var(crate::paths::TORCH_DEVICE_ENV_VAR)
                .ok()
                .and_then(|value| value.parse::<i32>().ok())
                .is_none_or(|value| value >= 0))
}
