# Local WhisperX worker

The Rust application launches whisperx_worker.py as a short-lived JSONL
process. The worker reads one request from stdin and emits one response on
stdout. Diagnostics belong on stderr.

The worker calls WhisperX with local_files_only=True. It does not download
models, contact a CDN, or write to the model directory. Model acquisition and
distribution are intentionally deferred.

The expected environment is the checked-out WhisperX project or an installed
WhisperX package with its Python dependencies available to the configured
Python executable. The Rust side supplies the model directory, model name,
device, compute type, and normalized WAV path.

This worker is not yet the complete production daemon: it is the narrow
single-request backend slice. Streaming progress, a persistent process, model
pooling, alignment, and cancellation remain later work items.
