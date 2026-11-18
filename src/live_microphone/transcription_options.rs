use crate::microphone_chunk_milliseconds::MicrophoneChunkMilliseconds;
use crate::native_whisper::cuda::CudaWhisperOptions;
use std::time::Duration;

#[derive(Clone, Copy)]
pub(crate) struct LiveTranscriptionOptions {
    pub(crate) cuda: CudaWhisperOptions,
    pub(crate) chunk_ms: MicrophoneChunkMilliseconds,
    pub(crate) keep_recording: bool,
    pub(crate) inactivity: Option<Duration>,
}
