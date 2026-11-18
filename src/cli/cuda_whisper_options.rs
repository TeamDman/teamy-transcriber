use crate::native_whisper::cuda::CudaWhisperOptions;
use eyre::Result;
use std::env;
use teamy_whisper_native::MAX_BATCH_SIZE;

pub(crate) fn resolve_cuda_whisper_options() -> Result<CudaWhisperOptions> {
    let defaults = CudaWhisperOptions::default();
    let device = match env::var("TEAMY_TRANSCRIBER_CUDA_DEVICE") {
        Err(env::VarError::NotPresent) => defaults.device,
        Ok(value) => value
            .parse()
            .map_err(|error| eyre::eyre!("invalid TEAMY_TRANSCRIBER_CUDA_DEVICE: {error}"))?,
        Err(error) => return Err(error.into()),
    };
    let tf32 = match env::var("TEAMY_TRANSCRIBER_CUDA_MATH") {
        Err(env::VarError::NotPresent) => defaults.tf32,
        Ok(value) if value == "tf32" => true,
        Ok(value) if value == "fp32" => false,
        _ => eyre::bail!("TEAMY_TRANSCRIBER_CUDA_MATH must be tf32 or fp32"),
    };
    let batch_size = match env::var("TEAMY_TRANSCRIBER_CUDA_BATCH_SIZE") {
        Err(env::VarError::NotPresent) => defaults.batch_size,
        Ok(value) => value.parse().map_err(|error| {
            eyre::eyre!("TEAMY_TRANSCRIBER_CUDA_BATCH_SIZE must be 1..={MAX_BATCH_SIZE}: {error}")
        })?,
        Err(error) => return Err(error.into()),
    };
    eyre::ensure!(
        (1..=MAX_BATCH_SIZE).contains(&batch_size),
        "TEAMY_TRANSCRIBER_CUDA_BATCH_SIZE must be 1..={MAX_BATCH_SIZE}"
    );
    Ok(CudaWhisperOptions {
        device,
        tf32,
        batch_size,
    })
}
