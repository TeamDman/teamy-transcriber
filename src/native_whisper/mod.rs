#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::default_trait_access,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    clippy::map_err_ignore,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::needless_pass_by_value,
    clippy::single_range_in_vec_init,
    clippy::similar_names,
    clippy::struct_field_names,
    clippy::trivially_copy_pass_by_ref,
    reason = "The native Whisper implementation preserves the verified Burn model and frontend contracts; these localized numerical casts, Burn serialization derives, and tensor helper signatures are audited compatibility exceptions."
)]

pub mod frontend;
pub mod model;
pub mod prepare;
#[cfg(feature = "tch-native")]
pub mod tch;
#[cfg(feature = "tch-native")]
pub mod tch_safetensors;
pub mod whisper;

// MSVC can discard the otherwise-unused `torch_cuda.dll` import from the
// LibTorch link. Keep the same one-symbol CUDA anchor used by teamy-tts so
// LibTorch initializes its CUDA backend when the executable starts.
#[cfg(all(windows, teamy_transcriber_cuda_link))]
unsafe extern "C" {
    fn teamy_transcriber_force_torch_cuda();
}

#[cfg(all(windows, teamy_transcriber_cuda_link))]
#[used]
static FORCE_TORCH_CUDA_LINK: unsafe extern "C" fn() = teamy_transcriber_force_torch_cuda;
