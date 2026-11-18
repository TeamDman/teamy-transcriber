use std::env;
use std::path::PathBuf;

fn main() {
    cargo_instruction(format_args!("cargo:rerun-if-changed=kernels/phones.cuh"));
    cargo_instruction(format_args!("cargo:rerun-if-changed=kernels/whisper.cu"));
    cargo_instruction(format_args!("cargo:rerun-if-env-changed=CUDA_PATH"));
    cargo_instruction(format_args!("cargo:rerun-if-env-changed=TEAMY_WHISPER_CUDA_ARCH"));
    let cuda = PathBuf::from(env::var_os("CUDA_PATH").expect("CUDA_PATH must name a CUDA toolkit"));
    let arch = env::var("TEAMY_WHISPER_CUDA_ARCH").unwrap_or_else(|_| "89".into());
    assert!(
        !arch.is_empty() && arch.chars().all(|c| c.is_ascii_digit()),
        "invalid CUDA architecture"
    );
    cc::Build::new()
        .cuda(true)
        .cudart("shared")
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-lineinfo")
        .flag(format!(
            "-gencode=arch=compute_{arch},code=[sm_{arch},compute_{arch}]"
        ))
        .include(cuda.join("include"))
        .file("kernels/whisper.cu")
        .compile("teamy_whisper_cuda");
    let lib = if cfg!(windows) {
        cuda.join("lib/x64")
    } else {
        cuda.join("lib64")
    };
    cargo_instruction(format_args!("cargo:rustc-link-search=native={}", lib.display()));
    cargo_instruction(format_args!("cargo:rustc-link-lib=cublas"));
}

#[expect(
    clippy::disallowed_macros,
    reason = "Build-script stdout is Cargo's instruction protocol"
)]
fn cargo_instruction(instruction: std::fmt::Arguments<'_>) {
    println!("{instruction}");
}
