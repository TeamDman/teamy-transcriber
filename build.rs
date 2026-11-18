use std::path::PathBuf;
use std::process::Command;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

fn main() {
    add_build_script_inputs();
    add_exe_resources();
    add_git_metadata();
    add_build_timestamp();
}

/// Re-run the build script when normal binary inputs change so embedded build metadata stays fresh.
fn add_build_script_inputs() {
    // Native dependency edits relink the application even when src/ is
    // unchanged. Track their source inputs so its embedded Git/build receipt
    // is regenerated too; never watch nested target/ build-output directories.
    for path in [
        "build.rs",
        "Cargo.toml",
        "Cargo.lock",
        "src",
        "examples",
        "native/Cargo.toml",
        "native/Cargo.lock",
        "native/build.rs",
        "native/src",
        "native/kernels",
    ] {
        cargo_instruction(format_args!("cargo:rerun-if-changed={path}"));
    }
}

/// Embeds Windows resources (like application icon) into the executable.
fn add_exe_resources() {
    cargo_instruction(format_args!("cargo:rerun-if-changed=resources"));

    embed_resource::compile("resources/app.rc", embed_resource::NONE)
        .manifest_required()
        .expect("failed to embed resources");
}

/// In your code you can now access git metadata using
/// ```rust
/// let git_rev = option_env!("GIT_REVISION").unwrap_or("unknown");
/// ```
fn add_git_metadata() {
    add_git_metadata_inputs();

    // Try to get a short git revision; on failure, set to "unknown".
    let rev =
        git_output(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let branch = git_output(&["branch", "--show-current"]).unwrap_or_else(|| {
        git_output(&["describe", "--tags", "--exact-match"])
            .unwrap_or_else(|| "detached".to_string())
    });
    let repository = git_output(&["remote", "get-url", "origin"])
        .or_else(|| std::env::var("CARGO_PKG_REPOSITORY").ok())
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = match git_output_allow_empty(&["status", "--short", "--untracked-files=no"]) {
        Some(status) if status.is_empty() => "clean",
        Some(_) => "dirty",
        None => "unknown",
    };

    cargo_instruction(format_args!("cargo:rustc-env=GIT_REVISION={rev}"));
    cargo_instruction(format_args!("cargo:rustc-env=GIT_BRANCH={branch}"));
    cargo_instruction(format_args!(
        "cargo:rustc-env=GIT_REPOSITORY_URL={repository}"
    ));
    cargo_instruction(format_args!("cargo:rustc-env=GIT_WORKTREE_STATUS={dirty}"));
}

/// Re-run the build script when the current git metadata changes.
fn add_git_metadata_inputs() {
    if let Some(git_config_path) = git_output(&["rev-parse", "--git-path", "config"]) {
        cargo_instruction(format_args!("cargo:rerun-if-changed={git_config_path}"));
    }

    if let Some(head_path) = git_output(&["rev-parse", "--git-path", "HEAD"]) {
        cargo_instruction(format_args!("cargo:rerun-if-changed={head_path}"));
    }

    if let Some(index_path) = git_output(&["rev-parse", "--git-path", "index"]) {
        cargo_instruction(format_args!("cargo:rerun-if-changed={index_path}"));
    }

    if let Some(head_ref) = git_output(&["symbolic-ref", "--quiet", "HEAD"])
        && let Some(head_ref_path) = git_output(&["rev-parse", "--git-path", &head_ref])
    {
        cargo_instruction(format_args!("cargo:rerun-if-changed={head_ref_path}"));
    }
}

fn git_output(args: &[&str]) -> Option<String> {
    git_output_allow_empty(args).filter(|s| !s.is_empty())
}

fn git_output_allow_empty(args: &[&str]) -> Option<String> {
    let mut command = Command::new("git");
    if let Some(manifest_dir) = std::env::var_os("CARGO_MANIFEST_DIR") {
        command
            .arg("-c")
            .arg(format!(
                "safe.directory={}",
                PathBuf::from(&manifest_dir).display()
            ))
            .arg("-C")
            .arg(manifest_dir);
    }
    command
        .args(args)
        .output()
        .ok()
        .and_then(|o| o.status.success().then_some(o.stdout))
        .and_then(|v| String::from_utf8(v).ok())
        .map(|s| s.trim().to_string())
}

/// Capture build time as a UTC instant so the runtime can render it in the user's local timezone.
fn add_build_timestamp() {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after Unix epoch")
        .as_secs();

    cargo_instruction(format_args!(
        "cargo:rustc-env=BUILD_TIMESTAMP_UNIX={timestamp}"
    ));
}

#[expect(
    clippy::disallowed_macros,
    reason = "Build-script stdout is Cargo's instruction protocol"
)]
fn cargo_instruction(instruction: std::fmt::Arguments<'_>) {
    println!("{instruction}");
}
