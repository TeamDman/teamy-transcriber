<#
.SYNOPSIS
Build and install the CUDA version of Teamy Transcriber.
.DESCRIPTION
Installs into CARGO_INSTALL_ROOT, then CARGO_HOME, then the user's .cargo
directory unless -Root is supplied. Stages CUDA runtime DLLs beside the
executable and verifies it can start without CUDA directories on PATH.
Requires Rust, the Windows C++ build tools and a CUDA toolkit. Model weights
and existing application settings are preserved; no weights are downloaded.
.EXAMPLE
./update.ps1
.EXAMPLE
./update.ps1 -Root C:\path\to\isolated-install -CudaRoot C:\path\to\cuda
#>
[CmdletBinding()]
param(
    [string]$Root,
    [string]$CudaRoot = $env:CUDA_PATH
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($Root)) {
    # Match Cargo's usual install location. An explicit root remains useful
    # for side-by-side installations and isolated verification.
    $Root = if (-not [string]::IsNullOrWhiteSpace($env:CARGO_INSTALL_ROOT)) {
        $env:CARGO_INSTALL_ROOT
    } elseif (-not [string]::IsNullOrWhiteSpace($env:CARGO_HOME)) {
        $env:CARGO_HOME
    } else {
        Join-Path ([Environment]::GetFolderPath('UserProfile')) '.cargo'
    }
}
& (Join-Path $PSScriptRoot 'tools/install-cuda-native.ps1') -Root $Root -CudaRoot $CudaRoot
