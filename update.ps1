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
