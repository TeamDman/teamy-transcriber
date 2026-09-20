[CmdletBinding()]
param(
    [ValidateSet('tch', 'cuda')][string]$Backend = 'cuda',
    [string]$Root,
    [string]$CudaRoot = $env:CUDA_PATH
)

$ErrorActionPreference = 'Stop'
if ($Backend -eq 'cuda') {
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
} else {
    $installArgs = @('install', '--path', $PSScriptRoot, '--locked', '--no-default-features', '--features', 'tch-native', '--bin', 'teamy-transcriber', '--force')
    if (-not [string]::IsNullOrWhiteSpace($Root)) { $installArgs += @('--root', $Root) }
    & cargo @installArgs
    if ($LASTEXITCODE -ne 0) { throw 'Installation failed.' }
}
