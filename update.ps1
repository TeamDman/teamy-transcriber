[CmdletBinding()]
param(
    [ValidateSet('tch', 'cuda')][string]$Backend = 'tch',
    [string]$Root,
    [string]$CudaRoot = $env:CUDA_PATH
)

$ErrorActionPreference = 'Stop'
if ($Backend -eq 'cuda') {
    if ([string]::IsNullOrWhiteSpace($Root)) {
        throw 'Native CUDA installation requires -Root <install-directory>; choose a separate directory to keep its runtime DLLs together.'
    }
    & (Join-Path $PSScriptRoot 'tools/install-cuda-native.ps1') -Root $Root -CudaRoot $CudaRoot
} else {
    $installArgs = @('install', '--path', $PSScriptRoot, '--locked')
    if (-not [string]::IsNullOrWhiteSpace($Root)) { $installArgs += @('--root', $Root) }
    & cargo @installArgs
    if ($LASTEXITCODE -ne 0) { throw 'Installation failed.' }
}
