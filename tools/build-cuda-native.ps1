[CmdletBinding()]
param([string]$CudaRoot = $env:CUDA_PATH)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
if ([string]::IsNullOrWhiteSpace($CudaRoot)) { throw 'Set CUDA_PATH or pass -CudaRoot.' }
$cuda = (Resolve-Path -LiteralPath $CudaRoot).Path
$runtime = @((Join-Path $cuda 'bin/x64'), (Join-Path $cuda 'bin')) | Where-Object {
    (Test-Path -LiteralPath $_) -and @(Get-ChildItem -LiteralPath $_ -Filter 'cublas64_*.dll').Count -gt 0
} | Select-Object -First 1
if (-not $runtime) { throw 'CUDA runtime DLLs were not found beneath the toolkit.' }
$savedCuda = $env:CUDA_PATH
Push-Location $repo
try {
    $env:CUDA_PATH = $cuda
    & cargo build --release --locked --no-default-features --features cuda-native
    if ($LASTEXITCODE -ne 0) { throw 'Native release build failed.' }
    $metadata = & cargo metadata --no-deps --format-version 1 | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw 'Could not resolve the Cargo target directory.' }
    $output = Join-Path $metadata.target_directory 'release'
    foreach ($pattern in @('cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll')) {
        $files = @(Get-ChildItem -LiteralPath $runtime -Filter $pattern -File)
        if ($files.Count -eq 0) { throw "Missing CUDA runtime dependency: $pattern" }
        foreach ($file in $files) { Copy-Item -LiteralPath $file.FullName -Destination $output -Force }
    }
    Write-Output (Join-Path $output 'teamy-transcriber.exe')
} finally {
    Pop-Location
    $env:CUDA_PATH = $savedCuda
}
