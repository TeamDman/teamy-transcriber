[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Root,
    [string]$CudaRoot = $env:CUDA_PATH
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$installRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Root)
$destination = Join-Path $installRoot 'bin'
if ([string]::IsNullOrWhiteSpace($CudaRoot)) { throw 'Set CUDA_PATH or pass -CudaRoot.' }
$cuda = (Resolve-Path -LiteralPath $CudaRoot).Path

# Build and resolve all dependencies before changing the installation. This
# script never changes PATH, downloads weights, or registers a background app.
& (Join-Path $PSScriptRoot 'build-cuda-native.ps1') -CudaRoot $cuda
Push-Location $repo
try {
    $metadataJson = & cargo metadata --no-deps --format-version 1
    if ($LASTEXITCODE -ne 0) { throw 'Could not resolve the Cargo target directory.' }
    $metadata = $metadataJson | ConvertFrom-Json
    $release = Join-Path $metadata.target_directory 'release'
    $dependencies = @()
    foreach ($pattern in @('cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll')) {
        $files = @(Get-ChildItem -LiteralPath $release -Filter $pattern -File)
        if ($files.Count -ne 1) { throw "Expected one staged dependency matching $pattern; clean up obsolete staged versions first." }
        $dependencies += $files
    }
    # Cargo's bin directory may be shared with other applications. Reuse only
    # identical libraries, rather than replacing a DLL used by another program.
    foreach ($file in $dependencies) {
        $target = Join-Path $destination $file.Name
        if ((Test-Path -LiteralPath $target) -and
            (Get-FileHash -LiteralPath $target).Hash -ne (Get-FileHash -LiteralPath $file.FullName).Hash) {
            throw "A different $($file.Name) already exists in $destination. Choose a separate -Root."
        }
    }
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    foreach ($file in $dependencies) {
        $target = Join-Path $destination $file.Name
        if (-not (Test-Path -LiteralPath $target)) { Copy-Item -LiteralPath $file.FullName -Destination $target }
    }
    $savedCuda = $env:CUDA_PATH
    try {
        $env:CUDA_PATH = $cuda
        & cargo install --path $repo --root $installRoot --locked --no-default-features --features cuda-native --bin teamy-transcriber --force
        if ($LASTEXITCODE -ne 0) { throw 'Native installation failed.' }
    } finally {
        $env:CUDA_PATH = $savedCuda
    }
    Write-Output "Installed native CUDA transcription: $(Join-Path $destination 'teamy-transcriber.exe')"
    Write-Output "To select it in a terminal, put $destination before other installations on PATH."
} finally {
    Pop-Location
}
