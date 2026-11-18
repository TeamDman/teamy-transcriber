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
$savedPath = $env:PATH
try {
    # The cc crate supplies MSVC to the CUDA build, but its nvcc -E probe
    # also needs to find cl.exe on PATH before that configuration is applied.
    if (-not (Get-Command cl.exe -CommandType Application -ErrorAction SilentlyContinue)) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
        if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) {
            throw 'Visual Studio Installer was not found. Install the Visual Studio C++ build tools.'
        }

        $compiler = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe' |
            Select-Object -First 1
        if ($LASTEXITCODE -ne 0 -or -not $compiler) {
            throw 'The x64 MSVC compiler was not found. Install the Visual Studio C++ build tools.'
        }

        $compilerDirectory = Split-Path -Path $compiler -Parent
        $env:PATH = "$compilerDirectory;$savedPath"
    }

    # cc already gets the compiler family from MSVC for actual CUDA builds.
    # Suppress only the redundant NVCC family probe warnings, including cached ones.
    & (Join-Path $PSScriptRoot 'tools/install-cuda-native.ps1') -Root $Root -CudaRoot $CudaRoot 2>&1 |
        ForEach-Object {
            $message = $_.ToString()
            if ($message -match '^warning: teamy-whisper-native@[^:]+: Compiler family detection failed (?:since it does not define|due to error: (?:ToolFamilyMacroNotFound:|ToolExecError:.*"nvcc" "-E" .*detect_compiler_family\.c))') {
                return
            }

            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                [Console]::Error.WriteLine($message)
            } else {
                $_
            }
        }
} finally {
    $env:PATH = $savedPath
}
