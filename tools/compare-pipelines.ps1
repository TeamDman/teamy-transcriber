[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$NativeWorkflowExe,
    [Parameter(Mandatory)][string]$RustPipelineExe,
    [Parameter(Mandatory)][string]$NativeModel,
    [Parameter(Mandatory)][string]$CanonicalModel,
    [Parameter(Mandatory)][string]$Ct2Model,
    [Parameter(Mandatory)][string]$SileroRepo,
    [Parameter(Mandatory)][string]$SileroOnnx,
    [Parameter(Mandatory)][string]$OnnxRuntimeDll,
    [Parameter(Mandatory)][string]$Inputs,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [Parameter(Mandatory)][string]$PythonDllDirectory,
    [string]$Python = 'python',
    [string]$ModelId = 'openai/whisper-large-v3',
    [ValidateRange(1, 8)][int]$NativeBatch = 8,
    [ValidateRange(1, 16)][int]$RustBatch = 8,
    [ValidateRange(1, 32)][int]$PythonBatch = 16,
    [ValidateRange(2, 10)][int]$Repetitions = 2,
    [ValidateRange(1, 9)][int]$Rounds = 3
)

$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $OutputDirectory) { throw 'Choose a fresh output directory.' }
$out = [IO.Path]::GetFullPath($OutputDirectory)
$pythonScript = Join-Path $PSScriptRoot 'benchmark_whisperx_pipeline.py'
$assets = @($NativeWorkflowExe, $RustPipelineExe, $SileroOnnx, $OnnxRuntimeDll, $Inputs,
    $pythonScript, (Join-Path $PSScriptRoot 'rust-whisperx-bench/Cargo.lock'),
    (Join-Path $NativeModel 'model.safetensors'), (Join-Path $CanonicalModel 'model.safetensors'),
    (Join-Path $CanonicalModel 'generation_config.json'), (Join-Path $NativeModel 'generation_config.json'),
    (Join-Path $NativeModel 'vad/silero.safetensors'), (Join-Path $Ct2Model 'model.bin'))
foreach ($path in $assets) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing artifact: $path" }
}
New-Item -ItemType Directory -Path $out | Out-Null
$hashes = @(Get-FileHash -LiteralPath $assets -Algorithm SHA256)
$hashes | ConvertTo-Json | Set-Content "$out/artifacts.json"
foreach ($file in @('model.safetensors', 'generation_config.json')) {
    $nativePath = [IO.Path]::GetFullPath((Join-Path $NativeModel $file))
    $canonicalPath = [IO.Path]::GetFullPath((Join-Path $CanonicalModel $file))
    $nativeHash = @($hashes | Where-Object Path -eq $nativePath)[0].Hash
    $canonicalHash = @($hashes | Where-Object Path -eq $canonicalPath)[0].Hash
    if (-not $nativeHash -or $nativeHash -ne $canonicalHash) { throw "Native/canonical $file differ." }
}
$paths = if ([IO.Path]::GetExtension($Inputs) -eq '.json') {
    @(Get-Content -LiteralPath $Inputs -Raw | ConvertFrom-Json)
} else { @($Inputs) }
Get-FileHash -LiteralPath $paths -Algorithm SHA256 | ConvertTo-Json | Set-Content "$out/inputs.json"
@{ native_batch=$NativeBatch; rust_batch=$RustBatch; python_batch=$PythonBatch;
    repetitions=$Repetitions; rounds=$Rounds; model_id=$ModelId;
    scope='WAV-to-VAD/greedy-English-ASR/assembly. Native additionally imports, persists and exports. No word alignment/diarization. OS file cache retained. Rust FP32 active-row batching; native TF32/FP32 storage; Python CT2 FP16.'
} | ConvertTo-Json | Set-Content "$out/settings.json"

$savedEnvironment = @{}
foreach ($name in @('ORT_DYLIB_PATH', 'TEAMY_TRANSCRIBER_CUDA_DEVICE',
    'TEAMY_TRANSCRIBER_CUDA_BATCH_SIZE', 'TEAMY_TRANSCRIBER_CUDA_MATH')) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
try {
    $env:ORT_DYLIB_PATH = $OnnxRuntimeDll
    $env:TEAMY_TRANSCRIBER_CUDA_DEVICE = '0'
    $env:TEAMY_TRANSCRIBER_CUDA_BATCH_SIZE = [string]$NativeBatch
    $env:TEAMY_TRANSCRIBER_CUDA_MATH = 'tf32'
    $engines = @('native', 'python', 'rust')
    for ($round = 0; $round -lt $Rounds; $round++) {
        for ($slot = 0; $slot -lt $engines.Count; $slot++) {
            $engine = $engines[($round + $slot) % $engines.Count]
            $name = "$engine-$round"
            & nvidia-smi --query-gpu=name,driver_version,memory.used,utilization.gpu --format=csv > "$out/$name-gpu.csv"
            if ($LASTEXITCODE -ne 0) { throw 'Could not record GPU state.' }
            $watch = [Diagnostics.Stopwatch]::StartNew()
            switch ($engine) {
                'native' {
                    & $NativeWorkflowExe $NativeModel $Inputs "$out/$name-home" $Repetitions resident > "$out/$name.json" 2> "$out/$name.log"
                }
                'rust' {
                    & $RustPipelineExe $CanonicalModel $SileroOnnx $Inputs $Repetitions $RustBatch fp32 $ModelId > "$out/$name.json" 2> "$out/$name.log"
                }
                'python' {
                    & $Python $pythonScript $Ct2Model $Inputs "$out/$name.json" --silero-repo $SileroRepo --generation-config (Join-Path $CanonicalModel 'generation_config.json') --dll-dir $PythonDllDirectory --batch-sizes $PythonBatch --repeats $Repetitions > "$out/$name.log" 2>&1
                }
            }
            $code = $LASTEXITCODE
            $watch.Stop()
            @{whole_process_ms=$watch.Elapsed.TotalMilliseconds; exit_code=$code; order=$slot} |
                ConvertTo-Json | Set-Content "$out/$name-process.json"
            if ($code -ne 0) { throw "$name failed; inspect $out/$name.log" }
            Write-Output "$name completed"
        }
    }
} finally {
    foreach ($name in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
}
