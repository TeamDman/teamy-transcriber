[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$NativeExe,
    [Parameter(Mandatory)][string]$NativeModel,
    [Parameter(Mandatory)][string]$Python,
    [Parameter(Mandatory)][string]$Ct2Model,
    [Parameter(Mandatory)][string]$Corpus,
    [Parameter(Mandatory)][string]$Output,
    [Parameter(Mandatory)][string]$Ct2DllDirectory,
    [ValidateRange(1, 10)][int]$Rounds = 3,
    [ValidateRange(2, 20)][int]$Repeats = 4,
    [ValidateSet('fp32', 'tf32')][string]$Math = 'tf32'
)

$ErrorActionPreference = 'Stop'
$native = (Resolve-Path -LiteralPath $NativeExe).Path
$model = (Resolve-Path -LiteralPath $NativeModel).Path
$pythonExe = (Resolve-Path -LiteralPath $Python).Path
$ct2 = (Resolve-Path -LiteralPath $Ct2Model).Path
$corpusPath = (Resolve-Path -LiteralPath $Corpus).Path
$manifest = Join-Path $corpusPath 'manifest.json'
$wavList = Join-Path $corpusPath 'wav-list.json'
if ((Test-Path -LiteralPath $Output) -and @(Get-ChildItem -LiteralPath $Output).Count) {
    throw 'Choose an empty output directory to preserve previous measurements.'
}
$out = (New-Item -ItemType Directory -Path $Output -Force).FullName
$savedOffline = $env:HF_HUB_OFFLINE
$metadata = [ordered]@{
    scope = 'ASR only; same local audio/weights, English greedy, batch one; no VAD/alignment/diarization'
    started_utc = [DateTime]::UtcNow.ToString('o')
    native_exe_sha256 = (Get-FileHash -LiteralPath $native -Algorithm SHA256).Hash
    native_model_hashes = @(Get-ChildItem -LiteralPath $model -Filter 'model*.safetensors' -File | Sort-Object Name | ForEach-Object { [ordered]@{name=$_.Name; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash} })
    reference_model_sha256 = (Get-FileHash -LiteralPath (Join-Path $ct2 'model.bin') -Algorithm SHA256).Hash
    reference_script_sha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'benchmark_ct2.py') -Algorithm SHA256).Hash
    manifest_sha256 = (Get-FileHash -LiteralPath $manifest -Algorithm SHA256).Hash
    rounds = $Rounds
    repeats = $Repeats
    native_math = $Math
    reference_compute_type = 'float16'
    file_cache_policy = 'OS file cache retained; fresh processes in alternating order; no simultaneous inference'
}
$summaries = @()
try {
    $env:HF_HUB_OFFLINE = '1'
    for ($round = 0; $round -lt $Rounds; $round++) {
        $nativeReceipt = Join-Path $out "native-$round.json"
        $referenceReceipt = Join-Path $out "reference-$round.json"
        $order = if ($round % 2 -eq 0) { @('native', 'reference') } else { @('reference', 'native') }
        foreach ($backend in $order) {
            Write-Host "Round $($round + 1)/${Rounds}: $backend"
            & nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv,noheader > (Join-Path $out "gpu-$round-$backend.csv")
            if ($LASTEXITCODE -ne 0) { throw 'Could not capture GPU state.' }
            if ($backend -eq 'native') {
                $nativeArguments = @($model, $wavList, "$Repeats", '-')
                if ($Math -eq 'tf32') { $nativeArguments += 'tf32' }
                & $native @nativeArguments > $nativeReceipt 2> (Join-Path $out "native-$round.log")
            } else {
                & $pythonExe (Join-Path $PSScriptRoot 'benchmark_ct2.py') $ct2 $manifest --wav-input --compute-type float16 --repeats $Repeats --dll-dir $Ct2DllDirectory > $referenceReceipt 2> (Join-Path $out "reference-$round.log")
            }
            if ($LASTEXITCODE -ne 0) { throw "$backend failed; see the round log." }
        }
        $summary = Join-Path $out "comparison-$round.json"
        & $pythonExe (Join-Path $PSScriptRoot 'summarize_asr.py') $manifest $nativeReceipt $referenceReceipt > $summary
        if ($LASTEXITCODE -ne 0) { throw 'Comparison failed.' }
        $summaries += (Get-Content -LiteralPath $summary -Raw | ConvertFrom-Json | Select-Object -ExcludeProperty rows)
    }
    [ordered]@{ metadata = $metadata; comparisons = $summaries } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $out 'summary.json')
} finally {
    $env:HF_HUB_OFFLINE = $savedOffline
}
