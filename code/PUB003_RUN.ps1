param(
    [Parameter(Mandatory = $true)]
    [string]$ResultsRoot,

    [Parameter(Mandatory = $true)]
    [string]$ManuscriptIn,

    [string]$ScriptsRoot = $PSScriptRoot,

    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

# PUBLIC PORTABLE WRAPPER — no machine-specific absolute paths
# This wrapper changes invocation paths only. It does not alter the PUB-003
# publication builder, manuscript patcher, or frozen scientific protocol.

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $ResultsRoot "pub003_publication_assets"
}

$Builder = Join-Path $ScriptsRoot "pub003_build_publication_assets.py"
$Patcher = Join-Path $ScriptsRoot "pub003_patch_manuscript.py"
$ManuscriptOut = Join-Path $OutputDir "Experimental_Mathematics_manuscript_v0_5_PUB003.docx"

if (-not (Test-Path -LiteralPath $Builder)) {
    throw "PUB-003 asset builder not found: $Builder"
}
if (-not (Test-Path -LiteralPath $Patcher)) {
    throw "PUB-003 manuscript patcher not found: $Patcher"
}
if (-not (Test-Path -LiteralPath $ResultsRoot)) {
    throw "Results root not found: $ResultsRoot"
}
if (-not (Test-Path -LiteralPath $ManuscriptIn)) {
    throw "Input manuscript not found: $ManuscriptIn"
}

python $Builder `
  --results-root $ResultsRoot `
  --scripts-root $ScriptsRoot `
  --output-dir $OutputDir `
  --max-copy-mib 256

if ($LASTEXITCODE -ne 0) {
    throw "PUB-003 asset builder failed."
}

Get-Content (Join-Path $OutputDir "pub003_verdict.txt")

python $Patcher `
  --manuscript-in $ManuscriptIn `
  --assets-dir $OutputDir `
  --manuscript-out $ManuscriptOut

if ($LASTEXITCODE -ne 0) {
    throw "PUB-003 manuscript patch failed."
}

Write-Host ""
Write-Host "PUB-003 outputs:"
Write-Host "  $OutputDir"
Write-Host "  $ManuscriptOut"
Write-Host ""
Write-Host "This public wrapper requires explicit local input paths and contains no machine-specific defaults."