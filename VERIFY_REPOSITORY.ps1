param(
    [string]$RepositoryRoot = $PSScriptRoot
)

$ErrorActionPreference = "Stop"

$ShaFile = Join-Path $RepositoryRoot "manifests\SHA256SUMS_REPOSITORY.txt"

if (-not (Test-Path -LiteralPath $ShaFile)) {
    throw "SHA-256 inventory not found: $ShaFile"
}

$Failures = @()
$Checked = 0

foreach ($line in (Get-Content -LiteralPath $ShaFile)) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }

    if ($line -notmatch '^([0-9a-fA-F]{64})\s{2}(.+)$') {
        throw "Malformed SHA-256 line: $line"
    }

    $Expected = $Matches[1].ToLowerInvariant()
    $Rel = $Matches[2].Replace("/", "\")
    $Path = Join-Path $RepositoryRoot $Rel

    if (-not (Test-Path -LiteralPath $Path)) {
        $Failures += "$Rel :: MISSING"
        continue
    }

    $Actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    $Checked++

    if ($Actual -ne $Expected) {
        $Failures += "$Rel :: HASH_MISMATCH"
    }
}

if ($Failures.Count -gt 0) {
    Write-Host ""
    Write-Host "VERIFY_REPOSITORY = FAIL"
    $Failures | ForEach-Object { Write-Host "  $_" }
    exit 1
}

Write-Host ""
Write-Host "VERIFY_REPOSITORY = PASS"
Write-Host "Files checked: $Checked"