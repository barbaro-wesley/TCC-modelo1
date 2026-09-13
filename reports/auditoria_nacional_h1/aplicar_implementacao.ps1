$ErrorActionPreference = 'Stop'
$targetRoot = [IO.Path]::GetFullPath('C:/Users/Wesley.Barbaro/Documents/tcc-joao/tcc-mvp')
$sourceRoot = Join-Path $PSScriptRoot 'implementacao'
$changes = @(
    'src/vs_epl_krls/production.py',
    'scripts/06_train_s10_production.py',
    'src/vs_epl_krls/experiment.py',
    'scripts/34_s10_frozen_experiment.py',
    'configs/s10_nacional_h1.json',
    'tests/test_frozen_experiment.py',
    'docs/protocolo_tcc_nacional_h1.md',
    'reports/tcc_h1_protocol_v1/manifest.json',
    'reports/tcc_h1_protocol_v1/development_history.json',
    'reports/tcc_h1_protocol_v1/validacao.md',
    'reports/tcc_h1_smoke_v1/predictions.json',
    'reports/tcc_h1_smoke_v1/result.json',
    'reports/tcc_h1_smoke_v1/status.json'
)
$originalHashes = @{
    'src/vs_epl_krls/production.py' = '4fc7b7cacc7b44c5f728d01dd15b01371b2a407cbbd4ccad913c279ab3eb2e0a'
    'scripts/06_train_s10_production.py' = 'e22bada61c351d695d039b650a6da8cad2d7057d31be806e4c192e23ea1eac3f'
}
$manifest = Get-Content -LiteralPath (Join-Path $sourceRoot 'reports/tcc_h1_protocol_v1/manifest.json') -Raw | ConvertFrom-Json
# Preflight every path before making any change. Preserve unrelated local edits.
foreach ($entry in $manifest.source_hashes.PSObject.Properties) {
    if ($changes -contains $entry.Name) { continue }
    $currentPath = Join-Path $targetRoot $entry.Name
    if ((Get-FileHash -LiteralPath $currentPath -Algorithm SHA256).Hash -ne $entry.Value) {
        throw "Source changed since validation: $currentPath"
    }
}
foreach ($relative in $changes) {
    $destination = [IO.Path]::GetFullPath((Join-Path $targetRoot $relative))
    if (-not $destination.StartsWith($targetRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes target: $destination"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $sourceRoot $relative) -PathType Leaf)) {
        throw "Missing prepared file: $relative"
    }
    if ($originalHashes.ContainsKey($relative)) {
        if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne $originalHashes[$relative]) {
            throw "Modified file has changed since audit: $relative"
        }
    } elseif (Test-Path -LiteralPath $destination) {
        throw "Refusing to overwrite existing new-file destination: $relative"
    }
}
$backupRoot = Join-Path $PSScriptRoot 'originais_antes_da_implementacao'
foreach ($relative in $originalHashes.Keys) {
    $backup = Join-Path $backupRoot $relative
    New-Item -ItemType Directory -Path (Split-Path $backup -Parent) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $targetRoot $relative) -Destination $backup
}
foreach ($relative in $changes) {
    $destination = Join-Path $targetRoot $relative
    New-Item -ItemType Directory -Path (Split-Path $destination -Parent) -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $sourceRoot $relative) -Destination $destination
    if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne
        (Get-FileHash -LiteralPath (Join-Path $sourceRoot $relative) -Algorithm SHA256).Hash) {
        throw "Copy verification failed: $relative"
    }
}
Write-Output "Applied and hash-verified $($changes.Count) files in $targetRoot"
