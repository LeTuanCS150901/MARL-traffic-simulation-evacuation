param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExpRunsDir = $ScriptDir
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$RepoRoot = Resolve-Path (Join-Path $ProjectRoot "..")

$ExperimentConfig = Join-Path $ExpRunsDir "experiment.yaml"
$BestSeedsFile = Join-Path $ExpRunsDir "proposed\randers\best_seeds.yaml"
$FineTuneOutputDir = Join-Path $ExpRunsDir "finetune\randers"
$FineTuneHistory = Join-Path $FineTuneOutputDir "finetune_seed_history.csv"
$RunLog = Join-Path $ExpRunsDir "full_seed_finetune.log"

function Assert-LastExitCode {
    param(
        [string]$StepName
    )

    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE"
    }
}

Write-Host "=== MARL Seed Search + Anti-Overfit Fine-Tuning ==="
Write-Host "Repo root: $RepoRoot"
Write-Host "Experiment config: $ExperimentConfig"
Write-Host "Run log: $RunLog"
if (Test-Path $Python -PathType Container) {
    $Python = Join-Path $Python "python.exe"
}

Write-Host "Python: $Python"

if (-not $env:SUMO_HOME) {
    throw "SUMO_HOME is not set. Please set SUMO_HOME before running this script."
}

$env:PATH = "$env:SUMO_HOME\bin;$env:PATH"
$env:PYTHONPATH = "$env:SUMO_HOME\tools;$env:PYTHONPATH"

if (-not (Test-Path $ExperimentConfig)) {
    throw "Missing experiment config: $ExperimentConfig"
}

if (-not (Test-Path (Join-Path $ExpRunsDir "run_experiments.py"))) {
    throw "Missing seed-search runner: $(Join-Path $ExpRunsDir 'run_experiments.py')"
}

if (-not (Test-Path (Join-Path $ProjectRoot "routing_MARl\main.py"))) {
    throw "Missing MARL entrypoint: $(Join-Path $ProjectRoot 'routing_MARl\main.py')"
}

Start-Transcript -Path $RunLog -Append | Out-Null

try {
    Write-Host "`n[0/4] Checking Python/SUMO environment..."
    & $Python -c "import sys; print(sys.executable); import libsumo; print('libsumo OK')"
    if ($LASTEXITCODE -ne 0) {
        throw "libsumo import failed. Run this script with the Python environment that has SUMO/libsumo installed, for example: .\routing-MARL\exp_runs\run_full_seed_finetune.ps1 -Python C:\path\to\env\python.exe"
    }

    & $Python -c "import torch, numpy; print('torch/numpy OK')"
    if ($LASTEXITCODE -ne 0) {
        throw "Core MARL dependency import failed. The selected Python must import torch and numpy."
    }

    Write-Host "`n[1/4] Running complete-only seed search..."
    & $Python (Join-Path $ExpRunsDir "run_experiments.py") --config $ExperimentConfig
    Assert-LastExitCode "Seed search"

    Write-Host "`n[2/4] Validating best complete seeds..."
    if (-not (Test-Path $BestSeedsFile)) {
        throw "Missing best_seeds.yaml: $BestSeedsFile"
    }

    $BestSeedText = Get-Content $BestSeedsFile -Raw
    $CompleteSeedMatches = [regex]::Matches($BestSeedText, 'status:\s+"?complete"?')
    $CompleteSeedCount = $CompleteSeedMatches.Count

    Write-Host "Complete seeds found in best_seeds.yaml: $CompleteSeedCount"

    if ($CompleteSeedCount -lt 5) {
        throw "Candidate gate failed: need at least 5 complete seeds, found $CompleteSeedCount. Increase max_seed_windows or rerun with a new seed_start."
    }

    Write-Host "`n[3/4] Running anti-overfit fine-tuning..."
    New-Item -ItemType Directory -Force -Path $FineTuneOutputDir | Out-Null

    & $Python (Join-Path $ProjectRoot "routing_MARl\main.py") `
        --mode finetune `
        --testcase randers `
        --index 10 `
        --episode 401 `
        --finetune-episodes 50 `
        --steps 7500 `
        --save_freq 10 `
        --output-dir $FineTuneOutputDir `
        --best-seeds-file $BestSeedsFile `
        --seed-pool-size 5 `
        --seed 123

    Assert-LastExitCode "Fine-tuning"

    Write-Host "`n[4/4] Validating fine-tune seed diversity..."
    if (-not (Test-Path $FineTuneHistory)) {
        throw "Missing fine-tune seed history: $FineTuneHistory"
    }

    $Rows = Import-Csv $FineTuneHistory
    if ($Rows.Count -eq 0) {
        throw "Fine-tune seed history is empty."
    }

    $DistinctSeeds = @($Rows | Select-Object -ExpandProperty seed -Unique)
    Write-Host "Fine-tune episodes logged: $($Rows.Count)"
    Write-Host "Distinct seeds used: $($DistinctSeeds -join ', ')"

    if ($Rows.Count -ge 20) {
        for ($i = 0; $i -le ($Rows.Count - 20); $i++) {
            $Window = $Rows[$i..($i + 19)]
            $WindowDistinct = @($Window | Select-Object -ExpandProperty seed -Unique)
            if ($WindowDistinct.Count -lt 3) {
                throw "Anti-overfit gate failed: episodes $($i + 1)-$($i + 20) used fewer than 3 distinct seeds."
            }
        }
    }

    Write-Host "`nDONE."
    Write-Host "Best seeds: $BestSeedsFile"
    Write-Host "Fine-tune output: $FineTuneOutputDir"
    Write-Host "Seed history: $FineTuneHistory"
}
finally {
    Stop-Transcript | Out-Null
}
