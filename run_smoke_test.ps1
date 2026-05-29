param(
    [string]$PythonExecutable = "python",
    [int]$Seed = 0,
    [int]$McSamples = 5
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path -LiteralPath $PythonExecutable)) {
    throw "Python executable not found: $PythonExecutable"
}

Push-Location $ProjectRoot
try {
    Write-Host "Running isolated GBSD smoke test from: $ProjectRoot"
    Write-Host "Python: $PythonExecutable"
    Write-Host "Seed: $Seed; MC samples: $McSamples"
    Write-Host "Outputs: results\smoke_test (not formal paper results)"

    & $PythonExecutable "experiments\generate_all.py" `
        --all `
        --execute `
        --preset smoke `
        --seed $Seed `
        --n-mc $McSamples `
        --python $PythonExecutable
    if ($LASTEXITCODE -ne 0) {
        throw "Smoke execution failed with exit code $LASTEXITCODE."
    }

    & $PythonExecutable "experiments\summarize.py" `
        --input "results/smoke_test" `
        --output "results/smoke_tables" `
        --allow-incomplete
    if ($LASTEXITCODE -ne 0) {
        throw "Smoke summary generation failed with exit code $LASTEXITCODE."
    }

    & $PythonExecutable "paper_tools\verify_consistency.py" `
        --summary-dir "results/smoke_tables" `
        --allow-incomplete
    if ($LASTEXITCODE -ne 0) {
        throw "Smoke consistency check failed with exit code $LASTEXITCODE."
    }

    Write-Host ""
    Write-Host "Smoke test completed."
    Write-Host "Inspect per-seed outputs: $ProjectRoot\results\smoke_test"
    Write-Host "Inspect summary tables:   $ProjectRoot\results\smoke_tables"
}
finally {
    Pop-Location
}
