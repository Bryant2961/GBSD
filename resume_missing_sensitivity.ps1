param(
    [string]$PythonExecutable = "python",
    [string]$Seeds = "0,1,2,3,4",
    [int]$McSamples = 200
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Push-Location $ProjectRoot
try {
    & $PythonExecutable "experiments\resume_missing_sensitivity.py" `
        --execute `
        --preset full `
        --seeds $Seeds `
        --n-mc $McSamples `
        --python $PythonExecutable
    if ($LASTEXITCODE -ne 0) {
        throw "Missing sensitivity resume failed with exit code $LASTEXITCODE."
    }

    & $PythonExecutable "experiments\summarize.py" `
        --input "results/unified_blind_protocol" `
        --output "results/paper_tables"
    if ($LASTEXITCODE -ne 0) {
        throw "Summary generation failed with exit code $LASTEXITCODE."
    }

    & $PythonExecutable "paper_tools\generate_tables.py" `
        --summary-dir "results/paper_tables" `
        --output-dir "results/paper_tables/generated"
    if ($LASTEXITCODE -ne 0) {
        throw "Paper table generation failed with exit code $LASTEXITCODE."
    }

    & $PythonExecutable "paper_tools\generate_figures.py" `
        --summary-dir "results/paper_tables" `
        --output-dir "results/paper_figures"
    if ($LASTEXITCODE -ne 0) {
        throw "Paper figure manifest generation failed with exit code $LASTEXITCODE."
    }

    & $PythonExecutable "paper_tools\verify_consistency.py" `
        --summary-dir "results/paper_tables"
    if ($LASTEXITCODE -ne 0) {
        throw "Consistency verification failed with exit code $LASTEXITCODE."
    }

    & $PythonExecutable "paper_tools\verify_submission_assets.py"
    if ($LASTEXITCODE -ne 0) {
        throw "Submission asset verification failed with exit code $LASTEXITCODE."
    }

    & $PythonExecutable "paper_tools\export_submission_assets.py" `
        --output "results/submission_assets"
    if ($LASTEXITCODE -ne 0) {
        throw "Submission asset export failed with exit code $LASTEXITCODE."
    }

    Write-Host ""
    Write-Host "Missing sensitivity results have been resumed and integrated."
    Write-Host "Unified results: $ProjectRoot\results\unified_blind_protocol"
    Write-Host "Paper tables:    $ProjectRoot\results\paper_tables"
}
finally {
    Pop-Location
}
