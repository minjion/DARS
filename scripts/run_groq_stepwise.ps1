param(
    [string[]]$Suites = @("workspace", "slack", "travel", "banking"),
    [string]$PythonExe = "",
    [switch]$SkipPreflight
)

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not $PythonExe) {
    $PythonExe = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-Path $PythonExe)) {
        $PythonExe = Join-Path $Root "venv\Scripts\python.exe"
    }
}

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found: $PythonExe"
    exit 1
}

if (-not $env:GROQ_API_KEY) {
    Write-Error "Set GROQ_API_KEY in the environment before running this script."
    exit 1
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Running Groq collection one suite at a time" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

foreach ($suite in $Suites) {
    Write-Host "\n--- Suite: $suite ---" -ForegroundColor Yellow
    $args = @(
        "data\collect_real_traces.py",
        "--suite", $suite,
        "--output", "data\real_traces.jsonl",
        "--seed-output", "data\real_seed_traces.jsonl"
    )
    if ($SkipPreflight) {
        $args += "--skip-preflight"
    }

    & $PythonExe @args
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Suite '$suite' returned exit code $LASTEXITCODE. Stopping stepwise run."
        exit $LASTEXITCODE
    }
}

Write-Host "All requested suites completed stepwise." -ForegroundColor Green
