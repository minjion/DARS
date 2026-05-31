param (
    [string]$ModelId = "llama-3.3-70b-versatile",
    [string]$Suite = "workspace",
    [string]$Attack = "none",
    [string]$InjectionTask = "",
    [string]$UserTask = "",
    [string]$PythonExe = ""
)

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if (-not $PythonExe) {
    $PythonExe = Join-Path $Root "venv\Scripts\python.exe"
}

if (-not (Test-Path $PythonExe)) {
    Write-Error "Python executable not found: $PythonExe"
    exit 1
}

if (-not $env:GROQ_API_KEY) {
    Write-Error "Set GROQ_API_KEY in the environment before running this script."
    exit 1
}

if ($ModelId -eq "llama3-70b-8192") {
    Write-Error "Groq model 'llama3-70b-8192' is decommissioned. Use 'llama-3.3-70b-versatile'."
    exit 1
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " Running AgentDojo through Groq-compatible API" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$env:OPENAI_API_KEY = $env:GROQ_API_KEY
$env:OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
$env:GROQ_MODEL_ID = $ModelId
$env:PYTHONIOENCODING = "utf-8"

$Command = @(
    "-m", "agentdojo.scripts.benchmark",
    "-s", $Suite,
    "--model", "LOCAL",
    "--model-id", $ModelId,
    "--module-to-load", "scripts.agentdojo_groq_patch",
    "--force-rerun"
)

if ($Attack -and $Attack.ToLowerInvariant() -ne "none") {
    $Command += @("--attack", $Attack)
}
if ($InjectionTask) {
    $Command += @("--injection-task", $InjectionTask)
}
if ($UserTask) {
    $Command += @("--user-task", $UserTask)
}

& $PythonExe @Command

if ($LASTEXITCODE -ne 0) {
    Write-Error "AgentDojo benchmark failed with exit code $LASTEXITCODE. Not parsing logs from this failed run."
    exit $LASTEXITCODE
}

Write-Host "Done. Raw benchmark logs are under runs/." -ForegroundColor Green
Write-Host "Run 'python scripts/parse_agentdojo_logs.py' to convert logs to DARS JSONL." -ForegroundColor Yellow
