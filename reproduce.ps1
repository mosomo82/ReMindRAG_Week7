# reproduce.ps1 — Single-command reproducibility script for ReMindRAG (Windows/PowerShell)
# Mirrors reproduce.sh for Windows users.
# Usage:
#   .\reproduce.ps1                      # native mode
#   .\reproduce.ps1 -Docker              # build & run via Docker
#   .\reproduce.ps1 -SkipModelDownload   # skip heavy model downloads
#
# Prerequisites: Python 3.13, pip, and (optionally) Docker Desktop

param (
    [switch]$Docker,
    [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"

$ScriptDir   = $PSScriptRoot
$ArtifactsDir = Join-Path $ScriptDir "artifacts"
$LogFile      = Join-Path $ArtifactsDir "reproduce_run.log"
$ResultFile   = Join-Path $ArtifactsDir "smoke_test_result.json"

# ── Bootstrap directories ─────────────────────────────────────────────────────
@($ArtifactsDir,
  (Join-Path $ScriptDir "logs"),
  (Join-Path $ScriptDir "model_cache"),
  (Join-Path $ScriptDir "Rag_Cache")) | ForEach-Object {
    if (-not (Test-Path $_)) { New-Item -ItemType Directory -Path $_ | Out-Null }
}

# ── Helpers ───────────────────────────────────────────────────────────────────
function Log {
    param([string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

function Fail {
    param([string]$Message)
    Log "FAIL: $Message"
    $json = '{"status":"FAIL","error":"' + $Message.Replace('"','\"') + '"}'
    Set-Content -Path $ResultFile -Value $json
    exit 1
}

Log "====== ReMindRAG Reproducibility Run (Windows) ======"
Log "Host: $($env:COMPUTERNAME) / $([System.Environment]::OSVersion.VersionString)"
Log "Args: Docker=$Docker  SkipModelDownload=$SkipModelDownload"

# ── Docker mode ───────────────────────────────────────────────────────────────
if ($Docker) {
    Log "Building Docker image..."
    docker build -t remindrag:latest . 2>&1 | Tee-Object -FilePath $LogFile -Append

    Log "Running smoke test via Docker..."
    $result = docker run --rm remindrag:latest 2>&1 | Tee-Object -FilePath $LogFile -Append

    if ($result -match "SUCCESS") {
        Log "Docker smoke test PASSED"
        $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        Set-Content -Path $ResultFile -Value "{`"status`":`"PASS`",`"mode`":`"docker`",`"timestamp`":`"$ts`"}"
        exit 0
    } else {
        Fail "Docker smoke test failed"
    }
}

# ── Native mode ───────────────────────────────────────────────────────────────

# Step 1: Check Python version
Log "Checking Python version..."
try {
    $pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
} catch {
    Fail "Python not found. Install Python 3.13 and ensure it is on your PATH."
}
Log "Python version: $pyVersion"
if ($pyVersion -ne "3.13") {
    Log "WARNING: Expected Python 3.13, got $pyVersion. Proceeding anyway."
}

# Step 2: Check dependencies
Log "Checking core dependencies..."
$depCheck = python -c "import torch, transformers, chromadb, sentence_transformers, nltk, openai" 2>&1
if ($LASTEXITCODE -ne 0) {
    Log $depCheck
    Fail "Missing dependencies. Run: pip install -r requirements.txt"
}
Log "Dependencies OK."

# Step 3: Check / download NLTK punkt_tab
Log "Verifying NLTK punkt_tab..."
$nltkCheck = python -c "import nltk; nltk.data.find('tokenizers/punkt_tab')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Log "punkt_tab not found — downloading..."
    python -c "import nltk; nltk.download('punkt_tab', quiet=True)" 2>&1 | Add-Content $LogFile
}

# Step 4: Run smoke test
Log "Running smoke test (tests/smoke_test.py)..."
$smokeOutput = python (Join-Path $ScriptDir "tests\smoke_test.py") 2>&1 |
    Tee-Object -FilePath $LogFile -Append |
    Out-String

if ($smokeOutput -match "SUCCESS") {
    $status = "PASS"
    Log "Smoke test PASSED"
} else {
    $status = "FAIL"
    Log "Smoke test FAILED"
}

# Step 5: Write result artifact
$ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$resultJson = @"
{
  "status": "$status",
  "mode": "native",
  "python_version": "$pyVersion",
  "timestamp": "$ts",
  "log": "$($LogFile.Replace('\','\\'))"
}
"@
Set-Content -Path $ResultFile -Value $resultJson
Log "Result written to $ResultFile"
Log "====== Done ======"

if ($status -eq "PASS") { exit 0 } else { exit 1 }
