#!/usr/bin/env bash
# reproduce.sh — Single-command reproducibility script for ReMindRAG
# Usage: bash reproduce.sh [--docker] [--skip-model-download]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"
LOG_FILE="$ARTIFACTS_DIR/reproduce_run.log"
RESULT_FILE="$ARTIFACTS_DIR/smoke_test_result.json"

mkdir -p "$ARTIFACTS_DIR"
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/model_cache"
mkdir -p "$SCRIPT_DIR/Rag_Cache"

# ── Logging ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
fail() { log "FAIL: $*"; echo '{"status":"FAIL","error":"'"$*"'"}' > "$RESULT_FILE"; exit 1; }

log "====== ReMindRAG Reproducibility Run ======"
log "Host: $(uname -a)"
log "Python: $(python --version 2>&1 || echo 'not found')"
log "Args: $*"

# ── Parse flags ───────────────────────────────────────────────────────────────
USE_DOCKER=false
SKIP_MODELS=false
for arg in "$@"; do
  case $arg in
    --docker) USE_DOCKER=true ;;
    --skip-model-download) SKIP_MODELS=true ;;
  esac
done

# ── Docker mode ───────────────────────────────────────────────────────────────
if [ "$USE_DOCKER" = true ]; then
  log "Building Docker image..."
  docker build -t remindrag:latest . 2>&1 | tee -a "$LOG_FILE"
  log "Running smoke test via Docker..."
  RESULT=$(docker run --rm remindrag:latest 2>&1 | tee -a "$LOG_FILE")
  if echo "$RESULT" | grep -q "SUCCESS"; then
    log "Docker smoke test PASSED"
    echo '{"status":"PASS","mode":"docker","timestamp":"'"$(date -u +%Y-%m-%dT%H:%M:%SZ)"'"}' > "$RESULT_FILE"
    exit 0
  else
    fail "Docker smoke test failed"
  fi
fi

# ── Native mode ───────────────────────────────────────────────────────────────

# Step 1: Check Python version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
log "Python version: $PYTHON_VERSION"
if [[ "$PYTHON_VERSION" != "3.13" ]]; then
  log "WARNING: Expected Python 3.13, got $PYTHON_VERSION. Proceeding anyway."
fi

# Step 2: Check dependencies
log "Checking dependencies..."
python -c "import torch, transformers, chromadb, sentence_transformers, nltk, openai" \
  2>&1 | tee -a "$LOG_FILE" \
  || fail "Missing dependencies. Run: pip install -r requirements.txt"

# Step 3: Check NLTK data
log "Verifying NLTK punkt_tab..."
python -c "import nltk; nltk.data.find('tokenizers/punkt_tab')" 2>/dev/null \
  || python -c "import nltk; nltk.download('punkt_tab', quiet=True)"

# Step 4: Run smoke test
log "Running smoke test (tests/smoke_test.py)..."
SMOKE_OUTPUT=$(python "$SCRIPT_DIR/tests/smoke_test.py" 2>&1 | tee -a "$LOG_FILE")

if echo "$SMOKE_OUTPUT" | grep -q "SUCCESS"; then
  STATUS="PASS"
  log "Smoke test PASSED"
else
  STATUS="FAIL"
  log "Smoke test FAILED"
fi

# Step 5: Write results artifact
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cat > "$RESULT_FILE" <<EOF
{
  "status": "$STATUS",
  "mode": "native",
  "python_version": "$PYTHON_VERSION",
  "timestamp": "$TIMESTAMP",
  "log": "$LOG_FILE"
}
EOF

log "Result written to $RESULT_FILE"
log "====== Done ======"

[ "$STATUS" = "PASS" ] && exit 0 || exit 1
