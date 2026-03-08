# RUN.md — How to Run ReMindRAG

## Option 1: Docker (Recommended — Single Command, No Setup)

```bash
# Build the image
docker build -t remindrag:latest .

# Run the smoke test (no API key needed)
docker run --rm remindrag:latest

# Expected output:
# SUCCESS: RAG pipeline successfully retrieved chunks from the database.
```

### With your API key and model cache (persistent):
```bash
docker run --rm -p 5000:5000 \
  -v "$(pwd)/api_key.json:/app/api_key.json:ro" \
  -v "$(pwd)/model_cache:/app/model_cache" \
  -e HF_TOKEN="your_hf_token_here" \
  remindrag:latest \
  python /app/example/example.py
```

**Windows PowerShell:**
```powershell
docker run --rm -p 5000:5000 `
  -v "${PWD}\api_key.json:/app/api_key.json:ro" `
  -v "${PWD}\model_cache:/app/model_cache" `
  -e HF_TOKEN="your_hf_token_here" `
  remindrag:latest `
  python /app/example/example.py
```

---

## Option 2: Native Python

### Step 1: Environment Setup

```bash
conda create -n ReMindRag python==3.13.2
conda activate ReMindRag
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

**CPU-only (no GPU):**
```bash
pip install -r requirements.txt
```

### Step 2: Credentials

1. Copy the template:
```bash
cp api_key.json.example api_key.json
```
2. Fill in your OpenAI key in `api_key.json`:
```json
[{"base_url": "https://api.openai.com/v1", "api_key": "sk-proj-..."}]
```
3. Login to HuggingFace (one-time):
```bash
huggingface-cli login
```

### Step 3: Run

```bash
# Smoke test (no API key needed — uses mock agents)
python tests/smoke_test.py

# Full 17-test reproducibility audit
python tests/test_reproducibility.py

# Example with real API
python example/example.py

# WebUI
python -m ReMindRag.webui.webui
# Open http://127.0.0.1:5000
```

---

## Option 3: Makefile (All-in-one)

```bash
make setup    # Create venv and install dependencies
make smoke    # Run smoke test
make test     # Run full test suite (17 tests)
make docker   # Build Docker image and run smoke test
make clean    # Remove cache and generated files
```

---

## Evaluation Scripts

```bash
# LooGLE evaluation (requires OpenAI API key + dataset)
python eval/eval_LooGLE.py \
  --title_index 0 \
  --data_type longdep_qa \
  --question_type origin \
  --seed 42 \
  --judge_model_name gpt-4o-mini

# HotpotQA evaluation
python eval/eval_Hotpot.py \
  --seed 42 \
  --judge_model_name gpt-4o-mini
```

---

## Config-Driven Execution

All parameters can be set in `config.yaml` instead of CLI flags:

```bash
# Use config file
python eval/eval_LooGLE.py --config config.yaml

# Or override a single value
python eval/eval_LooGLE.py --config config.yaml --seed 123
```

---

## Expected Outputs

| Run | Output Location | Contents |
|-----|-----------------|----------|
| Smoke test | `artifacts/smoke_test_result.json` | Pass/fail + timestamp |
| Full test suite | stdout | 17 test results |
| LooGLE eval | `eval/database/eval-long/<index>/input.json` | Per-question results |
| WebUI | `http://127.0.0.1:5000` | Interactive chat |
| Logs | `logs/` | Dated log files |
