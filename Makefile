# Makefile — ReMindRAG convenience targets
# Usage: make <target>
# Note: On Windows, use Git Bash or WSL to run these targets.

PYTHON = python
PIP = pip
VENV_DIR = venv
ACTIVATE_UNIX = . $(VENV_DIR)/bin/activate
ACTIVATE_WIN = $(VENV_DIR)\Scripts\activate.bat

.PHONY: all setup smoke test eval docker clean help

## Default target
all: smoke

## help: Show this help message
help:
	@echo "ReMindRAG — Available Makefile targets:"
	@echo "  make setup      Create venv and install all dependencies"
	@echo "  make smoke      Run the quick smoke test (no API key needed)"
	@echo "  make test       Run the full 26-test suite (reproducibility + smoke + variance)"
	@echo "  make eval       Run the constrained benchmark subset (~$0.70, needs API key)"
	@echo "  make streamlit  Run the standalone Streamlit WebUI"
	@echo "  make docker     Build Docker image and run smoke test"
	@echo "  make clean      Remove generated cache files and artifacts"

## setup: Create virtual environment and install pinned dependencies
setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/Scripts/pip install --upgrade pip
	$(VENV_DIR)/Scripts/pip install -r requirements.txt \
		--extra-index-url https://download.pytorch.org/whl/cu126
	$(VENV_DIR)/Scripts/python -c \
		"import nltk; nltk.download('punkt_tab', quiet=True)"
	@echo "Setup complete. Activate with: $(ACTIVATE_WIN)"

## smoke: Run the quick smoke test with mock agents (no API key needed)
smoke:
	mkdir -p artifacts logs Rag_Cache model_cache
	$(PYTHON) tests/smoke_test.py
	@echo "Smoke test result written to artifacts/smoke_test_result.json"

## test: Run all 26 mock-only tests (no API key needed)
test:
	$(PYTHON) tests/test_reproducibility.py -v
	$(PYTHON) tests/test_benchmark_smoke.py -v
	$(PYTHON) tests/test_repro_variance.py -v

## eval: Run the constrained benchmark subset (LooGLE 5-title + Hotpot 50-question, ~$0.70)
eval:
	$(PYTHON) eval/run_eval_subset.py --seed 42 --judge_model_name gpt-4o-mini
	@echo "Results written to eval_results.json"

## streamlit: Run the standalone Streamlit WebUI
streamlit:
	$(VENV_DIR)/Scripts/streamlit run streamlit_app.py

## docker: Build Docker image and run the in-container smoke test
docker:
	docker build -t remindrag:latest .
	docker run --rm remindrag:latest
	@echo "Docker smoke test complete."

## clean: Remove generated cache, pycache, and large local artifacts
clean:
	@echo "Removing __pycache__ directories..."
	find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "Removing evaluation database cache..."
	rm -rf eval/database/eval-long eval/database/eval-short
	@echo "Removing run logs..."
	rm -f logs/*.log eval/logs/*.log
	@echo "Done. Run 'make setup' to reinstall dependencies."
