# Lab 7: Team Report

## Division of Responsibilities

Our team successfully executed Lab 7 by dividing the workload into three distinct, non-overlapping phases, allowing for parallel execution and individual accountability.

### Tony Nguyen (Member 1) — Related Work Reproduction & Core Pipeline Audit

**Primary Ownership:** ReMindRAG Reproduction and Environmental Architecture

- **Repo Selection & Setup:** Selected ReMindRAG as the related work, cloned the repository, and audited the initial environment.
- **Dependency Management:** Identified and fixed critical containerization blockers (e.g., missing CUDA package URLs, UTF-16 encoding errors in `requirements.txt`).
- **Reproducibility Auditing:** Authored the `REPRO_AUDIT.md` document, logging 14 distinct reproducibility failures categorized by severity.
- **Automation:** Developed the `reproduce.sh` script and `Makefile` to allow single-command execution of the RAG pipeline.
- **Dockerization:** Created a multi-stage Dockerfile with `.dockerignore` to completely isolate the execution environment from local state drift.

### Daniel Evans (Member 2) — LLM Integration & Security Enhancements

**Primary Ownership:** Multi-Model Support and API Management

- **Model Integration:** Integrated Anthropic and Gemini as alternative LLM providers in addition to the existing OpenAI implementation, creating dedicated modules for each (`ReMindRag/llms/anthropic_api.py`, `ReMindRag/llms/gemini_api.py`).
- **UI Updates:** Updated the Streamlit application interface (`streamlit_app.py`) to allow dynamic, on-the-fly selection of the active language model.
- **Security & Credential Management:** Hardened repository security by removing `api_key.json` from version control tracking, configuring `.gitignore`, and providing an `api_key.json.example` template for user setup.

### Joel Vinas (Member 3) — Advanced Prompting & Quality Enhancements

**Primary Ownership:** LLM Reasoning and System Robustness

- **Reasoning Injection:** Designed the upgrade path for implementing Chain-of-Thought (CoT) and Self-Consistency methodologies into the entity extraction agent.
- **Cost Control:** Refactored the evaluation orchestrator (`eval_LooGLE.py`) to allow dynamic switching of the judge model (e.g., overriding hardcoded `gpt-4o` with `gpt-4o-mini`).
- **System Improvements:** Integrated the `--seed` parameter globally to enforce strict deterministic randomness across all PyTorch and NumPy operations.
