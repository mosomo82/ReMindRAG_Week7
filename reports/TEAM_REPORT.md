# Lab 7: Team Report

## Division of Responsibilities

Our team successfully executed Lab 7 by dividing the workload into three distinct, non-overlapping phases, allowing for parallel execution and individual accountability.

## Project Component Ownership & Division of Labor Summary

To ensure rapid iteration and clear accountability, we mapped our division of labor directly to the core architectural components of the codebase:

| Project Component                | Primary Owner | Description of Responsibilities                                                                                                                                                            |
| :------------------------------- | :------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Environment & Infrastructure** | Tony Nguyen   | Authored `Makefile`, `reproduce.sh/ps1`, Dockerfile, `.env.template`, and test suite (`test_reproducibility.py`); fixed PyTorch/NLTK dependency bugs and Streamlit Cloud deployments.      |
| **Evaluation Metrics & Testing** | Daniel Evans  | Built `tests/smoke_test.py`, integrated quantitative metrics (ROUGE, BLEU, BERTScore), and designed entity faithfulness scoring templates.                                                 |
| **AI Agents & Cost Control**     | Joel Vinas    | Audited LLM orchestration (`eval_LooGLE.py`), integrated dynamic judge model selection (e.g., `gpt-4o-mini`), applied global graph deterministic seeds, and designed CoT injection points. |

### Tony Nguyen (Member 1) — Related Work Reproduction & Core Pipeline Audit

**Primary Ownership:** ReMindRAG Reproduction and Environmental Architecture

- **Repo Selection & Setup:** Selected ReMindRAG as the related work, cloned the repository, and audited the initial environment.
- **Dependency Management:** Identified and fixed critical containerization blockers (e.g., missing CUDA package URLs, UTF-16 encoding errors in `requirements.txt`).
- **Reproducibility Auditing:** Authored the `REPRO_AUDIT.md` document, logging 17 distinct reproducibility failures categorized by severity.
- **Automation:** Developed the `reproduce.sh` script (Linux/macOS) and `reproduce.ps1` (Windows/PowerShell) to allow single-command execution of the RAG pipeline.
- **Dockerization & Cloud Deployment:** Created a multi-stage Dockerfile and debugged Streamlit Cloud deployment failures by resolving PyTorch PyPI wheel conflicts (`+cu126`).
- **Dependency Auditing:** Fixed a hidden tokenizer crash where NLTK failed to chunk Markdown tables, and mapped missing architectural dependencies (`einops` for Nomic MoE).

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
