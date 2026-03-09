# Lab 7: Team Report

## Division of Responsibilities

Our team successfully executed Lab 7 by dividing the workload into three distinct, non-overlapping phases, allowing for parallel execution and individual accountability.

### Tony Nguyen (Member 1) — Related Work Reproduction & Core Pipeline Audit
**Primary Ownership:** ReMindRAG Reproduction and Environmental Architecture
*   **Repo Selection & Setup:** Selected ReMindRAG as the related work, cloned the repository, and audited the initial environment.
*   **Dependency Management:** Identified and fixed critical containerization blockers (e.g., missing CUDA package URLs, UTF-16 encoding errors in `requirements.txt`) and resolved 8 missing dependencies (`flask`, `streamlit`, `openai`, etc.) that crashed the original WebUI.
*   **Streamlit UI Integration:** Built a standalone interactive Streamlit application to visualize the RAG process, complete with real-time agentic trace logs and dynamic PyVis Knowledge Graph rendering.
*   **Visualization Bugfixes:** Refactored the core graph generation logic to remove rigid, hardcoded node coordinates, allowing PyVis's `ForceAtlas2` physics engine to correctly render circular, interactive graphs instead of static lines.
*   **Reproducibility Auditing:** Authored the `REPRO_AUDIT.md` document, logging 20 distinct reproducibility failures categorized by severity, including HuggingFace token restrictions and vector dimension mismatches.
*   **Automation:** Developed the `reproduce.sh` script and `Makefile` to allow single-command execution of the RAG pipeline.
*   **Dockerization:** Created a multi-stage Dockerfile with `.dockerignore` to completely isolate the execution environment from local state drift, configured to launch the UI on port 8501.

### Daniel Evans (Member 2) — Evaluation Metrics & Baseline Implementation
**Primary Ownership:** Quantitative Benchmarking
*   **Metric Expansion:** Designed the implementation plan to replace binary pass/fail RAG validation with robust quantitative metrics (ROUGE-L, BLEU, BERTScore).
*   **Testing Infrastructure:** Developed the `smoke_test.py` script to validate end-to-end traversal without requiring API keys.
*   **Faithfulness Scoring:** Prototyped the prompt template required to verify that extracted knowledge graph entities are factually grounded in the source text.

### Joel Vinas (Member 3) — Advanced Prompting & Quality Enhancements
**Primary Ownership:** LLM Reasoning and System Robustness
*   **Reasoning Injection:** Designed the upgrade path for implementing Chain-of-Thought (CoT) and Self-Consistency methodologies into the entity extraction agent.
*   **Cost Control:** Refactored the evaluation orchestrator (`eval_LooGLE.py`) to allow dynamic switching of the judge model (e.g., overriding hardcoded `gpt-4o` with `gpt-4o-mini`).
*   **System Improvements:** Integrated the `--seed` parameter globally to enforce strict deterministic randomness across all PyTorch and NumPy operations.
