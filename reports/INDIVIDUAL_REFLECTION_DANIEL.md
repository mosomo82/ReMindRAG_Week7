# Individual Reflection: Daniel Evans

**Specific Technical Contributions**
While I initially planned to focus on Evaluation Metrics (ROUGE-L, BLEU, BERTScore) and creating scripts like `smoke_test.py`, unavoidable circumstances required me to pivot my focus to core system usability and security. My primary technical contribution was integrating Anthropic and Gemini as alternative LLM providers alongside the existing OpenAI implementation. To achieve this, I added modules (`ReMindRag/llms/anthropic_api.py` and `ReMindRag/llms/gemini_api.py`) and updated our Streamlit application (`streamlit_app.py`) to allow dynamic, on-the-fly model switching. Furthermore, I hardened the repository's security by establishing a secure API key handling protocol—removing tracked keys, updating `.gitignore`, and providing an `api_key.json.example` template.

**Challenges Encountered**
A significant challenge was securely managing API configurations for multiple providers while ensuring the Streamlit UI remained intuitive. Refactoring the system to load credentials locally via `api_key.json` instead of relying on hardcoded values or manual user input required careful state management updates. Additionally, we faced severe API rate limits and quotas, particularly on the free tiers for Gemini and Anthropic. This friction heavily motivated the need for dynamic model switching, allowing us to pivot to a different provider whenever one was temporarily exhausted.

**What I Learned About Reproducibility**
This lab taught me that a project is not truly reproducible if it relies exclusively on a single, expensive proprietary model or requires users to expose their personal credentials insecurely. I learned that reproducibility encompasses configuration security—if users have to hunt through code to replace hardcoded keys, the risk of accidental commits and broken pipelines increases. Standardizing secrets management and expanding model support drastically lowers the barrier to entry for others attempting to run our code.

**How Agentic AI Tools Influenced My Workflow**
Collaborating with Agentic AI tools accelerated the process of scaffolding new LLM wrappers. Instead of manually parsing the Anthropic and Gemini API documentation from scratch, the AI helped map their varied response formats into the standard structure our RAG pipeline expected. The AI was also instrumental in auditing the repository for security flaws, quickly identifying the need for a robust `.gitignore` and helping implement the `api_key.json.example` pattern.

**How Reproducing Related Work Strengthened My Understanding**
Adapting the ReMindRAG repository to work with alternative models exposed the mechanical dependencies of the original framework on OpenAI's specific formatting quirks. It taught me that components tightly coupled to a single provider make a system brittle. Building abstraction layers for different LLMs deepened my understanding of system architecture and the necessity of modular, loosely-coupled design in modern AI applications.
