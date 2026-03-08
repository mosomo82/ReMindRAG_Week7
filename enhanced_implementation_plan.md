# Lab 7 Enhancement Plan — ReMindRAG

> **Team:** Tony Nguyen (`mosomo82`), Member A (`member-a`), Member B (`member-b`)
> **Repository:** `kilgrims/ReMindRAG`
> **Base Branch:** `main`

---

## Current State (Already Completed by Tony)

| File | Status | Owner |
|---|---|---|
| `ReMindRag/rag_main.py` | ✅ Fixed — makedirs, os.path.join, typo fix | Tony |
| `ReMindRag/generator/preprocess.py` | ✅ Fixed — `daatabase_description` typo | Tony |
| `ReMindRag/webui/webui.py` | ✅ Fixed — temp/ dir creation | Tony |
| `example/example.py` | ✅ Fixed — `__file__`-relative paths | Tony |
| `eval/eval_LooGLE.py` | ✅ Fixed — paths, `--judge_model_name`, `--seed` | Tony |
| `eval/eval_Hotpot.py` | ✅ Fixed — paths, `--judge_model_name` | Tony |
| `eval/start_LooGLE.py` | ✅ Fixed — resume guard added | Tony |
| `requirements_repro.txt` | ✅ Fixed — UTF-8 re-encoded | Tony |
| `README.md` | ✅ Updated — install docs, HF auth, trust_remote_code | Tony |
| `Dockerfile` | ✅ Created — multi-stage, NLTK data, runtime dirs | Tony |
| `.dockerignore` | ✅ Created — excludes stale cache data | Tony |
| `test_reproducibility.py` | ✅ Created — 17 automated tests, all passing | Tony |
| `repro_test.py` | ✅ Verified — mock smoke test passes | Tony |

**Existing eval structure:** Binary pass/fail via GPT-4o judge. No quantitative metrics.

---

## Updated Division of Labor

| Member | Scope | New Files | Modified Files |
|---|---|---|---|
| **Member A** | Evaluation metrics & baseline comparison | `eval/metrics/__init__.py`, `eval/metrics/faithfulness.py`, `eval/baseline_rag.py` | `eval/eval_LooGLE.py`, `eval/eval_Hotpot.py`, `requirements.txt` |
| **Member B** | Prompting improvements & LLM caching | `ReMindRag/generator/self_consistency.py`, `ReMindRag/llms/cached_agent.py` | `ReMindRag/generator/prompts.py`, `ReMindRag/generator/preprocess.py`, `ReMindRag/database/data_extract.py`, `eval/eval_LooGLE.py` |

> [!IMPORTANT]
> After this work the project will have: **6 quantitative metrics** (ROUGE-L, BLEU, BERTScore, faithfulness, retrieval precision, latency) + **naive RAG baseline** + **CoT prompting** + **self-consistency voting** + **LLM response caching**.

---

## Phase 0: Git Setup (Both Members)

### 0.1 — Clone & Branch
```bash
git clone https://github.com/kilgrims/ReMindRAG.git
cd ReMindRAG
git checkout main && git pull origin main

# Each member creates their branch:
git checkout -b member-a/eval-metrics     # Member A
git checkout -b member-b/prompt-quality   # Member B
```

### 0.2 — Local Environment
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

### 0.3 — Verify Existing Code
```bash
python test_reproducibility.py   # All 17 tests pass
python repro_test.py             # SUCCESS: RAG pipeline retrieved chunks
```

---

## Phase 1: Evaluation Metrics (Member A)

---

### 👤 Member A — 6 New Metrics + Baseline

#### Step 1: Install New Dependencies

```bash
pip install rouge-score bert-score
```

Add to `requirements.txt`:
```
rouge-score==0.1.2
bert-score==0.3.13
```

Commit:
```bash
git add requirements.txt
git commit -m "deps: Add rouge-score and bert-score for evaluation metrics"
```

---

#### Step 2: Create `eval/metrics/__init__.py`

```python
"""Evaluation metrics for ReMindRAG."""
```

---

#### Step 3: Create `eval/metrics/faithfulness.py`

```python
"""LLM-based faithfulness scoring — detects hallucination."""
import json

FAITHFULNESS_PROMPT = """
You are evaluating whether an AI answer is fully supported by the retrieved context.

Context:
{context}

Answer:
{answer}

Does the answer contain ONLY information present in the context?
Reply with JSON: {{"faithful": true/false, "unsupported_claims": ["..."]}}
"""

def faithfulness_score(agent, context_chunks: list, answer: str) -> dict:
    context = "\n\n".join(str(c) for c in context_chunks)
    response = agent.generate_response(
        "", [{"role": "user", "content": FAITHFULNESS_PROMPT.format(
            context=context, answer=answer)}]
    )
    try:
        # Extract JSON from response
        start = response.index("{")
        end = response.rindex("}") + 1
        return json.loads(response[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"faithful": None, "unsupported_claims": [], "parse_error": True}
```

Commit:
```bash
git add eval/metrics/__init__.py eval/metrics/faithfulness.py
git commit -m "feat: Add faithfulness scorer for hallucination detection"
```

---

#### Step 4: Add ROUGE-L, BLEU, BERTScore, Retrieval Precision, Latency to `eval/eval_LooGLE.py`

Add these imports at the top of `eval/eval_LooGLE.py`:

```python
import time
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smooth = SmoothingFunction().method1
```

Add these helper functions below `set_seed()`:

```python
def compute_rouge_bleu(prediction: str, reference: str) -> dict:
    """Compute ROUGE-L and BLEU between prediction and reference."""
    rouge_result = rouge.score(reference, prediction)['rougeL'].fmeasure
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
    return {"rougeL": round(rouge_result, 4), "bleu": round(bleu, 4)}

def retrieval_precision(retrieved_chunks: dict, gold_answer: str,
                        threshold: float = 0.3) -> dict:
    """Check what fraction of retrieved chunks are relevant to the gold answer."""
    gold_keywords = set(gold_answer.lower().split())
    hits = sum(
        1 for chunk in retrieved_chunks.values()
        if len(gold_keywords & set(chunk.lower().split())) / max(len(gold_keywords), 1) >= threshold
    )
    precision = hits / max(len(retrieved_chunks), 1)
    return {"retrieval_precision": round(precision, 4),
            "chunks_retrieved": len(retrieved_chunks)}
```

Inside the evaluation loop, **wrap the query call with timing** and **log all metrics**:

```python
# Before the query call:
start_time = time.perf_counter()

# (existing query call here)

# After getting the response:
latency_ms = int((time.perf_counter() - start_time) * 1000)

# Compute metrics against gold answer:
text_metrics = compute_rouge_bleu(response, gold_answer)
ret_metrics = retrieval_precision(chunk_summaries, gold_answer)

# Add to the result dict:
result["rougeL"] = text_metrics["rougeL"]
result["bleu"] = text_metrics["bleu"]
result["retrieval_precision"] = ret_metrics["retrieval_precision"]
result["chunks_retrieved"] = ret_metrics["chunks_retrieved"]
result["latency_ms"] = latency_ms
```

After the evaluation loop ends, compute BERTScore over all predictions:

```python
from bert_score import score as bert_score_fn

all_predictions = [r["response"] for r in results]
all_references = [r["gold_answer"] for r in results]

if all_predictions:
    P, R, F1 = bert_score_fn(all_predictions, all_references,
                              lang="en", model_type="distilbert-base-uncased",
                              verbose=False)
    avg_bertscore = round(F1.mean().item(), 4)
    print(f"Average BERTScore F1: {avg_bertscore}")

    # Write summary
    summary = {
        "avg_rougeL": round(sum(r["rougeL"] for r in results) / len(results), 4),
        "avg_bleu": round(sum(r["bleu"] for r in results) / len(results), 4),
        "avg_bertscore_f1": avg_bertscore,
        "avg_latency_ms": int(sum(r["latency_ms"] for r in results) / len(results)),
        "total_questions": len(results)
    }
```

Commit:
```bash
git add eval/eval_LooGLE.py
git commit -m "feat: Add ROUGE-L, BLEU, BERTScore, retrieval precision, latency metrics"
```

---

#### Step 5: Create `eval/baseline_rag.py` — Naive RAG Baseline

```python
"""
Flat ChromaDB similarity search baseline (no knowledge graph traversal).
Query -> embed -> top-k chunks -> generate answer directly.
Uses identical prompts to ReMindRAG for fair comparison.
"""
import sys, os, json, time, argparse
from datetime import datetime

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EVAL_DIR)
sys.path.insert(0, REPO_ROOT)

from ReMindRag.llms import OpenaiAgent
from ReMindRag.embeddings import HgEmbedding
from ReMindRag.generator.prompts import generate_rag_ans_prompt

def naive_rag_query(database, embedding, generate_agent, query, top_k=5):
    """Simple vector search -> LLM generation, no graph traversal."""
    query_emb = embedding.sentence_embedding(query)

    # Direct ChromaDB similarity search
    results = database.entity_collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )

    if results and results['documents'] and results['documents'][0]:
        context = "\n\n".join(results['documents'][0])
    else:
        context = "No relevant context found."

    prompt = generate_rag_ans_prompt.format(
        chat_history="", query=query,
        rag_summary=context, edges=""
    )
    response = generate_agent.generate_response(
        "", [{"role": "user", "content": prompt}]
    )
    return response, context

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Naive RAG Baseline")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    # Initialize agents (same as eval_LooGLE.py)
    api_key_path = os.path.join(REPO_ROOT, "api_key.json")
    with open(api_key_path, "r") as f:
        api_keys = json.load(f)
    agent = OpenaiAgent(
        api_key=api_keys[0]["api_key"],
        base_url=api_keys[0]["base_url"],
        llm_model_name="gpt-4o-mini"
    )

    model_cache = os.path.join(REPO_ROOT, "model_cache")
    embedding = HgEmbedding("nomic-ai/nomic-embed-text-v2-moe", model_cache)

    print(f"Query: {args.query}")
    start = time.perf_counter()
    response, context = naive_rag_query(None, embedding, agent, args.query, args.top_k)
    latency = int((time.perf_counter() - start) * 1000)
    print(f"Response: {response}")
    print(f"Latency: {latency}ms")
```

Commit:
```bash
git add eval/baseline_rag.py
git commit -m "feat: Add naive RAG baseline for comparison with ReMindRAG"
```

---

#### Step 6: Push & Open PR
```bash
git push origin member-a/eval-metrics
```
**Open PR** → base: `main` ← compare: `member-a/eval-metrics`
- Title: `feat: Add 6 evaluation metrics + naive RAG baseline`
- Tag Member B for review

---

## Phase 2: Prompting & Quality (Member B)

---

### 👤 Member B — CoT + Self-Consistency + Few-Shot + Caching

#### Step 1: Update `ReMindRag/generator/prompts.py` — Chain-of-Thought

Replace the existing `generate_rag_ans_prompt`:

```python
generate_rag_ans_prompt = """
You are answering a question using retrieved knowledge graph context.

INSTRUCTIONS:
1. Think step-by-step through the retrieved chunks and relationship edges.
2. Identify which chunks are most relevant to the question.
3. Reason through any multi-hop connections in the edges.
4. Synthesize a final answer grounded ONLY in the retrieved context.

Retrieved Chunks:
{rag_summary}

Knowledge Graph Edges:
{edges}

Chat History:
{chat_history}

Question: {query}

```cot-ans
[STEP 1 - Relevant chunks]: Identify which chunks contain information about the question.
[STEP 2 - Edge reasoning]: Trace relationships through the knowledge graph edges.
[STEP 3 - Final answer]: Synthesize a grounded answer from the above reasoning.
```
"""
```

Commit:
```bash
git add ReMindRag/generator/prompts.py
git commit -m "feat: Add Chain-of-Thought structure to RAG answer prompt"
```

---

#### Step 2: Add Few-Shot Examples to Entity Extraction

In `ReMindRag/database/data_extract.py`, prepend few-shot examples to the entity extraction prompt:

```python
FEW_SHOT_EXAMPLES = """
Example 1:
Text: "Albert Einstein developed the theory of relativity in 1905."
Output:
{"entities": [{"name": "Albert Einstein", "type": "Person"},
              {"name": "theory of relativity", "type": "Concept"}],
 "relations": [{"source": "Albert Einstein", "relation": "developed",
                "target": "theory of relativity"}]}

Example 2:
Text: "The Eiffel Tower is located in Paris, France."
Output:
{"entities": [{"name": "Eiffel Tower", "type": "Landmark"},
              {"name": "Paris", "type": "Location"}],
 "relations": [{"source": "Eiffel Tower", "relation": "located_in",
                "target": "Paris"}]}

Now extract entities and relations from the following text:
"""

# Prepend to entity_extract_prompt in generate_entity_response():
def generate_entity_response(agent, chunk):
    full_prompt = FEW_SHOT_EXAMPLES + entity_extract_prompt
    response = agent.generate_response(full_prompt,
        [{"role": "user", "content": chunk["content"]}])
    ...
```

Commit:
```bash
git add ReMindRag/database/data_extract.py
git commit -m "feat: Add few-shot examples to entity extraction for stabler JSON output"
```

---

#### Step 3: Create `ReMindRag/generator/self_consistency.py`

```python
"""Self-Consistency voting for improved multi-hop QA accuracy.

Reference: Wang et al., 2022 — "Self-Consistency Improves Chain of
Thought Reasoning in Language Models"
"""
import collections
from ..llms import AgentBase

def self_consistent_answer(agent: AgentBase, system_prompt: str,
                           user_msg: str, n_samples: int = 3) -> str:
    """
    Generate n candidate answers, then majority-vote on the final answer.
    Falls back to first sample if answers are too diverse to cluster.

    Args:
        agent: Any AgentBase-compatible LLM agent.
        system_prompt: System prompt for the generation.
        user_msg: The user message / query.
        n_samples: Number of candidate answers to generate.

    Returns:
        The most common answer among the n samples.
    """
    responses = []
    for _ in range(n_samples):
        r = agent.generate_response(system_prompt,
                                    [{"role": "user", "content": user_msg}])
        responses.append(r.strip())

    # Majority vote: pick the most common response
    counter = collections.Counter(responses)
    return counter.most_common(1)[0][0]
```

Wire into `PreProcessing.generate_temp_response` — add optional flag:

```python
# In ReMindRag/generator/preprocess.py, modify generate_temp_response:
def generate_temp_response(self, chat_history_str, system_prompt,
                           rewritten_query, chunk_summary, edges,
                           use_self_consistency=False, sc_samples=3,
                           error_chat_history=None):
    # ... (existing chunk/edges formatting) ...

    input_msg = generate_rag_ans_prompt.format(...)

    if use_self_consistency and sc_samples > 1:
        from .self_consistency import self_consistent_answer
        response = self_consistent_answer(self.agent, system_prompt,
                                          input_msg, n_samples=sc_samples)
    else:
        response = self.agent.generate_response(system_prompt,
            [{"role": "user", "content": input_msg}] + (error_chat_history or []))
    return response
```

Commit:
```bash
git add ReMindRag/generator/self_consistency.py ReMindRag/generator/preprocess.py
git commit -m "feat: Add self-consistency voting with configurable sample count"
```

---

#### Step 4: Create `ReMindRag/llms/cached_agent.py`

```python
"""Disk-based LLM response cache for reproducible evaluation runs.

Hashes (system_prompt + chat_history) → caches response to disk.
Identical inputs always return identical outputs, eliminating API
non-determinism across evaluation runs.
"""
import hashlib
import json
import os
from . import AgentBase

class CachedAgent(AgentBase):
    """Wraps any AgentBase and caches responses to disk by prompt hash."""

    def __init__(self, agent: AgentBase, cache_dir: str = "./llm_cache"):
        self.agent = agent
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._hits = 0
        self._misses = 0

    def _cache_key(self, system_prompt: str, chat_history: list) -> str:
        payload = json.dumps({"s": system_prompt, "h": chat_history},
                             sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def generate_response(self, system_prompt: str, chat_history: list) -> str:
        key = self._cache_key(system_prompt, chat_history)
        cache_file = os.path.join(self.cache_dir, f"{key}.txt")

        if os.path.exists(cache_file):
            self._hits += 1
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()

        self._misses += 1
        response = self.agent.generate_response(system_prompt, chat_history)
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(response)
        return response

    def stats(self) -> dict:
        return {"cache_hits": self._hits, "cache_misses": self._misses,
                "hit_rate": round(self._hits / max(self._hits + self._misses, 1), 2)}
```

Commit:
```bash
git add ReMindRag/llms/cached_agent.py
git commit -m "feat: Add CachedAgent wrapper for deterministic eval reruns"
```

---

#### Step 5: Upgrade Judge Prompt in `eval/eval_LooGLE.py`

Replace the binary pass/fail judge with a rubric-based scorer:

```python
ANS_CHECK_PROMPT_V2 = """
You are evaluating an AI system's answer against a gold reference.

Gold Answer: {gold_answer}
System Answer: {system_answer}

Score the system answer on these dimensions (each 0-2):
1. Factual Correctness: Is the core fact correct?
2. Completeness: Does it cover all important aspects of the gold answer?
3. Conciseness: Is it free of irrelevant or hallucinated content?

Reply ONLY with JSON:
{{
  "factual_correctness": <0-2>,
  "completeness": <0-2>,
  "conciseness": <0-2>,
  "total": <0-6>,
  "pass": <true if total >= 4 else false>,
  "reasoning": "one sentence explanation"
}}
"""
```

Update the judge agent call to use `ANS_CHECK_PROMPT_V2` and parse the structured JSON response.

Commit:
```bash
git add eval/eval_LooGLE.py
git commit -m "feat: Upgrade judge to rubric-based scoring (0-6 scale)"
```

---

#### Step 6: Push & Open PR
```bash
git push origin member-b/prompt-quality
```
**Open PR** → base: `main` ← compare: `member-b/prompt-quality`
- Title: `feat: CoT prompting, self-consistency, few-shot extraction, LLM caching`
- Tag Member A for review

---

## Phase 3: Integration & Verification (Both Members)

> [!IMPORTANT]
> **Merge Phase 1 and Phase 2 PRs first**, then both members verify on a single branch.

### 3.1 — Create Verification Branch
```bash
git checkout main && git pull origin main
git checkout -b team/verification
```

### 3.2 — Sub-Tasks by Member

| Sub-Task | Owner | Description |
|---|---|---|
| Run LooGLE eval with all metrics | **Member A** | Execute `eval_LooGLE.py` on 5 titles, collect JSON results with all 6 metrics |
| Run baseline comparison | **Member A** | Run `baseline_rag.py` on same questions, compute delta vs ReMindRAG |
| Run eval with CoT + self-consistency | **Member B** | Execute same eval with `use_self_consistency=True`, compare accuracy boost |
| Run eval with CachedAgent | **Member B** | Execute eval twice — verify second run has 100% cache hits and identical outputs |
| Generate comparison table | **Both** | Merge results into a single comparison markdown table |

### 3.3 — Expected Output Format

Each question produces a JSON result:
```json
{
  "question": "What was the main cause of...?",
  "gold_answer": "The main cause was...",
  "remindrag_answer": "Based on the context...",
  "judge_pass": true,
  "judge_score": {"factual_correctness": 2, "completeness": 2, "conciseness": 1, "total": 5},
  "rougeL": 0.42,
  "bleu": 0.18,
  "bertscore_f1": 0.76,
  "faithfulness": true,
  "retrieval_precision": 0.80,
  "latency_ms": 3200,
  "baseline_answer": "The document mentions...",
  "baseline_rougeL": 0.28,
  "baseline_judge_score": 3
}
```

### 3.4 — Commit Results
```bash
git add eval/results/
git commit -m "results: Full evaluation with 6 metrics + baseline comparison"
git push origin team/verification
```
Open PR → merge after both review.

---

## Files Modified / Created Summary

### Member A
| File | Change |
|---|---|
| `eval/eval_LooGLE.py` | Add ROUGE-L, BLEU, BERTScore, retrieval precision, latency per question |
| `eval/eval_Hotpot.py` | Mirror same metric additions |
| `eval/metrics/__init__.py` | **NEW** — package init |
| `eval/metrics/faithfulness.py` | **NEW** — LLM-based faithfulness scorer |
| `eval/baseline_rag.py` | **NEW** — naive RAG baseline for comparison |
| `requirements.txt` | Add `rouge-score==0.1.2`, `bert-score==0.3.13` |

### Member B
| File | Change |
|---|---|
| `ReMindRag/generator/prompts.py` | CoT structure in `generate_rag_ans_prompt` |
| `ReMindRag/database/data_extract.py` | Few-shot examples in entity extraction |
| `ReMindRag/generator/self_consistency.py` | **NEW** — self-consistency voting wrapper |
| `ReMindRag/generator/preprocess.py` | Wire `use_self_consistency` flag |
| `ReMindRag/llms/cached_agent.py` | **NEW** — disk-based LLM response cache |
| `eval/eval_LooGLE.py` | Upgrade judge prompt to rubric-based scoring |

---

## Verification Plan

```bash
# 1. All existing tests still pass
python test_reproducibility.py    # 17/17 OK

# 2. Smoke test with mock agents
python repro_test.py              # SUCCESS

# 3. Eval with metrics (requires OpenAI credits)
python eval/eval_LooGLE.py \
    --title_index 0 --data_type longdep_qa \
    --question_type origin --seed 42 \
    --judge_model_name gpt-4o

# 4. Verify metrics JSON output
cat eval/database/eval-long/0/input.json | python -m json.tool
# Should contain: rougeL, bleu, bertscore_f1, faithfulness, latency_ms

# 5. Verify cache works (Member B)
python eval/eval_LooGLE.py --title_index 0 ...   # First run: all misses
python eval/eval_LooGLE.py --title_index 0 ...   # Second run: all hits, same output

# 6. Docker build still works
docker build -t remindrag:latest .
docker run --rm remindrag:latest  # SUCCESS

# PR history
# Minimum 4 PRs: Member A metrics, Member B prompting, team verification, docs
```

---

## Final Submission Checklist

| Deliverable | File | Owner |
|---|---|---|
| Reproducibility fixes (14 issues) | Multiple files | Tony ✅ |
| Dockerfile + .dockerignore | `Dockerfile`, `.dockerignore` | Tony ✅ |
| Automated test suite | `test_reproducibility.py` | Tony ✅ |
| ROUGE-L, BLEU, BERTScore metrics | `eval/eval_LooGLE.py` | Member A |
| Faithfulness scorer | `eval/metrics/faithfulness.py` | Member A |
| Retrieval precision + latency | `eval/eval_LooGLE.py` | Member A |
| Naive RAG baseline | `eval/baseline_rag.py` | Member A |
| CoT prompting | `ReMindRag/generator/prompts.py` | Member B |
| Few-shot entity extraction | `ReMindRag/database/data_extract.py` | Member B |
| Self-consistency voting | `ReMindRag/generator/self_consistency.py` | Member B |
| LLM response caching | `ReMindRag/llms/cached_agent.py` | Member B |
| Rubric-based judge prompt | `eval/eval_LooGLE.py` | Member B |
| Evaluation results JSON | `eval/results/` | Both |
| Updated `requirements.txt` | `requirements.txt` | Member A |
| Updated `README.md` | `README.md` | Both |
| `CONTRIBUTIONS.md` | `CONTRIBUTIONS.md` | Both |

---

## Grading Impact

| Enhancement | Difficulty | Expected Impact |
|---|---|---|
| ROUGE-L / BLEU | Low | Shows rigorous quantitative eval |
| BERTScore | Low | Semantic quality — more credible than text overlap |
| Faithfulness | Medium | Directly addresses hallucination concern |
| Baseline comparison | Medium | **Highest impact** — proves the system works better than naive RAG |
| CoT prompting | Low-Medium | +3-8% accuracy improvement typical |
| Self-consistency | Medium | +2-5% on multi-hop questions |
| Few-shot extraction | Low | Reduces JSON parse errors, cleaner knowledge graph |
| Judge rubric | Low | More nuanced evaluation, publishable format |
| LLM caching | Low | Reproducibility + cost saving, shows engineering maturity |
