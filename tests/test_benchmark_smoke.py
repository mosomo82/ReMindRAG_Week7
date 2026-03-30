"""
tests/test_benchmark_smoke.py — Benchmark Smoke Test Suite
===========================================================
5 unittest tests that validate the full ReMindRAG pipeline using a mock LLM.
Zero API cost: no OpenAI / HuggingFace token required.

Tests:
  1. test_1_pipeline_init        — ReMindRag object is created without errors
  2. test_2_chunk_retrieval      — load_file + generate_response returns ≥1 chunk
  3. test_3_keyword_overlap      — retrieved chunks contain keywords from the query
  4. test_4_answer_quality       — response string is non-empty, plain text (no fences)
  5. test_5_eval_schema          — generate_response returns the (str, list, list) schema

Run:
    python tests/test_benchmark_smoke.py
    python tests/test_benchmark_smoke.py -v
    # or from repo root:
    python -m pytest tests/test_benchmark_smoke.py -v   (if pytest is installed)
"""

import os
import sys
import shutil
import tempfile
import unittest
from datetime import datetime
from typing import List, Dict

# ---------------------------------------------------------------------------
# Path setup — works regardless of working directory
# ---------------------------------------------------------------------------
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, REPO_ROOT)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE = os.path.join(REPO_ROOT, "repro_model_cache")
EXAMPLE_DATA = os.path.join(REPO_ROOT, "example", "example_data.txt")

# ---------------------------------------------------------------------------
# Mock LLM — no API key, deterministic, zero cost
# ---------------------------------------------------------------------------
from ReMindRag.llms import AgentBase


class MockAgent(AgentBase):
    """Deterministic stub — mimics the real agent's response format."""

    def generate_response(self, system_prompt: str, chat_history: List[Dict]) -> str:
        sp = system_prompt.lower().replace("\n", " ")
        if "extract" in sp or "entity" in sp:
            return (
                '```json\n'
                '{"entities": [{"name": "Paladin", "type": "Class"}], "relations": []}\n'
                '```'
            )
        if "sufficient information" in sp:
            return "```cot-ans\nyes\n```"
        return (
            "```cot-ans\n"
            "A level 20 paladin gains the Sacred Oath capstone feature "
            "along with additional Divine Smite uses and Aura improvements.\n"
            "```"
        )


# ---------------------------------------------------------------------------
# Shared RAG instance (built once for the whole module)
# ---------------------------------------------------------------------------
_rag_instance = None
_tmp_dir = None


def _build_rag():
    """
    Builds and caches a ReMindRag instance with MockAgent and all-MiniLM-L6-v2.
    Loads example_data.txt into the RAG store.
    Returns (rag, tmp_dir) or raises unittest.SkipTest.
    """
    from ReMindRag.embeddings import HgEmbedding
    from ReMindRag.chunking import NaiveChunker
    from ReMindRag import ReMindRag
    from transformers import AutoTokenizer

    if not os.path.exists(EXAMPLE_DATA):
        raise unittest.SkipTest(f"example_data.txt not found at {EXAMPLE_DATA}")

    os.makedirs(MODEL_CACHE, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix="bench_smoke_")
    logs_dir = os.path.join(tmp_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    embedding = HgEmbedding(MODEL_NAME, MODEL_CACHE)
    chunker = NaiveChunker(MODEL_NAME, MODEL_CACHE, max_token_length=200)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE)
    agent = MockAgent()

    rag = ReMindRag(
        logger_level=30,
        log_path=os.path.join(logs_dir, "bench.log"),
        chunk_agent=agent,
        kg_agent=agent,
        generate_agent=agent,
        embedding=embedding,
        chunker=chunker,
        tokenizer=tokenizer,
        database_description="D&D Player Handbook — Benchmark Smoke Test",
        save_dir=tmp_dir,
    )
    rag.load_file(EXAMPLE_DATA, language="en")
    return rag, tmp_dir


def setUpModule():
    global _rag_instance, _tmp_dir
    _rag_instance, _tmp_dir = _build_rag()


def tearDownModule():
    if _tmp_dir:
        shutil.rmtree(_tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 1 — Pipeline initialises without errors
# ---------------------------------------------------------------------------
class Test1PipelineInit(unittest.TestCase):
    """ReMindRag instance must be created successfully."""

    def test_1_pipeline_init(self):
        self.assertIsNotNone(_rag_instance, "rag_instance is None")
        self.assertTrue(hasattr(_rag_instance, "database"), "Missing .database")
        self.assertTrue(hasattr(_rag_instance, "kg"), "Missing .kg")
        self.assertTrue(hasattr(_rag_instance, "preprocess"), "Missing .preprocess")


# ---------------------------------------------------------------------------
# Test 2 — Chunk retrieval
# ---------------------------------------------------------------------------
class Test2ChunkRetrieval(unittest.TestCase):
    """generate_response must retrieve ≥1 chunk from the loaded document."""

    def test_2_chunk_retrieval(self):
        query = "What does a level 20 paladin gain?"
        _, chunks, _ = _rag_instance.generate_response(query, force_do_rag=True)
        self.assertGreaterEqual(
            len(chunks), 1,
            f"Expected ≥1 chunk for '{query}', got {len(chunks)}"
        )


# ---------------------------------------------------------------------------
# Test 3 — Keyword overlap
# ---------------------------------------------------------------------------
class Test3KeywordOverlap(unittest.TestCase):
    """Retrieved chunk text must contain at least one keyword from the query."""

    def test_3_keyword_overlap(self):
        query = "What abilities does a paladin have?"
        keywords = {"paladin", "ability", "abilities", "divine", "smite", "oath"}
        _, chunks, _ = _rag_instance.generate_response(query, force_do_rag=True)

        if not chunks:
            self.skipTest("No chunks retrieved — skipping keyword overlap check")

        combined = ""
        for ch in chunks:
            if isinstance(ch, str):
                combined += ch.lower()
            elif isinstance(ch, dict):
                combined += " ".join(str(v) for v in ch.values()).lower()

        overlap = keywords & set(combined.split())
        self.assertGreater(
            len(overlap), 0,
            f"No keyword overlap found. Keywords: {keywords}. "
            f"First 200 chars: {combined[:200]!r}"
        )


# ---------------------------------------------------------------------------
# Test 4 — Answer quality
# ---------------------------------------------------------------------------
class Test4AnswerQuality(unittest.TestCase):
    """Response must be non-empty plain text with no raw markdown fences."""

    def test_4_answer_quality(self):
        query = "What is the paladin's Sacred Oath feature?"
        response, _, _ = _rag_instance.generate_response(query, force_do_rag=True)

        self.assertIsInstance(response, str, "Response must be a str")
        self.assertGreater(len(response.strip()), 0, "Response is empty")
        self.assertNotIn(
            "```", response,
            "Response contains raw markdown fences — pipeline should strip them"
        )


# ---------------------------------------------------------------------------
# Test 5 — Eval schema
# ---------------------------------------------------------------------------
class Test5EvalSchema(unittest.TestCase):
    """generate_response must return a 3-tuple of (str, list, list)."""

    def test_5_eval_schema(self):
        query = "Describe the paladin class."
        result = _rag_instance.generate_response(query, force_do_rag=True)

        self.assertIsInstance(result, tuple, "Return value must be a tuple")
        self.assertEqual(len(result), 3, "Return tuple must have 3 elements")

        response, chunks, edges = result
        self.assertIsInstance(response, str, "result[0] must be str")
        self.assertIsInstance(chunks, list, "result[1] must be list")
        self.assertIsInstance(edges, list, "result[2] must be list")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("  ReMindRAG Benchmark Smoke Test Suite")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        Test1PipelineInit,
        Test2ChunkRetrieval,
        Test3KeywordOverlap,
        Test4AnswerQuality,
        Test5EvalSchema,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    if result.wasSuccessful():
        print("  [PASS] ALL 5 BENCHMARK SMOKE TESTS PASSED.")
    else:
        failed = len(result.failures) + len(result.errors)
        print(f"  [FAIL] {failed} TEST(S) FAILED — review output above.")
    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)
