"""
tests/test_repro_variance.py — Reproducibility Variance Test Suite
====================================================================
4 unittest tests that run the same query TWICE and verify determinism:

  1. test_1_same_chunks          — chunk content is identical across runs
  2. test_2_same_chunk_count     — number of retrieved chunks is identical
  3. test_3_same_nodes           — KG node set is stable after both queries
  4. test_4_deterministic_answer — generated answer string is identical

Model: sentence-transformers/all-MiniLM-L6-v2  (no HF token needed)
LLM:   MockAgent (zero API cost)

Run:
    python tests/test_repro_variance.py
    python tests/test_repro_variance.py -v
"""

import os
import sys
import shutil
import tempfile
import unittest
from datetime import datetime
from typing import List, Dict

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, REPO_ROOT)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE = os.path.join(REPO_ROOT, "repro_model_cache")
EXAMPLE_DATA = os.path.join(REPO_ROOT, "example", "example_data.txt")

# Fixed query used for all variance tests
REPRO_QUERY = "What does a level 20 paladin gain?"


# ---------------------------------------------------------------------------
# Mock LLM — fully deterministic, zero API cost
# ---------------------------------------------------------------------------
from ReMindRag.llms import AgentBase


class MockAgent(AgentBase):
    """Deterministic stub — same input always produces same output."""

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
            "At level 20, a paladin gains the Sacred Oath capstone feature.\n"
            "```"
        )


# ---------------------------------------------------------------------------
# Module-level state: one RAG instance + two query results
# ---------------------------------------------------------------------------
_rag = None
_tmp_dir = None
_run1 = None   # (response, chunks, edges)
_run2 = None


def setUpModule():
    global _rag, _tmp_dir, _run1, _run2
    from ReMindRag.embeddings import HgEmbedding
    from ReMindRag.chunking import NaiveChunker
    from ReMindRag import ReMindRag
    from transformers import AutoTokenizer

    if not os.path.exists(EXAMPLE_DATA):
        raise unittest.SkipTest(f"example_data.txt not found at {EXAMPLE_DATA}")

    os.makedirs(MODEL_CACHE, exist_ok=True)
    _tmp_dir = tempfile.mkdtemp(prefix="repro_var_")
    logs_dir = os.path.join(_tmp_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    agent = MockAgent()
    embedding = HgEmbedding(MODEL_NAME, MODEL_CACHE)
    chunker = NaiveChunker(MODEL_NAME, MODEL_CACHE, max_token_length=200)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE)

    _rag = ReMindRag(
        logger_level=30,
        log_path=os.path.join(logs_dir, "repro_var.log"),
        chunk_agent=agent,
        kg_agent=agent,
        generate_agent=agent,
        embedding=embedding,
        chunker=chunker,
        tokenizer=tokenizer,
        database_description="D&D Player Handbook — Repro Variance Test",
        save_dir=_tmp_dir,
    )
    _rag.load_file(EXAMPLE_DATA, language="en")

    # Run the same query twice — results must be identical
    r1, c1, e1 = _rag.generate_response(REPRO_QUERY, force_do_rag=True)
    r2, c2, e2 = _rag.generate_response(REPRO_QUERY, force_do_rag=True)
    _run1 = (r1, c1, e1)
    _run2 = (r2, c2, e2)


def tearDownModule():
    if _tmp_dir:
        shutil.rmtree(_tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helper: normalise a chunk to a comparable key
# ---------------------------------------------------------------------------
def _chunk_key(chunk) -> str:
    if isinstance(chunk, str):
        return chunk.strip()
    if isinstance(chunk, dict):
        return str(chunk.get("id", chunk))
    return str(chunk)


# ---------------------------------------------------------------------------
# Test 1 — Same chunks
# ---------------------------------------------------------------------------
class Test1SameChunks(unittest.TestCase):
    """The set of retrieved chunks must be identical across two identical queries."""

    def test_1_same_chunks(self):
        _, c1, _ = _run1
        _, c2, _ = _run2
        keys1 = sorted(_chunk_key(ch) for ch in c1)
        keys2 = sorted(_chunk_key(ch) for ch in c2)
        self.assertEqual(
            keys1, keys2,
            f"Chunk content differs.\nRun 1: {keys1[:3]}\nRun 2: {keys2[:3]}"
        )


# ---------------------------------------------------------------------------
# Test 2 — Same chunk count
# ---------------------------------------------------------------------------
class Test2SameChunkCount(unittest.TestCase):
    """Number of retrieved chunks must be identical across two identical queries."""

    def test_2_same_chunk_count(self):
        _, c1, _ = _run1
        _, c2, _ = _run2
        self.assertEqual(
            len(c1), len(c2),
            f"Chunk count differs: run 1={len(c1)}, run 2={len(c2)}"
        )


# ---------------------------------------------------------------------------
# Test 3 — Same KG nodes
# ---------------------------------------------------------------------------
class Test3SameNodes(unittest.TestCase):
    """KG node set must be stable across consecutive refreshes."""

    def test_3_same_nodes(self):
        if not hasattr(_rag.kg, "graph") or _rag.kg.graph is None:
            self.skipTest("KG .graph attribute not available")

        _rag.refresh_kg()
        nodes_a = set(str(n) for n in _rag.kg.graph.nodes())

        _rag.refresh_kg()
        nodes_b = set(str(n) for n in _rag.kg.graph.nodes())

        # Node set must not change between two consecutive refreshes
        self.assertEqual(
            nodes_a, nodes_b,
            f"KG node set changed after a second refresh — graph is not stable.\n"
            f"Before: {sorted(nodes_a)[:10]}\nAfter: {sorted(nodes_b)[:10]}"
        )

        # After loading a full document the graph must have at least one node
        # (nodes are keyed by numeric chunk ID, not by entity name)
        self.assertGreater(
            len(nodes_a), 0,
            "KG has no nodes after loading example_data.txt — check entity extraction"
        )


# ---------------------------------------------------------------------------
# Test 4 — Deterministic answer
# ---------------------------------------------------------------------------
class Test4DeterministicAnswer(unittest.TestCase):
    """Generated answer string must be identical for the same query."""

    def test_4_deterministic_answer(self):
        r1, _, _ = _run1
        r2, _, _ = _run2
        self.assertIsInstance(r1, str, "Run 1 response must be str")
        self.assertIsInstance(r2, str, "Run 2 response must be str")
        self.assertEqual(
            r1, r2,
            f"Answer is non-deterministic.\nRun 1: {r1!r}\nRun 2: {r2!r}"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("  ReMindRAG Reproducibility Variance Test Suite")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        Test1SameChunks,
        Test2SameChunkCount,
        Test3SameNodes,
        Test4DeterministicAnswer,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    if result.wasSuccessful():
        print("  [PASS] ALL 4 REPRO VARIANCE TESTS PASSED.")
    else:
        failed = len(result.failures) + len(result.errors)
        print(f"  [FAIL] {failed} TEST(S) FAILED — review output above.")
    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)
