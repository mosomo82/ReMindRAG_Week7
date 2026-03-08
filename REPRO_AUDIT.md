# REPRO_AUDIT.md — Reproducibility Audit: ReMindRAG

**Audit Date:** 2026-03-07
**Auditor:** Tony Nguyen (mosomo82)
**Automated Test Suite:** `tests/test_reproducibility.py` — **17/17 PASS**

---

## Summary

| Severity | Total Found | Fixed | Remaining |
|----------|-------------|-------|-----------|
| CRITICAL | 6 | 6 | 0 |
| HIGH | 4 | 4 | 0 |
| MEDIUM | 4 | 4 | 0 |
| **TOTAL** | **14** | **14** | **0** |

---

## CRITICAL — Would break on first run

| # | Issue | File | Fix Applied |
|---|-------|------|-------------|
| 1 | `requirements_repro.txt` encoded as UTF-16 LE — `pip` fails silently | `requirements_repro.txt` | Re-encoded to UTF-8 |
| 2 | Missing `--extra-index-url` for CUDA PyTorch wheels — `pip` cannot find packages | `README.md` | Documented in install steps |
| 3 | `model_cache/` directory never created by library — crashes on first run | `ReMindRag/rag_main.py` | `os.makedirs(exist_ok=True)` |
| 4 | `chroma_data/` directory never created by library — ChromaDB init fails | `ReMindRag/rag_main.py` | `os.makedirs(exist_ok=True)` |
| 5 | `logs/` directory never created — `FileNotFoundError` on first run | `example/example.py` | `os.makedirs(exist_ok=True)` |
| 6 | WebUI `temp/` directory never created — upload handler crashes | `ReMindRag/webui/webui.py` | `os.makedirs(exist_ok=True)` |

**Evidence:** Running `python example/example.py` on a fresh clone crashed with:
```
FileNotFoundError: [Errno 2] No such file or directory: './Rag_Cache/logs/...'
```
After fix: runs cleanly.

---

## HIGH — Silent failures or incorrect results

| # | Issue | File | Fix Applied |
|---|-------|------|-------------|
| 7 | Judge model hardcoded to `gpt-4o` — no way to use cheaper model | `eval/eval_LooGLE.py`, `eval/eval_Hotpot.py` | Added `--judge_model_name` CLI arg |
| 8 | LooGLE preprocessed dataset provenance undocumented — no Google Drive link | `README.md` | Added download link + instructions |
| 9 | `start_LooGLE.py` had no resume guard — re-ran all questions from scratch after interruption | `eval/start_LooGLE.py` | Added `if os.path.exists("input.json"): skip` guard |
| 10 | `daatabase_description` typo (3 a's) — silent wrong parameter assignment | `ReMindRag/rag_main.py`, `ReMindRag/generator/preprocess.py` | Fixed typo in both files |

**Evidence for #10:**
```python
# Before (wrong — extra 'a' meant the param was never used):
self.daatabase_description = database_description

# After (correct):
self.database_description = database_description
```

---

## MEDIUM — Environmental fragility

| # | Issue | File | Fix Applied |
|---|-------|------|-------------|
| 11 | String concatenation for paths (`save_dir + "/chroma_data"`) — breaks on Windows | `ReMindRag/rag_main.py` | Replaced with `os.path.join()` |
| 12 | Eval scripts used `sys.path.append('../')` — broke when not run from `eval/` directory | `eval/eval_LooGLE.py`, `eval/eval_Hotpot.py`, `example/example.py` | `EVAL_DIR = os.path.dirname(os.path.abspath(__file__))` |
| 13 | `trust_remote_code=True` requirement undocumented — hangs in non-interactive shells | `README.md` | Added warning + `TRUST_REMOTE_CODE=1` env var note |
| 14 | HuggingFace token only set ephemerally via `HF_TOKEN` — requires re-setting on each run | `README.md` | Added `huggingface-cli login` (persistent) instructions |

---

## Automated Test Coverage

Run: `python tests/test_reproducibility.py -v`

| Test | Checks | Status |
|------|--------|--------|
| `test_01_requirements_repro_is_utf8` | UTF-8 readable | ✅ PASS |
| `test_03_04_rag_main_creates_directories` | makedirs for chroma_data + model_cache | ✅ PASS |
| `test_05_log_directory_created` | logs/ creation | ✅ PASS |
| `test_07a_eval_loogle_has_judge_model_arg` | `--judge_model_name` arg | ✅ PASS |
| `test_07b_eval_hotpot_has_judge_model_arg` | `--judge_model_name` arg | ✅ PASS |
| `test_09_start_loogle_has_resume_guard` | resume guard phrase | ✅ PASS |
| `test_10_typo_fixed_in_preprocess` | no `daatabase_description` | ✅ PASS |
| `test_10b_typo_fixed_in_rag_main` | no `daatabase_description` | ✅ PASS |
| `test_11_rag_main_uses_os_path_join` | `os.path.join` used | ✅ PASS |
| `test_12_eval_loogle_has_file_relative_paths` | `__file__`-based paths | ✅ PASS |
| `test_12b_eval_hotpot_has_file_relative_paths` | `__file__`-based paths | ✅ PASS |
| `test_12c_example_has_file_relative_paths` | `__file__`-based paths | ✅ PASS |
| `test_14_readme_documents_hf_login` | `huggingface-cli login` | ✅ PASS |
| `test_seed_argument_exists` | `--seed` arg | ✅ PASS |
| `test_set_seed_function_exists` | `set_seed()` function | ✅ PASS |
| `test_seed_produces_deterministic_output` | same seed = same output | ✅ PASS |
| `test_full_rag_pipeline` | end-to-end with mock agents | ✅ PASS |

**Total: 17/17 PASS in 73 seconds.**

---

## Remaining Non-Determinism (Cannot Fix)

| Source | Impact | Mitigation |
|--------|--------|-----------|
| OpenAI API responses | Even at `temperature=0`, response can vary across model versions | `CachedAgent` in enhancement plan caches responses by prompt hash |
| OpenAI model updates | `gpt-4o-mini` weights may change silently | Pin model version where API supports it |
| HuggingFace model revisions | `nomic-embed-text-v2-moe` may be updated | Use `revision=` argument to pin specific commit SHA |
| CUDA non-determinism | Some GPU ops are inherently non-deterministic | `torch.use_deterministic_algorithms(True)` where possible |
