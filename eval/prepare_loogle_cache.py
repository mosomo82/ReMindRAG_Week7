"""
eval/prepare_loogle_cache.py
Generates the LooGLE-rewrite-data overlay (titles.json + per-title choice-format
and similar-data JSONs) from the upstream LooGLE-testdata .jsonl files.

Run ONCE from the repo root before running eval_LooGLE.py:
    venv\Scripts\python.exe eval/prepare_loogle_cache.py
"""
import json, os, sys

EVAL_DIR    = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(EVAL_DIR)
TESTDATA    = os.path.join(REPO_ROOT, "_loogle_tmp", "LooGLE-testdata")
CACHE_OUT   = os.path.join(EVAL_DIR, "dataset_cache", "LooGLE-rewrite-data")

DATA_TYPES  = ["shortdep_qa", "longdep_qa"]

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def extract_qa(raw_qa_str):
    """
    qa_pairs may be a JSON string or already a list.
    Returns list of dicts with at least 'Q' and 'A' keys.
    Falls back gracefully to empty list.
    """
    if isinstance(raw_qa_str, list):
        return raw_qa_str
    try:
        return json.loads(raw_qa_str)
    except Exception:
        return []

def build_choice_format_entry(qa):
    """Convert raw Q/A pair to the choice-format schema eval_LooGLE.py expects."""
    q = qa.get("Q") or qa.get("question") or qa.get("q") or ""
    a = qa.get("A") or qa.get("answer") or qa.get("a") or ""
    evidence = qa.get("evidence") or qa.get("S") or ""
    return {"question": q, "answer": a, "evidence": evidence}

def build_similar_entry(qa, idx):
    """Similar-data: slight rewording of question (stubbed — same Q for now)."""
    q = qa.get("Q") or qa.get("question") or qa.get("q") or ""
    a = qa.get("A") or qa.get("answer") or qa.get("a") or ""
    return {"question": q, "answer": a}

def process_data_type(dt):
    jsonl_path = os.path.join(TESTDATA, f"{dt}.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"  SKIP {dt}: {jsonl_path} not found")
        return []

    records = load_jsonl(jsonl_path)
    titles  = [r["title"] for r in records]
    print(f"  {dt}: {len(titles)} titles")

    choice_dir  = os.path.join(CACHE_OUT, "choice-format",  dt)
    similar_dir = os.path.join(CACHE_OUT, "similar-data",   dt)
    os.makedirs(choice_dir,  exist_ok=True)
    os.makedirs(similar_dir, exist_ok=True)

    for rec in records:
        title = rec["title"]
        raw   = rec.get("qa_pairs", "[]")
        qa_list = extract_qa(raw)

        choice_entries  = [build_choice_format_entry(q) for q in qa_list]
        similar_entries = [build_similar_entry(q, i) for i, q in enumerate(qa_list)]

        # Safe filename: replace path-unsafe chars
        safe = title.replace("/", "_").replace("\\", "_")

        with open(os.path.join(choice_dir,  f"{safe}.json"), "w", encoding="utf-8") as f:
            json.dump(choice_entries, f, ensure_ascii=False, indent=2)
        with open(os.path.join(similar_dir, f"{safe}.json"), "w", encoding="utf-8") as f:
            json.dump(similar_entries, f, ensure_ascii=False, indent=2)

    return titles

def main():
    if not os.path.isdir(TESTDATA):
        print(f"ERROR: Could not find upstream testdata at {TESTDATA}")
        print("Run: git clone --depth 1 https://github.com/bigai-nlco/LooGLE.git _loogle_tmp")
        sys.exit(1)

    print(f"Writing rewrite-data overlay to:\n  {CACHE_OUT}")
    os.makedirs(CACHE_OUT, exist_ok=True)

    all_titles_map = {}
    for dt in DATA_TYPES:
        print(f"\nProcessing {dt}...")
        titles = process_data_type(dt)
        for i, t in enumerate(titles):
            all_titles_map[str(i)] = t   # titles.json uses string keys

    # Write combined titles.json (used by start_LooGLE.py and eval_LooGLE.py)
    titles_path = os.path.join(CACHE_OUT, "titles.json")
    with open(titles_path, "w", encoding="utf-8") as f:
        json.dump(all_titles_map, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(all_titles_map)} titles to {titles_path}")
    print("LooGLE rewrite-data cache is ready.")

if __name__ == "__main__":
    main()
