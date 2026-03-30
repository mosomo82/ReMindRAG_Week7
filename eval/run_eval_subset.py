"""
eval/run_eval_subset.py
=======================
Constrained benchmark runner — ~$0.70 total API cost.

Runs:
  • LooGLE  shortdep_qa/origin — title indices 0-4  (5 titles)
  • HotpotQA origin            — question indices 0-49 (50 questions)

Both use --seed 42 --judge_model_name gpt-4o-mini.
Results are aggregated and written to eval_results.json at the repo root.

Prerequisites:
  1. api_key.json contains a real OpenAI key
  2. eval/dataset_cache/LooGLE-rewrite-data/ exists
       (run: python eval/prepare_loogle_cache.py)
  3. eval/dataset_cache/Hotpot/hotpot_dev_distractor_v1.json exists
       (download from: https://hotpotqa.github.io/)

Usage:
    venv\\Scripts\\python.exe eval/run_eval_subset.py
    venv\\Scripts\\python.exe eval/run_eval_subset.py --seed 42 --judge_model_name gpt-4o-mini
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime, timezone

EVAL_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EVAL_DIR)


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def check_api_key():
    key_path = os.path.join(REPO_ROOT, "api_key.json")
    if not os.path.exists(key_path):
        sys.exit(f"[ERROR] api_key.json not found at {key_path}")
    with open(key_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    key = data[0].get("api_key", "")
    if not key or key == "YOUR_API_KEY_HERE":
        sys.exit("[ERROR] api_key.json still has placeholder key — update it first.")
    print(f"[OK] api_key.json: key present (len={len(key)})")


def check_loogle_cache():
    overlay = os.path.join(EVAL_DIR, "dataset_cache", "LooGLE-rewrite-data")
    titles  = os.path.join(overlay, "titles.json")
    if not os.path.exists(titles):
        sys.exit(
            f"[ERROR] LooGLE rewrite cache missing at {overlay}\n"
            f"        Run first: python eval/prepare_loogle_cache.py"
        )
    with open(titles, "r", encoding="utf-8") as f:
        t = json.load(f)
    print(f"[OK] LooGLE cache: {len(t)} titles available")
    return t


def check_hotpot_cache():
    hp_path = os.path.join(EVAL_DIR, "dataset_cache", "Hotpot",
                           "hotpot_dev_distractor_v1.json")
    if not os.path.exists(hp_path):
        return False
    print(f"[OK] Hotpot cache found: {hp_path}")
    return True


# ---------------------------------------------------------------------------
# Single-title runner helpers
# ---------------------------------------------------------------------------

def run_loogle_title(idx, test_name, seed, judge_model, model_name, data_type, question_type):
    """Call eval_LooGLE.py for one title index. Returns (correct, total) or None."""
    script = os.path.join(EVAL_DIR, "eval_LooGLE.py")
    cmd = [
        sys.executable, script,
        "--title_index",     str(idx),
        "--test_name",       test_name,
        "--data_type",       data_type,
        "--question_type",   question_type,
        "--model_name",      model_name,
        "--judge_model_name", judge_model,
        "--seed",            str(seed),
    ]
    print(f"\n[LooGLE] title={idx}  cmd: {' '.join(cmd[2:])}")
    result = subprocess.run(cmd, cwd=EVAL_DIR)

    if result.returncode != 0:
        print(f"[WARN] title {idx} exited with code {result.returncode}")

    # Read result.txt written by eval_LooGLE.py
    result_txt = os.path.join(EVAL_DIR, "database", test_name, str(idx), "result.txt")
    if not os.path.exists(result_txt):
        print(f"[WARN] result.txt not found for title {idx}")
        return None

    correct = total = 0
    with open(result_txt, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Correct:"):
                try:
                    frac = line.split(":")[1].strip()
                    c, t = map(int, frac.split("/"))
                    correct, total = c, t
                except Exception:
                    pass
    return correct, total


def run_hotpot_question(idx, test_name, seed, judge_model, model_name, question_type):
    """Call eval_Hotpot.py for one question index. Returns 1 (correct) or 0."""
    script = os.path.join(EVAL_DIR, "eval_Hotpot.py")
    cmd = [
        sys.executable, script,
        "--title_index",      str(idx),
        "--test_name",        test_name,
        "--model_name",       model_name,
        "--judge_model_name", judge_model,
        "--question_type",    question_type,
        "--seed",             str(seed),
    ]
    print(f"\n[Hotpot] q={idx}  cmd: {' '.join(cmd[2:])}")
    result = subprocess.run(cmd, cwd=EVAL_DIR)

    if result.returncode != 0:
        print(f"[WARN] question {idx} exited with code {result.returncode}")

    # Read input.json written by eval_Hotpot.py
    input_json = os.path.join(EVAL_DIR, "database", test_name, str(idx), "input.json")
    if not os.path.exists(input_json):
        print(f"[WARN] input.json not found for question {idx}")
        return 0

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return 1 if data.get("check_response") == "True" else 0


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ReMindRAG constrained benchmark runner")
    parser.add_argument("--seed",             type=int, default=42,           help="Random seed")
    parser.add_argument("--judge_model_name", type=str, default="gpt-4o-mini", help="Judge LLM")
    parser.add_argument("--model_name",       type=str, default="gpt-4o-mini", help="Backbone LLM")
    parser.add_argument("--loogle_titles",    type=int, default=5,            help="Number of LooGLE titles (0–N-1)")
    parser.add_argument("--hotpot_questions", type=int, default=50,           help="Number of Hotpot questions (0–N-1)")
    parser.add_argument("--loogle_data_type",    type=str, default="shortdep_qa")
    parser.add_argument("--loogle_question_type", type=str, default="origin")
    parser.add_argument("--hotpot_question_type", type=str, default="origin")
    parser.add_argument("--test_name",        type=str, default="eval_subset")
    parser.add_argument("--skip_loogle",      action="store_true", help="Skip LooGLE run")
    parser.add_argument("--skip_hotpot",      action="store_true", help="Skip Hotpot run")
    args = parser.parse_args()

    print("=" * 65)
    print("  ReMindRAG Constrained Benchmark Runner")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  seed={args.seed}  judge={args.judge_model_name}")
    print("=" * 65)

    check_api_key()
    loogle_titles = check_loogle_cache()
    hotpot_ok     = check_hotpot_cache()

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results = {
        "run_date": run_date,
        "seed": args.seed,
        "judge_model": args.judge_model_name,
        "model": args.model_name,
    }

    # ------------------------------------------------------------------
    # LooGLE subset
    # ------------------------------------------------------------------
    if args.skip_loogle:
        print("\n[SKIP] LooGLE (--skip_loogle)")
        results["loogle"] = {"skipped": True}
    else:
        n_loogle = min(args.loogle_titles, len(loogle_titles))
        print(f"\n{'='*65}")
        print(f"  LooGLE {args.loogle_data_type}/{args.loogle_question_type} — {n_loogle} titles")
        print(f"{'='*65}")

        loogle_correct = loogle_total = 0
        for idx in range(n_loogle):
            pair = run_loogle_title(
                idx, args.test_name, args.seed, args.judge_model_name,
                args.model_name, args.loogle_data_type, args.loogle_question_type
            )
            if pair:
                loogle_correct += pair[0]
                loogle_total   += pair[1]

        f1_loogle = round(loogle_correct / loogle_total, 4) if loogle_total > 0 else None
        results["loogle"] = {
            "subset": f"{args.loogle_data_type}/{args.loogle_question_type}, titles 0-{n_loogle-1}",
            "correct": loogle_correct,
            "total":   loogle_total,
            "f1":      f1_loogle,
        }
        print(f"\n[LooGLE] correct={loogle_correct}/{loogle_total}  F1={f1_loogle}")

    # ------------------------------------------------------------------
    # HotpotQA subset
    # ------------------------------------------------------------------
    if args.skip_hotpot:
        print("\n[SKIP] Hotpot (--skip_hotpot)")
        results["hotpot"] = {"skipped": True}
    elif not hotpot_ok:
        print("\n[SKIP] Hotpot cache not found — skipping.")
        print("       Download from: https://hotpotqa.github.io/")
        print("       Place at: eval/dataset_cache/Hotpot/hotpot_dev_distractor_v1.json")
        results["hotpot"] = {"skipped": True, "reason": "dataset cache not found"}
    else:
        n_hotpot = args.hotpot_questions
        print(f"\n{'='*65}")
        print(f"  HotpotQA {args.hotpot_question_type} — {n_hotpot} questions")
        print(f"{'='*65}")

        hotpot_correct = 0
        for idx in range(n_hotpot):
            hotpot_correct += run_hotpot_question(
                idx, args.test_name, args.seed, args.judge_model_name,
                args.model_name, args.hotpot_question_type
            )

        f1_hotpot = round(hotpot_correct / n_hotpot, 4) if n_hotpot > 0 else None
        results["hotpot"] = {
            "subset":  f"{args.hotpot_question_type}, questions 0-{n_hotpot-1}",
            "correct": hotpot_correct,
            "total":   n_hotpot,
            "f1":      f1_hotpot,
        }
        print(f"\n[Hotpot] correct={hotpot_correct}/{n_hotpot}  F1={f1_hotpot}")

    # ------------------------------------------------------------------
    # Write results
    # ------------------------------------------------------------------
    out_path = os.path.join(REPO_ROOT, "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*65}")
    print(f"  Results written to: {out_path}")
    print(f"{'='*65}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
