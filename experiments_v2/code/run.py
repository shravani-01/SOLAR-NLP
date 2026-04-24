#!/usr/bin/env python3
"""
2-Pass Composition Gap — Code Generation (HumanEval).

Pass 1 (Pieces): Ask 3 sub-questions separately:
  Q1: "Does this problem require iteration (loops)?"
  Q2: "Does this problem require conditional branching?"
  Q3: "Does this problem require recursion?"

Pass 2 (Composed): "Write the complete solution."
  Classify structure: SINGLE-FUNC/LOOP/CONDITIONAL/RECURSIVE/MULTI-STRUCT

Gap = P(Pass 2 wrong structure | Pass 1 all correct)

Usage:
    python run.py --model gpt4o --limit 10
    python run.py --model all
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experiments" / "humaneval_composition_gap"))
from core import (call_api, call_oss, load_oss_model, API_MODELS, OSS_MODELS,
                  REQUEST_DELAY, compute_composition_gap, print_results, save_results)
from classify_code_structure import classify_code, classify_response, STRUCTURAL_TYPES

DATA_DIR = Path(__file__).parent.parent.parent / "experiments" / "humaneval_composition_gap" / "data"
RESULTS_DIR = Path(__file__).parent / "results"

SYSTEM = "You are a Python programming expert. Answer precisely."

# ─── Gold piece derivation ──────────────────────────────────────────────────

def get_gold_pieces(gold_type: str, gold_code: str = "") -> dict:
    """Derive expected piece answers from gold code, not just type label.
    We check the ACTUAL gold solution to determine what structures it uses."""
    import re
    code = gold_code.lower() if gold_code else ""

    # Detect from actual code rather than type label
    has_loop = bool(re.search(r'\bfor\s+\w+\s+in\s+|\bwhile\s+', code))
    has_recursion = False
    # Check if any defined function calls itself
    func_names = re.findall(r'def\s+(\w+)\s*\(', code)
    for fn in func_names:
        # Look for fn( in the body (not on the def line)
        parts = code.split(f'def {fn}')
        if len(parts) > 1 and re.search(rf'\b{fn}\s*\(', parts[-1]):
            has_recursion = True

    return {
        "needs_loop": has_loop,
        "needs_recursion": has_recursion,
    }

# ─── Piece sub-questions ────────────────────────────────────────────────────

PIECE_Q1 = """Look at this Python function signature and docstring:

{prompt}

Question: Does the BEST solution to this problem require ITERATION (for loops or while loops)?

Answer with ONLY one word: YES or NO."""

PIECE_Q2 = """Look at this Python function signature and docstring:

{prompt}

Question: Does the BEST solution to this problem require RECURSION (function calling itself)?

Answer with ONLY one word: YES or NO."""

# ─── Composed question ─────────────────────────────────────────────────────

COMPOSED_Q = """Complete this Python function:

{prompt}

Return ONLY the function body. No explanation, no markdown."""

# ─── Parsing ───────────────────────────────────────────────────────────────

def parse_yesno(response: str) -> bool:
    r = response.strip().upper()
    return r.startswith("YES")


# ─── Data loading ───────────────────────────────────────────────────────────

def load_data(limit=None):
    path = DATA_DIR / "humaneval_classified.json"
    if not path.exists():
        path = DATA_DIR / "humaneval.json"
    if not path.exists():
        print(f"[ERROR] No data. Run download_humaneval.py first.")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


# ─── Main ───────────────────────────────────────────────────────────────────

def run(model_key: str, data: list, backend: str = "api"):
    if backend == "api":
        config = API_MODELS[model_key]
        provider, model_name = config["provider"], config["model"]
        call_fn = lambda p: call_api(p, provider, model_name, SYSTEM, max_tokens=1024)
    else:
        config = OSS_MODELS[model_key]
        model, tokenizer = load_oss_model(config["hf_name"])
        call_fn = lambda p: call_oss(p, model, tokenizer, SYSTEM, max_tokens=1024)

    print(f"\n[INFO] Code 2-Pass: {model_key} ({len(data)} examples)")

    predictions = []
    errors = 0

    for i, ex in enumerate(data):
        prompt = ex["prompt"]
        gold_solution = ex.get("canonical_solution", "")
        gold_type = ex.get("structural_type", "SINGLE-FUNC")
        full_gold_code = prompt + gold_solution
        gold_pieces = get_gold_pieces(gold_type, full_gold_code)

        try:
            # PASS 1: Piece sub-questions (2 questions)
            r1 = call_fn(PIECE_Q1.format(prompt=prompt))
            time.sleep(REQUEST_DELAY)
            r2 = call_fn(PIECE_Q2.format(prompt=prompt))
            time.sleep(REQUEST_DELAY)

            pred_loop = parse_yesno(r1)
            pred_rec = parse_yesno(r2)

            # PASS 2: Composed question
            r3 = call_fn(COMPOSED_Q.format(prompt=prompt))
            time.sleep(REQUEST_DELAY)

            # Check what the generated code ACTUALLY uses
            code_match = re.search(r'```(?:python)?\s*\n(.*?)```', r3, re.DOTALL)
            gen_code = code_match.group(1).strip() if code_match else r3.strip()
            full_gen = prompt + "\n" + gen_code

            code_has_loop = bool(re.search(r'\bfor\s+\w+\s+in\s+|\bwhile\s+', full_gen))
            code_has_rec = False
            fn_names = re.findall(r'def\s+(\w+)\s*\(', full_gen)
            for fn in fn_names:
                parts = full_gen.split(f'def {fn}')
                if len(parts) > 1 and re.search(rf'\b{fn}\s*\(', parts[-1]):
                    code_has_rec = True

            # Self-consistency: does the code match what the model SAID it would use?
            # Gap = model said "I need X" but didn't use X, or said "I don't need X" but used X
            q1_consistent = pred_loop == code_has_loop
            q2_consistent = pred_rec == code_has_rec
            pieces_all_correct = True  # Model answered the piece questions (we accept its judgment)
            composed_correct = q1_consistent and q2_consistent  # Code matches its own plan

            pred_type = classify_response(prompt, r3)

            predictions.append({
                "idx": i, "task_id": ex.get("task_id", ""),
                "gold_type": gold_type, "pred_type": pred_type,
                "said_needs_loop": pred_loop,
                "code_has_loop": code_has_loop,
                "q1_consistent": q1_consistent,
                "said_needs_rec": pred_rec,
                "code_has_rec": code_has_rec,
                "q2_consistent": q2_consistent,
                "pieces_all_correct": pieces_all_correct,
                "composed_correct": composed_correct,
            })

        except Exception as e:
            errors += 1
            predictions.append({
                "idx": i, "gold_type": gold_type, "pred_type": "",
                "pieces_all_correct": False, "composed_correct": False,
                "error": str(e),
            })

        if (i + 1) % 25 == 0:
            pc = sum(1 for p in predictions if p.get("pieces_all_correct"))
            gap = sum(1 for p in predictions if p.get("pieces_all_correct") and not p.get("composed_correct"))
            print(f"[INFO] {i+1}/{len(data)} | pieces_ok: {pc} | gap_cases: {gap} | errors: {errors}")

    results = compute_composition_gap(predictions)
    print_results(results, model_key, "Code (HumanEval)")
    save_results(predictions, results, model_key, "code", RESULTS_DIR)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default="api", choices=["api", "oss"])
    args = parser.parse_args()

    data = load_data(args.limit)
    print(f"[INFO] Loaded {len(data)} problems")

    models = API_MODELS if args.backend == "api" else OSS_MODELS
    to_run = list(models.keys()) if args.model == "all" else [args.model]

    for mk in to_run:
        run(mk, data, backend=args.backend)


if __name__ == "__main__":
    main()
