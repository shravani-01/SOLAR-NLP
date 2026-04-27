#!/usr/bin/env python3
"""
Mechanism Experiment — Code (HumanEval).

Tests whether giving the model the correct code structure as a hint
closes the composition gap.

3 conditions:
  1. original     — reuse baseline gap from existing predictions
  2. hint_correct — "This solution requires a LOOP-based approach with iteration."
  3. hint_wrong   — "This solution requires a RECURSIVE approach."

Usage:
    python run_code.py --model gpt4o --limit 10
    python run_code.py --model all
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experiments" / "humaneval_composition_gap"))
from core import (call_api, call_oss, load_oss_model, API_MODELS, OSS_MODELS,
                  REQUEST_DELAY)
from classify_code_structure import classify_code, classify_response, STRUCTURAL_TYPES as CODE_TYPES
from core_mechanism import (load_baseline_predictions, get_gap_eligible,
                            pick_wrong_type, compute_mechanism_results,
                            save_mechanism_results, print_mechanism_comparison)

DATA_DIR = Path(__file__).parent.parent.parent / "experiments" / "humaneval_composition_gap" / "data"
RESULTS_DIR = Path(__file__).parent / "results"

SYSTEM = "You are a Python programming expert. Answer precisely."

# ─── Structure descriptions for hints ─────────────────────────────────────

STRUCTURE_HINTS = {
    "SINGLE-FUNC":  "a SIMPLE FUNCTION with straightforward logic — no loops, no recursion, "
                    "just direct computation or string/list operations. Use built-in functions where possible",
    "LOOP":         "a LOOP-BASED approach using for/while iteration. "
                    "Iterate over the input, accumulate results, and return. "
                    "Do NOT use recursion — iteration is the natural fit",
    "CONDITIONAL":  "CONDITIONAL BRANCHING with multiple if/elif/else cases. "
                    "The key challenge is handling different cases correctly. "
                    "Map out all edge cases before implementing",
    "RECURSIVE":    "a RECURSIVE approach where the function calls itself. "
                    "Define a base case and recursive step. "
                    "Do NOT use loops — recursion is the natural fit",
    "MULTI-STRUCT": "a MULTI-STRUCTURE approach combining loops with conditionals, "
                    "or nested loops, or helper functions. "
                    "Break the problem into sub-problems and compose the structures",
}

# ─── Mechanism prompts ────────────────────────────────────────────────────

COMPOSED_HINT = """Complete this Python function:

{prompt}

IMPORTANT STRUCTURAL NOTE: The best solution uses {hint}.

Return ONLY the function body. No explanation, no markdown."""


# ─── Parsing ──────────────────────────────────────────────────────────────

def parse_yesno(response: str) -> bool:
    return response.strip().upper().startswith("YES")


# ─── Data loading ─────────────────────────────────────────────────────────

def load_data(limit=None):
    path = DATA_DIR / "humaneval_classified.json"
    if not path.exists():
        path = DATA_DIR / "humaneval.json"
    if not path.exists():
        print(f"[ERROR] No data found.")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


# ─── Run mechanism conditions ─────────────────────────────────────────────

def run(model_key: str, data: list, backend: str = "api"):
    if backend == "api":
        config = API_MODELS[model_key]
        provider, model_name = config["provider"], config["model"]
        call_fn = lambda p: call_api(p, provider, model_name, SYSTEM, max_tokens=1024)
    else:
        config = OSS_MODELS[model_key]
        model, tokenizer = load_oss_model(config["hf_name"])
        call_fn = lambda p: call_oss(p, model, tokenizer, SYSTEM, max_tokens=1024)

    baseline_preds = load_baseline_predictions("code", model_key)
    eligible = get_gap_eligible(baseline_preds)

    if not eligible:
        print(f"[WARN] No gap-eligible examples for {model_key}.")
        return
    eligible_indices = set(p["idx"] for p in eligible)
    print(f"[INFO] {model_key}: {len(eligible)} gap-eligible examples")

    original_results = compute_mechanism_results(eligible, "original")

    # hint_correct condition
    print(f"\n[INFO] Running hint_correct condition...")
    hint_correct_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        prompt = ex["prompt"]
        gold_solution = ex.get("canonical_solution", "")
        gold_type = ex.get("structural_type", "SINGLE-FUNC")
        full_gold_code = prompt + gold_solution

        # Derive gold pieces from actual code
        gold_has_loop = bool(re.search(r'\bfor\s+\w+\s+in\s+|\bwhile\s+', full_gold_code.lower()))
        gold_has_rec = False
        fn_names = re.findall(r'def\s+(\w+)\s*\(', full_gold_code.lower())
        for fn in fn_names:
            parts = full_gold_code.lower().split(f'def {fn}')
            if len(parts) > 1 and re.search(rf'\b{fn}\s*\(', parts[-1]):
                gold_has_rec = True

        hint = STRUCTURE_HINTS.get(gold_type, STRUCTURE_HINTS["SINGLE-FUNC"])

        try:
            r = call_fn(COMPOSED_HINT.format(prompt=prompt, hint=hint))
            time.sleep(REQUEST_DELAY)

            # Check structural consistency (same as baseline)
            code_match = re.search(r'```(?:python)?\s*\n(.*?)```', r, re.DOTALL)
            gen_code = code_match.group(1).strip() if code_match else r.strip()
            full_gen = prompt + "\n" + gen_code

            code_has_loop = bool(re.search(r'\bfor\s+\w+\s+in\s+|\bwhile\s+', full_gen))
            code_has_rec = False
            fn_names_gen = re.findall(r'def\s+(\w+)\s*\(', full_gen)
            for fn in fn_names_gen:
                parts = full_gen.split(f'def {fn}')
                if len(parts) > 1 and re.search(rf'\b{fn}\s*\(', parts[-1]):
                    code_has_rec = True

            # For mechanism experiment: check if generated code matches GOLD structure
            # (not self-consistency like baseline, but gold-consistency)
            loop_match = (code_has_loop == gold_has_loop)
            rec_match = (code_has_rec == gold_has_rec)
            composed_correct = loop_match and rec_match

            pred_type = classify_response(prompt, r)

            hint_correct_preds.append({
                "idx": i, "gold_type": gold_type, "pred_type": pred_type,
                "code_has_loop": code_has_loop, "gold_has_loop": gold_has_loop,
                "code_has_rec": code_has_rec, "gold_has_rec": gold_has_rec,
                "composed_correct": composed_correct,
                "condition": "hint_correct",
                "hint_given": hint[:100],
            })
        except Exception as e:
            hint_correct_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if len(hint_correct_preds) % 25 == 0:
            correct = sum(1 for p in hint_correct_preds if p.get("composed_correct"))
            print(f"  [hint_correct] {len(hint_correct_preds)}/{len(eligible_indices)} | correct: {correct}")

    hint_correct_results = compute_mechanism_results(hint_correct_preds, "hint_correct")

    # hint_wrong condition
    print(f"\n[INFO] Running hint_wrong condition...")
    hint_wrong_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        prompt = ex["prompt"]
        gold_solution = ex.get("canonical_solution", "")
        gold_type = ex.get("structural_type", "SINGLE-FUNC")
        full_gold_code = prompt + gold_solution

        gold_has_loop = bool(re.search(r'\bfor\s+\w+\s+in\s+|\bwhile\s+', full_gold_code.lower()))
        gold_has_rec = False
        fn_names = re.findall(r'def\s+(\w+)\s*\(', full_gold_code.lower())
        for fn in fn_names:
            parts = full_gold_code.lower().split(f'def {fn}')
            if len(parts) > 1 and re.search(rf'\b{fn}\s*\(', parts[-1]):
                gold_has_rec = True

        wrong_type = pick_wrong_type(gold_type, list(CODE_TYPES))
        hint = STRUCTURE_HINTS.get(wrong_type, STRUCTURE_HINTS["SINGLE-FUNC"])

        try:
            r = call_fn(COMPOSED_HINT.format(prompt=prompt, hint=hint))
            time.sleep(REQUEST_DELAY)

            code_match = re.search(r'```(?:python)?\s*\n(.*?)```', r, re.DOTALL)
            gen_code = code_match.group(1).strip() if code_match else r.strip()
            full_gen = prompt + "\n" + gen_code

            code_has_loop = bool(re.search(r'\bfor\s+\w+\s+in\s+|\bwhile\s+', full_gen))
            code_has_rec = False
            fn_names_gen = re.findall(r'def\s+(\w+)\s*\(', full_gen)
            for fn in fn_names_gen:
                parts = full_gen.split(f'def {fn}')
                if len(parts) > 1 and re.search(rf'\b{fn}\s*\(', parts[-1]):
                    code_has_rec = True

            loop_match = (code_has_loop == gold_has_loop)
            rec_match = (code_has_rec == gold_has_rec)
            composed_correct = loop_match and rec_match

            pred_type = classify_response(prompt, r)

            hint_wrong_preds.append({
                "idx": i, "gold_type": gold_type, "pred_type": pred_type,
                "code_has_loop": code_has_loop, "gold_has_loop": gold_has_loop,
                "code_has_rec": code_has_rec, "gold_has_rec": gold_has_rec,
                "composed_correct": composed_correct,
                "condition": "hint_wrong",
                "wrong_type_given": wrong_type,
                "hint_given": hint[:100],
            })
        except Exception as e:
            hint_wrong_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if len(hint_wrong_preds) % 25 == 0:
            correct = sum(1 for p in hint_wrong_preds if p.get("composed_correct"))
            print(f"  [hint_wrong] {len(hint_wrong_preds)}/{len(eligible_indices)} | correct: {correct}")

    hint_wrong_results = compute_mechanism_results(hint_wrong_preds, "hint_wrong")

    all_results = {
        "original": original_results,
        "hint_correct": hint_correct_results,
        "hint_wrong": hint_wrong_results,
        "predictions": {
            "hint_correct": hint_correct_preds,
            "hint_wrong": hint_wrong_preds,
        },
    }

    print_mechanism_comparison(all_results, model_key, "Code")
    save_mechanism_results(all_results, model_key, "code", RESULTS_DIR)
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default="api", choices=["api", "oss"])
    args = parser.parse_args()

    data = load_data(args.limit)
    print(f"[INFO] Loaded {len(data)} code problems")

    models = API_MODELS if args.backend == "api" else OSS_MODELS
    to_run = list(models.keys()) if args.model == "all" else [args.model]

    for mk in to_run:
        run(mk, data, backend=args.backend)


if __name__ == "__main__":
    main()
