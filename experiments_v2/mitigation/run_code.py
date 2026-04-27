#!/usr/bin/env python3
"""
Mitigation Experiment — Code (Section 8).

Structure-Aware Prompting for code generation.

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
from classify_code_structure import classify_response
from core_mitigation import (load_baseline_predictions, get_gap_eligible,
                             compute_mitigation_results, save_mitigation_results,
                             print_mitigation_comparison)

DATA_DIR = Path(__file__).parent.parent.parent / "experiments" / "humaneval_composition_gap" / "data"
RESULTS_DIR = Path(__file__).parent / "results"

SYSTEM = "You are a Python programming expert. Answer precisely."

STAGE1_PROMPT = """Look at this Python function signature and docstring:

{prompt}

Before writing code, analyze the STRUCTURAL APPROACH needed:
1. Does this require ITERATION (for/while loops)?
2. Does this require RECURSION (function calls itself)?
3. Does this require CONDITIONAL BRANCHING (multiple if/elif/else)?
4. Does this require HELPER FUNCTIONS or NESTED STRUCTURES?
5. What is the best overall approach?

Format:
ITERATION: [YES/NO and why]
RECURSION: [YES/NO and why]
CONDITIONALS: [YES/NO and why]
APPROACH: [SINGLE-FUNC / LOOP / CONDITIONAL / RECURSIVE / MULTI-STRUCT]"""

STAGE2_PROMPT = """Complete this Python function:

{prompt}

You have already analyzed the structural approach:
{structure_analysis}

Follow the approach you identified. Return ONLY the function body. No explanation, no markdown."""

COT_STRUCTURE_PROMPT = """Complete this Python function:

{prompt}

THINK ABOUT STRUCTURE FIRST:
Step 1: What data structures does the input/output suggest?
Step 2: Does this need iteration, recursion, or direct computation?
Step 3: What edge cases need conditional handling?
Step 4: Plan your approach, then implement.

Return ONLY the function body after your plan. No markdown."""


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

    original_results = compute_mitigation_results(eligible, "original")

    # Derive gold structural info for each example
    def get_gold_structure(ex):
        gold_solution = ex.get("canonical_solution", "")
        full_gold_code = (ex["prompt"] + gold_solution).lower()
        has_loop = bool(re.search(r'\bfor\s+\w+\s+in\s+|\bwhile\s+', full_gold_code))
        has_rec = False
        fn_names = re.findall(r'def\s+(\w+)\s*\(', full_gold_code)
        for fn in fn_names:
            parts = full_gold_code.split(f'def {fn}')
            if len(parts) > 1 and re.search(rf'\b{fn}\s*\(', parts[-1]):
                has_rec = True
        return has_loop, has_rec

    def check_gen_structure(prompt_text, response):
        code_match = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
        gen_code = code_match.group(1).strip() if code_match else response.strip()
        full_gen = (prompt_text + "\n" + gen_code).lower()
        has_loop = bool(re.search(r'\bfor\s+\w+\s+in\s+|\bwhile\s+', full_gen))
        has_rec = False
        fn_names = re.findall(r'def\s+(\w+)\s*\(', full_gen)
        for fn in fn_names:
            parts = full_gen.split(f'def {fn}')
            if len(parts) > 1 and re.search(rf'\b{fn}\s*\(', parts[-1]):
                has_rec = True
        return has_loop, has_rec

    # self_structure
    print(f"\n[INFO] Running self_structure condition...")
    self_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        prompt = ex["prompt"]
        gold_type = ex.get("structural_type", "SINGLE-FUNC")
        gold_loop, gold_rec = get_gold_structure(ex)

        try:
            r1 = call_fn(STAGE1_PROMPT.format(prompt=prompt))
            time.sleep(REQUEST_DELAY)
            r2 = call_fn(STAGE2_PROMPT.format(prompt=prompt, structure_analysis=r1))
            time.sleep(REQUEST_DELAY)

            gen_loop, gen_rec = check_gen_structure(prompt, r2)
            composed_correct = (gen_loop == gold_loop) and (gen_rec == gold_rec)

            self_preds.append({
                "idx": i, "gold_type": gold_type,
                "gold_has_loop": gold_loop, "gen_has_loop": gen_loop,
                "gold_has_rec": gold_rec, "gen_has_rec": gen_rec,
                "composed_correct": composed_correct,
                "condition": "self_structure",
                "stage1_response": r1[:400],
            })
        except Exception as e:
            self_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if len(self_preds) % 25 == 0:
            correct = sum(1 for p in self_preds if p.get("composed_correct"))
            print(f"  [self_structure] {len(self_preds)}/{len(eligible_indices)} | correct: {correct}")

    self_results = compute_mitigation_results(self_preds, "self_structure")

    # cot_structure
    print(f"\n[INFO] Running cot_structure condition...")
    cot_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        prompt = ex["prompt"]
        gold_type = ex.get("structural_type", "SINGLE-FUNC")
        gold_loop, gold_rec = get_gold_structure(ex)

        try:
            r = call_fn(COT_STRUCTURE_PROMPT.format(prompt=prompt))
            time.sleep(REQUEST_DELAY)

            gen_loop, gen_rec = check_gen_structure(prompt, r)
            composed_correct = (gen_loop == gold_loop) and (gen_rec == gold_rec)

            cot_preds.append({
                "idx": i, "gold_type": gold_type,
                "gold_has_loop": gold_loop, "gen_has_loop": gen_loop,
                "gold_has_rec": gold_rec, "gen_has_rec": gen_rec,
                "composed_correct": composed_correct,
                "condition": "cot_structure",
                "response": r[:500],
            })
        except Exception as e:
            cot_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if len(cot_preds) % 25 == 0:
            correct = sum(1 for p in cot_preds if p.get("composed_correct"))
            print(f"  [cot_structure] {len(cot_preds)}/{len(eligible_indices)} | correct: {correct}")

    cot_results = compute_mitigation_results(cot_preds, "cot_structure")

    all_results = {
        "original": original_results,
        "self_structure": self_results,
        "cot_structure": cot_results,
        "predictions": {"self_structure": self_preds, "cot_structure": cot_preds},
    }

    print_mitigation_comparison(all_results, model_key, "Code")
    save_mitigation_results(all_results, model_key, "code", RESULTS_DIR)
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
