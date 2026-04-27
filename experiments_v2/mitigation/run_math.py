#!/usr/bin/env python3
"""
Mitigation Experiment — Math (Section 8).

Structure-Aware Prompting for math problem solving.

Usage:
    python run_math.py --model gpt4o --limit 10
    python run_math.py --model all
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from core import (call_api, call_oss, load_oss_model, API_MODELS, OSS_MODELS,
                  REQUEST_DELAY)
from core_mitigation import (load_baseline_predictions, get_gap_eligible,
                             compute_mitigation_results, save_mitigation_results,
                             print_mitigation_comparison)

DATA_DIR = Path(__file__).parent.parent.parent / "experiments" / "gsm8k_composition_gap" / "data"
RESULTS_DIR = Path(__file__).parent / "results"

SYSTEM = "You are a math expert. Answer precisely."

STAGE1_PROMPT = """Read this math problem:

"{question}"

Before solving, analyze the STRUCTURE of this problem:
1. How many STEPS are needed? (1, 2, 3, or more)
2. What OPERATIONS are involved? (addition, subtraction, multiplication, division, comparison, percentage)
3. What is the REASONING PATTERN? (single calculation, multi-step chain, ratio/proportion, comparison of quantities, system of constraints)

Format your response as:
STEPS: [number]
OPERATIONS: [list]
PATTERN: [SINGLE-OP / MULTI-STEP / RATIO-PROP / COMPARISON / SYSTEM]"""

STAGE2_PROMPT = """Solve this math problem:

"{question}"

You have already analyzed the structure:
{structure_analysis}

Follow the structure you identified. Solve step by step.
On the final line, write your answer as a single number after "#### "."""

COT_STRUCTURE_PROMPT = """Solve this math problem:

"{question}"

THINK ABOUT THE STRUCTURE FIRST:
Step 0 (PLAN): How many operations are needed? What type of reasoning? Map out the steps before computing.
Then execute each step in order.

On the final line, write your answer as a single number after "#### "."""


def extract_answer(response: str) -> str:
    match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', response)
    if match:
        return match.group(1).replace(',', '')
    nums = re.findall(r'[\d,]+(?:\.\d+)?', response)
    return nums[-1].replace(',', '') if nums else ""


def extract_gold_answer(answer_text: str) -> str:
    match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', answer_text)
    if match:
        return match.group(1).replace(',', '')
    return ""


def load_data(limit=None):
    test_path = DATA_DIR / "test_classified.json"
    if not test_path.exists():
        test_path = DATA_DIR / "test.json"
    if not test_path.exists():
        print(f"[ERROR] No data at {DATA_DIR}")
        sys.exit(1)
    with open(test_path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


def run(model_key: str, data: list, backend: str = "api"):
    if backend == "api":
        config = API_MODELS[model_key]
        provider, model_name = config["provider"], config["model"]
        call_fn = lambda p: call_api(p, provider, model_name, SYSTEM)
    else:
        config = OSS_MODELS[model_key]
        model, tokenizer = load_oss_model(config["hf_name"])
        call_fn = lambda p: call_oss(p, model, tokenizer, SYSTEM)

    baseline_preds = load_baseline_predictions("math", model_key)
    eligible = get_gap_eligible(baseline_preds)

    if not eligible:
        print(f"[WARN] No gap-eligible examples for {model_key}.")
        return
    eligible_indices = set(p["idx"] for p in eligible)
    print(f"[INFO] {model_key}: {len(eligible)} gap-eligible examples")

    original_results = compute_mitigation_results(eligible, "original")

    # self_structure
    print(f"\n[INFO] Running self_structure condition...")
    self_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        question = ex.get("question", "")
        gold_answer_text = ex.get("full_answer", ex.get("answer", ""))
        gold_type = ex.get("structural_type", "MULTI-STEP")
        gold_answer = str(ex.get("final_answer", "")) if ex.get("final_answer") else extract_gold_answer(gold_answer_text)
        gold_answer = gold_answer.replace(",", "")

        try:
            r1 = call_fn(STAGE1_PROMPT.format(question=question))
            time.sleep(REQUEST_DELAY)
            r2 = call_fn(STAGE2_PROMPT.format(question=question, structure_analysis=r1))
            time.sleep(REQUEST_DELAY)

            pred_answer = extract_answer(r2)
            composed_correct = pred_answer == gold_answer

            self_preds.append({
                "idx": i, "gold_type": gold_type,
                "gold_answer": gold_answer, "pred_answer": pred_answer,
                "composed_correct": composed_correct,
                "condition": "self_structure",
                "stage1_response": r1[:300],
            })
        except Exception as e:
            self_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if len(self_preds) % 50 == 0:
            correct = sum(1 for p in self_preds if p.get("composed_correct"))
            print(f"  [self_structure] {len(self_preds)}/{len(eligible_indices)} | correct: {correct}")

    self_results = compute_mitigation_results(self_preds, "self_structure")

    # cot_structure
    print(f"\n[INFO] Running cot_structure condition...")
    cot_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        question = ex.get("question", "")
        gold_answer_text = ex.get("full_answer", ex.get("answer", ""))
        gold_type = ex.get("structural_type", "MULTI-STEP")
        gold_answer = str(ex.get("final_answer", "")) if ex.get("final_answer") else extract_gold_answer(gold_answer_text)
        gold_answer = gold_answer.replace(",", "")

        try:
            r = call_fn(COT_STRUCTURE_PROMPT.format(question=question))
            time.sleep(REQUEST_DELAY)

            pred_answer = extract_answer(r)
            composed_correct = pred_answer == gold_answer

            cot_preds.append({
                "idx": i, "gold_type": gold_type,
                "gold_answer": gold_answer, "pred_answer": pred_answer,
                "composed_correct": composed_correct,
                "condition": "cot_structure",
                "response": r[:500],
            })
        except Exception as e:
            cot_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if len(cot_preds) % 50 == 0:
            correct = sum(1 for p in cot_preds if p.get("composed_correct"))
            print(f"  [cot_structure] {len(cot_preds)}/{len(eligible_indices)} | correct: {correct}")

    cot_results = compute_mitigation_results(cot_preds, "cot_structure")

    all_results = {
        "original": original_results,
        "self_structure": self_results,
        "cot_structure": cot_results,
        "predictions": {"self_structure": self_preds, "cot_structure": cot_preds},
    }

    print_mitigation_comparison(all_results, model_key, "Math")
    save_mitigation_results(all_results, model_key, "math", RESULTS_DIR)
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default="api", choices=["api", "oss"])
    args = parser.parse_args()

    data = load_data(args.limit)
    print(f"[INFO] Loaded {len(data)} math examples")

    models = API_MODELS if args.backend == "api" else OSS_MODELS
    to_run = list(models.keys()) if args.model == "all" else [args.model]

    for mk in to_run:
        run(mk, data, backend=args.backend)


if __name__ == "__main__":
    main()
