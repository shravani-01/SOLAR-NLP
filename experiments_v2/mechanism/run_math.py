#!/usr/bin/env python3
"""
Mechanism Experiment — Math (GSM8K).

Tests whether giving the model the correct problem structure as a hint
closes the composition gap.

3 conditions:
  1. original     — reuse baseline gap from existing predictions
  2. hint_correct — "This is a MULTI-STEP problem requiring 3 operations: addition, multiplication."
  3. hint_wrong   — "This is a SINGLE-OP problem requiring 1 subtraction."

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
from core_mechanism import (load_baseline_predictions, get_gap_eligible,
                            pick_wrong_type, compute_mechanism_results,
                            save_mechanism_results, print_mechanism_comparison)

DATA_DIR = Path(__file__).parent.parent.parent / "experiments" / "gsm8k_composition_gap" / "data"
RESULTS_DIR = Path(__file__).parent / "results"

SYSTEM = "You are a math expert. Answer precisely."

MATH_TYPES = ["SINGLE-OP", "MULTI-STEP", "RATIO-PROP", "COMPARISON", "SYSTEM"]

# ─── Structure descriptions for hints ─────────────────────────────────────

def get_structure_hint(gold_type: str, question: str, answer: str) -> str:
    """Generate a detailed structure hint based on the gold type and content."""
    # Count operations from gold answer
    ops = []
    combined = (question + " " + answer).lower()
    if re.search(r'\+|add|total|sum|together|combined|more than', combined):
        ops.append("addition")
    if re.search(r'\-|subtract|minus|left|remain|fewer|less than|difference', combined):
        ops.append("subtraction")
    if re.search(r'\*|×|multiply|times|each|per|every|double|triple|twice', combined):
        ops.append("multiplication")
    if re.search(r'\/|÷|divide|split|shared|half|third|quarter', combined):
        ops.append("division")
    if not ops:
        ops = ["arithmetic"]

    step_count = len(re.findall(r'<<.*?>>', answer)) if '<<' in answer else len(ops)

    hints = {
        "SINGLE-OP":  f"a SINGLE-OPERATION problem solvable in one step using {ops[0]}",
        "MULTI-STEP": f"a MULTI-STEP problem requiring {max(step_count, 2)} sequential operations ({', '.join(ops[:3])})",
        "RATIO-PROP": f"a RATIO/PROPORTION problem involving percentages or ratios, requiring {', '.join(ops[:2])}",
        "COMPARISON": f"a COMPARISON problem that compares two quantities, requiring {', '.join(ops[:2])}",
        "SYSTEM":     f"a SYSTEM problem with multiple interacting constraints, requiring {max(step_count, 3)} steps ({', '.join(ops[:3])})",
    }
    return hints.get(gold_type, hints["MULTI-STEP"])


def get_wrong_hint(wrong_type: str) -> str:
    """Generate a misleading structure hint."""
    hints = {
        "SINGLE-OP":  "a SINGLE-OPERATION problem solvable in one step with simple subtraction",
        "MULTI-STEP": "a MULTI-STEP problem requiring 4 sequential operations (addition, multiplication, subtraction, division)",
        "RATIO-PROP": "a RATIO/PROPORTION problem involving percentages with multiplication and division",
        "COMPARISON": "a COMPARISON problem that requires finding the difference between two quantities",
        "SYSTEM":     "a SYSTEM problem with 5 interacting variables and multiple constraints",
    }
    return hints.get(wrong_type, hints["SINGLE-OP"])


# ─── Mechanism prompts ────────────────────────────────────────────────────

COMPOSED_HINT = """Solve this math problem:

"{question}"

IMPORTANT STRUCTURAL NOTE: Analysis shows this is {hint}.

Show your work step by step. On the final line, write your answer as a single number after "#### "."""


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


# ─── Data loading ─────────────────────────────────────────────────────────

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


# ─── Run mechanism conditions ─────────────────────────────────────────────

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

    # Original gap
    original_results = compute_mechanism_results(eligible, "original")

    # hint_correct condition
    print(f"\n[INFO] Running hint_correct condition...")
    hint_correct_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        question = ex.get("question", "")
        gold_answer_text = ex.get("full_answer", ex.get("answer", ""))
        gold_type = ex.get("structural_type", "MULTI-STEP")
        gold_answer = str(ex.get("final_answer", "")) if ex.get("final_answer") else extract_gold_answer(gold_answer_text)
        gold_answer = gold_answer.replace(",", "")

        hint = get_structure_hint(gold_type, question, gold_answer_text)

        try:
            r = call_fn(COMPOSED_HINT.format(question=question, hint=hint))
            time.sleep(REQUEST_DELAY)

            pred_answer = extract_answer(r)
            composed_correct = pred_answer == gold_answer

            hint_correct_preds.append({
                "idx": i, "gold_type": gold_type,
                "gold_answer": gold_answer, "pred_answer": pred_answer,
                "composed_correct": composed_correct,
                "condition": "hint_correct",
                "hint_given": hint,
            })
        except Exception as e:
            hint_correct_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if len(hint_correct_preds) % 50 == 0:
            correct = sum(1 for p in hint_correct_preds if p.get("composed_correct"))
            print(f"  [hint_correct] {len(hint_correct_preds)}/{len(eligible_indices)} | correct: {correct}")

    hint_correct_results = compute_mechanism_results(hint_correct_preds, "hint_correct")

    # hint_wrong condition
    print(f"\n[INFO] Running hint_wrong condition...")
    hint_wrong_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        question = ex.get("question", "")
        gold_answer_text = ex.get("full_answer", ex.get("answer", ""))
        gold_type = ex.get("structural_type", "MULTI-STEP")
        gold_answer = str(ex.get("final_answer", "")) if ex.get("final_answer") else extract_gold_answer(gold_answer_text)
        gold_answer = gold_answer.replace(",", "")

        wrong_type = pick_wrong_type(gold_type, MATH_TYPES)
        hint = get_wrong_hint(wrong_type)

        try:
            r = call_fn(COMPOSED_HINT.format(question=question, hint=hint))
            time.sleep(REQUEST_DELAY)

            pred_answer = extract_answer(r)
            composed_correct = pred_answer == gold_answer

            hint_wrong_preds.append({
                "idx": i, "gold_type": gold_type,
                "gold_answer": gold_answer, "pred_answer": pred_answer,
                "composed_correct": composed_correct,
                "condition": "hint_wrong",
                "wrong_type_given": wrong_type,
                "hint_given": hint,
            })
        except Exception as e:
            hint_wrong_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if len(hint_wrong_preds) % 50 == 0:
            correct = sum(1 for p in hint_wrong_preds if p.get("composed_correct"))
            print(f"  [hint_wrong] {len(hint_wrong_preds)}/{len(eligible_indices)} | correct: {correct}")

    hint_wrong_results = compute_mechanism_results(hint_wrong_preds, "hint_wrong")

    # Compare and save
    all_results = {
        "original": original_results,
        "hint_correct": hint_correct_results,
        "hint_wrong": hint_wrong_results,
        "predictions": {
            "hint_correct": hint_correct_preds,
            "hint_wrong": hint_wrong_preds,
        },
    }

    print_mechanism_comparison(all_results, model_key, "Math")
    save_mechanism_results(all_results, model_key, "math", RESULTS_DIR)
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
