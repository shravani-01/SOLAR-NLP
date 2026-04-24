#!/usr/bin/env python3
"""
2-Pass Composition Gap — Math (GSM8K).

Pass 1 (Pieces): Ask 2 sub-questions separately:
  Q1: "What numbers/quantities are given in this problem?"
  Q2: "What math operations are needed (add, subtract, multiply, divide)?"

Pass 2 (Composed): "Solve the problem and show your work."

Gap = P(Pass 2 wrong answer | Pass 1 all correct)

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
from core import (call_api, call_oss, load_oss_model, API_MODELS, OSS_MODELS,
                  REQUEST_DELAY, compute_composition_gap, print_results, save_results)

DATA_DIR = Path(__file__).parent.parent.parent / "experiments" / "gsm8k_composition_gap" / "data"
RESULTS_DIR = Path(__file__).parent / "results"

SYSTEM = "You are a math expert. Answer precisely."

# ─── Piece sub-questions ────────────────────────────────────────────────────

PIECE_Q1 = """Read this math problem:

"{question}"

List ALL the numbers and quantities mentioned in the problem. Just list them, comma-separated."""

PIECE_Q2 = """Read this math problem:

"{question}"

What math OPERATIONS are needed to solve this? Choose from: addition, subtraction, multiplication, division, comparison, percentage.

List only the operations needed, comma-separated."""

# ─── Composed question ─────────────────────────────────────────────────────

COMPOSED_Q = """Solve this math problem:

"{question}"

Show your work step by step. On the final line, write your answer as a single number after "#### "."""

# ─── Piece evaluation ──────────────────────────────────────────────────────

def extract_gold_numbers(question: str, answer: str) -> set:
    """Extract numbers from question."""
    nums = set()
    for m in re.finditer(r'\b(\d+(?:\.\d+)?)\b', question):
        nums.add(m.group(1))
    return nums


def parse_numbers_response(response: str) -> set:
    """Extract numbers from model's response to Q1."""
    nums = set()
    for m in re.finditer(r'\b(\d+(?:\.\d+)?)\b', response):
        nums.add(m.group(1))
    return nums


def extract_gold_ops(question: str, answer: str) -> set:
    """Extract operations from question + gold answer."""
    ops = set()
    combined = (question + " " + answer).lower()
    if re.search(r'\+|add|total|sum|together|combined|more than|gave|bought|additional', combined):
        ops.add("addition")
    if re.search(r'\-|subtract|minus|left|remain|fewer|less than|difference|lost|spent|gave away', combined):
        ops.add("subtraction")
    if re.search(r'\*|×|multiply|times|each|per|every|double|triple|twice', combined):
        ops.add("multiplication")
    if re.search(r'\/|÷|divide|split|shared|average|per|half|third|quarter', combined):
        ops.add("division")
    if re.search(r'%|percent|ratio|proportion|fraction', combined):
        ops.add("percentage")
    return ops if ops else {"arithmetic"}


def parse_ops_response(response: str) -> set:
    """Parse operations from model's response to Q2."""
    ops = set()
    r = response.lower()
    if "addition" in r or "add" in r:
        ops.add("addition")
    if "subtraction" in r or "subtract" in r:
        ops.add("subtraction")
    if "multiplication" in r or "multiply" in r:
        ops.add("multiplication")
    if "division" in r or "divide" in r:
        ops.add("division")
    if "percentage" in r or "percent" in r:
        ops.add("percentage")
    if "comparison" in r or "compare" in r:
        ops.add("comparison")
    return ops if ops else {"arithmetic"}


def extract_answer(response: str) -> str:
    """Extract numerical answer from #### format."""
    match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', response)
    if match:
        return match.group(1).replace(',', '')
    # Fallback: last number in response
    nums = re.findall(r'[\d,]+(?:\.\d+)?', response)
    return nums[-1].replace(',', '') if nums else ""


def extract_gold_answer(answer_text: str) -> str:
    """Extract gold answer."""
    match = re.search(r'####\s*([\d,]+(?:\.\d+)?)', answer_text)
    if match:
        return match.group(1).replace(',', '')
    return ""


# ─── Data loading ───────────────────────────────────────────────────────────

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


# ─── Main ───────────────────────────────────────────────────────────────────

def run(model_key: str, data: list, backend: str = "api"):
    if backend == "api":
        config = API_MODELS[model_key]
        provider, model_name = config["provider"], config["model"]
        call_fn = lambda p: call_api(p, provider, model_name, SYSTEM)
    else:
        config = OSS_MODELS[model_key]
        model, tokenizer = load_oss_model(config["hf_name"])
        call_fn = lambda p: call_oss(p, model, tokenizer, SYSTEM)

    print(f"\n[INFO] Math 2-Pass: {model_key} ({len(data)} examples)")

    predictions = []
    errors = 0

    for i, ex in enumerate(data):
        question = ex.get("question", "")
        gold_answer_text = ex.get("full_answer", ex.get("answer", ""))
        gold_type = ex.get("structural_type", "MULTI-STEP")
        # Try final_answer field first (GSM8K classified format)
        gold_answer = str(ex.get("final_answer", "")) if ex.get("final_answer") else extract_gold_answer(gold_answer_text)
        gold_answer = gold_answer.replace(",", "")
        gold_numbers = extract_gold_numbers(question, gold_answer_text)
        gold_ops = extract_gold_ops(question, gold_answer_text)

        try:
            # PASS 1: Piece sub-questions
            r1 = call_fn(PIECE_Q1.format(question=question))
            time.sleep(REQUEST_DELAY)
            r2 = call_fn(PIECE_Q2.format(question=question))
            time.sleep(REQUEST_DELAY)

            pred_numbers = parse_numbers_response(r1)
            pred_ops = parse_ops_response(r2)

            # Check pieces: model found most numbers and right operation types
            q1_correct = len(gold_numbers & pred_numbers) >= len(gold_numbers) * 0.7 if gold_numbers else True
            q2_correct = len(gold_ops & pred_ops) >= 1 if gold_ops else True
            pieces_all_correct = q1_correct and q2_correct

            # PASS 2: Composed question
            r3 = call_fn(COMPOSED_Q.format(question=question))
            time.sleep(REQUEST_DELAY)

            pred_answer = extract_answer(r3)
            composed_correct = pred_answer == gold_answer

            predictions.append({
                "idx": i, "question": question[:200],
                "gold_type": gold_type,
                "gold_answer": gold_answer, "pred_answer": pred_answer,
                "q1_numbers_correct": q1_correct,
                "q2_ops_correct": q2_correct,
                "pieces_all_correct": pieces_all_correct,
                "composed_correct": composed_correct,
            })

        except Exception as e:
            errors += 1
            predictions.append({
                "idx": i, "gold_type": gold_type,
                "pieces_all_correct": False, "composed_correct": False,
                "error": str(e),
            })

        if (i + 1) % 50 == 0:
            pc = sum(1 for p in predictions if p.get("pieces_all_correct"))
            gap = sum(1 for p in predictions if p.get("pieces_all_correct") and not p.get("composed_correct"))
            print(f"[INFO] {i+1}/{len(data)} | pieces_ok: {pc} | gap_cases: {gap} | errors: {errors}")

    results = compute_composition_gap(predictions)
    print_results(results, model_key, "Math (GSM8K)")
    save_results(predictions, results, model_key, "math", RESULTS_DIR)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default="api", choices=["api", "oss"])
    args = parser.parse_args()

    data = load_data(args.limit)
    print(f"[INFO] Loaded {len(data)} examples")

    models = API_MODELS if args.backend == "api" else OSS_MODELS
    to_run = list(models.keys()) if args.model == "all" else [args.model]

    for mk in to_run:
        run(mk, data, backend=args.backend)


if __name__ == "__main__":
    main()
