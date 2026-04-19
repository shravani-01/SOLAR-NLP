#!/usr/bin/env python3
"""
Run LLM baselines on GSM8K for Composition Gap analysis.

Tests piece-level extraction (numbers, operations, answer) vs
structure-level classification (solution pattern type).

Usage:
    python run_baselines.py --model gpt4o
    python run_baselines.py --model gpt4o-cot
    python run_baselines.py --model deepseek
    python run_baselines.py --model deepseek-cot
    python run_baselines.py --model all
    python run_baselines.py --model gpt4o --limit 50
"""

import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

from classify_math_structure import classify_math, STRUCTURAL_TYPES

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

# Load .env
ENV_PATH = Path(__file__).parent.parent / "spider_composition_gap" / ".env"
if not ENV_PATH.exists():
    ENV_PATH = Path(__file__).parent / ".env"
if ENV_PATH.exists():
    with open(ENV_PATH) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

MODELS = {
    "gpt4o": {"provider": "openai", "model": "gpt-4o", "cot": False},
    "gpt4o-cot": {"provider": "openai", "model": "gpt-4o", "cot": True},
    "deepseek": {"provider": "deepseek", "model": "deepseek-chat", "cot": False},
    "deepseek-cot": {"provider": "deepseek", "model": "deepseek-chat", "cot": True},
}

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


# ─── Prompt templates ────────────────────────────────────────────────────────

ZERO_SHOT_PROMPT = """Solve this math problem:

{question}

Show your work step by step. On the final line, write your answer as a single number after "#### " (e.g., #### 42)."""


COT_PROMPT = """Solve this math problem:

{question}

Think step by step:
1. What quantities are given?
2. What operations are needed?
3. What is the structural pattern? (single operation, multi-step, ratio/percentage, comparison, or iterative)
4. Work through each step.
5. On the final line, write your answer as a single number after "#### " (e.g., #### 42)."""


def build_prompt(question: str, cot: bool) -> str:
    template = COT_PROMPT if cot else ZERO_SHOT_PROMPT
    return template.format(question=question)


# ─── API calls ───────────────────────────────────────────────────────────────

def call_openai(prompt: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a math expert. Solve problems step by step and give the final numerical answer."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def call_deepseek(prompt: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a math expert. Solve problems step by step and give the final numerical answer."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def call_model(prompt: str, provider: str, model: str) -> str:
    if provider == "openai":
        return call_openai(prompt, model)
    elif provider == "deepseek":
        return call_deepseek(prompt, model)
    raise ValueError(f"Unknown provider: {provider}")


# ─── Response parsing ────────────────────────────────────────────────────────

def extract_answer(response: str) -> str:
    """Extract final numerical answer from response."""
    if not response:
        return ""

    # Look for #### pattern
    match = re.search(r'####\s*([\d,\.]+)', response)
    if match:
        return match.group(1).replace(",", "")

    # Look for "answer is X" pattern
    match = re.search(r'(?:answer|result|total)\s+(?:is|=)\s*\$?\s*([\d,\.]+)', response, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    # Look for boxed answer \boxed{X}
    match = re.search(r'\\boxed\{([\d,\.]+)\}', response)
    if match:
        return match.group(1).replace(",", "")

    # Last number in the response
    numbers = re.findall(r'[\d,]+\.?\d*', response)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def extract_numbers(text: str) -> set:
    """Extract all numbers mentioned in text."""
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    return set(numbers)


def extract_operations(text: str) -> set:
    """Extract arithmetic operations used."""
    ops = set()
    if re.search(r'[\+]|add|plus|sum|total|combined', text, re.IGNORECASE):
        ops.add("ADD")
    if re.search(r'[\-]|subtract|minus|less|difference|remain|left', text, re.IGNORECASE):
        ops.add("SUB")
    if re.search(r'[\*×]|multipl|times|product', text, re.IGNORECASE):
        ops.add("MUL")
    if re.search(r'[\/÷]|divid|per|each|split|share', text, re.IGNORECASE):
        ops.add("DIV")
    if re.search(r'percent|%|ratio|fraction|half|third|quarter', text, re.IGNORECASE):
        ops.add("PCT")
    return ops


def extract_steps_from_response(response: str) -> list:
    """Extract solution steps from model response."""
    lines = response.strip().split('\n')
    steps = [l.strip() for l in lines if l.strip() and not l.strip().startswith('####')]
    return steps


def classify_response(response: str) -> str:
    """Classify the structural type of the model's solution approach."""
    steps = extract_steps_from_response(response)
    return classify_math("", response, steps)


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_baseline(model_key: str, test_data: list, limit: int = None):
    """Run a single baseline."""
    config = MODELS[model_key]
    provider = config["provider"]
    model = config["model"]
    cot = config["cot"]

    print(f"\n{'='*60}")
    print(f"  Running: {model_key} ({model}, CoT={cot})")
    print(f"{'='*60}")

    if provider == "openai" and not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not set.")
        return None
    if provider == "deepseek" and not DEEPSEEK_API_KEY:
        print("[ERROR] DEEPSEEK_API_KEY not set.")
        return None

    examples = test_data[:limit] if limit else test_data
    print(f"[INFO] Processing {len(examples)} examples...")

    predictions = []
    errors = 0

    for i, example in enumerate(examples):
        question = example["question"]
        gold_answer = example.get("final_answer", "")
        gold_type = example.get("structural_type", "MULTI-STEP")

        prompt = build_prompt(question, cot)

        try:
            raw_response = call_model(prompt, provider, model)
            pred_answer = extract_answer(raw_response)
            pred_type = classify_response(raw_response)

            # Piece-level: extract numbers and operations
            gold_numbers = extract_numbers(question)
            pred_numbers = extract_numbers(raw_response)
            gold_ops = extract_operations(example.get("full_answer", ""))
            pred_ops = extract_operations(raw_response)

            # Answer correctness
            try:
                answer_correct = abs(float(pred_answer) - float(gold_answer)) < 0.01 if pred_answer and gold_answer else False
            except (ValueError, TypeError):
                answer_correct = pred_answer.strip() == gold_answer.strip()

            predictions.append({
                "idx": i,
                "question": question,
                "gold_answer": gold_answer,
                "gold_type": gold_type,
                "pred_answer": pred_answer,
                "pred_type": pred_type,
                "answer_correct": answer_correct,
                "gold_numbers": list(gold_numbers),
                "pred_numbers": list(pred_numbers),
                "gold_ops": list(gold_ops),
                "pred_ops": list(pred_ops),
                "raw_response": raw_response,
            })
        except Exception as e:
            errors += 1
            predictions.append({
                "idx": i,
                "question": question,
                "gold_answer": gold_answer,
                "gold_type": gold_type,
                "pred_answer": "",
                "pred_type": "SINGLE-OP",
                "answer_correct": False,
                "gold_numbers": [],
                "pred_numbers": [],
                "gold_ops": [],
                "pred_ops": [],
                "raw_response": f"ERROR: {str(e)}",
                "error": True,
            })

        if (i + 1) % 25 == 0:
            print(f"[INFO] Progress: {i + 1}/{len(examples)} (errors: {errors})")

        time.sleep(REQUEST_DELAY)

    print(f"[INFO] Done. {len(predictions)} predictions, {errors} errors.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RESULTS_DIR / f"predictions_{model_key}.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"[INFO] Predictions saved to {pred_path}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run GSM8K baselines for Composition Gap")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Load classified test data
    test_path = DATA_DIR / "test_classified.json"
    if not test_path.exists():
        test_path = DATA_DIR / "test.json"
    if not test_path.exists():
        print("[ERROR] No GSM8K data found. Run download_gsm8k.py first.")
        sys.exit(1)

    with open(test_path) as f:
        test_data = json.load(f)
    print(f"[INFO] Loaded {len(test_data)} examples")

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    all_predictions = {}
    for model_key in models_to_run:
        predictions = run_baseline(model_key, test_data, args.limit)
        if predictions:
            all_predictions[model_key] = predictions

    if all_predictions:
        print(f"\n{'='*60}")
        print(f"  Summary")
        print(f"{'='*60}")
        for model_key, preds in all_predictions.items():
            n = len(preds)
            errs = sum(1 for p in preds if p.get("error"))
            correct = sum(1 for p in preds if p.get("answer_correct"))
            type_correct = sum(1 for p in preds if p["gold_type"] == p["pred_type"])
            print(f"  {model_key}: {n} examples, {errs} errors, "
                  f"answer acc: {correct/n:.3f}, structure acc: {type_correct/n:.3f}")


if __name__ == "__main__":
    main()
