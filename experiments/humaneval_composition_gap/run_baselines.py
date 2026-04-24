#!/usr/bin/env python3
"""
Run API-based LLM baselines on HumanEval for Composition Gap analysis.

Measures:
  - Piece-level: identifier extraction, operation identification
  - Structure-level: structural type classification (via classify_code_structure)

Models (need API keys in .env):
  - gpt4o, gpt4o-cot        (OpenAI GPT-4o)
  - deepseek, deepseek-cot   (DeepSeek-chat)

Usage:
    python run_baselines.py --model gpt4o
    python run_baselines.py --model gpt4o-cot
    python run_baselines.py --model all
    python run_baselines.py --model gpt4o --limit 10
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

from classify_code_structure import classify_code, classify_response, STRUCTURAL_TYPES

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

# Load .env
for _env_candidate in [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / "spider_composition_gap" / ".env",
]:
    if _env_candidate.exists():
        with open(_env_candidate) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _val = _line.split("=", 1)
                    os.environ.setdefault(_key.strip(), _val.strip())
        break

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

MODELS = {
    "gpt4o":        {"provider": "openai",   "model": "gpt-4o",        "cot": False},
    "gpt4o-cot":    {"provider": "openai",   "model": "gpt-4o",        "cot": True},
    "deepseek":     {"provider": "deepseek", "model": "deepseek-chat", "cot": False},
    "deepseek-cot": {"provider": "deepseek", "model": "deepseek-chat", "cot": True},
}

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


# ─── Prompt templates ────────────────────────────────────────────────────────

ZERO_SHOT_PROMPT = """Complete this Python function:

{prompt}

Return ONLY the function body (the code that goes after the function signature). No explanation, no markdown."""

COT_PROMPT = """Complete this Python function:

{prompt}

Think step by step:
1. What inputs does the function take?
2. What operations are needed (loops, conditionals, recursion)?
3. What is the structural pattern? (simple computation, loop-based, conditional branching, recursive, or multi-structure)
4. Write the solution.

After your reasoning, write the final code inside a ```python``` block."""


# ─── Piece extraction ───────────────────────────────────────────────────────

def extract_gold_pieces(prompt: str, solution: str) -> dict:
    """Extract piece-level information from gold solution."""
    full_code = prompt + solution

    # Identifiers: function names, variable names
    func_names = set(re.findall(r'def\s+(\w+)\s*\(', full_code))
    # Variables assigned
    var_names = set(re.findall(r'(\w+)\s*=\s*', solution))
    # Parameters from prompt
    param_match = re.search(r'def\s+\w+\s*\(([^)]*)\)', prompt)
    params = set()
    if param_match:
        for p in param_match.group(1).split(','):
            p = p.strip().split(':')[0].split('=')[0].strip()
            if p:
                params.add(p)

    identifiers = func_names | var_names | params

    # Operations: what kinds of operations are used
    operations = set()
    if re.search(r'[\+\-\*\/\%]', solution):
        operations.add('arithmetic')
    if re.search(r'\b(len|sum|min|max|sorted|reversed|enumerate|zip|map|filter)\b', solution):
        operations.add('builtin')
    if re.search(r'\.(append|extend|insert|pop|remove|sort|reverse|split|join|strip|replace)\b', solution):
        operations.add('method')
    if re.search(r'\bfor\s+\w+\s+in\s+', solution):
        operations.add('iteration')
    if re.search(r'\bif\s+', solution):
        operations.add('branching')
    if re.search(r'\breturn\b', solution):
        operations.add('return')
    if re.search(r'\[.*\s+for\s+', solution):
        operations.add('comprehension')
    if any(fname in solution.split('def ')[0] if 'def ' in solution else solution
           for fname in func_names):
        operations.add('recursion')

    return {
        'identifiers': sorted(identifiers),
        'operations': sorted(operations),
    }


def extract_pred_pieces(prompt: str, response: str) -> dict:
    """Extract piece-level information from model response."""
    # Extract code from response
    code_match = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        code = response.strip()

    return extract_gold_pieces(prompt, code)


# ─── API calls ──────────────────────────────────────────────────────────────

def call_model(prompt: str, provider: str, model: str) -> str:
    from openai import OpenAI
    if provider == "openai":
        client = OpenAI(api_key=OPENAI_API_KEY)
    elif provider == "deepseek":
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a Python programming expert. Write clean, correct code."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ─── Data loading ───────────────────────────────────────────────────────────

def load_data(limit: int = None):
    classified_path = DATA_DIR / "humaneval_classified.json"
    if not classified_path.exists():
        raw_path = DATA_DIR / "humaneval.json"
        if not raw_path.exists():
            print("[ERROR] No data found. Run download_humaneval.py first.")
            sys.exit(1)
        classified_path = raw_path

    with open(classified_path) as f:
        data = json.load(f)

    if limit:
        data = data[:limit]
    return data


# ─── Main pipeline ──────────────────────────────────────────────────────────

def run_baseline(model_key: str, data: list):
    config = MODELS[model_key]
    provider = config["provider"]
    model = config["model"]
    use_cot = config["cot"]

    print(f"\n{'='*60}")
    print(f"  HumanEval Baseline: {model_key} ({model})")
    print(f"{'='*60}")

    if provider == "openai" and not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not set.")
        return None
    if provider == "deepseek" and not DEEPSEEK_API_KEY:
        print("[ERROR] DEEPSEEK_API_KEY not set.")
        return None

    prompt_template = COT_PROMPT if use_cot else ZERO_SHOT_PROMPT
    print(f"[INFO] Processing {len(data)} problems ({'CoT' if use_cot else 'zero-shot'})...")

    predictions = []
    errors = 0
    start_time = time.time()

    for i, example in enumerate(data):
        task_id = example["task_id"]
        prompt = example["prompt"]
        gold_solution = example.get("canonical_solution", "")
        gold_type = example.get("structural_type", "SINGLE-FUNC")

        try:
            # Build prompt
            formatted = prompt_template.format(prompt=prompt)
            response = call_model(formatted, provider, model)
            time.sleep(REQUEST_DELAY)

            # Classify response
            pred_type = classify_response(prompt, response)

            # Extract pieces
            gold_pieces = extract_gold_pieces(prompt, gold_solution)
            pred_pieces = extract_pred_pieces(prompt, response)

            predictions.append({
                "idx": i,
                "task_id": task_id,
                "gold_type": gold_type,
                "pred_type": pred_type,
                "gold_identifiers": gold_pieces["identifiers"],
                "pred_identifiers": pred_pieces["identifiers"],
                "gold_ops": gold_pieces["operations"],
                "pred_ops": pred_pieces["operations"],
                "raw_response": response[:2000],
            })

        except Exception as e:
            errors += 1
            predictions.append({
                "idx": i,
                "task_id": task_id,
                "gold_type": gold_type,
                "pred_type": "SINGLE-FUNC",
                "gold_identifiers": [],
                "pred_identifiers": [],
                "gold_ops": [],
                "pred_ops": [],
                "raw_response": "",
                "error": str(e),
            })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(data) - i - 1) / rate if rate > 0 else 0
            correct = sum(1 for p in predictions if p["pred_type"] == p["gold_type"])
            print(f"[INFO] Progress: {i+1}/{len(data)} "
                  f"(errors: {errors}, struct_acc: {correct/(i+1):.3f}, ETA: {eta/60:.1f} min)")

    elapsed = time.time() - start_time
    print(f"[INFO] Done. {len(predictions)} predictions, {errors} errors, {elapsed/60:.1f} min")

    # Save predictions
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RESULTS_DIR / f"predictions_{model_key}.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"[INFO] Predictions saved to {pred_path}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="HumanEval Composition Gap Baselines (API)")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data = load_data(args.limit)
    print(f"[INFO] Loaded {len(data)} problems")

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        run_baseline(model_key, data)


if __name__ == "__main__":
    main()
