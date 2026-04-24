#!/usr/bin/env python3
"""
Run OSS LLM baselines on HumanEval for Composition Gap analysis (GPU required).

Models:
  - qwen7b, qwen7b-cot       (Qwen2.5-7B-Instruct)
  - llama8b, llama8b-cot     (Llama-3.1-8B-Instruct)

Usage:
    python run_baselines_oss.py --model qwen7b
    python run_baselines_oss.py --model llama8b
    python run_baselines_oss.py --model all
    python run_baselines_oss.py --model qwen7b --limit 10
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

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from classify_code_structure import classify_code, classify_response, STRUCTURAL_TYPES

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

MODELS = {
    "qwen7b":       {"hf_name": "Qwen/Qwen2.5-7B-Instruct",          "cot": False},
    "qwen7b-cot":   {"hf_name": "Qwen/Qwen2.5-7B-Instruct",          "cot": True},
    "llama8b":      {"hf_name": "meta-llama/Llama-3.1-8B-Instruct",   "cot": False},
    "llama8b-cot":  {"hf_name": "meta-llama/Llama-3.1-8B-Instruct",   "cot": True},
}

# ─── Prompt templates ────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a Python programming expert. Write clean, correct code."

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


# ─── Piece extraction (same as API version) ─────────────────────────────────

def extract_gold_pieces(prompt: str, solution: str) -> dict:
    full_code = prompt + solution
    func_names = set(re.findall(r'def\s+(\w+)\s*\(', full_code))
    var_names = set(re.findall(r'(\w+)\s*=\s*', solution))
    param_match = re.search(r'def\s+\w+\s*\(([^)]*)\)', prompt)
    params = set()
    if param_match:
        for p in param_match.group(1).split(','):
            p = p.strip().split(':')[0].split('=')[0].strip()
            if p:
                params.add(p)
    identifiers = func_names | var_names | params

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

    return {'identifiers': sorted(identifiers), 'operations': sorted(operations)}


def extract_pred_pieces(prompt: str, response: str) -> dict:
    code_match = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    code = code_match.group(1).strip() if code_match else response.strip()
    return extract_gold_pieces(prompt, code)


# ─── Model loading ──────────────────────────────────────────────────────────

_loaded_models = {}

def load_model(hf_name: str):
    if hf_name in _loaded_models:
        return _loaded_models[hf_name]

    print(f"[INFO] Loading {hf_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"[INFO] Loaded. Device: {model.device}")

    _loaded_models[hf_name] = (model, tokenizer)
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, use_cot: bool) -> str:
    template = COT_PROMPT if use_cot else ZERO_SHOT_PROMPT
    user_msg = template.format(prompt=prompt)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=None,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


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
    hf_name = config["hf_name"]
    use_cot = config["cot"]

    print(f"\n{'='*60}")
    print(f"  HumanEval Baseline (OSS): {model_key}")
    print(f"{'='*60}")

    model, tokenizer = load_model(hf_name)
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
            response = generate_response(model, tokenizer, prompt, use_cot)
            pred_type = classify_response(prompt, response)
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RESULTS_DIR / f"predictions_{model_key}.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"[INFO] Predictions saved to {pred_path}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="HumanEval Composition Gap Baselines (OSS/GPU)")
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
