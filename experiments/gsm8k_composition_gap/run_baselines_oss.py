#!/usr/bin/env python3
"""
Run open-source LLM baselines on GSM8K for Composition Gap analysis.

Runs locally on GPU using transformers (no API keys needed).

Models:
  - qwen7b, qwen7b-cot       (Qwen2.5-7B-Instruct)
  - qwen72b, qwen72b-cot     (Qwen2.5-72B-Instruct, 4-bit quantized)

Usage (on RunPod):
    python run_baselines_oss.py --model qwen7b
    python run_baselines_oss.py --model qwen7b-cot
    python run_baselines_oss.py --model qwen72b
    python run_baselines_oss.py --model qwen72b-cot
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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from classify_math_structure import classify_math, STRUCTURAL_TYPES

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

MODELS = {
    "qwen7b": {
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "cot": False,
        "quantize": False,
    },
    "qwen7b-cot": {
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "cot": True,
        "quantize": False,
    },
    "qwen72b": {
        "hf_name": "Qwen/Qwen2.5-72B-Instruct",
        "cot": False,
        "quantize": True,
    },
    "qwen72b-cot": {
        "hf_name": "Qwen/Qwen2.5-72B-Instruct",
        "cot": True,
        "quantize": True,
    },
}

MAX_NEW_TOKENS = 1024


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


# ─── Model loading & inference ──────────────────────────────────────────────

def load_model(hf_name: str, quantize: bool):
    """Load model and tokenizer."""
    print(f"[INFO] Loading {hf_name} (quantize={quantize})...")

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    print(f"[INFO] Model loaded. Device: {model.device}")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str) -> str:
    """Generate a response from the model."""
    messages = [
        {"role": "system", "content": "You are a math expert. Solve problems step by step and give the final numerical answer."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=None,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


# ─── Response parsing ────────────────────────────────────────────────────────

def extract_answer(response: str) -> str:
    """Extract final numerical answer from response."""
    if not response:
        return ""

    match = re.search(r'####\s*([\d,\.]+)', response)
    if match:
        return match.group(1).replace(",", "")

    match = re.search(r'(?:answer|result|total)\s+(?:is|=)\s*\$?\s*([\d,\.]+)', response, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")

    match = re.search(r'\\boxed\{([\d,\.]+)\}', response)
    if match:
        return match.group(1).replace(",", "")

    numbers = re.findall(r'[\d,]+\.?\d*', response)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def extract_numbers(text: str) -> set:
    """Extract all numbers mentioned in text."""
    return set(re.findall(r'\b\d+(?:\.\d+)?\b', text))


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


def classify_response(question: str, response: str) -> str:
    """Classify the structural type of the model's solution approach.
    Uses the original question for pattern detection (ratio, comparison, system)
    and the response for operation counting."""
    steps = [l.strip() for l in response.strip().split('\n') if l.strip() and not l.strip().startswith('####')]
    return classify_math(question, response, steps)


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_baseline(model_key: str, test_data: list, limit: int = None):
    """Run a single OSS baseline."""
    config = MODELS[model_key]
    hf_name = config["hf_name"]
    cot = config["cot"]
    quantize = config["quantize"]

    print(f"\n{'='*60}")
    print(f"  Running: {model_key} ({hf_name}, CoT={cot})")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = load_model(hf_name, quantize)

    examples = test_data[:limit] if limit else test_data
    print(f"[INFO] Processing {len(examples)} examples...")

    predictions = []
    errors = 0
    start_time = time.time()

    for i, example in enumerate(examples):
        question = example["question"]
        gold_answer = example.get("final_answer", "")
        gold_type = example.get("structural_type", "MULTI-STEP")

        prompt = build_prompt(question, cot)

        try:
            raw_response = generate_response(model, tokenizer, prompt)
            pred_answer = extract_answer(raw_response)
            pred_type = classify_response(question, raw_response)

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
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(examples) - i - 1) / rate if rate > 0 else 0
            print(f"[INFO] Progress: {i + 1}/{len(examples)} "
                  f"(errors: {errors}, {rate:.1f} ex/s, ETA: {eta/60:.1f} min)")

    elapsed = time.time() - start_time
    print(f"[INFO] Done. {len(predictions)} predictions, {errors} errors, "
          f"{elapsed/60:.1f} minutes")

    # Save predictions
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RESULTS_DIR / f"predictions_{model_key}.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"[INFO] Predictions saved to {pred_path}")

    # Free GPU memory before next model
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run OSS baselines on GSM8K for Composition Gap (GPU)")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()) + ["all"],
                        help="Which model to run")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples (for testing)")
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

    # Run in order: small models first, then large
    if args.model == "all":
        models_to_run = ["qwen7b", "qwen7b-cot", "qwen72b", "qwen72b-cot"]
    else:
        models_to_run = [args.model]

    all_predictions = {}
    for model_key in models_to_run:
        predictions = run_baseline(model_key, test_data, args.limit)
        if predictions:
            all_predictions[model_key] = predictions

    # Print summary
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
