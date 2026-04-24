#!/usr/bin/env python3
"""
Run OSS LLM baselines on FOLIO for Composition Gap analysis (GPU required).

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

from classify_logic_structure import classify_logic, classify_response, STRUCTURAL_TYPES

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

MODELS = {
    "qwen7b":       {"hf_name": "Qwen/Qwen2.5-7B-Instruct",          "cot": False},
    "qwen7b-cot":   {"hf_name": "Qwen/Qwen2.5-7B-Instruct",          "cot": True},
    "llama8b":      {"hf_name": "meta-llama/Llama-3.1-8B-Instruct",   "cot": False},
    "llama8b-cot":  {"hf_name": "meta-llama/Llama-3.1-8B-Instruct",   "cot": True},
}

SYSTEM_PROMPT = "You are a logical reasoning expert. Analyze premises carefully and determine if conclusions follow."

ZERO_SHOT_PROMPT = """Given the following premises, determine if the conclusion is True, False, or Unknown.

Premises:
{premises}

Conclusion: {conclusion}

Answer with ONLY one word: True, False, or Unknown."""

COT_PROMPT = """Given the following premises, determine if the conclusion is True, False, or Unknown.

Premises:
{premises}

Conclusion: {conclusion}

Think step by step:
1. What are the key entities and relationships in the premises?
2. What logical structure connects the premises? (direct implication, categorical syllogism, disjunctive elimination, conditional chain, or negation/contradiction)
3. Can the conclusion be derived from the premises?
4. Is the conclusion True, False, or Unknown?

After reasoning, state your final answer as: ANSWER: True/False/Unknown"""


# ─── Piece extraction (same as API version) ─────────────────────────────────

def extract_pieces_from_premises(premises: list) -> dict:
    entities = set()
    relations = set()
    for p in premises:
        words = p.split()
        for w in words:
            clean = re.sub(r'[^\w]', '', w)
            if clean and clean[0].isupper() and len(clean) > 1 and clean.lower() not in {
                'the', 'if', 'then', 'all', 'every', 'some', 'no', 'not', 'and', 'or',
                'is', 'are', 'has', 'have', 'will', 'can', 'either', 'neither', 'when',
                'whenever', 'therefore', 'however', 'also', 'but', 'true', 'false',
            }:
                entities.add(clean)
        rel_patterns = re.findall(r'\b(is|are|has|have|can|will|likes?|plays?|knows?|works?|lives?)\s+(\w+)', p.lower())
        for verb, obj in rel_patterns:
            relations.add(f"{verb}_{obj}")
    return {"entities": sorted(entities), "relations": sorted(relations)}


def extract_pieces_from_response(response: str) -> dict:
    entities = set()
    relations = set()
    words = response.split()
    for w in words:
        clean = re.sub(r'[^\w]', '', w)
        if clean and clean[0].isupper() and len(clean) > 1 and clean.lower() not in {
            'the', 'if', 'then', 'all', 'every', 'some', 'no', 'not', 'and', 'or',
            'is', 'are', 'has', 'have', 'true', 'false', 'unknown', 'answer', 'step',
            'conclusion', 'premise', 'therefore', 'since', 'because', 'given', 'based',
        }:
            entities.add(clean)
    rel_patterns = re.findall(r'\b(is|are|has|have|can|will)\s+(\w+)', response.lower())
    for verb, obj in rel_patterns:
        relations.add(f"{verb}_{obj}")
    return {"entities": sorted(entities), "relations": sorted(relations)}


def extract_answer(response: str) -> str:
    if not response:
        return "Unknown"
    match = re.search(r'ANSWER:\s*(True|False|Unknown)', response, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    last_line = response.strip().split('\n')[-1].strip()
    for label in ['True', 'False', 'Unknown']:
        if label.lower() in last_line.lower():
            return label
    first_word = response.split()[0].strip('.,;:') if response.split() else ""
    for label in ['True', 'False', 'Unknown']:
        if first_word.lower() == label.lower():
            return label
    return "Unknown"


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
        hf_name, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model.eval()
    print(f"[INFO] Loaded. Device: {model.device}")
    _loaded_models[hf_name] = (model, tokenizer)
    return model, tokenizer


def generate_response(model, tokenizer, premises: list, conclusion: str, use_cot: bool) -> str:
    template = COT_PROMPT if use_cot else ZERO_SHOT_PROMPT
    premises_text = "\n".join(f"- {p}" for p in premises)
    user_msg = template.format(premises=premises_text, conclusion=conclusion)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=512, temperature=None,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─── Data loading ───────────────────────────────────────────────────────────

def load_data(limit: int = None):
    classified_path = DATA_DIR / "folio_classified.json"
    if not classified_path.exists():
        for fallback in ["folio_validation.json", "folio_test.json", "folio_all.json"]:
            fb_path = DATA_DIR / fallback
            if fb_path.exists():
                classified_path = fb_path
                break
    if not classified_path.exists():
        print("[ERROR] No data found. Run download_folio.py first.")
        sys.exit(1)
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
    print(f"  FOLIO Baseline (OSS): {model_key}")
    print(f"{'='*60}")

    model, tokenizer = load_model(hf_name)
    print(f"[INFO] Processing {len(data)} examples ({'CoT' if use_cot else 'zero-shot'})...")

    predictions = []
    errors = 0
    start_time = time.time()

    for i, example in enumerate(data):
        premises = example.get("premises", [])
        conclusion = example.get("conclusion", "")
        gold_label = example.get("label", "Unknown")
        gold_type = example.get("structural_type", "MODUS-PONENS")

        try:
            response = generate_response(model, tokenizer, premises, conclusion, use_cot)
            pred_label = extract_answer(response)
            answer_correct = pred_label.lower() == str(gold_label).lower()
            pred_type = classify_response(premises, conclusion, response)
            gold_pieces = extract_pieces_from_premises(premises)
            pred_pieces = extract_pieces_from_response(response)

            predictions.append({
                "idx": i, "gold_type": gold_type, "pred_type": pred_type,
                "gold_label": str(gold_label), "pred_label": pred_label,
                "answer_correct": answer_correct,
                "gold_entities": gold_pieces["entities"],
                "pred_entities": pred_pieces["entities"],
                "gold_relations": gold_pieces["relations"],
                "pred_relations": pred_pieces["relations"],
                "raw_response": response[:2000],
            })

        except Exception as e:
            errors += 1
            predictions.append({
                "idx": i, "gold_type": gold_type, "pred_type": "MODUS-PONENS",
                "gold_label": str(gold_label), "pred_label": "Unknown",
                "answer_correct": False,
                "gold_entities": [], "pred_entities": [],
                "gold_relations": [], "pred_relations": [],
                "raw_response": "", "error": str(e),
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
    parser = argparse.ArgumentParser(description="FOLIO Composition Gap Baselines (OSS/GPU)")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    data = load_data(args.limit)
    print(f"[INFO] Loaded {len(data)} examples")

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
    for model_key in models_to_run:
        run_baseline(model_key, data)


if __name__ == "__main__":
    main()
