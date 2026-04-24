#!/usr/bin/env python3
"""
Run API-based LLM baselines on FOLIO for Composition Gap analysis.

Measures:
  - Piece-level: premise extraction, relation identification
  - Structure-level: reasoning structural type Macro F1
  - Answer accuracy: True/False/Unknown classification

Models (need API keys in .env):
  - gpt4o, gpt4o-cot        (OpenAI GPT-4o)
  - deepseek, deepseek-cot   (DeepSeek-chat)

Usage:
    python run_baselines.py --model gpt4o
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

from classify_logic_structure import classify_logic, classify_response, STRUCTURAL_TYPES

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


# ─── Piece extraction ───────────────────────────────────────────────────────

def extract_pieces_from_premises(premises: list) -> dict:
    """Extract piece-level information from premises."""
    entities = set()
    relations = set()

    for p in premises:
        # Extract entities (capitalized words, nouns)
        words = p.split()
        for i, w in enumerate(words):
            clean = re.sub(r'[^\w]', '', w)
            if clean and clean[0].isupper() and len(clean) > 1 and clean.lower() not in {
                'the', 'if', 'then', 'all', 'every', 'some', 'no', 'not', 'and', 'or',
                'is', 'are', 'has', 'have', 'will', 'can', 'either', 'neither', 'when',
                'whenever', 'therefore', 'however', 'also', 'but', 'true', 'false',
            }:
                entities.add(clean)

        # Extract relations (verbs + predicates)
        rel_patterns = re.findall(r'\b(is|are|has|have|can|will|likes?|plays?|knows?|works?|lives?|goes?|makes?|reads?|writes?|eats?|drinks?|visits?|attends?)\s+(\w+)', p.lower())
        for verb, obj in rel_patterns:
            relations.add(f"{verb}_{obj}")

    return {
        "entities": sorted(entities),
        "relations": sorted(relations),
    }


def extract_pieces_from_response(response: str) -> dict:
    """Extract pieces from model response (what entities/relations it mentions)."""
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

    return {
        "entities": sorted(entities),
        "relations": sorted(relations),
    }


def extract_answer(response: str) -> str:
    """Extract True/False/Unknown from model response."""
    if not response:
        return "Unknown"

    resp = response.strip()

    # Check for explicit ANSWER: format
    match = re.search(r'ANSWER:\s*(True|False|Unknown)', resp, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    # Check last line
    last_line = resp.strip().split('\n')[-1].strip()
    for label in ['True', 'False', 'Unknown']:
        if label.lower() in last_line.lower():
            return label

    # Check first word
    first_word = resp.split()[0].strip('.,;:') if resp.split() else ""
    for label in ['True', 'False', 'Unknown']:
        if first_word.lower() == label.lower():
            return label

    return "Unknown"


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
            {"role": "system", "content": "You are a logical reasoning expert. Analyze premises carefully and determine if conclusions follow."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


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
    provider = config["provider"]
    model = config["model"]
    use_cot = config["cot"]

    print(f"\n{'='*60}")
    print(f"  FOLIO Baseline: {model_key} ({model})")
    print(f"{'='*60}")

    if provider == "openai" and not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not set.")
        return None
    if provider == "deepseek" and not DEEPSEEK_API_KEY:
        print("[ERROR] DEEPSEEK_API_KEY not set.")
        return None

    prompt_template = COT_PROMPT if use_cot else ZERO_SHOT_PROMPT
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
            premises_text = "\n".join(f"- {p}" for p in premises)
            formatted = prompt_template.format(premises=premises_text, conclusion=conclusion)
            response = call_model(formatted, provider, model)
            time.sleep(REQUEST_DELAY)

            # Extract answer
            pred_label = extract_answer(response)
            answer_correct = pred_label.lower() == str(gold_label).lower()

            # Classify reasoning structure
            pred_type = classify_response(premises, conclusion, response)

            # Extract pieces
            gold_pieces = extract_pieces_from_premises(premises)
            pred_pieces = extract_pieces_from_response(response)

            predictions.append({
                "idx": i,
                "gold_type": gold_type,
                "pred_type": pred_type,
                "gold_label": str(gold_label),
                "pred_label": pred_label,
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
                "idx": i,
                "gold_type": gold_type,
                "pred_type": "MODUS-PONENS",
                "gold_label": str(gold_label),
                "pred_label": "Unknown",
                "answer_correct": False,
                "gold_entities": [],
                "pred_entities": [],
                "gold_relations": [],
                "pred_relations": [],
                "raw_response": "",
                "error": str(e),
            })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(data) - i - 1) / rate if rate > 0 else 0
            correct = sum(1 for p in predictions if p["pred_type"] == p["gold_type"])
            ans_correct = sum(1 for p in predictions if p.get("answer_correct"))
            print(f"[INFO] Progress: {i+1}/{len(data)} "
                  f"(errors: {errors}, struct_acc: {correct/(i+1):.3f}, "
                  f"ans_acc: {ans_correct/(i+1):.3f}, ETA: {eta/60:.1f} min)")

    elapsed = time.time() - start_time
    print(f"[INFO] Done. {len(predictions)} predictions, {errors} errors, {elapsed/60:.1f} min")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RESULTS_DIR / f"predictions_{model_key}.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)
    print(f"[INFO] Predictions saved to {pred_path}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description="FOLIO Composition Gap Baselines (API)")
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
