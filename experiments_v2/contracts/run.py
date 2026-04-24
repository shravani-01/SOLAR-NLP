#!/usr/bin/env python3
"""
2-Pass Composition Gap — Contracts.

Pass 1 (Pieces): Ask 2 sub-questions separately:
  Q1: "Is this mandatory or discretionary language?"
  Q2: "Does this sentence contain a condition (if/when/unless)?"

Pass 2 (Composed): "What constraint type is this?"
  (HARD / SOFT / HARD-CONDITIONAL / SOFT-CONDITIONAL / NON-CONSTRAINT)

Gap = P(Pass 2 wrong | Pass 1 all correct)

Usage:
    python run.py --model gpt4o --limit 10
    python run.py --model all
    python run.py --model all --backend oss
"""

import csv
import json
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import (call_api, call_oss, load_oss_model, API_MODELS, OSS_MODELS,
                  REQUEST_DELAY, compute_composition_gap, print_results, save_results)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

CONSTRAINT_TYPES = ["HARD", "SOFT", "HARD-CONDITIONAL", "SOFT-CONDITIONAL", "NON-CONSTRAINT"]

SYSTEM = "You are a contract analysis expert. Answer precisely."

# ─── Piece sub-questions ────────────────────────────────────────────────────

PIECE_Q1 = """Read this labor contract sentence:

"{text}"

Question: Is the language MANDATORY (shall/must/required) or DISCRETIONARY (may/can/should) or NEITHER?

Answer with ONLY one word: MANDATORY, DISCRETIONARY, or NEITHER."""

PIECE_Q2 = """Read this labor contract sentence:

"{text}"

Question: Does this sentence contain a CONDITION (if/when/unless/provided that/in the event)?

Answer with ONLY one word: YES or NO."""

# ─── Composed question ─────────────────────────────────────────────────────

COMPOSED_Q = """Classify this labor contract sentence into one of 5 constraint types:

"{text}"

Types:
- HARD: Mandatory (shall/must), no conditions
- SOFT: Discretionary (may/can), no conditions
- HARD-CONDITIONAL: Mandatory WITH a condition (if/when/unless)
- SOFT-CONDITIONAL: Discretionary WITH a condition
- NON-CONSTRAINT: Not a constraint (definition, description)

Answer with ONLY the type name."""

# ─── Gold piece answers ────────────────────────────────────────────────────

def get_gold_pieces(gold_type: str) -> dict:
    """Derive gold piece answers from the gold constraint type."""
    modality_map = {
        "HARD": "MANDATORY", "SOFT": "DISCRETIONARY",
        "HARD-CONDITIONAL": "MANDATORY", "SOFT-CONDITIONAL": "DISCRETIONARY",
        "NON-CONSTRAINT": "NEITHER",
    }
    condition_map = {
        "HARD": "NO", "SOFT": "NO",
        "HARD-CONDITIONAL": "YES", "SOFT-CONDITIONAL": "YES",
        "NON-CONSTRAINT": "NO",
    }
    return {
        "modality": modality_map.get(gold_type, "NEITHER"),
        "has_condition": condition_map.get(gold_type, "NO"),
    }


def parse_modality(response: str) -> str:
    r = response.strip().upper()
    if "MANDATORY" in r:
        return "MANDATORY"
    if "DISCRETIONARY" in r:
        return "DISCRETIONARY"
    return "NEITHER"


def parse_condition(response: str) -> str:
    r = response.strip().upper()
    if r.startswith("YES"):
        return "YES"
    return "NO"


def parse_type(response: str) -> str:
    r = response.strip().upper()
    for ct in CONSTRAINT_TYPES:
        if ct in r:
            return ct
    return "NON-CONSTRAINT"


# ─── Data loading ───────────────────────────────────────────────────────────

def load_data(limit=None):
    test_path = Path(__file__).parent.parent.parent / "data" / "annotated" / "splits" / "test.csv"
    if not test_path.exists():
        # Try v1 location
        test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        print(f"[ERROR] No test data found")
        sys.exit(1)

    examples = []
    with open(test_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("constraint_type") in CONSTRAINT_TYPES:
                examples.append({
                    "text": row["raw_text"],
                    "gold_type": row["constraint_type"],
                })
    if limit:
        examples = examples[:limit]
    return examples


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

    print(f"\n[INFO] Contracts 2-Pass: {model_key} ({len(data)} examples)")

    predictions = []
    errors = 0

    for i, ex in enumerate(data):
        text = ex["text"]
        gold_type = ex["gold_type"]
        gold_pieces = get_gold_pieces(gold_type)

        try:
            # PASS 1: Piece sub-questions
            r1 = call_fn(PIECE_Q1.format(text=text))
            time.sleep(REQUEST_DELAY)
            r2 = call_fn(PIECE_Q2.format(text=text))
            time.sleep(REQUEST_DELAY)

            pred_modality = parse_modality(r1)
            pred_condition = parse_condition(r2)

            q1_correct = pred_modality == gold_pieces["modality"]
            q2_correct = pred_condition == gold_pieces["has_condition"]
            pieces_all_correct = q1_correct and q2_correct

            # PASS 2: Composed question
            r3 = call_fn(COMPOSED_Q.format(text=text))
            time.sleep(REQUEST_DELAY)

            pred_type = parse_type(r3)
            composed_correct = pred_type == gold_type

            predictions.append({
                "idx": i,
                "gold_type": gold_type,
                "pred_type": pred_type,
                "gold_modality": gold_pieces["modality"],
                "pred_modality": pred_modality,
                "q1_correct": q1_correct,
                "gold_condition": gold_pieces["has_condition"],
                "pred_condition": pred_condition,
                "q2_correct": q2_correct,
                "pieces_all_correct": pieces_all_correct,
                "composed_correct": composed_correct,
            })

        except Exception as e:
            errors += 1
            predictions.append({
                "idx": i, "gold_type": gold_type, "pred_type": "",
                "pieces_all_correct": False, "composed_correct": False,
                "error": str(e),
            })

        if (i + 1) % 50 == 0:
            pc = sum(1 for p in predictions if p.get("pieces_all_correct"))
            cc = sum(1 for p in predictions if p.get("composed_correct"))
            gap_so_far = sum(1 for p in predictions if p.get("pieces_all_correct") and not p.get("composed_correct"))
            print(f"[INFO] {i+1}/{len(data)} | pieces_ok: {pc} | composed_ok: {cc} | gap_cases: {gap_so_far} | errors: {errors}")

    results = compute_composition_gap(predictions)
    print_results(results, model_key, "Contracts")
    save_results(predictions, results, model_key, "contracts", RESULTS_DIR)
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
        run(mk, data, args.backend)


if __name__ == "__main__":
    main()
