#!/usr/bin/env python3
"""
Mechanism Experiment — Contracts.

Tests whether giving the model the correct structural type as a hint
closes the composition gap.

3 conditions:
  1. original     — reuse baseline gap from existing predictions
  2. hint_correct — "This clause uses MANDATORY language and contains a CONDITION. Classify it."
  3. hint_wrong   — "This clause uses DISCRETIONARY language and has no condition. Classify it."

Usage:
    python run_contracts.py --model gpt4o --limit 10
    python run_contracts.py --model all
    python run_contracts.py --model all --backend oss
"""

import csv
import json
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

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "annotated" / "splits"
RESULTS_DIR = Path(__file__).parent / "results"

CONSTRAINT_TYPES = ["HARD", "SOFT", "HARD-CONDITIONAL", "SOFT-CONDITIONAL", "NON-CONSTRAINT"]

SYSTEM = "You are a contract analysis expert. Answer precisely."

# ─── Type descriptions for hints ──────────────────────────────────────────

TYPE_HINTS = {
    "HARD":             "uses MANDATORY language (shall/must) and has NO conditions",
    "SOFT":             "uses DISCRETIONARY language (may/can) and has NO conditions",
    "HARD-CONDITIONAL": "uses MANDATORY language (shall/must) and CONTAINS a condition (if/when/unless)",
    "SOFT-CONDITIONAL": "uses DISCRETIONARY language (may/can) and CONTAINS a condition (if/when/unless)",
    "NON-CONSTRAINT":   "is NOT a constraint — it is a definition or description",
}

# ─── Mechanism prompts ────────────────────────────────────────────────────

COMPOSED_HINT = """Classify this labor contract sentence into one of 5 constraint types:

"{text}"

IMPORTANT STRUCTURAL NOTE: Analysis shows this clause {hint}.

Types:
- HARD: Mandatory (shall/must), no conditions
- SOFT: Discretionary (may/can), no conditions
- HARD-CONDITIONAL: Mandatory WITH a condition (if/when/unless)
- SOFT-CONDITIONAL: Discretionary WITH a condition
- NON-CONSTRAINT: Not a constraint (definition, description)

Answer with ONLY the type name."""


def parse_type(response: str) -> str:
    r = response.strip().upper()
    for ct in CONSTRAINT_TYPES:
        if ct in r:
            return ct
    return "NON-CONSTRAINT"


# ─── Data loading ─────────────────────────────────────────────────────────

def load_data(limit=None):
    test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        test_path = Path(__file__).parent.parent / "contracts" / "data" / "test.csv"
    if not test_path.exists():
        print(f"[ERROR] No test data found at {DATA_DIR}")
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


# ─── Run mechanism conditions ─────────────────────────────────────────────

def run(model_key: str, data: list, backend: str = "api"):
    # Setup model
    if backend == "api":
        config = API_MODELS[model_key]
        provider, model_name = config["provider"], config["model"]
        call_fn = lambda p: call_api(p, provider, model_name, SYSTEM)
    else:
        config = OSS_MODELS[model_key]
        model, tokenizer = load_oss_model(config["hf_name"])
        call_fn = lambda p: call_oss(p, model, tokenizer, SYSTEM)

    # Step 1: Load baseline predictions and get gap-eligible subset
    baseline_preds = load_baseline_predictions("contracts", model_key)
    eligible = get_gap_eligible(baseline_preds)

    if not eligible:
        print(f"[WARN] No gap-eligible examples for {model_key}. Running full eval instead.")
        eligible_indices = set(range(len(data)))
    else:
        eligible_indices = set(p["idx"] for p in eligible)
        print(f"[INFO] {model_key}: {len(eligible)} gap-eligible examples (out of {len(baseline_preds)})")

    # Step 2: Get original gap from baseline
    original_preds = [p for p in eligible]
    original_results = compute_mechanism_results(original_preds, "original")

    # Step 3: Run hint_correct condition
    print(f"\n[INFO] Running hint_correct condition...")
    hint_correct_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        gold_type = ex["gold_type"]
        hint = TYPE_HINTS[gold_type]

        try:
            r = call_fn(COMPOSED_HINT.format(text=ex["text"], hint=hint))
            time.sleep(REQUEST_DELAY)
            pred_type = parse_type(r)
            hint_correct_preds.append({
                "idx": i,
                "gold_type": gold_type,
                "pred_type": pred_type,
                "composed_correct": pred_type == gold_type,
                "condition": "hint_correct",
                "hint_given": hint,
            })
        except Exception as e:
            hint_correct_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if (len(hint_correct_preds)) % 50 == 0:
            correct = sum(1 for p in hint_correct_preds if p.get("composed_correct"))
            print(f"  [hint_correct] {len(hint_correct_preds)}/{len(eligible_indices)} | correct: {correct}")

    hint_correct_results = compute_mechanism_results(hint_correct_preds, "hint_correct")

    # Step 4: Run hint_wrong condition
    print(f"\n[INFO] Running hint_wrong condition...")
    hint_wrong_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        gold_type = ex["gold_type"]
        wrong_type = pick_wrong_type(gold_type, CONSTRAINT_TYPES)
        hint = TYPE_HINTS[wrong_type]

        try:
            r = call_fn(COMPOSED_HINT.format(text=ex["text"], hint=hint))
            time.sleep(REQUEST_DELAY)
            pred_type = parse_type(r)
            hint_wrong_preds.append({
                "idx": i,
                "gold_type": gold_type,
                "pred_type": pred_type,
                "composed_correct": pred_type == gold_type,
                "condition": "hint_wrong",
                "wrong_type_given": wrong_type,
                "hint_given": hint,
            })
        except Exception as e:
            hint_wrong_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if (len(hint_wrong_preds)) % 50 == 0:
            correct = sum(1 for p in hint_wrong_preds if p.get("composed_correct"))
            print(f"  [hint_wrong] {len(hint_wrong_preds)}/{len(eligible_indices)} | correct: {correct}")

    hint_wrong_results = compute_mechanism_results(hint_wrong_preds, "hint_wrong")

    # Step 5: Compare and save
    all_results = {
        "original": original_results,
        "hint_correct": hint_correct_results,
        "hint_wrong": hint_wrong_results,
        "predictions": {
            "hint_correct": hint_correct_preds,
            "hint_wrong": hint_wrong_preds,
        },
    }

    print_mechanism_comparison(all_results, model_key, "Contracts")
    save_mechanism_results(all_results, model_key, "contracts", RESULTS_DIR)
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default="api", choices=["api", "oss"])
    args = parser.parse_args()

    data = load_data(args.limit)
    print(f"[INFO] Loaded {len(data)} contract examples")

    models = API_MODELS if args.backend == "api" else OSS_MODELS
    to_run = list(models.keys()) if args.model == "all" else [args.model]

    for mk in to_run:
        run(mk, data, args.backend)


if __name__ == "__main__":
    main()
