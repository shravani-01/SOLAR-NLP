#!/usr/bin/env python3
"""
Mitigation Experiment — Contracts (Section 8).

Structure-Aware Prompting: prompt the model to first identify the
implicit structure, then use that to classify the constraint type.

3 conditions:
  1. original        — baseline (reuse existing predictions)
  2. self_structure   — 2-stage: model identifies modality+condition, then classifies
  3. cot_structure    — single prompt with structure-aware CoT instructions

Usage:
    python run_contracts.py --model gpt4o --limit 10
    python run_contracts.py --model all
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
from core_mitigation import (load_baseline_predictions, get_gap_eligible,
                             compute_mitigation_results, save_mitigation_results,
                             print_mitigation_comparison)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "annotated" / "splits"
RESULTS_DIR = Path(__file__).parent / "results"

CONSTRAINT_TYPES = ["HARD", "SOFT", "HARD-CONDITIONAL", "SOFT-CONDITIONAL", "NON-CONSTRAINT"]

SYSTEM = "You are a contract analysis expert. Answer precisely."

# ─── Stage 1: Structure identification prompt ─────────────────────────────

STAGE1_PROMPT = """Read this labor contract sentence:

"{text}"

Before classifying it, first identify its STRUCTURAL properties:

1. MODALITY: Is the language MANDATORY (shall/must/required), DISCRETIONARY (may/can/should), or NEITHER?
2. CONDITION: Does it contain a CONDITION (if/when/unless/provided that)?

Answer in this exact format:
MODALITY: [MANDATORY/DISCRETIONARY/NEITHER]
CONDITION: [YES/NO]"""

# ─── Stage 2: Structure-informed classification ───────────────────────────

STAGE2_PROMPT = """Classify this labor contract sentence into one of 5 constraint types:

"{text}"

You have already identified its structure:
{structure_analysis}

Using this structural analysis, the constraint type is:
- HARD: Mandatory (shall/must), no conditions
- SOFT: Discretionary (may/can), no conditions
- HARD-CONDITIONAL: Mandatory WITH a condition (if/when/unless)
- SOFT-CONDITIONAL: Discretionary WITH a condition
- NON-CONSTRAINT: Not a constraint (definition, description)

Answer with ONLY the type name."""

# ─── CoT Structure prompt (single call) ──────────────────────────────────

COT_STRUCTURE_PROMPT = """Classify this labor contract sentence into one of 5 constraint types.

"{text}"

THINK STEP BY STEP about the STRUCTURE:
Step 1: Is the language mandatory (shall/must) or discretionary (may/can) or neither?
Step 2: Does the sentence contain a condition (if/when/unless/provided that)?
Step 3: Combine your answers — mandatory + no condition = HARD, discretionary + no condition = SOFT, etc.

Types:
- HARD: Mandatory, no conditions
- SOFT: Discretionary, no conditions
- HARD-CONDITIONAL: Mandatory WITH condition
- SOFT-CONDITIONAL: Discretionary WITH condition
- NON-CONSTRAINT: Not a constraint

Show your reasoning, then on the final line write ONLY the type name."""


def parse_type(response: str) -> str:
    r = response.strip().upper()
    # Check last line first (for CoT responses)
    lines = r.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        for ct in CONSTRAINT_TYPES:
            if ct in line:
                return ct
    # Fallback: check whole response
    for ct in CONSTRAINT_TYPES:
        if ct in r:
            return ct
    return "NON-CONSTRAINT"


def parse_stage1(response: str) -> str:
    """Parse Stage 1 structural analysis response."""
    return response.strip()


# ─── Data loading ─────────────────────────────────────────────────────────

def load_data(limit=None):
    test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        test_path = Path(__file__).parent.parent / "contracts" / "data" / "test.csv"
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


# ─── Run mitigation conditions ────────────────────────────────────────────

def run(model_key: str, data: list, backend: str = "api"):
    if backend == "api":
        config = API_MODELS[model_key]
        provider, model_name = config["provider"], config["model"]
        call_fn = lambda p: call_api(p, provider, model_name, SYSTEM)
    else:
        config = OSS_MODELS[model_key]
        model, tokenizer = load_oss_model(config["hf_name"])
        call_fn = lambda p: call_oss(p, model, tokenizer, SYSTEM)

    # Load baseline and get gap-eligible
    baseline_preds = load_baseline_predictions("contracts", model_key)
    eligible = get_gap_eligible(baseline_preds)

    if not eligible:
        print(f"[WARN] No gap-eligible examples for {model_key}.")
        return
    eligible_indices = set(p["idx"] for p in eligible)
    print(f"[INFO] {model_key}: {len(eligible)} gap-eligible examples")

    original_results = compute_mitigation_results(eligible, "original")

    # ─── self_structure condition (2-stage pipeline) ──────────────────────
    print(f"\n[INFO] Running self_structure condition (2-stage)...")
    self_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        try:
            # Stage 1: Identify structure
            r1 = call_fn(STAGE1_PROMPT.format(text=ex["text"]))
            time.sleep(REQUEST_DELAY)

            # Stage 2: Classify using identified structure
            r2 = call_fn(STAGE2_PROMPT.format(text=ex["text"], structure_analysis=r1))
            time.sleep(REQUEST_DELAY)

            pred_type = parse_type(r2)
            self_preds.append({
                "idx": i,
                "gold_type": ex["gold_type"],
                "pred_type": pred_type,
                "composed_correct": pred_type == ex["gold_type"],
                "condition": "self_structure",
                "stage1_response": r1[:300],
                "stage2_response": r2[:200],
            })
        except Exception as e:
            self_preds.append({
                "idx": i, "gold_type": ex["gold_type"],
                "composed_correct": False, "error": str(e),
            })

        if len(self_preds) % 50 == 0:
            correct = sum(1 for p in self_preds if p.get("composed_correct"))
            print(f"  [self_structure] {len(self_preds)}/{len(eligible_indices)} | correct: {correct}")

    self_results = compute_mitigation_results(self_preds, "self_structure")

    # ─── cot_structure condition (single prompt) ──────────────────────────
    print(f"\n[INFO] Running cot_structure condition (single prompt)...")
    cot_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        try:
            r = call_fn(COT_STRUCTURE_PROMPT.format(text=ex["text"]))
            time.sleep(REQUEST_DELAY)

            pred_type = parse_type(r)
            cot_preds.append({
                "idx": i,
                "gold_type": ex["gold_type"],
                "pred_type": pred_type,
                "composed_correct": pred_type == ex["gold_type"],
                "condition": "cot_structure",
                "response": r[:500],
            })
        except Exception as e:
            cot_preds.append({
                "idx": i, "gold_type": ex["gold_type"],
                "composed_correct": False, "error": str(e),
            })

        if len(cot_preds) % 50 == 0:
            correct = sum(1 for p in cot_preds if p.get("composed_correct"))
            print(f"  [cot_structure] {len(cot_preds)}/{len(eligible_indices)} | correct: {correct}")

    cot_results = compute_mitigation_results(cot_preds, "cot_structure")

    # Save all
    all_results = {
        "original": original_results,
        "self_structure": self_results,
        "cot_structure": cot_results,
        "predictions": {
            "self_structure": self_preds,
            "cot_structure": cot_preds,
        },
    }

    print_mitigation_comparison(all_results, model_key, "Contracts")
    save_mitigation_results(all_results, model_key, "contracts", RESULTS_DIR)
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
