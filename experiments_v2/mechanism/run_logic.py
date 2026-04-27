#!/usr/bin/env python3
"""
Mechanism Experiment — Logic (FOLIO).

Tests whether giving the model the correct reasoning structure as a hint
closes the composition gap.

3 conditions:
  1. original     — reuse baseline gap from existing predictions
  2. hint_correct — "This requires MODUS PONENS reasoning: identify the conditional, verify the antecedent."
  3. hint_wrong   — "This requires DISJUNCTIVE reasoning: eliminate one branch of the disjunction."

Usage:
    python run_logic.py --model gpt4o --limit 10
    python run_logic.py --model all
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experiments" / "folio_composition_gap"))
from core import (call_api, call_oss, load_oss_model, API_MODELS, OSS_MODELS,
                  REQUEST_DELAY)
from core_mechanism import (load_baseline_predictions, get_gap_eligible,
                            pick_wrong_type, compute_mechanism_results,
                            save_mechanism_results, print_mechanism_comparison)

DATA_DIR = Path(__file__).parent.parent.parent / "experiments" / "folio_composition_gap" / "data"
RESULTS_DIR = Path(__file__).parent / "results"

SYSTEM = "You are a logic expert. Answer precisely."

LOGIC_TYPES = ["MODUS-PONENS", "SYLLOGISM", "DISJUNCTIVE", "CONDITIONAL-CHAIN", "NEGATION"]

# ─── Structure descriptions for hints ─────────────────────────────────────

STRUCTURE_HINTS = {
    "MODUS-PONENS": "MODUS PONENS reasoning (if P then Q; P is true; therefore Q). "
                    "Identify the conditional rule (if-then), verify that the antecedent holds, "
                    "then conclude the consequent",
    "SYLLOGISM":    "SYLLOGISTIC reasoning (All X are Y; Z is X; therefore Z is Y). "
                    "Identify the universal rule, check category membership, "
                    "then apply the rule to the specific case",
    "DISJUNCTIVE":  "DISJUNCTIVE reasoning (either P or Q; not P; therefore Q). "
                    "Identify the disjunction, determine which branch is eliminated, "
                    "then conclude the remaining branch",
    "CONDITIONAL-CHAIN": "CONDITIONAL CHAIN reasoning (if A then B; if B then C; therefore if A then C). "
                         "Trace the chain of conditionals from start to end, "
                         "linking each consequent to the next antecedent",
    "NEGATION":     "NEGATION-based reasoning (not all X are Y; Z is X; therefore Z might not be Y). "
                    "Carefully track negation scope — 'not all' ≠ 'none', "
                    "'cannot' ≠ 'does not'. Apply De Morgan's laws where needed",
}

# ─── Mechanism prompts ────────────────────────────────────────────────────

COMPOSED_HINT = """Given these premises:

{premises}

Conclusion: {conclusion}

IMPORTANT STRUCTURAL NOTE: This problem requires {hint}.

Is the conclusion True, False, or Unknown?

Answer with ONLY one word: True, False, or Unknown."""


def parse_answer(response: str) -> str:
    r = response.strip()
    for label in ["True", "False", "Unknown"]:
        if label.lower() in r.lower():
            return label
    return "Unknown"


# ─── Data loading ─────────────────────────────────────────────────────────

def load_data(limit=None):
    path = DATA_DIR / "folio_classified.json"
    if not path.exists():
        for fb in ["folio_validation.json", "folio_all.json"]:
            fb_path = DATA_DIR / fb
            if fb_path.exists():
                path = fb_path
                break
    if not path.exists():
        print(f"[ERROR] No data found.")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


# ─── Run mechanism conditions ─────────────────────────────────────────────

def run(model_key: str, data: list, backend: str = "api"):
    if backend == "api":
        config = API_MODELS[model_key]
        provider, model_name = config["provider"], config["model"]
        call_fn = lambda p: call_api(p, provider, model_name, SYSTEM)
    else:
        config = OSS_MODELS[model_key]
        model, tokenizer = load_oss_model(config["hf_name"])
        call_fn = lambda p: call_oss(p, model, tokenizer, SYSTEM)

    baseline_preds = load_baseline_predictions("logic", model_key)
    eligible = get_gap_eligible(baseline_preds)

    if not eligible:
        print(f"[WARN] No gap-eligible examples for {model_key}.")
        return
    eligible_indices = set(p["idx"] for p in eligible)
    print(f"[INFO] {model_key}: {len(eligible)} gap-eligible examples")

    original_results = compute_mechanism_results(eligible, "original")

    # hint_correct condition
    print(f"\n[INFO] Running hint_correct condition...")
    hint_correct_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        premises = ex.get("premises", [])
        conclusion = ex.get("conclusion", "")
        gold_label = str(ex.get("label", "Unknown"))
        gold_type = ex.get("structural_type", "MODUS-PONENS")
        premises_text = "\n".join(f"- {p}" for p in premises)

        hint = STRUCTURE_HINTS.get(gold_type, STRUCTURE_HINTS["MODUS-PONENS"])

        try:
            r = call_fn(COMPOSED_HINT.format(premises=premises_text, conclusion=conclusion, hint=hint))
            time.sleep(REQUEST_DELAY)

            pred_label = parse_answer(r)
            composed_correct = pred_label.lower() == gold_label.lower()

            hint_correct_preds.append({
                "idx": i, "gold_type": gold_type,
                "gold_label": gold_label, "pred_label": pred_label,
                "composed_correct": composed_correct,
                "condition": "hint_correct",
                "hint_given": hint[:100],
            })
        except Exception as e:
            hint_correct_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if len(hint_correct_preds) % 25 == 0:
            correct = sum(1 for p in hint_correct_preds if p.get("composed_correct"))
            print(f"  [hint_correct] {len(hint_correct_preds)}/{len(eligible_indices)} | correct: {correct}")

    hint_correct_results = compute_mechanism_results(hint_correct_preds, "hint_correct")

    # hint_wrong condition
    print(f"\n[INFO] Running hint_wrong condition...")
    hint_wrong_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        premises = ex.get("premises", [])
        conclusion = ex.get("conclusion", "")
        gold_label = str(ex.get("label", "Unknown"))
        gold_type = ex.get("structural_type", "MODUS-PONENS")
        premises_text = "\n".join(f"- {p}" for p in premises)

        wrong_type = pick_wrong_type(gold_type, LOGIC_TYPES)
        hint = STRUCTURE_HINTS.get(wrong_type, STRUCTURE_HINTS["MODUS-PONENS"])

        try:
            r = call_fn(COMPOSED_HINT.format(premises=premises_text, conclusion=conclusion, hint=hint))
            time.sleep(REQUEST_DELAY)

            pred_label = parse_answer(r)
            composed_correct = pred_label.lower() == gold_label.lower()

            hint_wrong_preds.append({
                "idx": i, "gold_type": gold_type,
                "gold_label": gold_label, "pred_label": pred_label,
                "composed_correct": composed_correct,
                "condition": "hint_wrong",
                "wrong_type_given": wrong_type,
                "hint_given": hint[:100],
            })
        except Exception as e:
            hint_wrong_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if len(hint_wrong_preds) % 25 == 0:
            correct = sum(1 for p in hint_wrong_preds if p.get("composed_correct"))
            print(f"  [hint_wrong] {len(hint_wrong_preds)}/{len(eligible_indices)} | correct: {correct}")

    hint_wrong_results = compute_mechanism_results(hint_wrong_preds, "hint_wrong")

    all_results = {
        "original": original_results,
        "hint_correct": hint_correct_results,
        "hint_wrong": hint_wrong_results,
        "predictions": {
            "hint_correct": hint_correct_preds,
            "hint_wrong": hint_wrong_preds,
        },
    }

    print_mechanism_comparison(all_results, model_key, "Logic")
    save_mechanism_results(all_results, model_key, "logic", RESULTS_DIR)
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default="api", choices=["api", "oss"])
    args = parser.parse_args()

    data = load_data(args.limit)
    print(f"[INFO] Loaded {len(data)} logic examples")

    models = API_MODELS if args.backend == "api" else OSS_MODELS
    to_run = list(models.keys()) if args.model == "all" else [args.model]

    for mk in to_run:
        run(mk, data, backend=args.backend)


if __name__ == "__main__":
    main()
