#!/usr/bin/env python3
"""
Mitigation Experiment — Logic (Section 8).

Structure-Aware Prompting for logical reasoning.

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
from core import (call_api, call_oss, load_oss_model, API_MODELS, OSS_MODELS,
                  REQUEST_DELAY)
from core_mitigation import (load_baseline_predictions, get_gap_eligible,
                             compute_mitigation_results, save_mitigation_results,
                             print_mitigation_comparison)

DATA_DIR = Path(__file__).parent.parent.parent / "experiments" / "folio_composition_gap" / "data"
RESULTS_DIR = Path(__file__).parent / "results"

SYSTEM = "You are a logic expert. Answer precisely."

STAGE1_PROMPT = """Read these premises:

{premises}

Before evaluating any conclusion, analyze the LOGICAL STRUCTURE:
1. What REASONING PATTERN connects these premises?
   - MODUS PONENS (if P then Q; P; therefore Q)
   - SYLLOGISM (all X are Y; Z is X; therefore Z is Y)
   - DISJUNCTIVE (either P or Q; not P; therefore Q)
   - CONDITIONAL CHAIN (if A then B; if B then C; therefore if A then C)
   - NEGATION (involves 'not all', 'cannot', 'never', contradictions)

2. What are the KEY ENTITIES and their RELATIONSHIPS?

3. What INFERENCE STEPS are needed?

Format:
PATTERN: [your choice]
ENTITIES: [key entities]
STEPS: [inference steps needed]"""

STAGE2_PROMPT = """Given these premises:

{premises}

Conclusion: {conclusion}

You have already analyzed the logical structure:
{structure_analysis}

Apply the reasoning pattern you identified to evaluate the conclusion.
Is the conclusion True, False, or Unknown?

Answer with ONLY one word: True, False, or Unknown."""

COT_STRUCTURE_PROMPT = """Given these premises:

{premises}

Conclusion: {conclusion}

THINK ABOUT THE LOGICAL STRUCTURE:
Step 1: What type of reasoning is needed? (if-then? universal? disjunction? chain? negation?)
Step 2: Identify the key logical relationships between premises.
Step 3: Trace the inference chain from premises to conclusion.
Step 4: Does the conclusion follow? Consider: True, False, or Unknown.

Show your structural reasoning, then on the final line answer: True, False, or Unknown."""


def parse_answer(response: str) -> str:
    lines = response.strip().split('\n')
    for line in reversed(lines):
        for label in ["True", "False", "Unknown"]:
            if label.lower() in line.lower():
                return label
    for label in ["True", "False", "Unknown"]:
        if label.lower() in response.lower():
            return label
    return "Unknown"


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

    original_results = compute_mitigation_results(eligible, "original")

    # self_structure
    print(f"\n[INFO] Running self_structure condition...")
    self_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        premises = ex.get("premises", [])
        conclusion = ex.get("conclusion", "")
        gold_label = str(ex.get("label", "Unknown"))
        gold_type = ex.get("structural_type", "MODUS-PONENS")
        premises_text = "\n".join(f"- {p}" for p in premises)

        try:
            r1 = call_fn(STAGE1_PROMPT.format(premises=premises_text))
            time.sleep(REQUEST_DELAY)
            r2 = call_fn(STAGE2_PROMPT.format(premises=premises_text, conclusion=conclusion, structure_analysis=r1))
            time.sleep(REQUEST_DELAY)

            pred_label = parse_answer(r2)
            composed_correct = pred_label.lower() == gold_label.lower()

            self_preds.append({
                "idx": i, "gold_type": gold_type,
                "gold_label": gold_label, "pred_label": pred_label,
                "composed_correct": composed_correct,
                "condition": "self_structure",
                "stage1_response": r1[:400],
            })
        except Exception as e:
            self_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if len(self_preds) % 25 == 0:
            correct = sum(1 for p in self_preds if p.get("composed_correct"))
            print(f"  [self_structure] {len(self_preds)}/{len(eligible_indices)} | correct: {correct}")

    self_results = compute_mitigation_results(self_preds, "self_structure")

    # cot_structure
    print(f"\n[INFO] Running cot_structure condition...")
    cot_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        premises = ex.get("premises", [])
        conclusion = ex.get("conclusion", "")
        gold_label = str(ex.get("label", "Unknown"))
        gold_type = ex.get("structural_type", "MODUS-PONENS")
        premises_text = "\n".join(f"- {p}" for p in premises)

        try:
            r = call_fn(COT_STRUCTURE_PROMPT.format(premises=premises_text, conclusion=conclusion))
            time.sleep(REQUEST_DELAY)

            pred_label = parse_answer(r)
            composed_correct = pred_label.lower() == gold_label.lower()

            cot_preds.append({
                "idx": i, "gold_type": gold_type,
                "gold_label": gold_label, "pred_label": pred_label,
                "composed_correct": composed_correct,
                "condition": "cot_structure",
                "response": r[:500],
            })
        except Exception as e:
            cot_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if len(cot_preds) % 25 == 0:
            correct = sum(1 for p in cot_preds if p.get("composed_correct"))
            print(f"  [cot_structure] {len(cot_preds)}/{len(eligible_indices)} | correct: {correct}")

    cot_results = compute_mitigation_results(cot_preds, "cot_structure")

    all_results = {
        "original": original_results,
        "self_structure": self_results,
        "cot_structure": cot_results,
        "predictions": {"self_structure": self_preds, "cot_structure": cot_preds},
    }

    print_mitigation_comparison(all_results, model_key, "Logic")
    save_mitigation_results(all_results, model_key, "logic", RESULTS_DIR)
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
