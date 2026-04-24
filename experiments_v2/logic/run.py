#!/usr/bin/env python3
"""
2-Pass Composition Gap — Logical Reasoning (FOLIO).

Pass 1 (Pieces): Ask 2 sub-questions separately:
  Q1: "What are the key entities and their properties in these premises?"
  Q2: "What logical connective/structure is used? (if-then, all-are, either-or, chain, negation)"

Pass 2 (Composed): "Is the conclusion True, False, or Unknown?"

Gap = P(Pass 2 wrong | Pass 1 all correct)

Usage:
    python run.py --model gpt4o --limit 10
    python run.py --model all
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experiments" / "folio_composition_gap"))
from core import (call_api, call_oss, load_oss_model, API_MODELS, OSS_MODELS,
                  REQUEST_DELAY, compute_composition_gap, print_results, save_results)
from classify_logic_structure import classify_logic, STRUCTURAL_TYPES

DATA_DIR = Path(__file__).parent.parent.parent / "experiments" / "folio_composition_gap" / "data"
RESULTS_DIR = Path(__file__).parent / "results"

SYSTEM = "You are a logic expert. Answer precisely."

# ─── Piece sub-questions ────────────────────────────────────────────────────

PIECE_Q1 = """Read these premises:

{premises}

Question: List ALL key entities (people, things, categories) mentioned. Just list them, comma-separated."""

PIECE_Q2 = """Read these premises:

{premises}

What LOGICAL STRUCTURE connects these premises? Choose the BEST match:

- IF-THEN: Direct conditional (if P then Q)
- ALL-ARE: Universal categorical (all X are Y, every X has Y)
- EITHER-OR: Disjunction (either P or Q)
- CHAIN: Multiple linked conditionals (if A then B, if B then C)
- NEGATION: Negation-heavy (not all, cannot, never, contradiction)

Answer with ONLY one choice: IF-THEN, ALL-ARE, EITHER-OR, CHAIN, or NEGATION."""

# ─── Composed question ─────────────────────────────────────────────────────

COMPOSED_Q = """Given these premises:

{premises}

Conclusion: {conclusion}

Is the conclusion True, False, or Unknown?

Answer with ONLY one word: True, False, or Unknown."""

# ─── Gold piece derivation ──────────────────────────────────────────────────

STRUCTURE_TO_LOGIC = {
    "MODUS-PONENS": "IF-THEN",
    "SYLLOGISM": "ALL-ARE",
    "DISJUNCTIVE": "EITHER-OR",
    "CONDITIONAL-CHAIN": "CHAIN",
    "NEGATION": "NEGATION",
}

def get_gold_entities(premises: list) -> set:
    """Extract gold entities from premises."""
    entities = set()
    for p in premises:
        words = p.split()
        for w in words:
            clean = re.sub(r'[^\w]', '', w)
            if clean and clean[0].isupper() and len(clean) > 1 and clean.lower() not in {
                'the', 'if', 'then', 'all', 'every', 'some', 'no', 'not', 'and', 'or',
                'is', 'are', 'has', 'have', 'will', 'can', 'either', 'neither', 'when',
                'whenever', 'therefore', 'however', 'also', 'but', 'true', 'false',
                'because', 'since', 'unless', 'only', 'both', 'each', 'any', 'none',
            }:
                entities.add(clean)
    return entities


def parse_entities(response: str) -> set:
    """Parse entities from model response."""
    entities = set()
    for part in re.split(r'[,\n;]', response):
        word = part.strip()
        if word and len(word) > 1:
            # Take significant words
            for w in word.split():
                clean = re.sub(r'[^\w]', '', w)
                if clean and len(clean) > 1 and clean[0].isupper():
                    entities.add(clean)
    return entities


def parse_logic_type(response: str) -> str:
    r = response.strip().upper()
    for lt in ["IF-THEN", "ALL-ARE", "EITHER-OR", "CHAIN", "NEGATION"]:
        if lt in r:
            return lt
    return "IF-THEN"


def parse_answer(response: str) -> str:
    r = response.strip()
    for label in ["True", "False", "Unknown"]:
        if label.lower() in r.lower():
            return label
    return "Unknown"


# ─── Data loading ───────────────────────────────────────────────────────────

def load_data(limit=None):
    path = DATA_DIR / "folio_classified.json"
    if not path.exists():
        for fb in ["folio_validation.json", "folio_all.json"]:
            fb_path = DATA_DIR / fb
            if fb_path.exists():
                path = fb_path
                break
    if not path.exists():
        print(f"[ERROR] No data. Run download_folio.py first.")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


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

    print(f"\n[INFO] Logic 2-Pass: {model_key} ({len(data)} examples)")

    predictions = []
    errors = 0

    for i, ex in enumerate(data):
        premises = ex.get("premises", [])
        conclusion = ex.get("conclusion", "")
        gold_label = str(ex.get("label", "Unknown"))
        gold_type = ex.get("structural_type", "MODUS-PONENS")
        gold_logic = STRUCTURE_TO_LOGIC.get(gold_type, "IF-THEN")
        gold_entities = get_gold_entities(premises)

        premises_text = "\n".join(f"- {p}" for p in premises)

        try:
            # PASS 1: Piece sub-questions
            r1 = call_fn(PIECE_Q1.format(premises=premises_text))
            time.sleep(REQUEST_DELAY)
            r2 = call_fn(PIECE_Q2.format(premises=premises_text))
            time.sleep(REQUEST_DELAY)

            pred_entities = parse_entities(r1)
            pred_logic = parse_logic_type(r2)

            # Q1: check entity overlap (at least 50% of gold entities found)
            q1_correct = (len(gold_entities & pred_entities) >= len(gold_entities) * 0.5) if gold_entities else True
            # Q2: check logic type
            q2_correct = pred_logic == gold_logic
            pieces_all_correct = q1_correct and q2_correct

            # PASS 2: Composed question
            r3 = call_fn(COMPOSED_Q.format(premises=premises_text, conclusion=conclusion))
            time.sleep(REQUEST_DELAY)

            pred_label = parse_answer(r3)
            composed_correct = pred_label.lower() == gold_label.lower()

            predictions.append({
                "idx": i, "gold_type": gold_type,
                "gold_label": gold_label, "pred_label": pred_label,
                "gold_logic_type": gold_logic, "pred_logic_type": pred_logic,
                "q1_entities_correct": q1_correct,
                "q2_logic_correct": q2_correct,
                "pieces_all_correct": pieces_all_correct,
                "composed_correct": composed_correct,
            })

        except Exception as e:
            errors += 1
            predictions.append({
                "idx": i, "gold_type": gold_type,
                "pieces_all_correct": False, "composed_correct": False,
                "error": str(e),
            })

        if (i + 1) % 25 == 0:
            pc = sum(1 for p in predictions if p.get("pieces_all_correct"))
            gap = sum(1 for p in predictions if p.get("pieces_all_correct") and not p.get("composed_correct"))
            print(f"[INFO] {i+1}/{len(data)} | pieces_ok: {pc} | gap_cases: {gap} | errors: {errors}")

    results = compute_composition_gap(predictions)
    print_results(results, model_key, "Logic (FOLIO)")
    save_results(predictions, results, model_key, "logic", RESULTS_DIR)
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
        run(mk, data, backend=args.backend)


if __name__ == "__main__":
    main()
