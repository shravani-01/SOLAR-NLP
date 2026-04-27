#!/usr/bin/env python3
"""
Mechanism Experiment — SQL (Spider).

Tests whether giving the model the correct query structure as a hint
closes the composition gap.

3 conditions:
  1. original     — reuse baseline gap from existing predictions
  2. hint_correct — "This query requires a JOIN structure using tables X, Y."
  3. hint_wrong   — "This query requires a SIMPLE structure with no joins."

Usage:
    python run_sql.py --model gpt4o --limit 10
    python run_sql.py --model all
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experiments" / "spider_composition_gap"))
from core import (call_api, call_oss, load_oss_model, API_MODELS, OSS_MODELS,
                  REQUEST_DELAY)
from classify_sql_structure import classify_sql, STRUCTURAL_TYPES as SQL_TYPES
from core_mechanism import (load_baseline_predictions, get_gap_eligible,
                            pick_wrong_type, compute_mechanism_results,
                            save_mechanism_results, print_mechanism_comparison)

DATA_DIR = Path(__file__).parent.parent.parent / "experiments" / "spider_composition_gap" / "data"
RESULTS_DIR = Path(__file__).parent / "results"

SYSTEM = "You are a SQL expert. Answer precisely."

# ─── Structure descriptions for hints ─────────────────────────────────────

STRUCTURE_HINTS = {
    "SIMPLE":    "a SIMPLE single-table query with basic SELECT and WHERE",
    "JOIN":      "a JOIN query that combines multiple tables using JOIN operations",
    "NESTED":    "a NESTED query with subqueries (SELECT inside SELECT)",
    "SET-OP":    "a SET OPERATION query using UNION, INTERSECT, or EXCEPT",
    "MULTI-AGG": "a MULTI-AGGREGATION query using GROUP BY with multiple aggregate functions",
}

# ─── Mechanism prompts ────────────────────────────────────────────────────

COMPOSED_HINT = """Write a SQL query for this question.

Schema:
{schema}

Question: "{question}"

IMPORTANT STRUCTURAL NOTE: Analysis shows this requires {hint}.

Return ONLY the SQL query. No explanation."""

def extract_tables_from_sql(sql: str) -> set:
    sql_upper = sql.upper()
    tables = set()
    for m in re.finditer(r'\b(?:FROM|JOIN)\s+(\w+)', sql_upper):
        tables.add(m.group(1))
    return tables


# ─── Data loading ─────────────────────────────────────────────────────────

def load_data(limit=None):
    dev_path = DATA_DIR / "dev_classified.json"
    if not dev_path.exists():
        dev_path = DATA_DIR / "dev.json"
    if not dev_path.exists():
        print(f"[ERROR] No data at {DATA_DIR}")
        sys.exit(1)

    with open(dev_path) as f:
        data = json.load(f)

    schemas = {}
    tables_path = DATA_DIR / "tables.json"
    if tables_path.exists():
        with open(tables_path) as f:
            for db in json.load(f):
                if "schema_text" in db:
                    schemas[db["db_id"]] = db["schema_text"]

    if limit:
        data = data[:limit]
    return data, schemas


# ─── Run mechanism conditions ─────────────────────────────────────────────

def run(model_key: str, data: list, schemas: dict, backend: str = "api"):
    if backend == "api":
        config = API_MODELS[model_key]
        provider, model_name = config["provider"], config["model"]
        call_fn = lambda p: call_api(p, provider, model_name, SYSTEM)
    else:
        config = OSS_MODELS[model_key]
        model, tokenizer = load_oss_model(config["hf_name"])
        call_fn = lambda p: call_oss(p, model, tokenizer, SYSTEM)

    # Step 1: Load baseline predictions
    baseline_preds = load_baseline_predictions("sql", model_key)
    eligible = get_gap_eligible(baseline_preds)

    if not eligible:
        print(f"[WARN] No gap-eligible examples for {model_key}.")
        return
    eligible_indices = set(p["idx"] for p in eligible)
    print(f"[INFO] {model_key}: {len(eligible)} gap-eligible examples (out of {len(baseline_preds)})")

    # Build gold type map from baseline predictions
    idx_to_gold_type = {p["idx"]: p.get("gold_type", "") for p in baseline_preds}

    # Original gap
    original_results = compute_mechanism_results(eligible, "original")

    # hint_correct condition
    print(f"\n[INFO] Running hint_correct condition...")
    hint_correct_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        gold_sql = ex.get("query", "")
        gold_type = classify_sql(gold_sql)
        schema = schemas.get(ex.get("db_id", ""), "-- Schema not available")
        question = ex.get("question", "")
        hint = STRUCTURE_HINTS.get(gold_type, STRUCTURE_HINTS["SIMPLE"])

        try:
            r = call_fn(COMPOSED_HINT.format(schema=schema, question=question, hint=hint))
            time.sleep(REQUEST_DELAY)

            code_match = re.search(r'```(?:sql)?\s*\n?(.*?)```', r, re.DOTALL | re.IGNORECASE)
            pred_sql = code_match.group(1).strip() if code_match else r.strip()
            pred_type = classify_sql(pred_sql)
            composed_correct = pred_type == gold_type

            hint_correct_preds.append({
                "idx": i, "gold_type": gold_type, "pred_type": pred_type,
                "composed_correct": composed_correct,
                "condition": "hint_correct",
                "hint_given": hint,
                "pred_sql": pred_sql[:500],
            })
        except Exception as e:
            hint_correct_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if len(hint_correct_preds) % 50 == 0:
            correct = sum(1 for p in hint_correct_preds if p.get("composed_correct"))
            print(f"  [hint_correct] {len(hint_correct_preds)}/{len(eligible_indices)} | correct: {correct}")

    hint_correct_results = compute_mechanism_results(hint_correct_preds, "hint_correct")

    # hint_wrong condition
    print(f"\n[INFO] Running hint_wrong condition...")
    hint_wrong_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        gold_sql = ex.get("query", "")
        gold_type = classify_sql(gold_sql)
        schema = schemas.get(ex.get("db_id", ""), "-- Schema not available")
        question = ex.get("question", "")
        wrong_type = pick_wrong_type(gold_type, list(SQL_TYPES))
        hint = STRUCTURE_HINTS.get(wrong_type, STRUCTURE_HINTS["SIMPLE"])

        try:
            r = call_fn(COMPOSED_HINT.format(schema=schema, question=question, hint=hint))
            time.sleep(REQUEST_DELAY)

            code_match = re.search(r'```(?:sql)?\s*\n?(.*?)```', r, re.DOTALL | re.IGNORECASE)
            pred_sql = code_match.group(1).strip() if code_match else r.strip()
            pred_type = classify_sql(pred_sql)
            composed_correct = pred_type == gold_type

            hint_wrong_preds.append({
                "idx": i, "gold_type": gold_type, "pred_type": pred_type,
                "composed_correct": composed_correct,
                "condition": "hint_wrong",
                "wrong_type_given": wrong_type,
                "hint_given": hint,
                "pred_sql": pred_sql[:500],
            })
        except Exception as e:
            hint_wrong_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if len(hint_wrong_preds) % 50 == 0:
            correct = sum(1 for p in hint_wrong_preds if p.get("composed_correct"))
            print(f"  [hint_wrong] {len(hint_wrong_preds)}/{len(eligible_indices)} | correct: {correct}")

    hint_wrong_results = compute_mechanism_results(hint_wrong_preds, "hint_wrong")

    # Compare and save
    all_results = {
        "original": original_results,
        "hint_correct": hint_correct_results,
        "hint_wrong": hint_wrong_results,
        "predictions": {
            "hint_correct": hint_correct_preds,
            "hint_wrong": hint_wrong_preds,
        },
    }

    print_mechanism_comparison(all_results, model_key, "SQL")
    save_mechanism_results(all_results, model_key, "sql", RESULTS_DIR)
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default="api", choices=["api", "oss"])
    args = parser.parse_args()

    data, schemas = load_data(args.limit)
    print(f"[INFO] Loaded {len(data)} SQL examples, {len(schemas)} schemas")

    models = API_MODELS if args.backend == "api" else OSS_MODELS
    to_run = list(models.keys()) if args.model == "all" else [args.model]

    for mk in to_run:
        run(mk, data, schemas, args.backend)


if __name__ == "__main__":
    main()
