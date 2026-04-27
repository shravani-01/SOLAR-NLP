#!/usr/bin/env python3
"""
Mitigation Experiment — SQL (Section 8).

Structure-Aware Prompting for SQL query generation.

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
from core_mitigation import (load_baseline_predictions, get_gap_eligible,
                             compute_mitigation_results, save_mitigation_results,
                             print_mitigation_comparison)

DATA_DIR = Path(__file__).parent.parent.parent / "experiments" / "spider_composition_gap" / "data"
RESULTS_DIR = Path(__file__).parent / "results"

SYSTEM = "You are a SQL expert. Answer precisely."

# ─── Stage 1: Structure identification ────────────────────────────────────

STAGE1_PROMPT = """Given this database schema and question, identify what SQL STRUCTURE is needed.

Schema:
{schema}

Question: "{question}"

Before writing the query, analyze the structure needed:
1. TABLES: How many tables are involved? Is a JOIN needed?
2. NESTING: Does this need a subquery (SELECT inside SELECT)?
3. SET-OPS: Does this need UNION, INTERSECT, or EXCEPT?
4. AGGREGATION: Does this need GROUP BY with aggregate functions?

Based on this analysis, what query STRUCTURE is needed?
Answer: SIMPLE, JOIN, NESTED, SET-OP, or MULTI-AGG

Format your response as:
TABLES: [your analysis]
NESTING: [your analysis]
SET-OPS: [your analysis]
AGGREGATION: [your analysis]
STRUCTURE: [SIMPLE/JOIN/NESTED/SET-OP/MULTI-AGG]"""

# ─── Stage 2: Structure-informed query ────────────────────────────────────

STAGE2_PROMPT = """Write a SQL query for this question.

Schema:
{schema}

Question: "{question}"

You have already analyzed the structure needed:
{structure_analysis}

Using this structural plan, write the query.
Return ONLY the SQL query. No explanation."""

# ─── CoT Structure prompt ────────────────────────────────────────────────

COT_STRUCTURE_PROMPT = """Write a SQL query for this question.

Schema:
{schema}

Question: "{question}"

THINK ABOUT STRUCTURE FIRST:
Step 1: How many tables are needed? List them.
Step 2: Do I need JOINs to connect tables?
Step 3: Do I need subqueries (nested SELECT)?
Step 4: Do I need set operations (UNION/INTERSECT/EXCEPT)?
Step 5: Do I need GROUP BY with aggregations?
Step 6: Write the query matching the structure I identified.

Show your structural reasoning, then write the final SQL query after "QUERY:"."""


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


def extract_sql(response: str) -> str:
    """Extract SQL from response (handles code blocks, QUERY: prefix, etc.)."""
    # Try code block first
    code_match = re.search(r'```(?:sql)?\s*\n?(.*?)```', response, re.DOTALL | re.IGNORECASE)
    if code_match:
        return code_match.group(1).strip()
    # Try QUERY: prefix
    query_match = re.search(r'QUERY:\s*\n?(.*?)$', response, re.DOTALL | re.IGNORECASE)
    if query_match:
        sql = query_match.group(1).strip()
        # Remove any trailing non-SQL text
        lines = []
        for line in sql.split('\n'):
            if line.strip().upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE',
                                                 'FROM', 'WHERE', 'JOIN', 'LEFT', 'RIGHT',
                                                 'INNER', 'GROUP', 'ORDER', 'HAVING', 'LIMIT',
                                                 'UNION', 'INTERSECT', 'EXCEPT', '(', ')')):
                lines.append(line)
            elif lines:  # Already started collecting SQL
                lines.append(line)
        return '\n'.join(lines).strip() if lines else sql
    return response.strip()


def run(model_key: str, data: list, schemas: dict, backend: str = "api"):
    if backend == "api":
        config = API_MODELS[model_key]
        provider, model_name = config["provider"], config["model"]
        call_fn = lambda p: call_api(p, provider, model_name, SYSTEM)
    else:
        config = OSS_MODELS[model_key]
        model, tokenizer = load_oss_model(config["hf_name"])
        call_fn = lambda p: call_oss(p, model, tokenizer, SYSTEM)

    baseline_preds = load_baseline_predictions("sql", model_key)
    eligible = get_gap_eligible(baseline_preds)

    if not eligible:
        print(f"[WARN] No gap-eligible examples for {model_key}.")
        return
    eligible_indices = set(p["idx"] for p in eligible)
    print(f"[INFO] {model_key}: {len(eligible)} gap-eligible examples")

    original_results = compute_mitigation_results(eligible, "original")

    # ─── self_structure (2-stage) ─────────────────────────────────────────
    print(f"\n[INFO] Running self_structure condition...")
    self_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        gold_sql = ex.get("query", "")
        gold_type = classify_sql(gold_sql)
        schema = schemas.get(ex.get("db_id", ""), "-- Schema not available")
        question = ex.get("question", "")

        try:
            r1 = call_fn(STAGE1_PROMPT.format(schema=schema, question=question))
            time.sleep(REQUEST_DELAY)

            r2 = call_fn(STAGE2_PROMPT.format(schema=schema, question=question, structure_analysis=r1))
            time.sleep(REQUEST_DELAY)

            pred_sql = extract_sql(r2)
            pred_type = classify_sql(pred_sql)
            composed_correct = pred_type == gold_type

            self_preds.append({
                "idx": i, "gold_type": gold_type, "pred_type": pred_type,
                "composed_correct": composed_correct,
                "condition": "self_structure",
                "stage1_response": r1[:400],
                "pred_sql": pred_sql[:500],
            })
        except Exception as e:
            self_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if len(self_preds) % 50 == 0:
            correct = sum(1 for p in self_preds if p.get("composed_correct"))
            print(f"  [self_structure] {len(self_preds)}/{len(eligible_indices)} | correct: {correct}")

    self_results = compute_mitigation_results(self_preds, "self_structure")

    # ─── cot_structure (single prompt) ────────────────────────────────────
    print(f"\n[INFO] Running cot_structure condition...")
    cot_preds = []
    for i, ex in enumerate(data):
        if i not in eligible_indices:
            continue

        gold_sql = ex.get("query", "")
        gold_type = classify_sql(gold_sql)
        schema = schemas.get(ex.get("db_id", ""), "-- Schema not available")
        question = ex.get("question", "")

        try:
            r = call_fn(COT_STRUCTURE_PROMPT.format(schema=schema, question=question))
            time.sleep(REQUEST_DELAY)

            pred_sql = extract_sql(r)
            pred_type = classify_sql(pred_sql)
            composed_correct = pred_type == gold_type

            cot_preds.append({
                "idx": i, "gold_type": gold_type, "pred_type": pred_type,
                "composed_correct": composed_correct,
                "condition": "cot_structure",
                "response": r[:600],
                "pred_sql": pred_sql[:500],
            })
        except Exception as e:
            cot_preds.append({
                "idx": i, "gold_type": gold_type,
                "composed_correct": False, "error": str(e),
            })

        if len(cot_preds) % 50 == 0:
            correct = sum(1 for p in cot_preds if p.get("composed_correct"))
            print(f"  [cot_structure] {len(cot_preds)}/{len(eligible_indices)} | correct: {correct}")

    cot_results = compute_mitigation_results(cot_preds, "cot_structure")

    all_results = {
        "original": original_results,
        "self_structure": self_results,
        "cot_structure": cot_results,
        "predictions": {
            "self_structure": self_preds,
            "cot_structure": cot_preds,
        },
    }

    print_mitigation_comparison(all_results, model_key, "SQL")
    save_mitigation_results(all_results, model_key, "sql", RESULTS_DIR)
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default="api", choices=["api", "oss"])
    args = parser.parse_args()

    data, schemas = load_data(args.limit)
    print(f"[INFO] Loaded {len(data)} SQL examples")

    models = API_MODELS if args.backend == "api" else OSS_MODELS
    to_run = list(models.keys()) if args.model == "all" else [args.model]

    for mk in to_run:
        run(mk, data, schemas, args.backend)


if __name__ == "__main__":
    main()
