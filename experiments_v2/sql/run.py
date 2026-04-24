#!/usr/bin/env python3
"""
2-Pass Composition Gap — SQL (Spider).

Pass 1 (Pieces): Ask 3 sub-questions separately:
  Q1: "What tables are needed?"
  Q2: "What conditions/filters apply?"
  Q3: "What aggregations are needed?"

Pass 2 (Composed): "Write the full SQL query."
  Classify structure: SIMPLE/JOIN/NESTED/SET-OP/MULTI-AGG

Gap = P(Pass 2 wrong structure | Pass 1 all correct)

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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "experiments" / "spider_composition_gap"))
from core import (call_api, call_oss, load_oss_model, API_MODELS, OSS_MODELS,
                  REQUEST_DELAY, compute_composition_gap, print_results, save_results)
from classify_sql_structure import classify_sql, STRUCTURAL_TYPES

DATA_DIR = Path(__file__).parent.parent.parent / "experiments" / "spider_composition_gap" / "data"
RESULTS_DIR = Path(__file__).parent / "results"

SYSTEM = "You are a SQL expert. Answer precisely."

SQL_TYPES = STRUCTURAL_TYPES

# ─── Piece sub-questions ────────────────────────────────────────────────────

PIECE_Q1 = """Given this database schema and question, what TABLES are needed?

Schema:
{schema}

Question: "{question}"

List ONLY the table names, comma-separated. Nothing else."""

PIECE_Q2 = """Given this database schema and question, what WHERE/HAVING CONDITIONS are needed?

Schema:
{schema}

Question: "{question}"

List each condition briefly (e.g., "age > 30", "name = 'John'"). If no conditions, say NONE."""

PIECE_Q3 = """Given this database schema and question, what AGGREGATIONS are needed?

Schema:
{schema}

Question: "{question}"

List any COUNT, SUM, AVG, MAX, MIN, GROUP BY needed. If none, say NONE."""

# ─── Composed question ─────────────────────────────────────────────────────

COMPOSED_Q = """Write a SQL query for this question.

Schema:
{schema}

Question: "{question}"

Return ONLY the SQL query. No explanation."""

# ─── Piece evaluation ──────────────────────────────────────────────────────

def extract_tables_from_sql(sql: str) -> set:
    """Extract table names from SQL."""
    sql_upper = sql.upper()
    tables = set()
    for m in re.finditer(r'\b(?:FROM|JOIN)\s+(\w+)', sql_upper):
        tables.add(m.group(1))
    return tables


def parse_tables_response(response: str) -> set:
    """Parse table names from model response."""
    # Remove markdown, punctuation, split by comma/newline
    r = response.strip().upper()
    r = re.sub(r'[`\*\-\d\.\)]', '', r)
    tables = set()
    for part in re.split(r'[,\n;]', r):
        word = part.strip()
        if word and len(word) > 1 and word not in {'NONE', 'NO', 'TABLES', 'TABLE', 'THE', 'AND', 'OR'}:
            tables.add(word.split()[-1])  # Take last word (handles "the Students table")
    return tables


def check_conditions(gold_sql: str, pred_response: str) -> bool:
    """Check if predicted conditions match gold SQL conditions."""
    gold_upper = gold_sql.upper()
    has_where = "WHERE" in gold_upper
    has_having = "HAVING" in gold_upper

    pred_upper = pred_response.upper()
    pred_has_cond = pred_upper.strip() != "NONE" and len(pred_upper.strip()) > 3

    if has_where or has_having:
        return pred_has_cond  # Gold has conditions, pred should mention some
    else:
        return not pred_has_cond or "NONE" in pred_upper  # Gold has no conditions


def check_aggregations(gold_sql: str, pred_response: str) -> bool:
    """Check if predicted aggregations match gold SQL."""
    gold_upper = gold_sql.upper()
    gold_has_agg = bool(re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', gold_upper))
    gold_has_group = "GROUP BY" in gold_upper

    pred_upper = pred_response.upper()
    pred_has_agg = bool(re.search(r'\b(COUNT|SUM|AVG|MAX|MIN|GROUP)', pred_upper))
    pred_none = "NONE" in pred_upper and len(pred_upper.strip()) < 20

    if gold_has_agg or gold_has_group:
        return pred_has_agg
    else:
        return pred_none or not pred_has_agg


# ─── Data loading ───────────────────────────────────────────────────────────

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


# ─── Main ───────────────────────────────────────────────────────────────────

def run(model_key: str, data: list, schemas: dict, backend: str = "api"):
    if backend == "api":
        config = API_MODELS[model_key]
        provider, model_name = config["provider"], config["model"]
        call_fn = lambda p: call_api(p, provider, model_name, SYSTEM)
    else:
        config = OSS_MODELS[model_key]
        model, tokenizer = load_oss_model(config["hf_name"])
        call_fn = lambda p: call_oss(p, model, tokenizer, SYSTEM)

    print(f"\n[INFO] SQL 2-Pass: {model_key} ({len(data)} examples)")

    predictions = []
    errors = 0

    for i, ex in enumerate(data):
        db_id = ex.get("db_id", "")
        question = ex.get("question", "")
        gold_sql = ex.get("query", "")
        gold_type = classify_sql(gold_sql)
        schema = schemas.get(db_id, "-- Schema not available")

        try:
            # PASS 1: Piece sub-questions
            r1 = call_fn(PIECE_Q1.format(schema=schema, question=question))
            time.sleep(REQUEST_DELAY)
            r2 = call_fn(PIECE_Q2.format(schema=schema, question=question))
            time.sleep(REQUEST_DELAY)
            r3 = call_fn(PIECE_Q3.format(schema=schema, question=question))
            time.sleep(REQUEST_DELAY)

            # Check pieces
            gold_tables = extract_tables_from_sql(gold_sql)
            pred_tables = parse_tables_response(r1)
            q1_correct = len(gold_tables & pred_tables) >= len(gold_tables) * 0.5 if gold_tables else True
            q2_correct = check_conditions(gold_sql, r2)
            q3_correct = check_aggregations(gold_sql, r3)
            pieces_all_correct = q1_correct and q2_correct and q3_correct

            # PASS 2: Composed question
            r4 = call_fn(COMPOSED_Q.format(schema=schema, question=question))
            time.sleep(REQUEST_DELAY)

            # Extract and classify SQL
            code_match = re.search(r'```(?:sql)?\s*\n?(.*?)```', r4, re.DOTALL | re.IGNORECASE)
            pred_sql = code_match.group(1).strip() if code_match else r4.strip()
            pred_type = classify_sql(pred_sql)
            composed_correct = pred_type == gold_type

            predictions.append({
                "idx": i, "db_id": db_id, "question": question,
                "gold_type": gold_type, "pred_type": pred_type,
                "q1_tables_correct": q1_correct,
                "q2_conditions_correct": q2_correct,
                "q3_aggregations_correct": q3_correct,
                "pieces_all_correct": pieces_all_correct,
                "composed_correct": composed_correct,
                "pred_sql": pred_sql[:500],
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
            gap = sum(1 for p in predictions if p.get("pieces_all_correct") and not p.get("composed_correct"))
            print(f"[INFO] {i+1}/{len(data)} | pieces_ok: {pc} | gap_cases: {gap} | errors: {errors}")

    results = compute_composition_gap(predictions)
    print_results(results, model_key, "SQL (Spider)")
    save_results(predictions, results, model_key, "sql", RESULTS_DIR)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--backend", default="api", choices=["api", "oss"])
    args = parser.parse_args()

    data, schemas = load_data(args.limit)
    print(f"[INFO] Loaded {len(data)} examples, {len(schemas)} schemas")

    models = API_MODELS if args.backend == "api" else OSS_MODELS
    to_run = list(models.keys()) if args.model == "all" else [args.model]

    for mk in to_run:
        run(mk, data, schemas, args.backend)


if __name__ == "__main__":
    main()
