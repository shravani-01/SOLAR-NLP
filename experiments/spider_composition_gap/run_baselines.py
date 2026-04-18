#!/usr/bin/env python3
"""
Run LLM baselines on Spider dev set for Composition Gap analysis.

Tests 4 baselines:
  1. GPT-4o (zero-shot)
  2. GPT-4o + CoT
  3. DeepSeek-V3 (zero-shot)
  4. DeepSeek-V3 + CoT

Measures both piece-level accuracy (per SQL clause) and structure-level
accuracy (structural type classification) to quantify the Composition Gap.

Usage:
    python run_baselines.py --model gpt4o
    python run_baselines.py --model gpt4o-cot
    python run_baselines.py --model deepseek
    python run_baselines.py --model deepseek-cot
    python run_baselines.py --model all
    python run_baselines.py --model gpt4o --limit 50   # test with 50 examples
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

from classify_sql_structure import classify_sql, STRUCTURAL_TYPES

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# Model configs
MODELS = {
    "gpt4o": {
        "provider": "openai",
        "model": "gpt-4o",
        "cot": False,
    },
    "gpt4o-cot": {
        "provider": "openai",
        "model": "gpt-4o",
        "cot": True,
    },
    "deepseek": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "cot": False,
    },
    "deepseek-cot": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "cot": True,
    },
}

# Rate limiting
REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


# ─── Schema loading ─────────────────────────────────────────────────────────

def load_schemas(tables_path: Path) -> dict:
    """Load database schemas from tables.json into formatted strings."""
    with open(tables_path) as f:
        tables_data = json.load(f)

    schemas = {}
    for db in tables_data:
        db_id = db["db_id"]
        lines = []

        table_names = db.get("table_names_original", db.get("table_names", []))
        col_names = db.get("column_names_original", db.get("column_names", []))
        col_types = db.get("column_types", [])
        primary_keys = set(db.get("primary_keys", []))
        foreign_keys = db.get("foreign_keys", [])

        for t_idx, t_name in enumerate(table_names):
            cols = []
            for c_idx, (ct_idx, c_name) in enumerate(col_names):
                if ct_idx == t_idx:
                    c_type = col_types[c_idx] if c_idx < len(col_types) else "TEXT"
                    pk = " PRIMARY KEY" if c_idx in primary_keys else ""
                    cols.append(f"    {c_name} {c_type}{pk}")

            if cols:
                lines.append(f"CREATE TABLE {t_name} (")
                lines.append(",\n".join(cols))
                lines.append(");")

        # Foreign keys
        for fk_col, ref_col in foreign_keys:
            if fk_col < len(col_names) and ref_col < len(col_names):
                fk_t = table_names[col_names[fk_col][0]] if col_names[fk_col][0] < len(table_names) else "?"
                fk_c = col_names[fk_col][1]
                ref_t = table_names[col_names[ref_col][0]] if col_names[ref_col][0] < len(table_names) else "?"
                ref_c = col_names[ref_col][1]
                lines.append(f"-- FK: {fk_t}.{fk_c} REFERENCES {ref_t}.{ref_c}")

        schemas[db_id] = "\n".join(lines)

    return schemas


# ─── Prompt templates ────────────────────────────────────────────────────────

ZERO_SHOT_PROMPT = """Given the following database schema:

{schema}

Write a SQL query that answers this question:
"{question}"

Return ONLY the SQL query. No explanation, no markdown code blocks."""


COT_PROMPT = """Given the following database schema:

{schema}

Write a SQL query that answers this question:
"{question}"

Think step by step:
1. What tables are needed?
2. What columns should be selected?
3. What conditions (WHERE, HAVING) apply?
4. What structural pattern is needed? (simple select, join, subquery, set operation, aggregation)
5. Write the final SQL query.

After your analysis, write the final SQL query on the last line, starting with "SELECT" (or the appropriate SQL keyword)."""


def build_prompt(question: str, schema: str, cot: bool) -> str:
    """Build the prompt for a given question and schema."""
    template = COT_PROMPT if cot else ZERO_SHOT_PROMPT
    return template.format(schema=schema, question=question)


# ─── API calls ───────────────────────────────────────────────────────────────

def call_openai(prompt: str, model: str) -> str:
    """Call OpenAI API."""
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a SQL expert. Generate accurate SQL queries based on the given schema and question."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def call_deepseek(prompt: str, model: str) -> str:
    """Call DeepSeek API (OpenAI-compatible)."""
    from openai import OpenAI

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a SQL expert. Generate accurate SQL queries based on the given schema and question."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def call_model(prompt: str, provider: str, model: str) -> str:
    """Route to the appropriate API."""
    if provider == "openai":
        return call_openai(prompt, model)
    elif provider == "deepseek":
        return call_deepseek(prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ─── SQL extraction from response ───────────────────────────────────────────

def extract_sql_from_response(response: str) -> str:
    """Extract the SQL query from a model response (handles CoT output)."""
    if not response:
        return ""

    # Try to find SQL in markdown code blocks
    code_block = re.search(r'```(?:sql)?\s*\n?(.*?)\n?```', response, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()

    # For CoT: find the last SQL-like statement
    lines = response.strip().split('\n')
    sql_lines = []
    in_sql = False

    for line in reversed(lines):
        stripped = line.strip()
        upper = stripped.upper()

        # Check if this line looks like SQL
        if any(upper.startswith(kw) for kw in ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']):
            sql_lines.insert(0, stripped)
            in_sql = True
        elif in_sql and stripped and not stripped.startswith('#') and not stripped.startswith('//'):
            sql_lines.insert(0, stripped)
        elif in_sql:
            break

    if sql_lines:
        return ' '.join(sql_lines)

    # Fallback: return the whole response (might be just SQL)
    return response.strip()


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_baseline(model_key: str, dev_data: list, schemas: dict, limit: int = None):
    """Run a single baseline on the Spider dev set."""
    config = MODELS[model_key]
    provider = config["provider"]
    model = config["model"]
    cot = config["cot"]

    print(f"\n{'='*60}")
    print(f"  Running: {model_key} ({model}, CoT={cot})")
    print(f"{'='*60}")

    # Check API key
    if provider == "openai" and not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not set. Export it first.")
        return None
    if provider == "deepseek" and not DEEPSEEK_API_KEY:
        print("[ERROR] DEEPSEEK_API_KEY not set. Export it first.")
        return None

    examples = dev_data[:limit] if limit else dev_data
    print(f"[INFO] Processing {len(examples)} examples...")

    predictions = []
    errors = 0

    for i, example in enumerate(examples):
        db_id = example.get("db_id", "")
        question = example.get("question", "")
        gold_sql = example.get("query", "")
        gold_type = classify_sql(gold_sql)

        schema = schemas.get(db_id, "-- Schema not available")
        prompt = build_prompt(question, schema, cot)

        try:
            raw_response = call_model(prompt, provider, model)
            pred_sql = extract_sql_from_response(raw_response)
            pred_type = classify_sql(pred_sql)

            predictions.append({
                "idx": i,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "gold_type": gold_type,
                "pred_sql": pred_sql,
                "pred_type": pred_type,
                "raw_response": raw_response,
            })
        except Exception as e:
            errors += 1
            predictions.append({
                "idx": i,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "gold_type": gold_type,
                "pred_sql": "",
                "pred_type": "SIMPLE",
                "raw_response": f"ERROR: {str(e)}",
                "error": True,
            })

        # Progress
        if (i + 1) % 25 == 0:
            print(f"[INFO] Progress: {i + 1}/{len(examples)} (errors: {errors})")

        # Rate limiting
        time.sleep(REQUEST_DELAY)

    print(f"[INFO] Done. {len(predictions)} predictions, {errors} errors.")

    # Save predictions
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RESULTS_DIR / f"predictions_{model_key}.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"[INFO] Predictions saved to {pred_path}")

    return predictions


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Spider baselines for Composition Gap")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()) + ["all"],
                        help="Which model to run")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples (for testing)")
    args = parser.parse_args()

    # Load data
    dev_path = DATA_DIR / "dev_classified.json"
    if not dev_path.exists():
        dev_path = DATA_DIR / "dev.json"
    if not dev_path.exists():
        print("[ERROR] No Spider data found. Run download_spider.py first.")
        sys.exit(1)

    with open(dev_path) as f:
        dev_data = json.load(f)
    print(f"[INFO] Loaded {len(dev_data)} examples")

    # Load schemas
    # Try multiple possible locations for tables.json
    tables_candidates = [
        DATA_DIR / "tables.json",
        DATA_DIR / "spider" / "tables.json",
    ]
    schemas = {}
    for tables_path in tables_candidates:
        if tables_path.exists():
            schemas = load_schemas(tables_path)
            print(f"[INFO] Loaded {len(schemas)} database schemas from {tables_path}")
            break

    if not schemas:
        print("[WARN] No tables.json found. Prompts will not include schema info.")
        print("[INFO] Expected at:", tables_candidates)

    # Run baselines
    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    all_predictions = {}
    for model_key in models_to_run:
        predictions = run_baseline(model_key, dev_data, schemas, args.limit)
        if predictions:
            all_predictions[model_key] = predictions

    # Print summary
    if all_predictions:
        print(f"\n{'='*60}")
        print(f"  Summary")
        print(f"{'='*60}")
        for model_key, preds in all_predictions.items():
            n = len(preds)
            errors = sum(1 for p in preds if p.get("error"))
            type_correct = sum(1 for p in preds if p["gold_type"] == p["pred_type"])
            print(f"  {model_key}: {n} examples, {errors} errors, "
                  f"structure accuracy: {type_correct}/{n} ({type_correct/n:.3f})")


if __name__ == "__main__":
    main()
