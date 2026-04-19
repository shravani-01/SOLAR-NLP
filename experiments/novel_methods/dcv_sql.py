#!/usr/bin/env python3
"""
Decompose-Compose-Verify (DCV) Loop for Text-to-SQL (Spider).

A novel 3-step prompting strategy that explicitly bridges piece extraction
and structural composition for SQL generation.

Steps:
  1. DECOMPOSE — Extract SQL pieces (tables, columns, conditions, aggregations)
  2. COMPOSE   — Given pieces, determine structural pattern (SIMPLE/JOIN/NESTED/SET-OP/MULTI-AGG)
  3. VERIFY    — Generate SQL using the identified structure, verify consistency

Usage:
    python dcv_sql.py --model gpt4o --limit 50
    python dcv_sql.py --model deepseek --limit 50
    python dcv_sql.py --model all
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

sys.path.insert(0, str(Path(__file__).parent.parent / "spider_composition_gap"))
from classify_sql_structure import classify_sql, STRUCTURAL_TYPES

# ─── Config ──────────────────────────────────────────────────────────────────

SPIDER_DIR = Path(__file__).parent.parent / "spider_composition_gap"
DATA_DIR = SPIDER_DIR / "data"
RESULTS_DIR = Path(__file__).parent / "results"

# Load .env
for _env_candidate in [
    Path(__file__).parent / ".env",
    SPIDER_DIR / ".env",
]:
    if _env_candidate.exists():
        with open(_env_candidate) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _val = _line.split("=", 1)
                    os.environ.setdefault(_key.strip(), _val.strip())
        break

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

MODELS = {
    "gpt4o":    {"provider": "openai",   "model": "gpt-4o"},
    "deepseek": {"provider": "deepseek", "model": "deepseek-chat"},
}

SQL_TYPES = ["SIMPLE", "JOIN", "NESTED", "SET-OP", "MULTI-AGG"]

REQUESTS_PER_MINUTE = 20
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


# ─── Prompts ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a SQL expert. Follow instructions precisely."

DECOMPOSE_PROMPT = """Given this database schema and question, extract the SQL PIECES needed (do NOT write SQL yet).

Schema:
{schema}

Question: "{question}"

Extract these pieces:
1. TABLES: Which tables are needed?
2. COLUMNS: Which columns to SELECT?
3. CONDITIONS: What WHERE/HAVING conditions apply?
4. AGGREGATIONS: Any COUNT, SUM, AVG, MAX, MIN, GROUP BY?
5. ORDERING: Any ORDER BY, LIMIT?
6. RELATIONSHIPS: How do tables relate? (join keys, foreign keys)

Return ONLY a JSON object:
{{"tables": [...], "columns": [...], "conditions": [...], "aggregations": [...], "ordering": [...], "relationships": [...]}}"""

COMPOSE_PROMPT = """Given these extracted SQL pieces, determine the STRUCTURAL PATTERN needed.

Question: "{question}"

Extracted pieces:
{pieces_json}

Structural patterns:
- SIMPLE: Single table, no joins, no subqueries. Basic SELECT-WHERE.
- JOIN: Multiple tables connected via JOIN. Explicit or implicit joins.
- NESTED: Subqueries (SELECT inside SELECT, EXISTS, IN with subquery).
- SET-OP: UNION, INTERSECT, or EXCEPT combining multiple queries.
- MULTI-AGG: GROUP BY with HAVING, or multiple aggregation functions.

Based on the extracted pieces, what structural pattern is needed?

Return ONLY a JSON object:
{{"structural_type": "<TYPE>", "reasoning": "<1-2 sentence explanation>", "join_type": "<if JOIN: INNER/LEFT/etc, else null>", "nesting_type": "<if NESTED: IN/EXISTS/scalar, else null>"}}"""

VERIFY_AND_GENERATE_PROMPT = """Now generate the SQL query using the identified structural pattern.

Schema:
{schema}

Question: "{question}"

Extracted pieces: {pieces_json}
Structural pattern: {structural_type}
Reasoning: {reasoning}

IMPORTANT: Your SQL query MUST follow the {structural_type} pattern:
- If SIMPLE: Use only one table, no JOINs or subqueries.
- If JOIN: You MUST use JOIN to connect tables.
- If NESTED: You MUST use a subquery (SELECT within SELECT, EXISTS, or IN subquery).
- If SET-OP: You MUST use UNION, INTERSECT, or EXCEPT.
- If MULTI-AGG: You MUST use GROUP BY with HAVING or multiple aggregation functions.

Before writing the final SQL, verify:
1. Does the query use the correct structural pattern ({structural_type})?
2. Does it include all needed tables and conditions from the pieces?
3. Is it syntactically correct?

Return ONLY the final SQL query. No explanation, no markdown."""


# ─── API calls ──────────────────────────────────────────────────────────────

def call_model(prompt: str, provider: str, model: str) -> str:
    from openai import OpenAI
    if provider == "openai":
        client = OpenAI(api_key=OPENAI_API_KEY)
    elif provider == "deepseek":
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def parse_json_response(response: str) -> dict:
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if match:
        response = match.group(1).strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def extract_sql_from_response(response: str) -> str:
    if not response:
        return ""
    code_block = re.search(r'```(?:sql)?\s*\n?(.*?)\n?```', response, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()
    lines = response.strip().split('\n')
    sql_lines = []
    for line in reversed(lines):
        stripped = line.strip()
        upper = stripped.upper()
        if any(upper.startswith(kw) for kw in ['SELECT', 'WITH']):
            sql_lines.insert(0, stripped)
            break
        elif stripped:
            sql_lines.insert(0, stripped)
    if sql_lines:
        return ' '.join(sql_lines)
    return response.strip()


# ─── Data loading ───────────────────────────────────────────────────────────

def load_spider_data(limit: int = None):
    dev_path = DATA_DIR / "dev_classified.json"
    if not dev_path.exists():
        dev_path = DATA_DIR / "dev.json"
    if not dev_path.exists():
        print(f"[ERROR] No Spider data found at {DATA_DIR}")
        sys.exit(1)

    with open(dev_path) as f:
        dev_data = json.load(f)

    # Load schemas
    schemas = {}
    tables_path = DATA_DIR / "tables.json"
    if tables_path.exists():
        with open(tables_path) as f:
            for db in json.load(f):
                db_id = db["db_id"]
                if "schema_text" in db:
                    schemas[db_id] = db["schema_text"]

    if limit:
        dev_data = dev_data[:limit]
    return dev_data, schemas


# ─── DCV Pipeline ───────────────────────────────────────────────────────────

def run_dcv(model_key: str, dev_data: list, schemas: dict):
    config = MODELS[model_key]
    provider = config["provider"]
    model = config["model"]

    print(f"\n{'='*60}")
    print(f"  DCV Loop (SQL): {model_key} ({model})")
    print(f"{'='*60}")

    if provider == "openai" and not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not set.")
        return None
    if provider == "deepseek" and not DEEPSEEK_API_KEY:
        print("[ERROR] DEEPSEEK_API_KEY not set.")
        return None

    print(f"[INFO] Processing {len(dev_data)} examples (3 calls each)...")

    predictions = []
    errors = 0
    start_time = time.time()

    for i, example in enumerate(dev_data):
        db_id = example.get("db_id", "")
        question = example.get("question", "")
        gold_sql = example.get("query", "")
        gold_type = classify_sql(gold_sql)
        schema = schemas.get(db_id, "-- Schema not available")

        try:
            # Step 1: DECOMPOSE
            d_prompt = DECOMPOSE_PROMPT.format(schema=schema, question=question)
            d_resp = call_model(d_prompt, provider, model)
            pieces = parse_json_response(d_resp)
            time.sleep(REQUEST_DELAY)

            # Step 2: COMPOSE
            pieces_json = json.dumps(pieces, indent=2) if pieces else d_resp
            c_prompt = COMPOSE_PROMPT.format(question=question, pieces_json=pieces_json)
            c_resp = call_model(c_prompt, provider, model)
            compose_result = parse_json_response(c_resp)
            pred_structure = compose_result.get("structural_type", "SIMPLE").upper()
            reasoning = compose_result.get("reasoning", "")
            time.sleep(REQUEST_DELAY)

            # Normalize
            if pred_structure not in SQL_TYPES:
                pred_structure = "SIMPLE"

            # Step 3: VERIFY & GENERATE
            v_prompt = VERIFY_AND_GENERATE_PROMPT.format(
                schema=schema, question=question,
                pieces_json=pieces_json,
                structural_type=pred_structure,
                reasoning=reasoning,
            )
            v_resp = call_model(v_prompt, provider, model)
            pred_sql = extract_sql_from_response(v_resp)
            # Re-classify the actual generated SQL
            actual_type = classify_sql(pred_sql)
            time.sleep(REQUEST_DELAY)

            predictions.append({
                "idx": i,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "gold_type": gold_type,
                "pieces": pieces,
                "pred_structure": pred_structure,
                "pred_sql": pred_sql,
                "actual_type": actual_type,
                "reasoning": reasoning,
            })

        except Exception as e:
            errors += 1
            predictions.append({
                "idx": i,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "gold_type": gold_type,
                "pieces": {},
                "pred_structure": "SIMPLE",
                "pred_sql": "",
                "actual_type": "SIMPLE",
                "reasoning": "",
                "error": str(e),
            })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(dev_data) - i - 1) / rate if rate > 0 else 0
            struct_correct = sum(1 for p in predictions if p["actual_type"] == p["gold_type"])
            print(f"[INFO] Progress: {i+1}/{len(dev_data)} "
                  f"(errors: {errors}, struct_acc: {struct_correct/(i+1):.3f}, ETA: {eta/60:.1f} min)")

    elapsed = time.time() - start_time
    print(f"[INFO] Done. {len(predictions)} predictions, {errors} errors, {elapsed/60:.1f} min")

    # Evaluate structure-level
    tp, fp, fn = Counter(), Counter(), Counter()
    for p in predictions:
        if p["actual_type"] == p["gold_type"]:
            tp[p["gold_type"]] += 1
        else:
            fn[p["gold_type"]] += 1
            fp[p["actual_type"]] += 1

    f1s = {}
    for st in SQL_TYPES:
        prec = tp[st] / (tp[st] + fp[st]) if (tp[st] + fp[st]) > 0 else 0
        rec = tp[st] / (tp[st] + fn[st]) if (tp[st] + fn[st]) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s[st] = round(f1, 4)

    macro_f1 = sum(f1s.values()) / len(f1s)

    # Also check: did the model follow its own structural plan?
    followed_plan = sum(1 for p in predictions if p["pred_structure"] == p["actual_type"])

    print(f"\n  Structure-Level Results:")
    print(f"  {'Type':<15} {'F1':>8}")
    print(f"  {'-'*25}")
    for st in SQL_TYPES:
        print(f"  {st:<15} {f1s[st]:>8.4f}")
    print(f"  {'-'*25}")
    print(f"  {'MACRO F1':<15} {macro_f1:>8.4f}")
    print(f"  {'Followed plan':<15} {followed_plan}/{len(predictions)} ({followed_plan/len(predictions)*100:.1f}%)")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "method": "DCV",
        "domain": "sql",
        "model": model_key,
        "n_examples": len(predictions),
        "n_errors": errors,
        "structure_macro_f1": round(macro_f1, 4),
        "per_type_f1": f1s,
        "followed_plan_pct": round(followed_plan / len(predictions) * 100, 1),
        "timestamp": datetime.now().isoformat(),
    }
    results_path = RESULTS_DIR / f"dcv_sql_{model_key}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    pred_path = RESULTS_DIR / f"dcv_sql_predictions_{model_key}.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"[INFO] Results saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="DCV Loop for SQL (Spider)")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dev_data, schemas = load_spider_data(args.limit)
    print(f"[INFO] Loaded {len(dev_data)} examples, {len(schemas)} schemas")

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        run_dcv(model_key, dev_data, schemas)


if __name__ == "__main__":
    main()
