#!/usr/bin/env python3
"""
Probe-and-Prompt (PaP) for Text-to-SQL (Spider).

Since we don't have trained SQL probes, this uses a "self-probing" variant:
  1. First pass: Ask the model to ONLY predict the structural type (no SQL)
  2. Second pass: Generate SQL with explicit structural constraint from step 1

This mimics what the probe does (extract internal structural knowledge)
but uses a two-pass prompting approach instead of linear probes.

Usage:
    python pap_sql.py --model gpt4o --limit 50
    python pap_sql.py --model deepseek --limit 50
    python pap_sql.py --model all
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

REQUESTS_PER_MINUTE = 25
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


# ─── Prompts ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a SQL expert. Follow instructions precisely."

# Step 1: Self-probe — predict structure ONLY (no SQL generation)
PROBE_PROMPT = """Given this database schema and question, predict the SQL STRUCTURAL PATTERN needed.
Do NOT write any SQL. Just analyze the structure.

Schema:
{schema}

Question: "{question}"

Structural patterns:
- SIMPLE: Single table query. Just SELECT-WHERE, no joins or subqueries.
- JOIN: Multiple tables must be connected. Requires JOIN (explicit or implicit).
- NESTED: Requires a subquery (SELECT inside SELECT, EXISTS, IN with subquery).
- SET-OP: Requires UNION, INTERSECT, or EXCEPT to combine multiple result sets.
- MULTI-AGG: Requires GROUP BY with HAVING, or multiple aggregation functions.

Think about:
1. How many tables are involved?
2. Does the question ask to compare, combine, or filter by aggregates?
3. Is there a "not in" or "except" or "both" pattern?
4. Are there nested conditions (e.g., "more than the average")?

Return ONLY a JSON object:
{{"structural_type": "<TYPE>", "confidence": "<high|medium|low>", "reasoning": "<1-2 sentences>"}}"""

# Step 2: Generate SQL with structural constraint
GENERATE_PROMPT = """Write a SQL query for this question. You MUST use the specified structural pattern.

Schema:
{schema}

Question: "{question}"

REQUIRED structural pattern: **{structural_type}** (confidence: {confidence})
Reasoning: {reasoning}

Structural pattern rules:
- If SIMPLE: Use only ONE table. No JOINs, no subqueries, no set operations.
- If JOIN: You MUST use JOIN (or comma-separated tables with WHERE join condition).
- If NESTED: You MUST include a subquery (SELECT within SELECT, or EXISTS/IN with subquery).
- If SET-OP: You MUST use UNION, INTERSECT, or EXCEPT.
- If MULTI-AGG: You MUST use GROUP BY with HAVING, or have 2+ aggregate functions.

Return ONLY the SQL query. No explanation, no markdown code blocks."""


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


# ─── PaP Pipeline ───────────────────────────────────────────────────────────

def run_pap(model_key: str, dev_data: list, schemas: dict):
    config = MODELS[model_key]
    provider = config["provider"]
    model = config["model"]

    print(f"\n{'='*60}")
    print(f"  PaP Self-Probe (SQL): {model_key} ({model})")
    print(f"{'='*60}")

    if provider == "openai" and not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not set.")
        return None
    if provider == "deepseek" and not DEEPSEEK_API_KEY:
        print("[ERROR] DEEPSEEK_API_KEY not set.")
        return None

    print(f"[INFO] Processing {len(dev_data)} examples (2 calls each)...")

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
            # Step 1: SELF-PROBE — predict structure only
            probe_prompt = PROBE_PROMPT.format(schema=schema, question=question)
            probe_resp = call_model(probe_prompt, provider, model)
            probe_result = parse_json_response(probe_resp)
            pred_structure = probe_result.get("structural_type", "SIMPLE").upper()
            confidence = probe_result.get("confidence", "medium")
            reasoning = probe_result.get("reasoning", "")
            time.sleep(REQUEST_DELAY)

            if pred_structure not in SQL_TYPES:
                pred_structure = "SIMPLE"

            # Step 2: GENERATE with structural constraint
            gen_prompt = GENERATE_PROMPT.format(
                schema=schema, question=question,
                structural_type=pred_structure,
                confidence=confidence,
                reasoning=reasoning,
            )
            gen_resp = call_model(gen_prompt, provider, model)
            pred_sql = extract_sql_from_response(gen_resp)
            actual_type = classify_sql(pred_sql)
            time.sleep(REQUEST_DELAY)

            predictions.append({
                "idx": i,
                "db_id": db_id,
                "question": question,
                "gold_sql": gold_sql,
                "gold_type": gold_type,
                "probe_structure": pred_structure,
                "probe_confidence": confidence,
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
                "probe_structure": "SIMPLE",
                "probe_confidence": "low",
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
            probe_correct = sum(1 for p in predictions if p["probe_structure"] == p["gold_type"])
            print(f"[INFO] Progress: {i+1}/{len(dev_data)} "
                  f"(errors: {errors}, struct_acc: {struct_correct/(i+1):.3f}, "
                  f"probe_acc: {probe_correct/(i+1):.3f}, ETA: {eta/60:.1f} min)")

    elapsed = time.time() - start_time
    print(f"[INFO] Done. {len(predictions)} predictions, {errors} errors, {elapsed/60:.1f} min")

    # Evaluate
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

    # Probe accuracy (step 1 alone)
    probe_correct = sum(1 for p in predictions if p["probe_structure"] == p["gold_type"])
    # Did model follow its own probe?
    followed = sum(1 for p in predictions if p["probe_structure"] == p["actual_type"])

    print(f"\n  Structure-Level Results:")
    print(f"  {'Type':<15} {'F1':>8}")
    print(f"  {'-'*25}")
    for st in SQL_TYPES:
        print(f"  {st:<15} {f1s[st]:>8.4f}")
    print(f"  {'-'*25}")
    print(f"  {'MACRO F1':<15} {macro_f1:>8.4f}")
    print(f"  {'Probe acc':<15} {probe_correct/len(predictions):>8.3f}")
    print(f"  {'Followed probe':<15} {followed}/{len(predictions)} ({followed/len(predictions)*100:.1f}%)")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "method": "PaP-Self",
        "domain": "sql",
        "model": model_key,
        "n_examples": len(predictions),
        "n_errors": errors,
        "structure_macro_f1": round(macro_f1, 4),
        "per_type_f1": f1s,
        "probe_accuracy": round(probe_correct / len(predictions), 4),
        "followed_probe_pct": round(followed / len(predictions) * 100, 1),
        "timestamp": datetime.now().isoformat(),
    }
    results_path = RESULTS_DIR / f"pap_sql_{model_key}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    pred_path = RESULTS_DIR / f"pap_sql_predictions_{model_key}.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"[INFO] Results saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="PaP Self-Probe for SQL (Spider)")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    dev_data, schemas = load_spider_data(args.limit)
    print(f"[INFO] Loaded {len(dev_data)} examples, {len(schemas)} schemas")

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        run_pap(model_key, dev_data, schemas)


if __name__ == "__main__":
    main()
