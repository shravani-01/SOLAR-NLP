#!/usr/bin/env python3
"""
Run open-source LLM baselines on Spider dev set for Composition Gap analysis.

Runs locally on GPU using transformers (no API keys needed).

Models:
  1. Qwen2.5-7B (same model we fine-tuned for contracts)
  2. Llama-3.3-70B (same model from our contract baselines)
  3. Qwen2.5-7B + CoT
  4. Llama-3.3-70B + CoT

Usage (on RunPod):
    python run_baselines_oss.py --model qwen7b
    python run_baselines_oss.py --model qwen7b-cot
    python run_baselines_oss.py --model llama70b
    python run_baselines_oss.py --model llama70b-cot
    python run_baselines_oss.py --model all
    python run_baselines_oss.py --model qwen7b --limit 50   # test with 50 examples
"""

import json
import os
import re
import sys
import argparse
import time
from pathlib import Path
from collections import Counter
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from classify_sql_structure import classify_sql, STRUCTURAL_TYPES

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

MODELS = {
    "qwen7b": {
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "cot": False,
        "quantize": False,  # 7B fits in 16-bit on 102GB
    },
    "qwen7b-cot": {
        "hf_name": "Qwen/Qwen2.5-7B-Instruct",
        "cot": True,
        "quantize": False,
    },
    "qwen72b": {
        "hf_name": "Qwen/Qwen2.5-72B-Instruct",
        "cot": False,
        "quantize": True,  # 72B needs 4-bit quantization
    },
    "qwen72b-cot": {
        "hf_name": "Qwen/Qwen2.5-72B-Instruct",
        "cot": True,
        "quantize": True,
    },
}

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0


# ─── Schema loading ─────────────────────────────────────────────────────────

def load_schemas(tables_path: Path) -> dict:
    """Load database schemas from tables.json."""
    with open(tables_path) as f:
        tables_data = json.load(f)

    schemas = {}
    for db in tables_data:
        db_id = db["db_id"]
        if "schema_text" in db:
            schemas[db_id] = db["schema_text"]
        else:
            # Full format with table_names, column_names, etc.
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
    template = COT_PROMPT if cot else ZERO_SHOT_PROMPT
    return template.format(schema=schema, question=question)


# ─── SQL extraction ──────────────────────────────────────────────────────────

def extract_sql_from_response(response: str) -> str:
    """Extract SQL query from model response."""
    if not response:
        return ""

    # Try markdown code blocks
    code_block = re.search(r'```(?:sql)?\s*\n?(.*?)\n?```', response, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()

    # For CoT: find the last SQL statement
    lines = response.strip().split('\n')
    sql_lines = []
    in_sql = False

    for line in reversed(lines):
        stripped = line.strip()
        upper = stripped.upper()

        if any(upper.startswith(kw) for kw in ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']):
            sql_lines.insert(0, stripped)
            in_sql = True
        elif in_sql and stripped and not stripped.startswith('#') and not stripped.startswith('//'):
            sql_lines.insert(0, stripped)
        elif in_sql:
            break

    if sql_lines:
        return ' '.join(sql_lines)

    return response.strip()


# ─── Model loading ───────────────────────────────────────────────────────────

def load_model(hf_name: str, quantize: bool):
    """Load model and tokenizer."""
    print(f"[INFO] Loading {hf_name} (quantize={quantize})...")

    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            hf_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    model.eval()
    print(f"[INFO] Model loaded. Device: {model.device}")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, hf_name: str) -> str:
    """Generate a response from the model."""

    # Build chat messages
    messages = [
        {"role": "system", "content": "You are a SQL expert. Generate accurate SQL queries based on the given schema and question."},
        {"role": "user", "content": prompt},
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE if TEMPERATURE > 0 else None,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_baseline(model_key: str, dev_data: list, schemas: dict, limit: int = None):
    """Run a single OSS baseline."""
    config = MODELS[model_key]
    hf_name = config["hf_name"]
    cot = config["cot"]
    quantize = config["quantize"]

    print(f"\n{'='*60}")
    print(f"  Running: {model_key} ({hf_name}, CoT={cot})")
    print(f"{'='*60}")

    # Load model
    model, tokenizer = load_model(hf_name, quantize)

    examples = dev_data[:limit] if limit else dev_data
    print(f"[INFO] Processing {len(examples)} examples...")

    predictions = []
    errors = 0
    start_time = time.time()

    for i, example in enumerate(examples):
        db_id = example.get("db_id", "")
        question = example.get("question", "")
        gold_sql = example.get("query", "")
        gold_type = classify_sql(gold_sql)

        schema = schemas.get(db_id, "-- Schema not available")
        prompt = build_prompt(question, schema, cot)

        try:
            raw_response = generate_response(model, tokenizer, prompt, hf_name)
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
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(examples) - i - 1) / rate if rate > 0 else 0
            print(f"[INFO] Progress: {i + 1}/{len(examples)} "
                  f"(errors: {errors}, {rate:.1f} ex/s, ETA: {eta/60:.1f} min)")

    elapsed = time.time() - start_time
    print(f"[INFO] Done. {len(predictions)} predictions, {errors} errors, "
          f"{elapsed/60:.1f} minutes")

    # Save predictions
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RESULTS_DIR / f"predictions_{model_key}.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"[INFO] Predictions saved to {pred_path}")

    # Free GPU memory before next model
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run OSS Spider baselines on GPU")
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
        print("[WARN] No tables.json found. Run download_schemas.py first.")

    # Run baselines
    if args.model == "all":
        # Run in order: small models first, then large
        models_to_run = ["qwen7b", "qwen7b-cot", "qwen72b", "qwen72b-cot"]
    else:
        models_to_run = [args.model]

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
            errs = sum(1 for p in preds if p.get("error"))
            type_correct = sum(1 for p in preds if p["gold_type"] == p["pred_type"])
            print(f"  {model_key}: {n} examples, {errs} errors, "
                  f"structure accuracy: {type_correct}/{n} ({type_correct/n:.3f})")


if __name__ == "__main__":
    main()
