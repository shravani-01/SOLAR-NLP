#!/usr/bin/env python3
"""
Download Spider database schemas (tables.json).

The HuggingFace datasets version doesn't include tables.json,
so we need to download it from the official Spider release.
"""

import json
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def download_from_hf():
    """Extract schema info from the HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] Install datasets: pip install datasets")
        return False

    print("[INFO] Loading Spider dataset to extract schemas...")
    ds = load_dataset("xlangai/spider")

    # The HF version stores schema info per-example
    # We need to reconstruct tables.json from the per-example db_id fields
    # Each example has: db_id, question, query
    # But the full schema (table_names, column_names, etc.) is in the dataset features

    # Try to access the underlying data
    sample = ds["validation"][0]
    available_keys = list(sample.keys())
    print(f"[INFO] Available fields: {available_keys}")

    # Check if schema info is embedded
    if "sql" in available_keys:
        print(f"[INFO] SQL field type: {type(sample['sql'])}")
        if isinstance(sample["sql"], dict):
            print(f"[INFO] SQL keys: {list(sample['sql'].keys())}")

    return False  # We'll need another approach


def build_schemas_from_queries(dev_path: Path, train_path: Path) -> dict:
    """
    Build approximate schemas by parsing table/column names from gold SQL queries.
    This is a fallback when tables.json is not available.
    """
    import re

    print("[INFO] Building schemas from gold SQL queries...")

    all_data = []
    for path in [dev_path, train_path]:
        if path.exists():
            with open(path) as f:
                all_data.extend(json.load(f))

    # Group by db_id
    db_queries = {}
    for ex in all_data:
        db_id = ex.get("db_id", "unknown")
        if db_id not in db_queries:
            db_queries[db_id] = []
        db_queries[db_id].append(ex.get("query", ""))

    schemas = {}
    for db_id, queries in db_queries.items():
        tables = set()
        columns = {}  # table -> set of columns

        for sql in queries:
            sql_upper = sql.upper()

            # Extract table names from FROM and JOIN clauses
            for match in re.finditer(r'\bFROM\s+(\w+)', sql_upper):
                t = match.group(1)
                if t not in ('SELECT', 'WHERE', 'AS'):
                    tables.add(t)
            for match in re.finditer(r'\bJOIN\s+(\w+)', sql_upper):
                t = match.group(1)
                tables.add(t)

            # Extract column references (table.column or alias.column)
            for match in re.finditer(r'(\w+)\.(\w+)', sql):
                table_or_alias, col = match.group(1), match.group(2)
                t_upper = table_or_alias.upper()
                # Skip T1, T2 aliases — map them back if possible
                if t_upper not in columns:
                    columns[t_upper] = set()
                columns[t_upper].add(col.upper())

            # Extract column names from SELECT (non-qualified)
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper, re.DOTALL)
            if select_match:
                select_clause = select_match.group(1)
                # Remove aggregation functions
                select_clause = re.sub(r'(COUNT|SUM|AVG|MIN|MAX)\s*\(', '', select_clause)
                for col_match in re.finditer(r'\b([A-Z_][A-Z0-9_]*)\b', select_clause):
                    col = col_match.group(1)
                    if col not in ('DISTINCT', 'AS', 'FROM', 'SELECT', '*'):
                        pass  # Hard to assign to table without context

        # Build schema string
        lines = [f"-- Database: {db_id}"]
        for table in sorted(tables):
            table_cols = columns.get(table, set())
            # Also check aliases
            if table_cols:
                col_str = ", ".join(sorted(table_cols))
                lines.append(f"CREATE TABLE {table} ({col_str});")
            else:
                lines.append(f"CREATE TABLE {table} (...);")

        schemas[db_id] = "\n".join(lines)

    return schemas


def try_download_tables_json():
    """Try to download tables.json from alternative sources."""
    import urllib.request

    urls = [
        "https://raw.githubusercontent.com/taoyds/spider/master/tables.json",
        "https://raw.githubusercontent.com/ygan/spider/master/tables.json",
    ]

    tables_path = DATA_DIR / "tables.json"

    for url in urls:
        try:
            print(f"[INFO] Trying {url}...")
            urllib.request.urlretrieve(url, tables_path)
            # Verify it's valid JSON
            with open(tables_path) as f:
                data = json.load(f)
            print(f"[INFO] Downloaded tables.json with {len(data)} databases")
            return True
        except Exception as e:
            print(f"[WARN] Failed: {e}")
            if tables_path.exists():
                os.remove(tables_path)

    return False


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tables_path = DATA_DIR / "tables.json"
    if tables_path.exists():
        with open(tables_path) as f:
            data = json.load(f)
        print(f"[INFO] tables.json already exists with {len(data)} databases")
        return

    # Try direct download first
    print("[INFO] Attempting to download tables.json...")
    if try_download_tables_json():
        return

    # Fallback: build from queries
    print("[INFO] Direct download failed. Building schemas from SQL queries...")
    dev_path = DATA_DIR / "dev.json"
    train_path = DATA_DIR / "train.json"

    if not dev_path.exists():
        print("[ERROR] No data found. Run download_spider.py first.")
        return

    schemas = build_schemas_from_queries(dev_path, train_path)

    # Save as a simplified tables.json format
    tables_data = []
    for db_id, schema_str in schemas.items():
        tables_data.append({
            "db_id": db_id,
            "schema_text": schema_str,
        })

    with open(tables_path, "w") as f:
        json.dump(tables_data, f, indent=2)

    print(f"[INFO] Saved {len(tables_data)} database schemas to {tables_path}")
    print("[WARN] These are approximate schemas extracted from SQL queries.")
    print("[INFO] For best results, download the official tables.json from:")
    print("       https://yale-lily.github.io/spider")


if __name__ == "__main__":
    main()
