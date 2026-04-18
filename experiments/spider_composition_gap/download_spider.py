#!/usr/bin/env python3
"""
Download and prepare the Spider dataset for Composition Gap analysis.

Downloads Spider 1.0 dev set from HuggingFace and extracts it into
a format ready for our experiments.

Usage:
    python download_spider.py
"""

import json
import os
from pathlib import Path
from collections import Counter

# Try HuggingFace datasets first, fall back to manual download
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

DATA_DIR = Path(__file__).parent / "data"


def download_via_hf():
    """Download Spider via HuggingFace datasets library."""
    print("[INFO] Downloading Spider dataset via HuggingFace...")
    ds = load_dataset("xlangai/spider")

    # Save dev and train splits
    dev_data = []
    for example in ds["validation"]:
        dev_data.append({
            "db_id": example["db_id"],
            "question": example["question"],
            "query": example["query"],
            "sql": example.get("sql", {}),
        })

    train_data = []
    for example in ds["train"]:
        train_data.append({
            "db_id": example["db_id"],
            "question": example["question"],
            "query": example["query"],
            "sql": example.get("sql", {}),
        })

    return train_data, dev_data


def download_manual():
    """Download Spider manually via the official release."""
    import urllib.request
    import zipfile

    url = "https://drive.google.com/uc?export=download&id=1iRDVHLr6THdbqaNR0shL9MtRTEAsR1Tc"
    zip_path = DATA_DIR / "spider.zip"

    print(f"[INFO] Downloading Spider dataset...")
    print(f"[INFO] If automatic download fails, manually download from:")
    print(f"       https://yale-lily.github.io/spider")
    print(f"       and place dev.json in {DATA_DIR}/")

    try:
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(DATA_DIR)
        os.remove(zip_path)
    except Exception as e:
        print(f"[WARN] Auto-download failed: {e}")
        print(f"[INFO] Please manually download Spider and place files in {DATA_DIR}/")
        return None, None

    # Load the JSON files
    spider_dir = DATA_DIR / "spider"
    with open(spider_dir / "dev.json") as f:
        dev_data = json.load(f)
    with open(spider_dir / "train_spider.json") as f:
        train_data = json.load(f)

    return train_data, dev_data


def load_schemas(tables_path):
    """Load database schemas from tables.json."""
    with open(tables_path) as f:
        tables = json.load(f)

    schemas = {}
    for db in tables:
        db_id = db["db_id"]
        schema_lines = []

        # Build table descriptions
        table_names = db["table_names_original"]
        column_names = db["column_names_original"]  # [[table_idx, col_name], ...]
        column_types = db["column_types"]
        primary_keys = db.get("primary_keys", [])
        foreign_keys = db.get("foreign_keys", [])

        for t_idx, t_name in enumerate(table_names):
            cols = []
            for c_idx, (ct_idx, c_name) in enumerate(column_names):
                if ct_idx == t_idx:
                    c_type = column_types[c_idx] if c_idx < len(column_types) else "text"
                    pk_marker = " (PK)" if c_idx in primary_keys else ""
                    cols.append(f"  {c_name} {c_type}{pk_marker}")

            schema_lines.append(f"CREATE TABLE {t_name} (")
            schema_lines.append(",\n".join(cols))
            schema_lines.append(");")
            schema_lines.append("")

        # Add foreign key info
        if foreign_keys:
            schema_lines.append("-- Foreign Keys:")
            for fk_col, ref_col in foreign_keys:
                if fk_col < len(column_names) and ref_col < len(column_names):
                    fk_table = table_names[column_names[fk_col][0]] if column_names[fk_col][0] < len(table_names) else "?"
                    fk_name = column_names[fk_col][1]
                    ref_table = table_names[column_names[ref_col][0]] if column_names[ref_col][0] < len(table_names) else "?"
                    ref_name = column_names[ref_col][1]
                    schema_lines.append(f"-- {fk_table}.{fk_name} = {ref_table}.{ref_name}")

        schemas[db_id] = "\n".join(schema_lines)

    return schemas


def analyze_distribution(data, label="Dataset"):
    """Print basic statistics about the dataset."""
    print(f"\n{'='*60}")
    print(f"  {label} Statistics")
    print(f"{'='*60}")
    print(f"  Total examples: {len(data)}")

    # Count by database
    db_counts = Counter(ex.get("db_id", "unknown") for ex in data)
    print(f"  Unique databases: {len(db_counts)}")

    # Count by SQL query (unique queries)
    query_counts = Counter(ex.get("query", "").strip() for ex in data)
    print(f"  Unique SQL queries: {len(query_counts)}")

    # Show sample
    print(f"\n  Sample example:")
    sample = data[0]
    print(f"    DB: {sample.get('db_id', 'N/A')}")
    print(f"    Question: {sample.get('question', 'N/A')}")
    print(f"    SQL: {sample.get('query', 'N/A')}")
    print()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    dev_path = DATA_DIR / "dev.json"
    if dev_path.exists():
        print(f"[INFO] Spider dev set already exists at {dev_path}")
        with open(dev_path) as f:
            dev_data = json.load(f)
        analyze_distribution(dev_data, "Spider Dev Set")
        return

    # Try downloading
    if HAS_DATASETS:
        train_data, dev_data = download_via_hf()
    else:
        train_data, dev_data = download_manual()

    if dev_data is None:
        print("[ERROR] Could not download Spider dataset.")
        print("[INFO] Please install 'datasets' package: pip install datasets")
        print("[INFO] Or manually download from https://yale-lily.github.io/spider")
        return

    # Save processed data
    with open(DATA_DIR / "dev.json", "w") as f:
        json.dump(dev_data, f, indent=2)

    if train_data:
        with open(DATA_DIR / "train.json", "w") as f:
            json.dump(train_data, f, indent=2)

    print(f"[INFO] Saved {len(dev_data)} dev examples to {DATA_DIR}/dev.json")
    if train_data:
        print(f"[INFO] Saved {len(train_data)} train examples to {DATA_DIR}/train.json")

    analyze_distribution(dev_data, "Spider Dev Set")


if __name__ == "__main__":
    main()
