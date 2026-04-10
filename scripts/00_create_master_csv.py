"""
SOLAR — Script 00: Create Master CSV + Domain Splits
======================================================
Combines all domain annotated CSVs into one master file,
adds domain/split columns, performs document-level splitting
(80/10/10 train/val/test) so no document leaks across splits.

Outputs:
  data/annotated/all_domains_master.csv   — full combined dataset
  data/annotated/splits/train.csv         — training set
  data/annotated/splits/val.csv           — validation set
  data/annotated/splits/test.csv          — test set
  data/annotated/splits/{domain}_test.csv — per-domain test sets

Usage:
  python scripts/00_create_master_csv.py
  python scripts/00_create_master_csv.py --seed 42
"""

import hashlib
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT  = Path(__file__).parent.parent
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"
SPLITS_DIR    = ANNOTATED_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

DOMAINS = [
    "transit",
    "healthcare",
    "education",
    "municipal",
    "construction",
    "aviation",
    "building_services",
    "hospitality",
]

EXPECTED_COLUMNS = [
    "sentence_id",
    "source_doc",
    "page_num",
    "raw_text",
    "predicted_type",
    "signals",
    "confidence_score",
    "is_constraint",
    "constraint_type",
    "entities",
    "threshold",
    "exception",
    "notes",
]

SPLIT_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


def assign_split(source_doc: str, seed: int = 42) -> str:
    """
    Deterministically assign a document to train/val/test
    using a hash of the document name. This ensures:
      - Same document always gets the same split
      - No sentences from the same doc appear in multiple splits
      - Reproducible across runs
    """
    raw       = f"{seed}:{source_doc}"
    hash_val  = int(hashlib.md5(raw.encode()).hexdigest(), 16)
    bucket    = (hash_val % 100) / 100.0

    if bucket < SPLIT_RATIOS["train"]:
        return "train"
    elif bucket < SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]:
        return "val"
    else:
        return "test"


def load_domain(domain: str) -> pd.DataFrame:
    """Load all annotated CSVs for a domain and tag with domain name."""
    domain_dir = ANNOTATED_DIR / domain
    if not domain_dir.exists():
        print(f"  SKIP {domain} — directory not found")
        return pd.DataFrame()

    ann_files = sorted(domain_dir.glob("*_annotated.csv"))
    if not ann_files:
        print(f"  SKIP {domain} — no annotated files found")
        return pd.DataFrame()

    frames = []
    for f in ann_files:
        try:
            df = pd.read_csv(f, dtype={
                "is_constraint":   "object",
                "constraint_type": "object",
                "entities":        "object",
                "threshold":       "object",
                "exception":       "object",
                "notes":           "object",
                "signals":         "object",
            })
            df["domain"]     = domain
            df["source_file"] = f.name
            frames.append(df)
        except Exception as e:
            print(f"  ERROR reading {f.name}: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"  {domain:<20} {len(combined):>6} rows  "
          f"({len(ann_files)} files)")
    return combined


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all expected columns exist.
    Add missing ones as empty strings.
    Rename common variants.
    """
    rename_map = {
        "sentence_index": "sentence_id",
        "page":           "page_num",
        "text":           "raw_text",
        "type":           "predicted_type",
        "signal":         "signals",
        "score":          "confidence_score",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items()
                             if k in df.columns})

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df


def add_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Assign train/val/test split at document level.
    All sentences from the same source_doc get the same split.
    """
    df["split"] = df["source_doc"].apply(
        lambda doc: assign_split(str(doc), seed)
    )
    return df


def print_stats(df: pd.DataFrame):
    """Print summary statistics."""
    total = len(df)
    yes   = (df["is_constraint"] == "Yes").sum()

    print(f"\n{'='*60}")
    print(f"  MASTER CSV SUMMARY")
    print(f"{'='*60}")
    print(f"  Total rows          : {total:,}")
    print(f"  Confirmed constraints: {yes:,}  "
          f"({100*yes//total}%)")

    print(f"\n  By domain:")
    for domain in DOMAINS:
        d = df[df["domain"] == domain]
        if len(d) == 0:
            continue
        d_yes = (d["is_constraint"] == "Yes").sum()
        print(f"    {domain:<20} {len(d):>6}  "
              f"({d_yes:,} constraints)")

    print(f"\n  By split:")
    for split in ["train", "val", "test"]:
        s     = df[df["split"] == split]
        s_yes = (s["is_constraint"] == "Yes").sum()
        docs  = s["source_doc"].nunique()
        print(f"    {split:<8} {len(s):>6} rows  "
              f"{s_yes:>5} constraints  "
              f"{docs:>3} documents")

    print(f"\n  Constraint type distribution:")
    type_counts = df[df["is_constraint"] == "Yes"][
        "constraint_type"
    ].value_counts()
    for ctype, count in type_counts.items():
        print(f"    {str(ctype):<20} {count:>6}")

    print(f"{'='*60}\n")


def check_leakage(df: pd.DataFrame):
    """Verify no document appears in multiple splits."""
    doc_splits = df.groupby("source_doc")["split"].nunique()
    leakers    = doc_splits[doc_splits > 1]

    if len(leakers) > 0:
        print(f"\n  WARNING: {len(leakers)} documents appear "
              f"in multiple splits:")
        for doc in leakers.index[:5]:
            splits = df[df["source_doc"] == doc]["split"].unique()
            print(f"    {doc} → {splits}")
    else:
        print(f"  Split integrity check: PASSED ✓ "
              f"(no document leaks)")


def main():
    parser = argparse.ArgumentParser(
        description="SOLAR Step 00: Create master CSV + splits")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for split assignment (default: 42)")
    parser.add_argument(
        "--domains", nargs="+", default=DOMAINS,
        help="Domains to include (default: all)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("SOLAR — Step 00: Create Master CSV + Splits")
    print("="*60)
    print(f"\n  Loading {len(args.domains)} domains...")
    print(f"  Split seed: {args.seed}")
    print(f"  Strategy: document-level (no doc leaks across splits)\n")

    frames = []
    for domain in args.domains:
        df = load_domain(domain)
        if len(df) > 0:
            frames.append(df)

    if not frames:
        print("ERROR: No annotated files found.")
        return

    master = pd.concat(frames, ignore_index=True)
    master = normalize_columns(master)
    master = add_splits(master, seed=args.seed)

    master["semantic_variable_name"]    = ""
    master["exception_type_normalized"] = ""
    master["threshold_value"]           = ""
    master["threshold_unit"]            = ""
    master["threshold_direction"]       = ""

    col_order = [
        "sentence_id", "domain", "source_doc", "source_file",
        "page_num", "raw_text", "predicted_type", "signals",
        "confidence_score", "is_constraint", "constraint_type",
        "entities", "threshold", "exception", "notes",
        "semantic_variable_name", "exception_type_normalized",
        "threshold_value", "threshold_unit", "threshold_direction",
        "split",
    ]
    for col in col_order:
        if col not in master.columns:
            master[col] = ""
    master = master[[c for c in col_order if c in master.columns]]

    master_path = ANNOTATED_DIR / "all_domains_master.csv"
    master.to_csv(master_path, index=False)
    print(f"\n  Master CSV → {master_path.relative_to(PROJECT_ROOT)}")

    train = master[master["split"] == "train"]
    val   = master[master["split"] == "val"]
    test  = master[master["split"] == "test"]

    train.to_csv(SPLITS_DIR / "train.csv", index=False)
    val.to_csv(  SPLITS_DIR / "val.csv",   index=False)
    test.to_csv( SPLITS_DIR / "test.csv",  index=False)

    print(f"  Train split → splits/train.csv  ({len(train):,} rows)")
    print(f"  Val split   → splits/val.csv    ({len(val):,} rows)")
    print(f"  Test split  → splits/test.csv   ({len(test):,} rows)")

    print(f"\n  Per-domain test sets:")
    for domain in args.domains:
        domain_test = test[test["domain"] == domain]
        if len(domain_test) == 0:
            continue
        out_path = SPLITS_DIR / f"{domain}_test.csv"
        domain_test.to_csv(out_path, index=False)
        docs = domain_test["source_doc"].nunique()
        print(f"    {domain:<20} {len(domain_test):>5} rows  "
              f"{docs} documents → splits/{domain}_test.csv")

    check_leakage(master)
    print_stats(master)

    print(f"  Next: python scripts/05_build_ir.py --domain all")
    print(f"        (runs on all domains using enriched extraction)\n")


if __name__ == "__main__":
    main()