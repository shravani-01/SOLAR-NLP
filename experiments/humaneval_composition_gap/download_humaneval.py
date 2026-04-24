#!/usr/bin/env python3
"""
Download HumanEval dataset and classify structural types.

Downloads from Hugging Face and saves as JSON with structural labels.

Usage:
    python download_humaneval.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent / "data"


def download_and_classify():
    try:
        from datasets import load_dataset
    except ImportError:
        print("[ERROR] Install datasets: pip install datasets")
        sys.exit(1)

    from classify_code_structure import classify_code, STRUCTURAL_TYPES

    print("[INFO] Downloading HumanEval from Hugging Face...")
    ds = load_dataset("openai_humaneval", split="test")

    print(f"[INFO] Downloaded {len(ds)} problems")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Convert to list of dicts and classify
    data = []
    type_counts = Counter()

    for item in ds:
        entry = {
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "canonical_solution": item["canonical_solution"],
            "entry_point": item["entry_point"],
            "test": item["test"],
        }

        # Classify the canonical solution
        full_code = item["prompt"] + item["canonical_solution"]
        stype = classify_code(full_code)
        entry["structural_type"] = stype
        type_counts[stype] += 1

        data.append(entry)

    # Save raw
    raw_path = DATA_DIR / "humaneval.json"
    with open(raw_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved raw data to {raw_path}")

    # Save classified
    classified_path = DATA_DIR / "humaneval_classified.json"
    with open(classified_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Saved classified data to {classified_path}")

    # Print distribution
    total = len(data)
    print(f"\nStructural Type Distribution ({total} problems):")
    print(f"{'Type':<15} {'Count':>6} {'Pct':>8}")
    print(f"{'-'*30}")
    for stype in STRUCTURAL_TYPES:
        count = type_counts[stype]
        pct = count / total * 100
        print(f"{stype:<15} {count:>6} {pct:>7.1f}%")

    # Save distribution
    dist_path = DATA_DIR / "structural_type_distribution.json"
    with open(dist_path, "w") as f:
        json.dump({stype: type_counts[stype] for stype in STRUCTURAL_TYPES}, f, indent=2)

    print(f"\n[INFO] Done! {total} problems classified.")


if __name__ == "__main__":
    download_and_classify()
