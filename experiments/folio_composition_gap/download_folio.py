#!/usr/bin/env python3
"""
Download FOLIO dataset and classify structural types.

Downloads from Hugging Face and saves as JSON with structural labels.

Usage:
    python download_folio.py
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

    from classify_logic_structure import classify_logic, STRUCTURAL_TYPES

    print("[INFO] Downloading FOLIO from Hugging Face...")
    try:
        ds = load_dataset("yale-nlp/FOLIO")
    except Exception:
        try:
            ds = load_dataset("hitachi-nlp/FLD", "FOLIO")
        except Exception:
            # Last resort: download from GitHub raw
            print("[INFO] HF download failed. Trying GitHub...")
            try:
                import requests
                import tempfile
                urls = {
                    "train": "https://raw.githubusercontent.com/Yale-LILY/FOLIO/main/data/v0.0/folio-train.jsonl",
                    "validation": "https://raw.githubusercontent.com/Yale-LILY/FOLIO/main/data/v0.0/folio-validation.jsonl",
                }
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                all_items = []
                for split_name, url in urls.items():
                    resp = requests.get(url)
                    resp.raise_for_status()
                    for line in resp.text.strip().split('\n'):
                        if line.strip():
                            item = json.loads(line)
                            item["_split"] = split_name
                            all_items.append(item)
                    print(f"[INFO] Downloaded {split_name} from GitHub")

                # Convert to datasets-like structure
                from collections import defaultdict
                ds = defaultdict(list)
                for item in all_items:
                    ds[item.pop("_split")].append(item)
                ds = dict(ds)

                # Process directly since it's not a HF dataset object
                all_data = []
                type_counts = Counter()
                for split_name, items in ds.items():
                    print(f"[INFO] Processing {split_name}: {len(items)} examples")
                    for item in items:
                        premises = item.get("premises", [])
                        if isinstance(premises, str):
                            premises = [p.strip() + '.' for p in premises.split('.') if p.strip()]
                        conclusion = item.get("conclusion", "")
                        label = item.get("label", "Unknown")
                        entry = {
                            "premises": premises if isinstance(premises, list) else [premises],
                            "conclusion": conclusion,
                            "label": label,
                            "split": split_name,
                        }
                        if item.get("premises-FOL"):
                            entry["premises_fol"] = item["premises-FOL"]
                        if item.get("conclusion-FOL"):
                            entry["conclusion_fol"] = item["conclusion-FOL"]

                        from classify_logic_structure import classify_logic
                        stype = classify_logic(entry["premises"], conclusion)
                        entry["structural_type"] = stype
                        type_counts[stype] += 1
                        all_data.append(entry)

                # Save
                all_path = DATA_DIR / "folio_all.json"
                with open(all_path, "w") as f:
                    json.dump(all_data, f, indent=2)
                print(f"[INFO] Saved all data to {all_path}")

                for split_name in ds.keys():
                    split_data = [d for d in all_data if d["split"] == split_name]
                    split_path = DATA_DIR / f"folio_{split_name}.json"
                    with open(split_path, "w") as f:
                        json.dump(split_data, f, indent=2)

                classified_path = DATA_DIR / "folio_classified.json"
                eval_data = [d for d in all_data if d["split"] == "validation"]
                if not eval_data:
                    eval_data = all_data
                with open(classified_path, "w") as f:
                    json.dump(eval_data, f, indent=2)
                print(f"[INFO] Saved classified eval set ({len(eval_data)} examples)")

                total = len(all_data)
                print(f"\nStructural Type Distribution ({total} total):")
                print(f"{'Type':<20} {'Count':>6} {'Pct':>8}")
                print(f"{'-'*35}")
                from classify_logic_structure import STRUCTURAL_TYPES as ST
                for stype in ST:
                    count = type_counts[stype]
                    pct = count / total * 100
                    print(f"{stype:<20} {count:>6} {pct:>7.1f}%")

                dist_path = DATA_DIR / "structural_type_distribution.json"
                with open(dist_path, "w") as f:
                    json.dump({s: type_counts[s] for s in ST}, f, indent=2)

                print(f"\n[INFO] Done! {total} examples classified.")
                return  # Exit early, we handled everything

            except Exception as e2:
                print(f"[ERROR] All download methods failed: {e2}")
                sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_data = []
    type_counts = Counter()

    for split_name in ds.keys():
        split = ds[split_name]
        print(f"[INFO] Processing {split_name}: {len(split)} examples")

        for item in split:
            # Handle different field name conventions
            premises = item.get("premises", [])
            if isinstance(premises, str):
                premises = [p.strip() for p in premises.split('.') if p.strip()]

            conclusion = item.get("conclusion", "")
            label = item.get("label", "Unknown")
            fol_premises = item.get("premises-FOL", None)

            entry = {
                "premises": premises if isinstance(premises, list) else [premises],
                "conclusion": conclusion,
                "label": label,
                "split": split_name,
            }

            # Add FOL if available
            if fol_premises:
                entry["premises_fol"] = fol_premises
            if item.get("conclusion-FOL"):
                entry["conclusion_fol"] = item["conclusion-FOL"]

            # Classify
            stype = classify_logic(entry["premises"], conclusion, fol_premises)
            entry["structural_type"] = stype
            type_counts[stype] += 1

            all_data.append(entry)

    # Save all data
    all_path = DATA_DIR / "folio_all.json"
    with open(all_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"[INFO] Saved all data to {all_path}")

    # Save splits separately
    for split_name in ds.keys():
        split_data = [d for d in all_data if d["split"] == split_name]
        split_path = DATA_DIR / f"folio_{split_name}.json"
        with open(split_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"[INFO] Saved {split_name} ({len(split_data)} examples) to {split_path}")

    # Save classified version (use validation or test for evaluation)
    test_data = [d for d in all_data if d["split"] in ("test", "validation")]
    if not test_data:
        test_data = all_data  # If no test split, use all
    classified_path = DATA_DIR / "folio_classified.json"
    with open(classified_path, "w") as f:
        json.dump(test_data, f, indent=2)
    print(f"[INFO] Saved classified eval set ({len(test_data)} examples) to {classified_path}")

    # Print distribution
    total = len(all_data)
    print(f"\nStructural Type Distribution ({total} total):")
    print(f"{'Type':<20} {'Count':>6} {'Pct':>8}")
    print(f"{'-'*35}")
    for stype in STRUCTURAL_TYPES:
        count = type_counts[stype]
        pct = count / total * 100
        print(f"{stype:<20} {count:>6} {pct:>7.1f}%")

    # Save distribution
    dist_path = DATA_DIR / "structural_type_distribution.json"
    with open(dist_path, "w") as f:
        json.dump({stype: type_counts[stype] for stype in STRUCTURAL_TYPES}, f, indent=2)

    print(f"\n[INFO] Done! {total} examples classified.")


if __name__ == "__main__":
    download_and_classify()
