"""
SOLAR — Script 06: Prepare Training Data
==========================================
Converts constraint_ir.json into prompt-completion pairs
for Llama-3.1-8B-Instruct fine-tuning via LoRA.

Each record becomes:
  prompt     : instruction + raw sentence
  completion : structured JSON with type, entities, thresholds, exceptions

Split: 80% train / 10% val / 10% test
Stratified by constraint_type to preserve class balance.

Output:
  data/training/train.jsonl
  data/training/val.jsonl
  data/training/test.jsonl
  data/training/split_stats.json

Usage:
  python scripts/06_prepare_training_data.py
  python scripts/06_prepare_training_data.py --ir data/processed/constraint_ir.json
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_ROOT  = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAINING_DIR  = PROJECT_ROOT / "data" / "training"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

# ── Prompt template ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a constraint extraction system for transit operations.
Given a sentence from a transit labor agreement or operational policy document,
extract the following structured information in JSON format:

- constraint_type: HARD or SOFT
- hardness_subtype: HARD-CONDITIONAL, SOFT-APPROVAL, SOFT-CONDITIONAL, or same as constraint_type
- entities: {subject: [], authority: [], object: []}
- thresholds: [{variable, value, unit, direction}] or []
- exceptions: [{trigger, type}] or []

Return ONLY valid JSON. No explanation, no preamble, no markdown."""

USER_TEMPLATE = """Extract constraints from this sentence:

\"{sentence}\""""

ASSISTANT_TEMPLATE = """{json_output}"""


def build_completion(ir: dict) -> dict:
    """
    Build the target completion JSON from an IR record.
    Only includes fields the model needs to predict.
    """
    return {
        "constraint_type"  : ir["constraint_type"],
        "hardness_subtype" : ir["hardness_subtype"],
        "entities"         : ir["entities"],
        "thresholds"       : [
            {k: v for k, v in t.items() if k != "raw"}
            for t in ir["thresholds"]
        ],
        "exceptions"       : ir["exceptions"],
    }


def build_training_record(ir: dict) -> dict:
    """
    Convert one IR record into a Hugging Face-compatible
    prompt-completion training record.

    Format: ChatML / Llama-3 instruct format
    """
    completion = build_completion(ir)

    # Format as Llama-3 instruct chat template
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}"
        f"<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{USER_TEMPLATE.format(sentence=ir['raw_text'].strip())}"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    completion_str = json.dumps(completion, indent=2)

    return {
        "constraint_id"    : ir["constraint_id"],
        "constraint_type"  : ir["constraint_type"],
        "hardness_subtype" : ir["hardness_subtype"],
        "prompt"           : prompt,
        "completion"       : completion_str,
        # Full text for training — prompt + completion + end token
        "text"             : prompt + completion_str + "<|eot_id|>",
        "has_threshold"    : len(ir["thresholds"]) > 0 and
                             any(t["value"] is not None
                                 for t in ir["thresholds"]),
        "has_exception"    : len(ir["exceptions"]) > 0,
        "annotation_source": ir["annotation_source"],
        "source_doc"       : ir["source_doc"],
    }


def stratified_split(
    records: list[dict],
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    seed:        int   = 42,
) -> tuple[list, list, list]:
    """
    Split records into train/val/test sets stratified by constraint_type.
    Ensures class distribution is preserved across all three splits.
    """
    random.seed(seed)

    # Group by constraint_type
    by_type: dict[str, list] = defaultdict(list)
    for r in records:
        by_type[r["constraint_type"]].append(r)

    train, val, test = [], [], []

    for ctype, recs in by_type.items():
        random.shuffle(recs)
        n       = len(recs)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train += recs[:n_train]
        val   += recs[n_train:n_train + n_val]
        test  += recs[n_train + n_val:]

    # Shuffle each split so classes are interleaved
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


def save_jsonl(records: list[dict], path: Path) -> None:
    """Save a list of records as newline-delimited JSON."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"  Saved {len(records):>5} records → {path.relative_to(PROJECT_ROOT)}")


def print_split_stats(
    train: list, val: list, test: list, total: int
) -> dict:
    """Print detailed statistics about each split."""

    def split_stats(records: list, name: str) -> dict:
        type_counts  = Counter(r["constraint_type"] for r in records)
        subtype_counts = Counter(r["hardness_subtype"] for r in records)
        with_thresh  = sum(1 for r in records if r["has_threshold"])
        with_exc     = sum(1 for r in records if r["has_exception"])
        human_ann    = sum(1 for r in records
                          if r["annotation_source"] == "human")
        return {
            "name"        : name,
            "total"       : len(records),
            "type_counts" : dict(type_counts),
            "subtype_counts": dict(subtype_counts),
            "with_threshold": with_thresh,
            "with_exception": with_exc,
            "human_annotated": human_ann,
        }

    splits = {
        "train": split_stats(train, "train"),
        "val"  : split_stats(val,   "val"),
        "test" : split_stats(test,  "test"),
    }

    print(f"\n{'='*60}")
    print(f"  TRAINING DATA SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"  Total records    : {total}")
    print(f"\n  {'Split':<8} {'Total':>7} {'HARD':>6} {'SOFT':>6} "
          f"{'Thresh':>7} {'Except':>7} {'Human':>6}")
    print(f"  {'-'*55}")

    for split_name, s in splits.items():
        hard = s["type_counts"].get("HARD", 0)
        soft = s["type_counts"].get("SOFT", 0)
        print(f"  {split_name:<8} {s['total']:>7} {hard:>6} {soft:>6} "
              f"{s['with_threshold']:>7} {s['with_exception']:>7} "
              f"{s['human_annotated']:>6}")

    print(f"\n  Subtype distribution (train set):")
    for subtype, count in sorted(
        splits["train"]["subtype_counts"].items(),
        key=lambda x: -x[1]
    ):
        pct = 100 * count // splits["train"]["total"]
        print(f"    {subtype:<25} {count:>5}  ({pct}%)")

    return splits


def show_examples(records: list[dict], n: int = 2) -> None:
    """Print n example training records for visual inspection."""
    print(f"\n{'='*60}")
    print(f"  SAMPLE TRAINING RECORDS")
    print(f"{'='*60}")

    for i, r in enumerate(records[:n]):
        print(f"\n  Example {i+1}: [{r['constraint_type']}] {r['constraint_id']}")
        print(f"\n  PROMPT (last 200 chars):")
        print(f"  ...{r['prompt'][-200:]}")
        print(f"\n  COMPLETION:")
        # Pretty print the completion JSON
        try:
            comp = json.loads(r["completion"])
            for line in json.dumps(comp, indent=2).split("\n"):
                print(f"  {line}")
        except json.JSONDecodeError:
            print(f"  {r['completion']}")
        print(f"\n  has_threshold : {r['has_threshold']}")
        print(f"  has_exception : {r['has_exception']}")
        print(f"  annotation    : {r['annotation_source']}")


def main():
    parser = argparse.ArgumentParser(
        description="SOLAR Step 6: Prepare Training Data")
    parser.add_argument(
        "--ir",
        help="Path to constraint IR JSON (default: data/processed/constraint_ir.json)")
    parser.add_argument(
        "--train-ratio", type=float, default=0.80,
        help="Fraction for training set (default: 0.80)")
    parser.add_argument(
        "--val-ratio", type=float, default=0.10,
        help="Fraction for validation set (default: 0.10)")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("SOLAR — Step 06: Prepare Training Data")
    print("="*60)

    ir_path = (Path(args.ir) if args.ir
               else PROCESSED_DIR / "constraint_ir.json")

    if not ir_path.exists():
        print(f"\nERROR: {ir_path} not found")
        print("Run scripts/05_build_ir.py first.")
        return

    # Load IR records
    with open(ir_path) as f:
        ir_records = json.load(f)

    print(f"\n  IR records loaded : {len(ir_records)}")

    # Filter to only fully annotated records
    # Skip records where constraint_type is missing or unknown
    valid = [
        r for r in ir_records
        if r["constraint_type"] in ("HARD", "SOFT")
    ]
    skipped = len(ir_records) - len(valid)
    if skipped > 0:
        print(f"  Skipped {skipped} records with missing constraint_type")
    print(f"  Valid records     : {len(valid)}")

    # Build training records
    print(f"\n  Building prompt-completion pairs...")
    training_records = [build_training_record(ir) for ir in valid]

    # Verify all completions are valid JSON
    invalid_json = 0
    for r in training_records:
        try:
            json.loads(r["completion"])
        except json.JSONDecodeError:
            invalid_json += 1
    if invalid_json > 0:
        print(f"  ⚠ {invalid_json} records have invalid JSON completions")
    else:
        print(f"  ✅ All {len(training_records)} completions are valid JSON")

    # Stratified split
    print(f"\n  Splitting: "
          f"{int(args.train_ratio*100)}% train / "
          f"{int(args.val_ratio*100)}% val / "
          f"{int((1-args.train_ratio-args.val_ratio)*100)}% test")

    train, val, test = stratified_split(
        training_records,
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio,
        seed        = args.seed,
    )

    # Save splits
    print()
    save_jsonl(train, TRAINING_DIR / "train.jsonl")
    save_jsonl(val,   TRAINING_DIR / "val.jsonl")
    save_jsonl(test,  TRAINING_DIR / "test.jsonl")

    # Print stats
    split_stats = print_split_stats(train, val, test, len(training_records))

    # Save split stats as JSON for reference
    stats_path = TRAINING_DIR / "split_stats.json"
    with open(stats_path, "w") as f:
        json.dump(split_stats, f, indent=2)
    print(f"\n  Stats saved → {stats_path.relative_to(PROJECT_ROOT)}")

    # Show examples
    show_examples(train, n=2)

    # Final summary
    print(f"\n{'='*60}")
    print(f"  ✅ Training data ready")
    print(f"{'='*60}")
    print(f"  train.jsonl : {len(train)} records")
    print(f"  val.jsonl   : {len(val)} records")
    print(f"  test.jsonl  : {len(test)} records")
    print(f"\n  Each record contains:")
    print(f"    'text'       — full prompt+completion for training")
    print(f"    'prompt'     — input only (for inference)")
    print(f"    'completion' — target JSON output")
    print(f"\n  Next: Upload to Google Colab and run")
    print(f"  scripts/07_train_ir_extractor.ipynb\n")


if __name__ == "__main__":
    main()