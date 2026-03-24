"""
SOLAR — Script 10: Prepare CP-SAT Code Generation Training Data
================================================================
Converts constraint_ir.json into prompt-completion pairs for
fine-tuning Llama to generate CP-SAT code directly from IR JSON.

This is the training data for Phase 5 Part A — teaching Llama
to generate CP-SAT code, not just extract IR.

Input:  data/processed/constraint_ir.json  (with cpsat_code field)
Output: data/training/cpsat_train.jsonl
        data/training/cpsat_val.jsonl
        data/training/cpsat_test.jsonl
        data/training/cpsat_split_stats.json

Usage:
  python scripts/10_prepare_cpsat_training_data.py
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter

PROJECT_ROOT  = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAINING_DIR  = PROJECT_ROOT / "data" / "training"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

# ── Prompt template ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a CP-SAT constraint code generator for transit operations.
Given a structured constraint IR (JSON), generate executable Python CP-SAT code
using Google OR-Tools that implements the constraint.

Rules:
- Use model.Add() for hard constraints
- Use model.NewBoolVar() for boolean conditions
- Use OnlyEnforceIf() for conditional constraints
- Use slack variables for soft constraints
- Always include a comment with the constraint ID and source text
- Return ONLY executable Python code. No explanation, no markdown."""

USER_TEMPLATE = """Generate CP-SAT code for this constraint:

{ir_json}"""


def build_cpsat_training_record(ir: dict) -> dict | None:
    """
    Convert one IR record into a CP-SAT code generation
    prompt-completion pair.

    Returns None if the record has no verified cpsat_code.
    """
    # Only use verified FEASIBLE records as training examples
    if not ir.get("verified") or not ir.get("cpsat_code"):
        return None

    # Build clean IR for the prompt — exclude solver fields
    prompt_ir = {
        "constraint_id"   : ir["constraint_id"],
        "raw_text"        : ir["raw_text"],
        "constraint_type" : ir["constraint_type"],
        "hardness_subtype": ir["hardness_subtype"],
        "entities"        : ir["entities"],
        "thresholds"      : ir["thresholds"],
        "exceptions"      : ir["exceptions"],
    }

    ir_json_str = json.dumps(prompt_ir, indent=2)

    # Build Llama-3 instruct format prompt
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}"
        f"<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{USER_TEMPLATE.format(ir_json=ir_json_str)}"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    completion = ir["cpsat_code"]

    return {
        "constraint_id"   : ir["constraint_id"],
        "constraint_type" : ir["constraint_type"],
        "hardness_subtype": ir["hardness_subtype"],
        "template_type"   : ir.get("template_type", "unknown"),
        "prompt"          : prompt,
        "completion"      : completion,
        "text"            : prompt + completion + "<|eot_id|>",
        "has_threshold"   : len(ir.get("thresholds", [])) > 0 and
                            any(t.get("value") is not None
                                for t in ir.get("thresholds", [])),
        "has_exception"   : len(ir.get("exceptions", [])) > 0,
        "solver_status"   : ir.get("solver_status", "FEASIBLE"),
    }


def stratified_split(
    records: list[dict],
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    seed:        int   = 42,
) -> tuple[list, list, list]:
    """Split stratified by constraint_type."""
    random.seed(seed)

    by_type: dict[str, list] = {}
    for r in records:
        ctype = r["constraint_type"]
        by_type.setdefault(ctype, []).append(r)

    train, val, test = [], [], []
    for ctype, recs in by_type.items():
        random.shuffle(recs)
        n       = len(recs)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train  += recs[:n_train]
        val    += recs[n_train:n_train + n_val]
        test   += recs[n_train + n_val:]

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def save_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved {len(records):>5} records → {path.relative_to(PROJECT_ROOT)}")


def print_stats(train, val, test, total):
    print(f"\n{'='*55}")
    print(f"  CP-SAT TRAINING DATA SUMMARY")
    print(f"{'='*55}")
    print(f"  Total records     : {total}")
    print(f"\n  {'Split':<8} {'Total':>7} {'HARD':>6} {'SOFT':>6} "
          f"{'Thresh':>7} {'Except':>7}")
    print(f"  {'-'*45}")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        hard   = sum(1 for r in split if r["constraint_type"] == "HARD")
        soft   = sum(1 for r in split if r["constraint_type"] == "SOFT")
        thresh = sum(1 for r in split if r["has_threshold"])
        exc    = sum(1 for r in split if r["has_exception"])
        print(f"  {name:<8} {len(split):>7} {hard:>6} {soft:>6} "
              f"{thresh:>7} {exc:>7}")

    # Template type breakdown
    print(f"\n  Template types in train set:")
    tmpl_counts = Counter(r["template_type"] for r in train)
    for tmpl, count in tmpl_counts.most_common():
        pct = 100 * count // len(train)
        print(f"    {tmpl:<20} {count:>5}  ({pct}%)")
    print(f"{'='*55}")


def show_examples(records: list[dict], n: int = 2):
    print(f"\n  Sample CP-SAT training records:")
    for i, r in enumerate(records[:n]):
        print(f"\n  Example {i+1}: [{r['constraint_type']}] "
              f"{r['constraint_id']} ({r['template_type']})")
        print(f"\n  PROMPT (last 150 chars):")
        print(f"  ...{r['prompt'][-150:]}")
        print(f"\n  COMPLETION (CP-SAT code):")
        for line in r['completion'].split('\n')[:8]:
            print(f"  {line}")
        print(f"  ...")


def main():
    parser = argparse.ArgumentParser(
        description="SOLAR Step 10: Prepare CP-SAT Code Generation Training Data")
    parser.add_argument("--ir", default=None,
        help="Path to IR JSON (default: data/processed/constraint_ir.json)")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio",   type=float, default=0.10)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    print("\n" + "="*55)
    print("SOLAR — Step 10: Prepare CP-SAT Training Data")
    print("="*55)

    ir_path = (Path(args.ir) if args.ir
               else PROCESSED_DIR / "constraint_ir.json")

    if not ir_path.exists():
        print(f"ERROR: {ir_path} not found. Run 09_generate_cpsat.py first.")
        return

    # Load IR
    with open(ir_path) as f:
        ir_records = json.load(f)

    print(f"\n  IR records loaded  : {len(ir_records)}")

    # Build training records — only from verified FEASIBLE records
    print(f"  Building CP-SAT prompt-completion pairs...")
    training_records = []
    skipped = 0
    for ir in ir_records:
        record = build_cpsat_training_record(ir)
        if record:
            training_records.append(record)
        else:
            skipped += 1

    print(f"  Training records   : {len(training_records)}")
    if skipped > 0:
        print(f"  Skipped            : {skipped} (no verified cpsat_code)")

    if not training_records:
        print("\nERROR: No training records built.")
        print("Make sure you ran 09_generate_cpsat.py first.")
        return

    # Split
    train, val, test = stratified_split(
        training_records,
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio,
        seed        = args.seed,
    )

    # Save
    print()
    save_jsonl(train, TRAINING_DIR / "cpsat_train.jsonl")
    save_jsonl(val,   TRAINING_DIR / "cpsat_val.jsonl")
    save_jsonl(test,  TRAINING_DIR / "cpsat_test.jsonl")

    # Stats
    print_stats(train, val, test, len(training_records))

    # Save split stats
    stats_path = TRAINING_DIR / "cpsat_split_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "total": len(training_records),
            "train": len(train),
            "val"  : len(val),
            "test" : len(test),
            "skipped": skipped,
        }, f, indent=2)
    print(f"\n  Stats saved → {stats_path.relative_to(PROJECT_ROOT)}")

    # Show examples
    show_examples(train, n=2)

    print(f"\n{'='*55}")
    print(f"  CP-SAT training data ready!")
    print(f"{'='*55}")
    print(f"  cpsat_train.jsonl : {len(train)} records")
    print(f"  cpsat_val.jsonl   : {len(val)} records")
    print(f"  cpsat_test.jsonl  : {len(test)} records")
    print(f"\n  Each record contains:")
    print(f"    'text'       — full prompt+completion for training")
    print(f"    'prompt'     — IR JSON input (for inference)")
    print(f"    'completion' — CP-SAT code output (target)")
    print(f"\n  Next: Upload cpsat_train.jsonl + cpsat_val.jsonl to")
    print(f"  Google Drive and run 11_train_cpsat_generator.py on Colab\n")


if __name__ == "__main__":
    main()