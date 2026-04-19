#!/usr/bin/env python3
"""
Download and prepare the GSM8K dataset for Composition Gap analysis.

Usage:
    python download_gsm8k.py
"""

import json
from pathlib import Path
from collections import Counter

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

DATA_DIR = Path(__file__).parent / "data"


def extract_answer(answer_str: str) -> str:
    """Extract the final numerical answer from GSM8K answer string."""
    # GSM8K answers end with #### <number>
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip()
    return answer_str.strip()


def extract_steps(answer_str: str) -> list:
    """Extract solution steps from GSM8K answer string."""
    if "####" in answer_str:
        steps_part = answer_str.split("####")[0].strip()
    else:
        steps_part = answer_str.strip()

    steps = [s.strip() for s in steps_part.split("\n") if s.strip()]
    return steps


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    test_path = DATA_DIR / "test.json"
    if test_path.exists():
        with open(test_path) as f:
            data = json.load(f)
        print(f"[INFO] GSM8K test set already exists: {len(data)} examples")
        return

    if not HAS_DATASETS:
        print("[ERROR] Install datasets: pip install datasets")
        return

    print("[INFO] Downloading GSM8K dataset via HuggingFace...")
    ds = load_dataset("openai/gsm8k", "main")

    # Process test set
    test_data = []
    for example in ds["test"]:
        question = example["question"]
        full_answer = example["answer"]
        final_answer = extract_answer(full_answer)
        steps = extract_steps(full_answer)

        test_data.append({
            "question": question,
            "full_answer": full_answer,
            "final_answer": final_answer,
            "steps": steps,
            "n_steps": len(steps),
        })

    # Also save train set (might need for reference)
    train_data = []
    for example in ds["train"]:
        question = example["question"]
        full_answer = example["answer"]
        final_answer = extract_answer(full_answer)
        steps = extract_steps(full_answer)

        train_data.append({
            "question": question,
            "full_answer": full_answer,
            "final_answer": final_answer,
            "steps": steps,
            "n_steps": len(steps),
        })

    with open(DATA_DIR / "test.json", "w") as f:
        json.dump(test_data, f, indent=2)
    with open(DATA_DIR / "train.json", "w") as f:
        json.dump(train_data, f, indent=2)

    print(f"[INFO] Saved {len(test_data)} test examples to {DATA_DIR}/test.json")
    print(f"[INFO] Saved {len(train_data)} train examples to {DATA_DIR}/train.json")

    # Stats
    step_counts = Counter(ex["n_steps"] for ex in test_data)
    print(f"\n{'='*60}")
    print(f"  GSM8K Test Set Statistics")
    print(f"{'='*60}")
    print(f"  Total examples: {len(test_data)}")
    print(f"  Step count distribution:")
    for n_steps in sorted(step_counts.keys()):
        print(f"    {n_steps} steps: {step_counts[n_steps]} ({step_counts[n_steps]/len(test_data)*100:.1f}%)")
    print(f"\n  Sample:")
    sample = test_data[0]
    print(f"    Q: {sample['question'][:80]}")
    print(f"    A: {sample['final_answer']}")
    print(f"    Steps: {sample['n_steps']}")


if __name__ == "__main__":
    main()
