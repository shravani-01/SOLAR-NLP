#!/usr/bin/env python3
"""
Rule-based math solution structural type classifier.

Classifies each GSM8K problem by its solution's structural pattern:
  1. SINGLE-OP      — One arithmetic operation
  2. MULTI-STEP     — 2-3 sequential operations
  3. RATIO-PROP     — Ratios, percentages, proportional reasoning
  4. COMPARISON     — Compute values for multiple entities, then compare
  5. SYSTEM         — Multiple interacting quantities, iterative logic

Classification is based on the GOLD solution steps (not the question).
"""

import re
import json
import sys
from pathlib import Path
from collections import Counter


# ─── Structural Type Constants ───────────────────────────────────────────────

SINGLE_OP = "SINGLE-OP"
MULTI_STEP = "MULTI-STEP"
RATIO_PROP = "RATIO-PROP"
COMPARISON = "COMPARISON"
SYSTEM = "SYSTEM"

STRUCTURAL_TYPES = [SINGLE_OP, MULTI_STEP, RATIO_PROP, COMPARISON, SYSTEM]


# ─── Classification helpers ─────────────────────────────────────────────────

def _count_operations(text: str) -> int:
    """Count arithmetic operations in solution text."""
    # Count explicit arithmetic: +, -, *, /, =
    ops = len(re.findall(r'[\+\-\*\/×÷]', text))
    # Count word-based operations
    ops += len(re.findall(r'\b(plus|minus|times|divided|multiplied|subtract|add)\b', text, re.IGNORECASE))
    # Count <<...>> calculation annotations (GSM8K format)
    ops += len(re.findall(r'<<(.+?)>>', text))
    return max(ops, 0)


def _has_ratio_proportion(question: str, answer: str) -> bool:
    """Check if problem involves ratios, percentages, or proportions."""
    combined = (question + " " + answer).lower()

    ratio_keywords = [
        r'\b\d+\s*%', r'percent', r'percentage',
        r'\bratio\b', r'\bproportion\b',
        r'\bfraction\b', r'\bhalf\b', r'\bthird\b', r'\bquarter\b',
        r'\bdouble\b', r'\btriple\b', r'\btwice\b', r'\bthrice\b',
        r'\b\d+/\d+\b',  # fractions like 2/3
        r'\btimes\s+as\s+(many|much|large|big|long|fast|heavy)',
        r'\b\d+\s*x\s+', r'\bdiscount\b', r'\bmarkup\b', r'\btax\b',
        r'\btip\b', r'\binterest\b', r'\brate\b',
    ]

    for pattern in ratio_keywords:
        if re.search(pattern, combined):
            return True
    return False


def _has_comparison(question: str, answer: str) -> bool:
    """Check if problem requires comparing quantities across entities."""
    combined = (question + " " + answer).lower()

    comparison_keywords = [
        r'\bhow\s+many\s+more\b', r'\bhow\s+much\s+more\b',
        r'\bhow\s+many\s+less\b', r'\bhow\s+much\s+less\b',
        r'\bwho\s+has\s+more\b', r'\bwho\s+has\s+less\b',
        r'\bmore\s+than\b.*\bless\s+than\b',
        r'\bdifference\s+between\b',
        r'\bcompare\b', r'\bcompared\s+to\b',
        r'\btogether\b.*\bhow\s+many\b',
        r'\bcombined\b', r'\btotal\b.*\bboth\b',
        r'\beach\b.*\bhow\s+many\b',
    ]

    for pattern in comparison_keywords:
        if re.search(pattern, combined):
            return True

    # Multiple named entities with quantities
    # Look for patterns like "Alice has X, Bob has Y"
    names = re.findall(r'\b[A-Z][a-z]+\b', question)
    unique_names = set(names)
    if len(unique_names) >= 2:
        # Multiple people/entities mentioned
        has_quantities = len(re.findall(r'\b\d+\b', question)) >= 2
        if has_quantities:
            return True

    return False


def _has_system(question: str, answer: str) -> bool:
    """Check if problem involves iterative/accumulative/system-like reasoning."""
    combined = (question + " " + answer).lower()

    system_keywords = [
        r'\beach\s+(day|week|month|year|hour|minute|time|round|turn)',
        r'\bper\s+(day|week|month|year|hour|minute)',
        r'\bevery\s+(day|week|month|year|hour|minute)',
        r'\bfirst\b.*\bthen\b.*\bfinally\b',
        r'\bday\s+\d+\b', r'\bweek\s+\d+\b',
        r'\bremaining\b.*\bremaining\b',
        r'\bleft\s+over\b.*\bleft\s+over\b',
        r'\brepeat\b', r'\bcycle\b', r'\bpattern\b',
        r'\bif\b.*\bthen\b.*\botherwise\b',
        r'\bfirst\s+\d+\b.*\bnext\s+\d+\b',
        r'\bround\s*\d+\b',
    ]

    for pattern in system_keywords:
        if re.search(pattern, combined):
            return True

    # Multiple time references suggest iterative reasoning
    time_refs = len(re.findall(r'\b(day|week|month|year|hour|morning|evening|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', combined))
    if time_refs >= 3:
        return True

    return False


def classify_math(question: str, full_answer: str, steps: list) -> str:
    """
    Classify a math problem into one of 5 structural types.

    Priority: SYSTEM > COMPARISON > RATIO-PROP > MULTI-STEP > SINGLE-OP
    """
    n_steps = len(steps)
    n_ops = _count_operations(full_answer)

    # Priority-based classification
    if _has_system(question, full_answer):
        return SYSTEM

    if _has_comparison(question, full_answer) and n_steps >= 2:
        return COMPARISON

    if _has_ratio_proportion(question, full_answer):
        return RATIO_PROP

    if n_steps >= 3 or n_ops >= 3:
        return MULTI_STEP

    if n_steps <= 1 and n_ops <= 1:
        return SINGLE_OP

    # Default: MULTI-STEP for anything with 2+ steps
    if n_steps >= 2:
        return MULTI_STEP

    return SINGLE_OP


# ─── Batch classification ────────────────────────────────────────────────────

def classify_dataset(data: list) -> list:
    """Classify all examples."""
    classified = []
    for example in data:
        struct_type = classify_math(
            example.get("question", ""),
            example.get("full_answer", ""),
            example.get("steps", []),
        )
        classified.append({
            **example,
            "structural_type": struct_type,
        })
    return classified


def print_distribution(classified: list, label: str = "Dataset"):
    """Print distribution of structural types."""
    counts = Counter(ex["structural_type"] for ex in classified)
    total = len(classified)

    print(f"\n{'='*60}")
    print(f"  {label} — Structural Type Distribution")
    print(f"{'='*60}")
    print(f"  {'Type':<15} {'Count':>6} {'Pct':>8}")
    print(f"  {'-'*35}")

    for stype in STRUCTURAL_TYPES:
        count = counts.get(stype, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {stype:<15} {count:>6} {pct:>7.1f}%")

    print(f"  {'-'*35}")
    print(f"  {'TOTAL':<15} {total:>6} {'100.0':>7}%")
    print()

    # Show samples per type
    print(f"  Sample problems per type:")
    print(f"  {'-'*55}")
    for stype in STRUCTURAL_TYPES:
        examples = [ex for ex in classified if ex["structural_type"] == stype]
        if examples:
            sample = examples[0]
            q_preview = sample["question"][:75] + "..." if len(sample["question"]) > 75 else sample["question"]
            print(f"  {stype}:")
            print(f"    Q: {q_preview}")
            print(f"    A: {sample.get('final_answer', 'N/A')}")
            print(f"    Steps: {sample.get('n_steps', 'N/A')}")
            print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    data_dir = Path(__file__).parent / "data"
    test_path = data_dir / "test.json"

    if not test_path.exists():
        print(f"[ERROR] Test set not found at {test_path}")
        print(f"[INFO] Run download_gsm8k.py first.")
        sys.exit(1)

    with open(test_path) as f:
        test_data = json.load(f)

    print(f"[INFO] Loaded {len(test_data)} examples from {test_path}")

    classified = classify_dataset(test_data)
    print_distribution(classified, "GSM8K Test Set")

    # Save
    output_path = data_dir / "test_classified.json"
    with open(output_path, "w") as f:
        json.dump(classified, f, indent=2)
    print(f"[INFO] Saved classified data to {output_path}")

    counts = Counter(ex["structural_type"] for ex in classified)
    dist_path = data_dir / "structural_type_distribution.json"
    with open(dist_path, "w") as f:
        json.dump(dict(counts), f, indent=2)
    print(f"[INFO] Saved distribution to {dist_path}")


if __name__ == "__main__":
    main()
