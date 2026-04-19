#!/usr/bin/env python3
"""
Rule-based math solution structural type classifier.

Classifies each GSM8K problem by its solution's structural pattern:
  1. SINGLE-OP      — 1-2 arithmetic operations, simple chain
  2. MULTI-STEP     — 3+ sequential operations, straightforward chain
  3. RATIO-PROP     — Core reasoning involves ratios, percentages, proportions
  4. COMPARISON     — Explicitly compares quantities (difference, who has more)
  5. SYSTEM         — Multiple interacting constraints, conditional/iterative logic

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

def _count_calc_ops(full_answer: str) -> int:
    """Count arithmetic operations. Prefers <<...>> annotations (gold format),
    falls back to counting arithmetic expressions for model responses."""
    # Gold format: <<...>> annotations
    calc_ops = len(re.findall(r'<<(.+?)>>', full_answer))
    if calc_ops > 0:
        return calc_ops

    # Model response fallback: count lines with arithmetic (=, +, -, *, /)
    arith_lines = 0
    for line in full_answer.split('\n'):
        line = line.strip()
        if re.search(r'\d+\s*[\+\-\*\/×÷=]\s*\d+', line):
            arith_lines += 1
    if arith_lines > 0:
        return arith_lines

    # Last resort: count step markers (1., 2., Step 1, etc.)
    step_markers = len(re.findall(r'(?:^|\n)\s*(?:\d+[\.\):]|Step\s+\d+)', full_answer))
    return max(step_markers, 1)


def _count_steps(steps: list) -> int:
    """Count meaningful solution steps."""
    meaningful = [s for s in steps if s.strip() and len(s.strip()) > 5]
    return len(meaningful)


def _has_ratio_proportion(question: str, answer: str) -> bool:
    """
    Check if the CORE reasoning involves ratios, percentages, or proportions.
    Must be the central mechanism, not just a passing mention of 'half'.
    """
    q_lower = question.lower()
    a_lower = answer.lower()

    # Strong ratio signals in the QUESTION (what's being asked)
    strong_q_patterns = [
        r'\b\d+\s*%', r'percent', r'percentage',
        r'\bratio\b', r'\bproportion\b',
        r'\bdiscount\b', r'\bmarkup\b', r'\btax\b',
        r'\btip\b', r'\binterest\s+rate\b',
        r'\btimes\s+as\s+(many|much|large|big|long|fast|heavy)',
        r'\btwice\s+as\b', r'\bthrice\s+as\b',
        r'\btriple\s+(?:the|his|her|their)\b',
        r'\bdouble\s+(?:the|his|her|their)\b',
    ]

    for pattern in strong_q_patterns:
        if re.search(pattern, q_lower):
            return True

    # Ratio operations in the ANSWER (how it's solved)
    answer_ratio_patterns = [
        r'\b\d+\s*/\s*\d+\s*=',  # "2/3 = ..."
        r'<<\d+/\d+=[^>]*>>',     # calc annotation with division
        r'\b\d+\s*\*\s*0\.\d+',   # multiply by decimal (percentage)
        r'<<[^>]*\*\s*0\.\d+[^>]*>>', # calc with decimal multiplication
        r'\b\d+%\s*(?:of|×|\*)', # "20% of ..."
        r'\bdivide\b.*\bby\b',    # "divide X by Y"
        r'\bmultiply\b.*\bby\s+0\.\d+', # "multiply by 0.5"
    ]

    ratio_op_count = 0
    for pattern in answer_ratio_patterns:
        ratio_op_count += len(re.findall(pattern, a_lower))

    if ratio_op_count >= 1:
        soft_q_patterns = [
            r'\bhalf\b', r'\bthird\b', r'\bquarter\b',
            r'\bfraction\b', r'\b\d+/\d+\b',
            r'\bdouble\b', r'\btriple\b', r'\btwice\b',
        ]
        for pattern in soft_q_patterns:
            if re.search(pattern, q_lower):
                return True

    # For model responses: if question has strong ratio signals, trust the question
    # (model responses won't have <<...>> annotations but the question still indicates the type)
    if q_lower:  # only if question is provided
        strong_count = 0
        for pattern in strong_q_patterns:
            if re.search(pattern, q_lower):
                strong_count += 1
        if strong_count >= 1:
            return True

    return False


def _has_comparison(question: str, answer: str) -> bool:
    """
    Check if problem EXPLICITLY asks to compare quantities.
    Only triggers on questions that ask for a difference or comparison.
    """
    q_lower = question.lower()

    comparison_question_patterns = [
        r'\bhow\s+many\s+more\b',
        r'\bhow\s+much\s+more\b',
        r'\bhow\s+many\s+(?:fewer|less)\b',
        r'\bhow\s+much\s+(?:fewer|less)\b',
        r'\bwho\s+(?:has|had|gets|got|earns|earned)\s+more\b',
        r'\bwho\s+(?:has|had|gets|got|earns|earned)\s+(?:fewer|less)\b',
        r'\bdifference\s+between\b',
        r'\bhow\s+many\s+times\s+(?:more|greater|larger)\b',
        r'\bcompare\b',
        r'\bmore\s+.*\bor\s+(?:fewer|less)\b',
    ]

    for pattern in comparison_question_patterns:
        if re.search(pattern, q_lower):
            return True

    return False


def _has_system(question: str, answer: str) -> bool:
    """
    Check if problem involves multi-phase processes, conditional logic,
    or iterative/accumulative reasoning over time periods.
    Requires STRONG signals.
    """
    combined = (question + " " + answer).lower()

    strong_patterns = [
        r'\bfirst\b.*\bthen\b.*\bfinally\b',
        r'\bif\b.*\bthen\b.*\b(?:otherwise|else)\b',
        r'\bday\s+\d+\b',
        r'\bweek\s+\d+\b',
        r'\bround\s*\d+\b',
        r'\brepeat\b',
        r'\bcycle\b',
        r'\bremaining\b.*\bthen\b',
        r'\bfirst\s+\d+\b.*\bnext\s+\d+\b',
        r'\bfirst\s+\d+\b.*\blast\s+\d+\b',
        r'\bfor\s+the\s+first\b.*\bfor\s+the\s+(?:next|rest|remaining)\b',
        r'\bfor\s+\d+\s+(?:days|weeks|months|hours)\b.*\bfor\s+\d+\s+(?:days|weeks|months|hours)\b',
    ]

    for pattern in strong_patterns:
        if re.search(pattern, combined):
            return True

    # Multiple "each/every/per" time references
    time_segments = re.findall(
        r'\b(?:each|every|per)\s+(?:day|week|month|year|hour|minute)\b', combined
    )
    if len(time_segments) >= 2:
        return True

    # Multiple days of the week mentioned
    days = re.findall(
        r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', combined
    )
    if len(set(days)) >= 3:
        return True

    return False


def classify_math(question: str, full_answer: str, steps: list) -> str:
    """
    Classify a math problem into one of 5 structural types.

    Priority: SYSTEM > COMPARISON > RATIO-PROP > (by operation count) MULTI-STEP / SINGLE-OP
    """
    n_ops = _count_calc_ops(full_answer)
    n_steps = _count_steps(steps)

    # Use calc ops if available, else fall back to step count
    complexity = n_ops if n_ops > 0 else n_steps

    # Priority-based classification (most complex first)
    if _has_system(question, full_answer):
        return SYSTEM

    if _has_comparison(question, full_answer):
        return COMPARISON

    if _has_ratio_proportion(question, full_answer):
        return RATIO_PROP

    # Fall through to complexity-based classification
    if complexity <= 2:
        return SINGLE_OP

    return MULTI_STEP


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
