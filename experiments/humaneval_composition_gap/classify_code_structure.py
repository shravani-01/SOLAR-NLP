#!/usr/bin/env python3
"""
Rule-based code solution structural type classifier.

Classifies each code generation problem by its solution's structural pattern:
  1. SINGLE-FUNC   — Simple computation, no loops/recursion/complex branching
  2. LOOP          — Requires iteration (for/while). List processing, accumulation
  3. CONDITIONAL   — Requires branching logic (if/elif/else). Edge cases, guards
  4. RECURSIVE     — Self-referential solution. Tree/graph, mathematical sequences
  5. MULTI-STRUCT  — Combines multiple patterns (nested loops, conditional+recursion, etc.)

Priority: MULTI-STRUCT > RECURSIVE > LOOP > CONDITIONAL > SINGLE-FUNC

Works on both gold canonical solutions and model-generated code.
"""

import re
import json
import sys
from pathlib import Path
from collections import Counter


# ─── Structural Type Constants ───────────────────────────────────────────────

SINGLE_FUNC = "SINGLE-FUNC"
LOOP = "LOOP"
CONDITIONAL = "CONDITIONAL"
RECURSIVE = "RECURSIVE"
MULTI_STRUCT = "MULTI-STRUCT"

STRUCTURAL_TYPES = [SINGLE_FUNC, LOOP, CONDITIONAL, RECURSIVE, MULTI_STRUCT]


# ─── Classification helpers ─────────────────────────────────────────────────

def _has_loop(code: str) -> bool:
    """Check if code contains loop structures."""
    # Match for/while at statement level (not inside strings/comments)
    lines = code.split('\n')
    for line in lines:
        stripped = line.strip()
        # Skip comments and strings
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        # for loops
        if re.match(r'\s*for\s+\w+\s+in\s+', line):
            return True
        # while loops
        if re.match(r'\s*while\s+', line):
            return True
    # Also check list comprehensions with for
    if re.search(r'\[.+\s+for\s+\w+\s+in\s+', code):
        return True
    return False


def _has_conditional(code: str) -> bool:
    """Check if code contains non-trivial conditional branching."""
    lines = code.split('\n')
    if_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if re.match(r'\s*if\s+', line) or re.match(r'\s*elif\s+', line):
            if_count += 1
        if re.match(r'\s*else\s*:', line):
            if_count += 1
    # At least one if statement indicates conditional logic
    return if_count >= 1


def _has_recursion(code: str) -> bool:
    """Check if code contains recursive calls."""
    # Find function definitions
    func_names = re.findall(r'def\s+(\w+)\s*\(', code)
    for name in func_names:
        # Check if function calls itself (excluding the def line)
        lines = code.split('\n')
        in_func = False
        for line in lines:
            if re.match(rf'\s*def\s+{name}\s*\(', line):
                in_func = True
                continue
            if in_func:
                # New function definition = left this function
                if re.match(r'def\s+\w+\s*\(', line.strip()):
                    in_func = False
                    continue
                # Check for recursive call
                if re.search(rf'\b{name}\s*\(', line) and not line.strip().startswith('#'):
                    return True
    return False


def _has_nested_structures(code: str) -> bool:
    """Check if code has genuinely nested structures (loop-in-loop, not just if-in-loop).

    A simple if-guard inside a for loop is normal code, not multi-structure.
    We only flag: nested loops, loop inside conditional branch, or 3+ distinct
    structural layers.
    """
    lines = code.split('\n')
    loop_stack = []  # Track only loop nesting

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        indent = len(line) - len(line.lstrip())

        # Pop loop stack for dedented lines
        while loop_stack and indent <= loop_stack[-1]:
            loop_stack.pop()

        # Only track loop-inside-loop as true nesting
        is_loop = False
        if re.match(r'for\s+\w+\s+in\s+', stripped):
            is_loop = True
        elif re.match(r'while\s+', stripped):
            is_loop = True

        if is_loop:
            if loop_stack:  # Already inside a loop = nested loops
                return True
            loop_stack.append(indent)

    # Also check for multiple list comprehensions with nested for
    nested_comp = re.findall(r'\[.+for\s+\w+\s+in\s+.+for\s+\w+\s+in\s+', code)
    if nested_comp:
        return True

    return False


def _count_structural_features(code: str) -> dict:
    """Count all structural features in the code."""
    return {
        'has_loop': _has_loop(code),
        'has_conditional': _has_conditional(code),
        'has_recursion': _has_recursion(code),
        'has_nested': _has_nested_structures(code),
    }


# ─── Main classifier ───────────────────────────────────────────────────────

def classify_code(code: str) -> str:
    """Classify a code solution into a structural type.

    Args:
        code: The Python code solution (gold or predicted).

    Returns:
        One of STRUCTURAL_TYPES.
    """
    if not code or not code.strip():
        return SINGLE_FUNC

    features = _count_structural_features(code)

    # MULTI-STRUCT: nested loops, or recursion + loop (truly complex structure)
    if features['has_nested']:
        return MULTI_STRUCT
    if features['has_recursion'] and features['has_loop']:
        return MULTI_STRUCT

    # RECURSIVE: self-referential (may have base-case if, that's normal)
    if features['has_recursion']:
        return RECURSIVE

    # LOOP: iteration (may have simple if-guards inside, that's normal loop code)
    if features['has_loop']:
        return LOOP

    # CONDITIONAL: non-trivial branching WITHOUT loops
    if features['has_conditional']:
        return CONDITIONAL

    # SINGLE-FUNC: simple computation
    return SINGLE_FUNC


def classify_response(prompt: str, response: str) -> str:
    """Classify a model's generated code response.

    Extracts code from the response (handling markdown blocks)
    and classifies the structural pattern.
    """
    if not response:
        return SINGLE_FUNC

    # Try to extract code from markdown blocks
    code_match = re.search(r'```(?:python)?\s*\n(.*?)```', response, re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
    else:
        # Try to find function definition
        func_match = re.search(r'(def\s+\w+\s*\(.*?)(?:\n\n|\Z)', response, re.DOTALL)
        if func_match:
            code = func_match.group(1).strip()
        else:
            code = response.strip()

    return classify_code(code)


# ─── Batch classification ───────────────────────────────────────────────────

def classify_dataset(data_path: str, output_path: str = None):
    """Classify all problems in a HumanEval dataset file."""
    with open(data_path) as f:
        # Handle both JSON and JSONL
        content = f.read().strip()
        if content.startswith('['):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.strip().split('\n') if line.strip()]

    type_counts = Counter()

    for item in data:
        code = item.get('canonical_solution', '')
        stype = classify_code(code)
        item['structural_type'] = stype
        type_counts[stype] += 1

    total = len(data)
    print(f"\nStructural Type Distribution ({total} problems):")
    print(f"{'Type':<15} {'Count':>6} {'Pct':>8}")
    print(f"{'-'*30}")
    for stype in STRUCTURAL_TYPES:
        count = type_counts[stype]
        pct = count / total * 100 if total > 0 else 0
        print(f"{stype:<15} {count:>6} {pct:>7.1f}%")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved to {output_path}")

    return data, type_counts


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify_code_structure.py <data.json> [output.json]")
        sys.exit(1)

    data_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    classify_dataset(data_path, output_path)
