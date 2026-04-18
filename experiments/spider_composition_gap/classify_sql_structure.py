#!/usr/bin/env python3
"""
Rule-based SQL structural type classifier.

Classifies each SQL query into one of 5 structural types:
  1. SIMPLE     — Single table, no JOIN/subquery/set-op
  2. JOIN       — Contains JOIN (no subquery/set-op)
  3. NESTED     — Contains subquery
  4. SET-OP     — Contains UNION/INTERSECT/EXCEPT
  5. MULTI-AGG  — GROUP BY + HAVING or multiple aggregations (no subquery/set-op)

Priority: SET-OP > NESTED > JOIN > MULTI-AGG > SIMPLE

This classifier works on raw SQL strings (no parsed AST needed).
"""

import re
import json
import sys
from pathlib import Path
from collections import Counter


# ─── SQL Structural Type Constants ───────────────────────────────────────────

SIMPLE = "SIMPLE"
JOIN = "JOIN"
NESTED = "NESTED"
SET_OP = "SET-OP"
MULTI_AGG = "MULTI-AGG"

STRUCTURAL_TYPES = [SIMPLE, JOIN, NESTED, SET_OP, MULTI_AGG]


# ─── Helper functions ────────────────────────────────────────────────────────

def _normalize_sql(sql: str) -> str:
    """Normalize SQL for reliable pattern matching."""
    # Remove extra whitespace
    sql = re.sub(r'\s+', ' ', sql.strip().upper())
    return sql


def _has_set_operation(sql: str) -> bool:
    """Check if SQL contains UNION, INTERSECT, or EXCEPT at the top level."""
    normalized = _normalize_sql(sql)

    # Remove content inside parentheses to avoid matching set ops in subqueries
    # We only want top-level set operations
    # Simple approach: check if set-op keywords appear outside of balanced parens
    depth = 0
    tokens = normalized.split()
    for i, token in enumerate(tokens):
        depth += token.count('(') - token.count(')')
        if depth == 0 and token in ('UNION', 'INTERSECT', 'EXCEPT'):
            return True
    return False


def _has_subquery(sql: str) -> bool:
    """Check if SQL contains a nested subquery."""
    normalized = _normalize_sql(sql)

    # Find SELECT keywords — if there's more than one, there's a subquery
    # Remove the first SELECT (the main query)
    select_positions = [m.start() for m in re.finditer(r'\bSELECT\b', normalized)]

    if len(select_positions) > 1:
        return True

    # Also check for EXISTS, IN (SELECT, NOT IN (SELECT patterns
    if re.search(r'\bEXISTS\s*\(', normalized):
        return True

    return False


def _has_join(sql: str) -> bool:
    """Check if SQL contains a JOIN clause."""
    normalized = _normalize_sql(sql)

    # Match various JOIN types
    join_patterns = [
        r'\bJOIN\b',
        r'\bINNER\s+JOIN\b',
        r'\bLEFT\s+JOIN\b',
        r'\bRIGHT\s+JOIN\b',
        r'\bOUTER\s+JOIN\b',
        r'\bCROSS\s+JOIN\b',
        r'\bNATURAL\s+JOIN\b',
    ]

    for pattern in join_patterns:
        if re.search(pattern, normalized):
            return True

    # Also detect implicit joins: FROM table1, table2 WHERE table1.x = table2.y
    # Check for multiple tables in FROM clause (comma-separated)
    from_match = re.search(r'\bFROM\b\s+(.+?)(?:\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)', normalized)
    if from_match:
        from_clause = from_match.group(1)
        # Remove subqueries from FROM clause
        depth = 0
        clean_from = []
        for char in from_clause:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif depth == 0:
                clean_from.append(char)
        clean_from = ''.join(clean_from)

        # Count comma-separated table references
        tables = [t.strip() for t in clean_from.split(',') if t.strip()]
        if len(tables) > 1:
            return True

    return False


def _has_multi_aggregation(sql: str) -> bool:
    """Check if SQL has GROUP BY + HAVING, or multiple aggregation functions."""
    normalized = _normalize_sql(sql)

    # GROUP BY + HAVING is a clear multi-aggregation pattern
    if re.search(r'\bGROUP\s+BY\b', normalized) and re.search(r'\bHAVING\b', normalized):
        return True

    # Count aggregation functions
    agg_functions = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
    agg_count = 0
    for func in agg_functions:
        agg_count += len(re.findall(rf'\b{func}\s*\(', normalized))

    # Multiple different aggregations in SELECT + GROUP BY
    if agg_count >= 2 and re.search(r'\bGROUP\s+BY\b', normalized):
        return True

    return False


def classify_sql(sql: str) -> str:
    """
    Classify a SQL query into one of 5 structural types.

    Priority: SET-OP > NESTED > JOIN > MULTI-AGG > SIMPLE
    """
    if not sql or not sql.strip():
        return SIMPLE

    # Apply priority order
    if _has_set_operation(sql):
        return SET_OP

    if _has_subquery(sql):
        return NESTED

    if _has_join(sql):
        return JOIN

    if _has_multi_aggregation(sql):
        return MULTI_AGG

    return SIMPLE


# ─── Batch classification ────────────────────────────────────────────────────

def classify_dataset(data: list) -> list:
    """
    Classify all examples in a Spider dataset.

    Args:
        data: List of Spider examples (each with 'query' field)

    Returns:
        List of examples with added 'structural_type' field
    """
    classified = []
    for example in data:
        sql = example.get("query", "")
        struct_type = classify_sql(sql)
        classified.append({
            **example,
            "structural_type": struct_type,
        })
    return classified


def print_distribution(classified: list, label: str = "Dataset"):
    """Print the distribution of structural types."""
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

    # Show examples per type
    print(f"  Sample queries per type:")
    print(f"  {'-'*55}")
    for stype in STRUCTURAL_TYPES:
        examples = [ex for ex in classified if ex["structural_type"] == stype]
        if examples:
            sample = examples[0]
            sql_preview = sample["query"][:80] + "..." if len(sample["query"]) > 80 else sample["query"]
            print(f"  {stype}:")
            print(f"    Q: {sample.get('question', 'N/A')[:70]}")
            print(f"    SQL: {sql_preview}")
            print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Classify Spider dev set and save results."""
    data_dir = Path(__file__).parent / "data"
    dev_path = data_dir / "dev.json"

    if not dev_path.exists():
        print(f"[ERROR] Dev set not found at {dev_path}")
        print(f"[INFO] Run download_spider.py first.")
        sys.exit(1)

    with open(dev_path) as f:
        dev_data = json.load(f)

    print(f"[INFO] Loaded {len(dev_data)} examples from {dev_path}")

    # Classify
    classified = classify_dataset(dev_data)
    print_distribution(classified, "Spider Dev Set")

    # Save classified data
    output_path = data_dir / "dev_classified.json"
    with open(output_path, "w") as f:
        json.dump(classified, f, indent=2)

    print(f"[INFO] Saved classified data to {output_path}")

    # Also save just the type distribution
    counts = Counter(ex["structural_type"] for ex in classified)
    dist_path = data_dir / "structural_type_distribution.json"
    with open(dist_path, "w") as f:
        json.dump(dict(counts), f, indent=2)
    print(f"[INFO] Saved distribution to {dist_path}")


if __name__ == "__main__":
    main()
