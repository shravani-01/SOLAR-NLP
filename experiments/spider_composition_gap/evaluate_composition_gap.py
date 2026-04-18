#!/usr/bin/env python3
"""
Evaluate Composition Gap on Spider Text-to-SQL predictions.

Computes:
  1. Piece-level accuracy: per-clause SQL component accuracy
  2. Structure-level accuracy: structural type classification Macro F1
  3. Composition Gap: difference between piece-level and structure-level

Usage:
    python evaluate_composition_gap.py                    # all models
    python evaluate_composition_gap.py --model gpt4o      # single model
    python evaluate_composition_gap.py --compare           # comparison table
"""

import json
import re
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

from classify_sql_structure import classify_sql, STRUCTURAL_TYPES

RESULTS_DIR = Path(__file__).parent / "results"


# ─── SQL Component Extraction (Piece-Level) ─────────────────────────────────

def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    if not sql:
        return ""
    sql = re.sub(r'\s+', ' ', sql.strip().upper())
    # Remove trailing semicolon
    sql = sql.rstrip(';').strip()
    return sql


def extract_select_columns(sql: str) -> set:
    """Extract SELECT clause columns/expressions."""
    normalized = normalize_sql(sql)
    match = re.search(r'\bSELECT\b\s+(DISTINCT\s+)?(.*?)\s+\bFROM\b', normalized, re.DOTALL)
    if not match:
        # SELECT without FROM (e.g., SELECT COUNT(*))
        match = re.search(r'\bSELECT\b\s+(DISTINCT\s+)?(.*?)(?:\s+\bWHERE\b|\s+\bGROUP\b|\s+\bORDER\b|\s+\bLIMIT\b|$)', normalized)
    if match:
        cols_str = match.group(2)
        # Split by comma, but respect parentheses
        cols = _split_respecting_parens(cols_str, ',')
        return {c.strip() for c in cols if c.strip()}
    return set()


def extract_where_conditions(sql: str) -> set:
    """Extract WHERE clause conditions."""
    normalized = normalize_sql(sql)
    match = re.search(r'\bWHERE\b\s+(.*?)(?:\s+\bGROUP\b|\s+\bORDER\b|\s+\bLIMIT\b|\s+\bHAVING\b|\s+\bUNION\b|\s+\bINTERSECT\b|\s+\bEXCEPT\b|$)', normalized)
    if match:
        where_str = match.group(1)
        # Split by AND/OR at top level
        conditions = re.split(r'\s+\bAND\b\s+|\s+\bOR\b\s+', where_str)
        return {c.strip() for c in conditions if c.strip()}
    return set()


def extract_groupby(sql: str) -> set:
    """Extract GROUP BY columns."""
    normalized = normalize_sql(sql)
    match = re.search(r'\bGROUP\s+BY\b\s+(.*?)(?:\s+\bHAVING\b|\s+\bORDER\b|\s+\bLIMIT\b|\s+\bUNION\b|\s+\bINTERSECT\b|\s+\bEXCEPT\b|$)', normalized)
    if match:
        cols_str = match.group(1)
        cols = _split_respecting_parens(cols_str, ',')
        return {c.strip() for c in cols if c.strip()}
    return set()


def extract_orderby(sql: str) -> set:
    """Extract ORDER BY columns."""
    normalized = normalize_sql(sql)
    match = re.search(r'\bORDER\s+BY\b\s+(.*?)(?:\s+\bLIMIT\b|\s+\bUNION\b|\s+\bINTERSECT\b|\s+\bEXCEPT\b|$)', normalized)
    if match:
        cols_str = match.group(1)
        cols = _split_respecting_parens(cols_str, ',')
        return {c.strip() for c in cols if c.strip()}
    return set()


def extract_tables(sql: str) -> set:
    """Extract table names from SQL."""
    normalized = normalize_sql(sql)
    tables = set()

    # FROM clause
    from_match = re.search(r'\bFROM\b\s+(\w+)', normalized)
    if from_match:
        tables.add(from_match.group(1))

    # JOIN clauses
    for match in re.finditer(r'\bJOIN\b\s+(\w+)', normalized):
        tables.add(match.group(1))

    # Aliases: FROM table AS T1
    for match in re.finditer(r'\bFROM\b\s+(\w+)\s+(?:AS\s+)?T\d+', normalized):
        tables.add(match.group(1))
    for match in re.finditer(r'\bJOIN\b\s+(\w+)\s+(?:AS\s+)?T\d+', normalized):
        tables.add(match.group(1))

    return tables


def extract_keywords(sql: str) -> set:
    """Extract SQL keywords used (structural indicators)."""
    normalized = normalize_sql(sql)
    keywords = set()

    kw_list = [
        'SELECT', 'DISTINCT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT',
        'JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN',
        'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT',
        'UNION', 'INTERSECT', 'EXCEPT',
        'IN', 'EXISTS', 'BETWEEN', 'LIKE',
        'COUNT', 'SUM', 'AVG', 'MIN', 'MAX',
        'ASC', 'DESC',
    ]

    for kw in kw_list:
        if re.search(r'\b' + kw.replace(' ', r'\s+') + r'\b', normalized):
            keywords.add(kw)

    return keywords


def _split_respecting_parens(s: str, delimiter: str) -> list:
    """Split a string by delimiter, respecting parentheses nesting."""
    parts = []
    current = []
    depth = 0
    for char in s:
        if char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
        elif char == delimiter and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append(''.join(current))
    return parts


# ─── Piece-Level Evaluation ─────────────────────────────────────────────────

def compute_set_f1(gold_set: set, pred_set: set) -> float:
    """Compute F1 between two sets."""
    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0

    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_piece_level_accuracy(gold_sql: str, pred_sql: str) -> dict:
    """
    Compute piece-level accuracy between gold and predicted SQL.

    Returns per-component F1 scores and overall piece F1.
    """
    components = {}

    # Table selection
    gold_tables = extract_tables(gold_sql)
    pred_tables = extract_tables(pred_sql)
    components["table_f1"] = compute_set_f1(gold_tables, pred_tables)

    # SELECT columns
    gold_select = extract_select_columns(gold_sql)
    pred_select = extract_select_columns(pred_sql)
    components["select_f1"] = compute_set_f1(gold_select, pred_select)

    # WHERE conditions
    gold_where = extract_where_conditions(gold_sql)
    pred_where = extract_where_conditions(pred_sql)
    components["where_f1"] = compute_set_f1(gold_where, pred_where)

    # GROUP BY
    gold_group = extract_groupby(gold_sql)
    pred_group = extract_groupby(pred_sql)
    components["groupby_f1"] = compute_set_f1(gold_group, pred_group)

    # ORDER BY
    gold_order = extract_orderby(gold_sql)
    pred_order = extract_orderby(pred_sql)
    components["orderby_f1"] = compute_set_f1(gold_order, pred_order)

    # Keywords
    gold_keywords = extract_keywords(gold_sql)
    pred_keywords = extract_keywords(pred_sql)
    components["keyword_f1"] = compute_set_f1(gold_keywords, pred_keywords)

    # Overall piece F1: average of all component F1s
    all_f1s = [v for v in components.values()]
    components["piece_f1"] = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0

    return components


# ─── Structure-Level Evaluation ──────────────────────────────────────────────

def compute_structure_metrics(predictions: list) -> dict:
    """
    Compute structure-level Macro F1 across 5 structural types.

    Returns per-type F1 and macro F1.
    """
    # Build confusion data
    per_type_tp = Counter()
    per_type_fp = Counter()
    per_type_fn = Counter()

    for pred in predictions:
        gold_type = pred["gold_type"]
        pred_type = pred["pred_type"]

        if gold_type == pred_type:
            per_type_tp[gold_type] += 1
        else:
            per_type_fn[gold_type] += 1
            per_type_fp[pred_type] += 1

    # Per-type precision, recall, F1
    per_type_f1 = {}
    for stype in STRUCTURAL_TYPES:
        tp = per_type_tp[stype]
        fp = per_type_fp[stype]
        fn = per_type_fn[stype]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_type_f1[stype] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }

    # Macro F1 (only over types with support > 0)
    f1_values = [v["f1"] for v in per_type_f1.values() if v["support"] > 0]
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    return {
        "per_type": per_type_f1,
        "macro_f1": round(macro_f1, 4),
    }


# ─── Full Evaluation ────────────────────────────────────────────────────────

def evaluate_model(model_key: str) -> dict:
    """Evaluate a single model's predictions."""
    pred_path = RESULTS_DIR / f"predictions_{model_key}.json"
    if not pred_path.exists():
        print(f"[WARN] No predictions found for {model_key} at {pred_path}")
        return None

    with open(pred_path) as f:
        predictions = json.load(f)

    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_key}")
    print(f"{'='*60}")
    print(f"  Total predictions: {len(predictions)}")
    errors = sum(1 for p in predictions if p.get("error"))
    print(f"  Errors: {errors}")

    # ── Piece-level evaluation ──
    piece_scores = defaultdict(list)
    for pred in predictions:
        if pred.get("error"):
            continue
        components = compute_piece_level_accuracy(pred["gold_sql"], pred["pred_sql"])
        for key, val in components.items():
            piece_scores[key].append(val)

    avg_piece = {}
    for key, vals in piece_scores.items():
        avg_piece[key] = round(sum(vals) / len(vals), 4) if vals else 0.0

    piece_f1 = avg_piece.get("piece_f1", 0.0)

    print(f"\n  Piece-Level Accuracy:")
    print(f"  {'Component':<15} {'F1':>8}")
    print(f"  {'-'*25}")
    for key in ["table_f1", "select_f1", "where_f1", "groupby_f1", "orderby_f1", "keyword_f1"]:
        print(f"  {key:<15} {avg_piece.get(key, 0):>8.4f}")
    print(f"  {'-'*25}")
    print(f"  {'PIECE F1':<15} {piece_f1:>8.4f}")

    # ── Structure-level evaluation ──
    struct_metrics = compute_structure_metrics(predictions)
    structure_f1 = struct_metrics["macro_f1"]

    print(f"\n  Structure-Level Accuracy:")
    print(f"  {'Type':<15} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Support':>8}")
    print(f"  {'-'*50}")
    for stype in STRUCTURAL_TYPES:
        info = struct_metrics["per_type"].get(stype, {})
        print(f"  {stype:<15} {info.get('f1', 0):>8.4f} {info.get('precision', 0):>8.4f} "
              f"{info.get('recall', 0):>8.4f} {info.get('support', 0):>8}")
    print(f"  {'-'*50}")
    print(f"  {'MACRO F1':<15} {structure_f1:>8.4f}")

    # ── Composition Gap ──
    gap = piece_f1 - structure_f1
    print(f"\n  ┌────────────────────────────────────────┐")
    print(f"  │  COMPOSITION GAP                        │")
    print(f"  │  Piece F1:     {piece_f1:>8.4f}                │")
    print(f"  │  Structure F1: {structure_f1:>8.4f}                │")
    print(f"  │  Gap:          {gap:>8.4f}  ({gap*100:.1f} points)  │")
    print(f"  └────────────────────────────────────────┘")

    # ── Build results object ──
    results = {
        "model": model_key,
        "n_examples": len(predictions),
        "n_errors": errors,
        "piece_level": avg_piece,
        "piece_f1": piece_f1,
        "structure_level": {
            stype: struct_metrics["per_type"][stype]["f1"]
            for stype in STRUCTURAL_TYPES
        },
        "structure_macro_f1": structure_f1,
        "composition_gap": round(gap, 4),
        "composition_gap_pct": round(gap * 100, 1),
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    results_path = RESULTS_DIR / f"results_{model_key}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to {results_path}")

    return results


def print_comparison_table(all_results: dict):
    """Print a comparison table across all models."""
    print(f"\n{'='*70}")
    print(f"  COMPOSITION GAP — SPIDER TEXT-TO-SQL")
    print(f"{'='*70}")
    print(f"  {'Model':<18} {'Piece F1':>10} {'Struct F1':>10} {'Gap':>10} {'Gap (pts)':>10}")
    print(f"  {'-'*60}")

    for model_key, results in sorted(all_results.items(), key=lambda x: -x[1]["composition_gap"]):
        print(f"  {model_key:<18} {results['piece_f1']:>10.4f} {results['structure_macro_f1']:>10.4f} "
              f"{results['composition_gap']:>10.4f} {results['composition_gap_pct']:>9.1f}")

    print(f"  {'-'*60}")

    # Average gap
    gaps = [r["composition_gap"] for r in all_results.values()]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0
    print(f"  {'AVERAGE':<18} {'':>10} {'':>10} {avg_gap:>10.4f} {avg_gap*100:>9.1f}")
    print()

    # Compare with labor contract results
    print(f"  Cross-Domain Comparison:")
    print(f"  {'-'*50}")
    print(f"  {'Domain':<20} {'Avg Composition Gap':>20}")
    print(f"  {'Labor Contracts':<20} {'~50 points':>20}")
    print(f"  {'Text-to-SQL':<20} {f'~{avg_gap*100:.0f} points':>20}")
    print()

    # Save comparison
    comparison = {
        "spider_results": {k: {
            "piece_f1": v["piece_f1"],
            "structure_macro_f1": v["structure_macro_f1"],
            "composition_gap": v["composition_gap"],
        } for k, v in all_results.items()},
        "average_gap": round(avg_gap, 4),
        "labor_contract_avg_gap": 0.50,
        "timestamp": datetime.now().isoformat(),
    }
    comp_path = RESULTS_DIR / "cross_domain_comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"[INFO] Cross-domain comparison saved to {comp_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Composition Gap on Spider")
    parser.add_argument("--model", type=str, default=None,
                        help="Evaluate a specific model (default: all available)")
    parser.add_argument("--compare", action="store_true",
                        help="Print cross-model comparison table")
    args = parser.parse_args()

    if args.model:
        results = evaluate_model(args.model)
    else:
        # Evaluate all available prediction files
        all_results = {}
        for pred_file in sorted(RESULTS_DIR.glob("predictions_*.json")):
            model_key = pred_file.stem.replace("predictions_", "")
            results = evaluate_model(model_key)
            if results:
                all_results[model_key] = results

        if all_results:
            print_comparison_table(all_results)
        else:
            print("[INFO] No prediction files found. Run run_baselines.py first.")

    if args.compare:
        all_results = {}
        for results_file in sorted(RESULTS_DIR.glob("results_*.json")):
            model_key = results_file.stem.replace("results_", "")
            with open(results_file) as f:
                all_results[model_key] = json.load(f)
        if all_results:
            print_comparison_table(all_results)


if __name__ == "__main__":
    main()
