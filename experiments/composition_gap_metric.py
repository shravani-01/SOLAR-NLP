#!/usr/bin/env python3
"""
Universal Composition Gap Metric — Per-Example Definition.

Following Press et al. (2023), the composition gap is defined as:

    Composition Gap = P(structure wrong | pieces correct)

    = count(pieces_correct AND structure_wrong) / count(pieces_correct)

This measures: "How often does the model fail at structural assembly
despite having identified all the right pieces?"

This is computed PER EXAMPLE, not as a difference of aggregate metrics.
It's directly comparable to Press et al.'s compositionality gap.

Usage:
    python composition_gap_metric.py --domain spider
    python composition_gap_metric.py --domain all
"""

import json
import re
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime


# ─── Piece correctness thresholds ──────────────────────────────────────────

PIECE_THRESHOLD = 0.5  # Minimum piece F1 to count as "pieces correct"


def compute_set_f1(gold: set, pred: set) -> float:
    """F1 between two sets."""
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    prec = tp / len(pred)
    rec = tp / len(gold)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


# ─── Domain-specific piece extractors ──────────────────────────────────────

def extract_sql_pieces(sql: str) -> dict:
    """Extract pieces from a SQL query string."""
    if not sql:
        return {"tables": set(), "operations": set()}

    sql_upper = sql.upper()

    # Tables: words after FROM or JOIN
    tables = set()
    for match in re.finditer(r'\b(?:FROM|JOIN)\s+(\w+)', sql_upper):
        tables.add(match.group(1))

    # Operations
    operations = set()
    if re.search(r'\bJOIN\b', sql_upper):
        operations.add("JOIN")
    if re.search(r'\bWHERE\b', sql_upper):
        operations.add("WHERE")
    if re.search(r'\bGROUP\s+BY\b', sql_upper):
        operations.add("GROUP_BY")
    if re.search(r'\bHAVING\b', sql_upper):
        operations.add("HAVING")
    if re.search(r'\bORDER\s+BY\b', sql_upper):
        operations.add("ORDER_BY")
    if re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql_upper):
        operations.add("AGGREGATE")
    if re.search(r'\bUNION\b|\bINTERSECT\b|\bEXCEPT\b', sql_upper):
        operations.add("SET_OP")
    if re.search(r'SELECT\s+.*\bSELECT\b', sql_upper):
        operations.add("SUBQUERY")
    if re.search(r'\bLIMIT\b', sql_upper):
        operations.add("LIMIT")

    return {"tables": tables, "operations": operations}


def get_piece_correctness_sql(pred: dict) -> tuple:
    """Check if SQL pieces are correct. Returns (pieces_correct, piece_f1)."""
    gold_pieces = extract_sql_pieces(pred.get("gold_sql", ""))
    pred_pieces = extract_sql_pieces(pred.get("pred_sql", ""))

    table_f1 = compute_set_f1(gold_pieces["tables"], pred_pieces["tables"])
    ops_f1 = compute_set_f1(gold_pieces["operations"], pred_pieces["operations"])
    piece_f1 = (table_f1 + ops_f1) / 2.0

    return piece_f1 >= PIECE_THRESHOLD, piece_f1


def get_piece_correctness_math(pred: dict) -> tuple:
    """Check if math pieces are correct."""
    gold_nums = set(str(n) for n in pred.get("gold_numbers", []))
    pred_nums = set(str(n) for n in pred.get("pred_numbers", []))
    gold_ops = set(pred.get("gold_ops", []))
    pred_ops = set(pred.get("pred_ops", []))

    num_f1 = compute_set_f1(gold_nums, pred_nums)
    ops_f1 = compute_set_f1(gold_ops, pred_ops)
    answer_correct = 1.0 if pred.get("answer_correct") else 0.0

    piece_f1 = (num_f1 + ops_f1 + answer_correct) / 3.0
    return piece_f1 >= PIECE_THRESHOLD, piece_f1


def get_piece_correctness_code(pred: dict) -> tuple:
    """Check if code pieces are correct."""
    gold_ids = set(pred.get("gold_identifiers", []))
    pred_ids = set(pred.get("pred_identifiers", []))
    gold_ops = set(pred.get("gold_ops", []))
    pred_ops = set(pred.get("pred_ops", []))

    id_f1 = compute_set_f1(gold_ids, pred_ids)
    ops_f1 = compute_set_f1(gold_ops, pred_ops)

    piece_f1 = (id_f1 + ops_f1) / 2.0
    return piece_f1 >= PIECE_THRESHOLD, piece_f1


def get_piece_correctness_logic(pred: dict) -> tuple:
    """Check if logic pieces are correct."""
    gold_ents = set(pred.get("gold_entities", []))
    pred_ents = set(pred.get("pred_entities", []))
    gold_rels = set(pred.get("gold_relations", []))
    pred_rels = set(pred.get("pred_relations", []))

    ent_f1 = compute_set_f1(gold_ents, pred_ents)
    rel_f1 = compute_set_f1(gold_rels, pred_rels)
    answer_correct = 1.0 if pred.get("answer_correct") else 0.0

    piece_f1 = (ent_f1 + rel_f1 + answer_correct) / 3.0
    return piece_f1 >= PIECE_THRESHOLD, piece_f1


def get_piece_correctness_contracts(pred: dict) -> tuple:
    """Check if contract pieces are correct.
    For contracts, we extract modality and condition signals from text.
    """
    # If prediction has explicit piece fields, use them
    if "gold_entities" in pred:
        gold_ents = set(pred.get("gold_entities", []))
        pred_ents = set(pred.get("pred_entities", []))
        ent_f1 = compute_set_f1(gold_ents, pred_ents)
        piece_f1 = ent_f1
        return piece_f1 >= PIECE_THRESHOLD, piece_f1

    # Fallback: extract from text and response
    text = pred.get("text", "").lower()
    response = pred.get("raw_response", "").lower() if pred.get("raw_response") else ""
    gold_type = pred.get("gold_type", "")

    # Extract modality pieces
    gold_pieces = set()
    pred_pieces = set()

    # Modality
    if re.search(r'\b(shall|must|required|obligat)', text):
        gold_pieces.add("mandatory")
    if re.search(r'\b(may|can|should|discretion|option)', text):
        gold_pieces.add("discretionary")
    if re.search(r'\b(if|when|unless|provided that|in the event|where)\b', text):
        gold_pieces.add("conditional")

    # Check if response mentions these
    if re.search(r'\b(shall|must|mandatory|required|obligat)', response):
        pred_pieces.add("mandatory")
    if re.search(r'\b(may|can|should|discretion|option)', response):
        pred_pieces.add("discretionary")
    if re.search(r'\b(if|when|unless|condition|provided|trigger)', response):
        pred_pieces.add("conditional")

    piece_f1 = compute_set_f1(gold_pieces, pred_pieces)
    return piece_f1 >= PIECE_THRESHOLD, piece_f1


# ─── Universal composition gap computation ──────────────────────────────────

PIECE_FUNCTIONS = {
    "spider": get_piece_correctness_sql,
    "sql": get_piece_correctness_sql,
    "gsm8k": get_piece_correctness_math,
    "math": get_piece_correctness_math,
    "humaneval": get_piece_correctness_code,
    "code": get_piece_correctness_code,
    "folio": get_piece_correctness_logic,
    "logic": get_piece_correctness_logic,
    "contracts": get_piece_correctness_contracts,
}


def compute_composition_gap(predictions: list, domain: str) -> dict:
    """Compute the per-example composition gap.

    Composition Gap = P(structure wrong | pieces correct)

    Returns detailed results including per-type breakdown.
    """
    piece_fn = PIECE_FUNCTIONS.get(domain)
    if not piece_fn:
        raise ValueError(f"Unknown domain: {domain}. Choose from: {list(PIECE_FUNCTIONS.keys())}")

    total = 0
    pieces_correct_count = 0
    structure_correct_count = 0
    both_correct = 0
    pieces_right_structure_wrong = 0

    per_type_stats = defaultdict(lambda: {"pieces_correct": 0, "struct_wrong_given_pieces": 0})
    piece_f1_list = []
    examples = []

    for pred in predictions:
        if pred.get("error"):
            continue

        total += 1
        gold_type = pred.get("gold_type", "")
        pred_type = pred.get("pred_type", "")

        pieces_correct, piece_f1 = piece_fn(pred)
        structure_correct = gold_type == pred_type

        piece_f1_list.append(piece_f1)

        if pieces_correct:
            pieces_correct_count += 1
            per_type_stats[gold_type]["pieces_correct"] += 1

            if not structure_correct:
                pieces_right_structure_wrong += 1
                per_type_stats[gold_type]["struct_wrong_given_pieces"] += 1
                examples.append({
                    "idx": pred.get("idx", -1),
                    "gold_type": gold_type,
                    "pred_type": pred_type,
                    "piece_f1": round(piece_f1, 3),
                })
            else:
                both_correct += 1

        if structure_correct:
            structure_correct_count += 1

    # The gap
    if pieces_correct_count > 0:
        composition_gap = pieces_right_structure_wrong / pieces_correct_count
    else:
        composition_gap = 0.0

    # Per-type gaps
    per_type_gap = {}
    for stype, stats in per_type_stats.items():
        if stats["pieces_correct"] > 0:
            per_type_gap[stype] = {
                "pieces_correct": stats["pieces_correct"],
                "struct_wrong": stats["struct_wrong_given_pieces"],
                "gap": round(stats["struct_wrong_given_pieces"] / stats["pieces_correct"], 4),
            }

    avg_piece_f1 = sum(piece_f1_list) / len(piece_f1_list) if piece_f1_list else 0

    return {
        "total_examples": total,
        "pieces_correct_count": pieces_correct_count,
        "pieces_correct_pct": round(pieces_correct_count / total * 100, 1) if total > 0 else 0,
        "structure_correct_count": structure_correct_count,
        "structure_accuracy": round(structure_correct_count / total, 4) if total > 0 else 0,
        "both_correct": both_correct,
        "pieces_right_structure_wrong": pieces_right_structure_wrong,
        "composition_gap": round(composition_gap, 4),
        "composition_gap_pct": round(composition_gap * 100, 1),
        "avg_piece_f1": round(avg_piece_f1, 4),
        "per_type_gap": per_type_gap,
        "failure_examples": examples[:20],  # Top 20 failure cases
    }


def print_results(results: dict, model: str, domain: str):
    """Pretty-print composition gap results."""
    print(f"\n{'='*60}")
    print(f"  Composition Gap: {model} on {domain}")
    print(f"{'='*60}")
    print(f"  Total examples:     {results['total_examples']}")
    print(f"  Avg piece F1:       {results['avg_piece_f1']:.4f}")
    print(f"  Pieces correct:     {results['pieces_correct_count']} ({results['pieces_correct_pct']}%)")
    print(f"  Structure correct:  {results['structure_correct_count']} ({results['structure_accuracy']:.1%})")
    print(f"  Both correct:       {results['both_correct']}")
    print(f"  Pieces OK, struct WRONG: {results['pieces_right_structure_wrong']}")

    print(f"\n  +{'='*46}+")
    print(f"  |  COMPOSITION GAP (Press et al. style)       |")
    print(f"  |                                              |")
    print(f"  |  P(struct wrong | pieces correct)            |")
    print(f"  |  = {results['pieces_right_structure_wrong']}/{results['pieces_correct_count']}")
    print(f"  |  = {results['composition_gap']:.4f} ({results['composition_gap_pct']:.1f}%)             |")
    print(f"  +{'='*46}+")

    if results['per_type_gap']:
        print(f"\n  Per-Type Gap:")
        print(f"  {'Type':<20} {'Pieces OK':>10} {'Struct Fail':>12} {'Gap':>8}")
        print(f"  {'-'*52}")
        for stype, stats in sorted(results['per_type_gap'].items(), key=lambda x: -x[1]['gap']):
            print(f"  {stype:<20} {stats['pieces_correct']:>10} {stats['struct_wrong']:>12} {stats['gap']:>8.1%}")


# ─── Domain runners ────────────────────────────────────────────────────────

DOMAIN_CONFIGS = {
    "spider": {
        "results_dir": "experiments/spider_composition_gap/results",
        "domain_key": "spider",
    },
    "gsm8k": {
        "results_dir": "experiments/gsm8k_composition_gap/results",
        "domain_key": "gsm8k",
    },
    "humaneval": {
        "results_dir": "experiments/humaneval_composition_gap/results",
        "domain_key": "humaneval",
    },
    "folio": {
        "results_dir": "experiments/folio_composition_gap/results",
        "domain_key": "folio",
    },
    "contracts": {
        "results_dir": "data/results/composition_gap",
        "domain_key": "contracts",
    },
}


def evaluate_domain(domain: str, base_dir: Path = None):
    """Evaluate all models for a domain."""
    if base_dir is None:
        base_dir = Path(__file__).parent.parent  # Go up from experiments/ to repo root

    config = DOMAIN_CONFIGS.get(domain)
    if not config:
        print(f"[ERROR] Unknown domain: {domain}")
        return {}

    results_dir = base_dir / config["results_dir"]
    if not results_dir.exists():
        print(f"[WARN] No results directory: {results_dir}")
        return {}

    all_results = {}

    for pred_file in sorted(results_dir.glob("predictions_*.json")):
        model_key = pred_file.stem.replace("predictions_", "")

        try:
            with open(pred_file) as f:
                predictions = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {pred_file}: {e}")
            continue

        results = compute_composition_gap(predictions, config["domain_key"])
        print_results(results, model_key, domain)

        # Save individual result
        result_path = results_dir / f"composition_gap_{model_key}.json"
        save_data = {
            "model": model_key,
            "domain": domain,
            "metric": "P(struct_wrong | pieces_correct)",
            **results,
            "timestamp": datetime.now().isoformat(),
        }
        # Remove examples for summary file
        save_data_summary = {k: v for k, v in save_data.items() if k != "failure_examples"}
        with open(result_path, "w") as f:
            json.dump(save_data_summary, f, indent=2, default=str)

        all_results[model_key] = results

    return all_results


def print_cross_domain_summary(all_domain_results: dict):
    """Print cross-domain comparison."""
    print(f"\n{'='*70}")
    print(f"  CROSS-DOMAIN COMPOSITION GAP SUMMARY")
    print(f"  Metric: P(structure wrong | pieces correct)")
    print(f"{'='*70}")
    print(f"  {'Domain':<15} {'Model':<18} {'Gap':>8} {'Pieces OK':>10} {'Struct Fail':>12}")
    print(f"  {'-'*65}")

    domain_avgs = {}

    for domain, models in sorted(all_domain_results.items()):
        gaps = []
        for model, results in sorted(models.items()):
            gap = results['composition_gap']
            gaps.append(gap)
            print(f"  {domain:<15} {model:<18} {gap:>7.1%} {results['pieces_correct_count']:>10} "
                  f"{results['pieces_right_structure_wrong']:>12}")
        if gaps:
            avg = sum(gaps) / len(gaps)
            domain_avgs[domain] = avg
            print(f"  {domain:<15} {'AVERAGE':<18} {avg:>7.1%}")
        print()

    if domain_avgs:
        print(f"  {'='*65}")
        print(f"  DOMAIN RANKING (by average composition gap):")
        for domain, avg in sorted(domain_avgs.items(), key=lambda x: -x[1]):
            bar = '#' * int(avg * 50)
            print(f"  {domain:<15} {avg:>7.1%}  {bar}")


def main():
    parser = argparse.ArgumentParser(description="Universal Composition Gap Metric")
    parser.add_argument("--domain", type=str, default="all",
                        choices=list(DOMAIN_CONFIGS.keys()) + ["all"])
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Piece F1 threshold for 'pieces correct' (default: 0.5)")
    args = parser.parse_args()

    global PIECE_THRESHOLD
    PIECE_THRESHOLD = args.threshold

    base_dir = Path(__file__).parent.parent  # repo root

    if args.domain == "all":
        all_domain_results = {}
        for domain in DOMAIN_CONFIGS:
            results = evaluate_domain(domain, base_dir)
            if results:
                all_domain_results[domain] = results
        if all_domain_results:
            print_cross_domain_summary(all_domain_results)
    else:
        evaluate_domain(args.domain, base_dir)


if __name__ == "__main__":
    main()
