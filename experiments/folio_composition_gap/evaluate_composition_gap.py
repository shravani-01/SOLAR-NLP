#!/usr/bin/env python3
"""
Evaluate Composition Gap on FOLIO logical reasoning predictions.

Computes:
  1. Piece-level: entity extraction F1, relation identification F1, answer accuracy
  2. Structure-level: reasoning structural type Macro F1
  3. Composition Gap: piece - structure

Usage:
    python evaluate_composition_gap.py
    python evaluate_composition_gap.py --model gpt4o
"""

import json
import re
import sys
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

from classify_logic_structure import classify_response, STRUCTURAL_TYPES

RESULTS_DIR = Path(__file__).parent / "results"


def compute_set_f1(gold_set: set, pred_set: set) -> float:
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


def evaluate_model(model_key: str) -> dict:
    pred_path = RESULTS_DIR / f"predictions_{model_key}.json"
    if not pred_path.exists():
        print(f"[WARN] No predictions found for {model_key}")
        return None

    with open(pred_path) as f:
        predictions = json.load(f)

    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_key}")
    print(f"{'='*60}")
    print(f"  Total predictions: {len(predictions)}")
    errors = sum(1 for p in predictions if p.get("error"))
    print(f"  Errors: {errors}")

    # ── Piece-level ──
    entity_f1s = []
    relation_f1s = []
    answer_accs = []

    for pred in predictions:
        if pred.get("error"):
            continue

        gold_ents = set(pred.get("gold_entities", []))
        pred_ents = set(pred.get("pred_entities", []))
        entity_f1s.append(compute_set_f1(gold_ents, pred_ents))

        gold_rels = set(pred.get("gold_relations", []))
        pred_rels = set(pred.get("pred_relations", []))
        relation_f1s.append(compute_set_f1(gold_rels, pred_rels))

        answer_accs.append(1.0 if pred.get("answer_correct") else 0.0)

    avg_entity_f1 = sum(entity_f1s) / len(entity_f1s) if entity_f1s else 0.0
    avg_relation_f1 = sum(relation_f1s) / len(relation_f1s) if relation_f1s else 0.0
    avg_answer_acc = sum(answer_accs) / len(answer_accs) if answer_accs else 0.0
    piece_f1 = (avg_entity_f1 + avg_relation_f1 + avg_answer_acc) / 3.0

    print(f"\n  Piece-Level Accuracy:")
    print(f"  {'Component':<25} {'Score':>8}")
    print(f"  {'-'*35}")
    print(f"  {'Entity extraction':<25} {avg_entity_f1:>8.4f}")
    print(f"  {'Relation ID':<25} {avg_relation_f1:>8.4f}")
    print(f"  {'Answer accuracy':<25} {avg_answer_acc:>8.4f}")
    print(f"  {'-'*35}")
    print(f"  {'PIECE F1':<25} {piece_f1:>8.4f}")

    # ── Structure-level ──
    per_type_tp = Counter()
    per_type_fp = Counter()
    per_type_fn = Counter()

    for pred in predictions:
        if pred.get("error"):
            continue
        gold_type = pred["gold_type"]
        pred_type = pred["pred_type"]
        if gold_type == pred_type:
            per_type_tp[gold_type] += 1
        else:
            per_type_fn[gold_type] += 1
            per_type_fp[pred_type] += 1

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

    f1_values = [v["f1"] for v in per_type_f1.values() if v["support"] > 0]
    structure_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    print(f"\n  Structure-Level Accuracy:")
    print(f"  {'Type':<20} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Support':>8}")
    print(f"  {'-'*55}")
    for stype in STRUCTURAL_TYPES:
        info = per_type_f1.get(stype, {})
        print(f"  {stype:<20} {info.get('f1', 0):>8.4f} {info.get('precision', 0):>8.4f} "
              f"{info.get('recall', 0):>8.4f} {info.get('support', 0):>8}")
    print(f"  {'-'*55}")
    print(f"  {'MACRO F1':<20} {structure_f1:>8.4f}")

    # ── Composition Gap ──
    gap = piece_f1 - structure_f1
    print(f"\n  +{'='*42}+")
    print(f"  |  COMPOSITION GAP                        |")
    print(f"  |  Piece F1:     {piece_f1:>8.4f}                |")
    print(f"  |  Structure F1: {structure_f1:>8.4f}                |")
    print(f"  |  Gap:          {gap:>8.4f}  ({gap*100:.1f} points)  |")
    print(f"  +{'='*42}+")

    results = {
        "model": model_key,
        "domain": "logical_reasoning",
        "dataset": "FOLIO",
        "n_examples": len(predictions),
        "n_errors": errors,
        "piece_level": {
            "entity_f1": round(avg_entity_f1, 4),
            "relation_f1": round(avg_relation_f1, 4),
            "answer_accuracy": round(avg_answer_acc, 4),
        },
        "piece_f1": round(piece_f1, 4),
        "structure_level": {
            stype: per_type_f1[stype]["f1"] for stype in STRUCTURAL_TYPES
        },
        "structure_macro_f1": round(structure_f1, 4),
        "composition_gap": round(gap, 4),
        "composition_gap_pct": round(gap * 100, 1),
        "timestamp": datetime.now().isoformat(),
    }

    results_path = RESULTS_DIR / f"results_{model_key}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to {results_path}")

    return results


def print_comparison(all_results: dict):
    print(f"\n{'='*70}")
    print(f"  COMPOSITION GAP — FOLIO LOGICAL REASONING")
    print(f"{'='*70}")
    print(f"  {'Model':<18} {'Piece F1':>10} {'Struct F1':>10} {'Gap':>10} {'Gap (pts)':>10}")
    print(f"  {'-'*60}")

    for model_key, results in sorted(all_results.items(), key=lambda x: -x[1]["composition_gap"]):
        print(f"  {model_key:<18} {results['piece_f1']:>10.4f} {results['structure_macro_f1']:>10.4f} "
              f"{results['composition_gap']:>10.4f} {results['composition_gap_pct']:>9.1f}")

    gaps = [r["composition_gap"] for r in all_results.values()]
    avg_gap = sum(gaps) / len(gaps) if gaps else 0
    print(f"  {'-'*60}")
    print(f"  {'AVERAGE':<18} {'':>10} {'':>10} {avg_gap:>10.4f} {avg_gap*100:>9.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    if args.model:
        evaluate_model(args.model)
    else:
        all_results = {}
        for pred_file in sorted(RESULTS_DIR.glob("predictions_*.json")):
            model_key = pred_file.stem.replace("predictions_", "")
            results = evaluate_model(model_key)
            if results:
                all_results[model_key] = results
        if all_results:
            print_comparison(all_results)
        else:
            print("[INFO] No predictions found. Run run_baselines.py first.")


if __name__ == "__main__":
    main()
