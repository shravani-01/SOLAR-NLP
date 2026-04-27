#!/usr/bin/env python3
"""
SOLAR-NLP — Mechanism Experiment Core.

Tests whether the composition gap is caused by STRUCTURE INFERENCE failure.

3 conditions on the COMPOSED question only:
  1. original   — baseline (reuse existing predictions)
  2. hint_correct — prepend correct structural type as hint
  3. hint_wrong   — prepend wrong structural type as hint

Hypothesis:
  - hint_correct closes gap → gap is from structure inference failure (causal evidence)
  - hint_wrong increases gap → model is sensitive to structural framing
  - hint_correct has no effect → gap is in reasoning itself, not structure inference

We only run on gap-ELIGIBLE examples (pieces_all_correct=True) for efficiency.
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime

random.seed(42)


def load_baseline_predictions(domain: str, model_key: str) -> list:
    """Load existing baseline predictions from experiments_v2."""
    results_dir = Path(__file__).parent.parent / domain / "results"
    pred_path = results_dir / f"predictions_{model_key}.json"
    if not pred_path.exists():
        print(f"[WARN] No baseline predictions: {pred_path}")
        return []
    with open(pred_path) as f:
        return json.load(f)


def get_gap_eligible(predictions: list) -> list:
    """Filter to examples where pieces_all_correct=True (gap-eligible subset)."""
    return [p for p in predictions if p.get("pieces_all_correct", False) and not p.get("error")]


def pick_wrong_type(gold_type: str, all_types: list) -> str:
    """Pick a random wrong structural type for the wrong-hint condition."""
    candidates = [t for t in all_types if t != gold_type]
    if not candidates:
        return all_types[0]
    return random.choice(candidates)


def compute_mechanism_results(predictions: list, condition: str) -> dict:
    """Compute gap metrics for a mechanism condition.

    NOTE: In mechanism experiment, ALL examples are gap-eligible by construction
    (pieces_all_correct=True in baseline). So:
      gap = P(composed_wrong) over the gap-eligible subset
    """
    total = 0
    composed_correct = 0
    per_type = defaultdict(lambda: {"total": 0, "correct": 0, "wrong": 0})

    for pred in predictions:
        if pred.get("error"):
            continue
        total += 1
        gold_type = pred.get("gold_type", "")
        per_type[gold_type]["total"] += 1

        if pred["composed_correct"]:
            composed_correct += 1
            per_type[gold_type]["correct"] += 1
        else:
            per_type[gold_type]["wrong"] += 1

    gap_cases = total - composed_correct
    gap_rate = gap_cases / total if total > 0 else 0.0

    per_type_results = {}
    for stype, stats in per_type.items():
        per_type_results[stype] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "wrong": stats["wrong"],
            "gap_rate": round(stats["wrong"] / stats["total"], 4) if stats["total"] > 0 else 0.0,
        }

    return {
        "condition": condition,
        "eligible_examples": total,
        "composed_correct": composed_correct,
        "gap_cases": gap_cases,
        "gap_rate": round(gap_rate, 4),
        "gap_pct": round(gap_rate * 100, 1),
        "per_type": per_type_results,
    }


def save_mechanism_results(all_results: dict, model_key: str, domain: str, results_dir: Path):
    """Save mechanism experiment results."""
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"mechanism_{domain}_{model_key}.json"
    summary = {
        "model": model_key,
        "domain": domain,
        "experiment": "mechanism_structure_hint",
        "timestamp": datetime.now().isoformat(),
        **all_results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


def print_mechanism_comparison(results: dict, model_key: str, domain: str):
    """Pretty-print comparison across 3 conditions."""
    print(f"\n{'='*70}")
    print(f"  MECHANISM EXPERIMENT — {domain.upper()} | {model_key}")
    print(f"{'='*70}")

    for cond in ["original", "hint_correct", "hint_wrong"]:
        r = results.get(cond, {})
        label = {
            "original": "Original (baseline)",
            "hint_correct": "Structure HINT (correct)",
            "hint_wrong": "Structure HINT (wrong)",
        }[cond]
        print(f"\n  {label}:")
        print(f"    Gap-eligible examples: {r.get('eligible_examples', 0)}")
        print(f"    Composed correct:      {r.get('composed_correct', 0)}")
        print(f"    Gap cases:             {r.get('gap_cases', 0)}")
        print(f"    GAP RATE:              {r.get('gap_pct', 0):.1f}%")

    # Delta analysis
    orig = results.get("original", {}).get("gap_pct", 0)
    hint = results.get("hint_correct", {}).get("gap_pct", 0)
    wrong = results.get("hint_wrong", {}).get("gap_pct", 0)

    print(f"\n  {'─'*50}")
    print(f"  DELTAS:")
    print(f"    hint_correct vs original:  {hint - orig:+.1f}pp  {'← GAP CLOSED!' if hint < orig * 0.5 else '← partial' if hint < orig else '← no effect'}")
    print(f"    hint_wrong vs original:    {wrong - orig:+.1f}pp  {'← gap increased' if wrong > orig else '← no increase'}")
    print(f"    hint_correct vs hint_wrong: {hint - wrong:+.1f}pp")

    if hint < orig * 0.5:
        print(f"\n  ★ CONCLUSION: Structure inference IS the bottleneck.")
        print(f"    Giving the model the correct structure closed ≥50% of the gap.")
    elif hint < orig * 0.8:
        print(f"\n  ◐ CONCLUSION: Structure inference is a PARTIAL bottleneck.")
        print(f"    Correct hint reduced gap by {orig - hint:.1f}pp but didn't eliminate it.")
    else:
        print(f"\n  ○ CONCLUSION: Structure inference is NOT the main bottleneck.")
        print(f"    Hint had minimal effect — gap may be in reasoning execution.")
