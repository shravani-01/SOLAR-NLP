#!/usr/bin/env python3
"""
SOLAR-NLP — Mitigation Experiment Core (Section 8).

Structure-Aware Prompting: Can we close the composition gap by
prompting the model to FIRST identify the implicit structure,
THEN solve the composed task using that structure?

Key difference from mechanism experiment (Section 7):
  - Section 7: We GIVE the model the correct structure (oracle test)
  - Section 8: We PROMPT the model to INFER the structure itself (practical mitigation)

2-Stage Pipeline:
  Stage 1: "What structural pattern does this problem require?"
  Stage 2: "Given that it requires [model's own answer], now solve it."

This tests whether explicit structure-awareness prompting can close the gap
WITHOUT oracle access to the gold type.

3 conditions:
  1. original          — baseline composed question (reuse existing)
  2. self_structure     — model identifies structure, then solves (pipeline)
  3. cot_structure      — CoT prompt specifically about structure (single call)
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime


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
    """Filter to examples where pieces_all_correct=True."""
    return [p for p in predictions if p.get("pieces_all_correct", False) and not p.get("error")]


def compute_mitigation_results(predictions: list, condition: str) -> dict:
    """Compute gap metrics for a mitigation condition."""
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


def save_mitigation_results(all_results: dict, model_key: str, domain: str, results_dir: Path):
    """Save mitigation experiment results."""
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"mitigation_{domain}_{model_key}.json"
    summary = {
        "model": model_key,
        "domain": domain,
        "experiment": "mitigation_structure_aware_prompting",
        "timestamp": datetime.now().isoformat(),
        **all_results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


def print_mitigation_comparison(results: dict, model_key: str, domain: str):
    """Pretty-print comparison across conditions."""
    print(f"\n{'='*70}")
    print(f"  MITIGATION EXPERIMENT — {domain.upper()} | {model_key}")
    print(f"{'='*70}")

    for cond in ["original", "self_structure", "cot_structure"]:
        r = results.get(cond, {})
        label = {
            "original":       "Original (baseline)",
            "self_structure":  "Self-Structure (2-stage pipeline)",
            "cot_structure":   "Structure-Aware CoT (single prompt)",
        }[cond]
        print(f"\n  {label}:")
        print(f"    Gap-eligible examples: {r.get('eligible_examples', 0)}")
        print(f"    Composed correct:      {r.get('composed_correct', 0)}")
        print(f"    Gap cases:             {r.get('gap_cases', 0)}")
        print(f"    GAP RATE:              {r.get('gap_pct', 0):.1f}%")

    orig = results.get("original", {}).get("gap_pct", 0)
    self_s = results.get("self_structure", {}).get("gap_pct", 0)
    cot_s = results.get("cot_structure", {}).get("gap_pct", 0)

    print(f"\n  {'─'*50}")
    print(f"  DELTAS:")
    print(f"    self_structure vs original:  {self_s - orig:+.1f}pp  {'← improved!' if self_s < orig else '← no improvement'}")
    print(f"    cot_structure vs original:   {cot_s - orig:+.1f}pp  {'← improved!' if cot_s < orig else '← no improvement'}")
    print(f"    self_structure vs cot:       {self_s - cot_s:+.1f}pp")

    # Compare with mechanism experiment if available
    mech_dir = Path(__file__).parent.parent / "mechanism" / "results"
    mech_path = mech_dir / f"mechanism_{domain}_{model_key}.json"
    if mech_path.exists():
        with open(mech_path) as f:
            mech = json.load(f)
        mech_hint = mech.get("hint_correct", {}).get("gap_pct", 0)
        print(f"\n  COMPARISON WITH MECHANISM (oracle hint):")
        print(f"    Oracle hint gap:       {mech_hint:.1f}%  (best possible with correct structure)")
        print(f"    Self-structure gap:    {self_s:.1f}%  (model infers own structure)")
        if mech_hint > 0:
            ceiling = (orig - mech_hint)  # max possible improvement
            actual_self = (orig - self_s)
            pct_of_ceiling = actual_self / ceiling * 100 if ceiling > 0 else 0
            print(f"    Self-structure captures {pct_of_ceiling:.0f}% of oracle improvement")
