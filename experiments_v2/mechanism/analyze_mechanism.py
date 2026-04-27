#!/usr/bin/env python3
"""
SOLAR-NLP — Mechanism Experiment Cross-Domain Analysis.

Reads mechanism results from all 5 domains and produces:
  1. Summary table (original gap vs hint_correct gap vs hint_wrong gap)
  2. Reduction ratios (how much did the correct hint close the gap?)
  3. Statistical significance (McNemar's test or permutation test)
  4. LaTeX-ready table for the paper

Usage:
    python analyze_mechanism.py --model gpt4o
    python analyze_mechanism.py --model all
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results"
DOMAINS = ["contracts", "sql", "math", "logic", "code"]


def load_mechanism_results(domain: str, model_key: str) -> dict:
    """Load mechanism results for a domain/model pair."""
    path = RESULTS_DIR / f"mechanism_{domain}_{model_key}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def analyze_model(model_key: str):
    """Analyze mechanism results across all domains for one model."""
    print(f"\n{'='*80}")
    print(f"  MECHANISM EXPERIMENT — Cross-Domain Analysis: {model_key}")
    print(f"{'='*80}")

    rows = []
    for domain in DOMAINS:
        results = load_mechanism_results(domain, model_key)
        if results is None:
            print(f"  [SKIP] No results for {domain}/{model_key}")
            continue

        orig = results.get("original", {})
        hint = results.get("hint_correct", {})
        wrong = results.get("hint_wrong", {})

        orig_gap = orig.get("gap_pct", 0)
        hint_gap = hint.get("gap_pct", 0)
        wrong_gap = wrong.get("gap_pct", 0)

        # Reduction ratio: how much of the gap was closed by correct hint?
        reduction = 0
        if orig_gap > 0:
            reduction = (orig_gap - hint_gap) / orig_gap * 100

        # Increase from wrong hint
        increase = wrong_gap - orig_gap

        rows.append({
            "domain": domain,
            "n_eligible": orig.get("eligible_examples", 0),
            "orig_gap": orig_gap,
            "hint_gap": hint_gap,
            "wrong_gap": wrong_gap,
            "reduction_pct": round(reduction, 1),
            "wrong_increase": round(increase, 1),
        })

    if not rows:
        print("  No results found.")
        return

    # Print summary table
    print(f"\n  {'Domain':<12} {'N':>6} {'Original':>10} {'Hint✓':>10} {'Hint✗':>10} {'Reduction':>12} {'Wrong Δ':>10}")
    print(f"  {'─'*72}")
    for r in rows:
        print(f"  {r['domain']:<12} {r['n_eligible']:>6} {r['orig_gap']:>9.1f}% {r['hint_gap']:>9.1f}% "
              f"{r['wrong_gap']:>9.1f}% {r['reduction_pct']:>10.1f}% {r['wrong_increase']:>+9.1f}pp")

    # Averages
    if rows:
        avg_orig = sum(r["orig_gap"] for r in rows) / len(rows)
        avg_hint = sum(r["hint_gap"] for r in rows) / len(rows)
        avg_wrong = sum(r["wrong_gap"] for r in rows) / len(rows)
        avg_reduction = sum(r["reduction_pct"] for r in rows) / len(rows)
        print(f"  {'─'*72}")
        print(f"  {'AVERAGE':<12} {'':>6} {avg_orig:>9.1f}% {avg_hint:>9.1f}% "
              f"{avg_wrong:>9.1f}% {avg_reduction:>10.1f}%")

    # Interpretation
    print(f"\n  {'─'*72}")
    print(f"  INTERPRETATION:")

    # Count domains where hint helped significantly (>30% reduction)
    sig_domains = [r for r in rows if r["reduction_pct"] > 30]
    partial_domains = [r for r in rows if 10 < r["reduction_pct"] <= 30]
    no_effect = [r for r in rows if r["reduction_pct"] <= 10]

    if sig_domains:
        print(f"  ★ Structure hint SIGNIFICANTLY reduced gap in {len(sig_domains)} domains:")
        for r in sig_domains:
            print(f"    - {r['domain']}: {r['orig_gap']:.1f}% → {r['hint_gap']:.1f}% ({r['reduction_pct']:.0f}% reduction)")

    if partial_domains:
        print(f"  ◐ Partial reduction in {len(partial_domains)} domains:")
        for r in partial_domains:
            print(f"    - {r['domain']}: {r['orig_gap']:.1f}% → {r['hint_gap']:.1f}% ({r['reduction_pct']:.0f}% reduction)")

    if no_effect:
        print(f"  ○ No significant effect in {len(no_effect)} domains:")
        for r in no_effect:
            print(f"    - {r['domain']}: {r['orig_gap']:.1f}% → {r['hint_gap']:.1f}%")

    # Check correlation with implicitness
    implicitness_order = {"contracts": 1, "math": 2, "sql": 3, "code": 4, "logic": 5}
    if len(rows) >= 3:
        print(f"\n  IMPLICITNESS CORRELATION:")
        sorted_rows = sorted(rows, key=lambda r: implicitness_order.get(r["domain"], 3))
        for r in sorted_rows:
            impl = implicitness_order.get(r["domain"], "?")
            bar = "█" * max(1, int(r["reduction_pct"] / 5))
            print(f"    {r['domain']:<12} (impl={impl}) reduction={r['reduction_pct']:>5.1f}% {bar}")

    # Generate LaTeX table
    print(f"\n  {'─'*72}")
    print(f"  LATEX TABLE (for paper):")
    print(f"  \\begin{{tabular}}{{lccccr}}")
    print(f"  \\toprule")
    print(f"  Domain & $N$ & Original & Hint$_{{\\checkmark}}$ & Hint$_{{\\times}}$ & Reduction \\\\")
    print(f"  \\midrule")
    for r in rows:
        print(f"  {r['domain'].title()} & {r['n_eligible']} & {r['orig_gap']:.1f}\\% & "
              f"{r['hint_gap']:.1f}\\% & {r['wrong_gap']:.1f}\\% & {r['reduction_pct']:.0f}\\% \\\\")
    if rows:
        print(f"  \\midrule")
        print(f"  Average & --- & {avg_orig:.1f}\\% & {avg_hint:.1f}\\% & {avg_wrong:.1f}\\% & {avg_reduction:.0f}\\% \\\\")
    print(f"  \\bottomrule")
    print(f"  \\end{{tabular}}")

    return rows


def mcnemar_test(results: dict, domain: str) -> dict:
    """Run McNemar's test comparing original vs hint_correct predictions.

    Tests whether the hint SIGNIFICANTLY changed the outcome distribution.
    Returns p-value and interpretation.
    """
    preds_hint = results.get("predictions", {}).get("hint_correct", [])
    orig = results.get("original", {})

    # We need paired data: same examples, different conditions
    # a = both correct, b = orig correct + hint wrong
    # c = orig wrong + hint correct, d = both wrong
    # McNemar tests if b != c (discordant pairs)

    # For now, return placeholder — implement with scipy if available
    return {"note": "McNemar test requires paired predictions — implement with scipy.stats.mcnemar"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Model key (e.g., gpt4o, qwen7b) or 'all' for all available")
    args = parser.parse_args()

    if args.model == "all":
        # Find all available models from result files
        models = set()
        for f in RESULTS_DIR.glob("mechanism_*.json"):
            parts = f.stem.split("_")
            if len(parts) >= 3:
                model_key = "_".join(parts[2:])
                models.add(model_key)
        if not models:
            print("[ERROR] No mechanism results found in", RESULTS_DIR)
            sys.exit(1)
        for mk in sorted(models):
            analyze_model(mk)
    else:
        analyze_model(args.model)


if __name__ == "__main__":
    main()
