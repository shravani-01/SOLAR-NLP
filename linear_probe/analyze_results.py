"""
SOLAR — Linear Probe: Step 3 — Analyze Results
=================================================
Reads probe_results.json from train_probes.py and generates:

  1. Summary tables (terminal + CSV)
  2. Composition Gap comparison: model output vs probe
  3. Layer-wise analysis (does structure info appear at different depths?)
  4. Per-class analysis (which constraint types are decodable?)
  5. Statistical significance tests
  6. Publication-ready figures (matplotlib)
  7. LaTeX table snippet for the paper

This script runs on CPU and requires no GPU.

Output:
  data/results/linear_probe/results/
    ├── summary_table.csv
    ├── composition_gap_comparison.csv
    ├── layer_analysis.csv
    ├── per_class_analysis.csv
    ├── latex_table.tex
    └── figures/
        ├── probe_vs_baseline.png
        ├── layer_wise_f1.png
        ├── composition_gap_comparison.png
        └── per_class_heatmap.png

Usage:
  python linear_probe/analyze_results.py \
      --results-dir data/results/linear_probe/probes/llama_3.1_8b/
"""

import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

VALID_TYPES = [
    "HARD", "SOFT", "HARD-CONDITIONAL",
    "SOFT-CONDITIONAL", "NON-CONSTRAINT",
]

# Baseline results from compare_baselines.py (Table 2)
BASELINE_RESULTS = {
    "deepseek-cot":  {"piece": 0.8481, "structure": 0.4268, "gap": 0.4213},
    "gpt4o-rag":     {"piece": 0.7952, "structure": 0.3986, "gap": 0.3966},
    "gpt4o-mini":    {"piece": 0.8924, "structure": 0.3935, "gap": 0.4989},
    "gpt4o":         {"piece": 0.9027, "structure": 0.3696, "gap": 0.5331},
    "gpt4o-cot":     {"piece": 0.8342, "structure": 0.3434, "gap": 0.4908},
    "llama3.3-70b":  {"piece": 0.9202, "structure": 0.3145, "gap": 0.6057},
    "deepseek":      {"piece": 0.9244, "structure": 0.3070, "gap": 0.6174},
}


def load_results(results_dir):
    """Load probe_results.json."""
    results_dir = Path(results_dir)
    results_file = results_dir / "probe_results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"No probe_results.json in {results_dir}")
    with open(results_file) as f:
        return json.load(f)


def print_summary_table(results, out_dir):
    """Print and save the main summary table."""
    probe_results = results["probe_results"]
    layers = sorted(probe_results.keys(), key=lambda x: int(x))

    print(f"\n{'='*80}")
    print(f"  SOLAR LINEAR PROBE — SUMMARY TABLE")
    print(f"  Model: {results['meta']['model']}")
    print(f"  Pooling: {results['meta']['pooling']}")
    print(f"{'='*80}")

    # Gather all task names
    task_names = set()
    for lr in probe_results.values():
        task_names.update(k for k, v in lr.items()
                         if isinstance(v, dict) and "macro_f1_mean" in v)
    task_names = sorted(task_names)

    # Header
    header = f"  {'Layer':<8}"
    for t in task_names:
        short = t[:12]
        header += f" {short:>13}"
    print(header)
    print(f"  {'─'*len(header)}")

    rows = []
    for li in layers:
        lr = probe_results[li]
        row = {"layer": int(li)}
        line = f"  {li:<8}"
        for t in task_names:
            if t in lr and "macro_f1_mean" in lr[t]:
                f1 = lr[t]["macro_f1_mean"]
                std = lr[t]["macro_f1_std"]
                line += f" {f1:.4f}±{std:.3f}"
                row[t] = f1
                row[f"{t}_std"] = std
            else:
                line += f"      —      "
                row[t] = None
        print(line)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary_table.csv", index=False)
    print(f"\n  Saved → summary_table.csv")
    return df


def print_composition_gap_comparison(results, out_dir):
    """Compare composition gap: baseline output vs linear probe."""
    probe_gaps = results.get("probe_composition_gap", {})
    if not probe_gaps:
        log.warning("No probe composition gap data found.")
        return

    print(f"\n{'='*80}")
    print(f"  COMPOSITION GAP COMPARISON")
    print(f"  (Baseline = model's own output; Probe = linear classifier on hidden states)")
    print(f"{'='*80}")

    # Find best probe layer
    best_layer = max(probe_gaps, key=lambda k: probe_gaps[k].get("structure_f1", 0))
    best_probe = probe_gaps[best_layer]

    print(f"\n  Best probe layer: {best_layer}")
    print(f"  Probe piece-avg F1:    {best_probe['piece_avg']:.4f}")
    print(f"  Probe structure F1:    {best_probe['structure_f1']:.4f}")
    print(f"  Probe composition gap: {best_probe['gap']:+.4f}")

    # Compare with baselines
    print(f"\n  {'Source':<20} {'Piece':>8} {'Structure':>10} {'Gap':>8}")
    print(f"  {'─'*50}")

    # Probe result
    print(f"  {'PROBE (best layer)':<20} {best_probe['piece_avg']:>8.4f} "
          f"{best_probe['structure_f1']:>10.4f} "
          f"{best_probe['gap']:>+8.4f}")

    # Baseline results
    for model, bl in sorted(BASELINE_RESULTS.items(),
                            key=lambda x: x[1]["structure"], reverse=True):
        print(f"  {model:<20} {bl['piece']:>8.4f} "
              f"{bl['structure']:>10.4f} {bl['gap']:>+8.4f}")

    # Key comparison
    avg_baseline_struct = np.mean([b["structure"] for b in BASELINE_RESULTS.values()])
    probe_struct = best_probe["structure_f1"]

    print(f"\n  {'─'*50}")
    print(f"  Avg baseline structure F1: {avg_baseline_struct:.4f}")
    print(f"  Probe structure F1:        {probe_struct:.4f}")
    improvement = probe_struct - avg_baseline_struct
    print(f"  Probe improvement:         {improvement:+.4f}")

    if improvement > 0.15:
        print(f"\n  ✓ CLAIM 3 SUPPORTED: Linear probe recovers significantly")
        print(f"    more structural information than the model outputs.")
        print(f"    The model 'knows' the structure internally but the")
        print(f"    output head fails to compose it.")
        verdict = "SUPPORTED"
    elif improvement > 0.05:
        print(f"\n  ~ PARTIAL SUPPORT: Probe recovers some additional")
        print(f"    structural info, but the improvement is modest.")
        verdict = "PARTIAL"
    else:
        print(f"\n  ✗ CLAIM 3 NOT SUPPORTED: Probe cannot recover structure")
        print(f"    from hidden states either. The failure is representational,")
        print(f"    not just at the output head.")
        verdict = "NOT_SUPPORTED"

    # Save comparison
    rows = [{"source": "PROBE (best layer)",
             "piece": best_probe["piece_avg"],
             "structure": best_probe["structure_f1"],
             "gap": best_probe["gap"],
             "layer": best_layer}]
    for model, bl in BASELINE_RESULTS.items():
        rows.append({"source": model, **bl, "layer": None})
    pd.DataFrame(rows).to_csv(out_dir / "composition_gap_comparison.csv",
                               index=False)

    return verdict


def print_selectivity(results):
    """Print selectivity analysis (Hewitt & Liang 2019)."""
    selectivity = results.get("selectivity", {})
    if not selectivity:
        log.info("No selectivity data (control task was skipped).")
        return

    print(f"\n{'='*80}")
    print(f"  SELECTIVITY (Hewitt & Liang 2019)")
    print(f"  Selectivity = real_F1 - control_F1")
    print(f"  Higher = probe is reading genuine information, not memorizing")
    print(f"{'='*80}")
    print(f"  {'Layer':<8} {'Real F1':>9} {'Control F1':>11} "
          f"{'Selectivity':>12} {'p-value':>9}")
    print(f"  {'─'*55}")

    for li in sorted(selectivity.keys(), key=lambda x: int(x)):
        s = selectivity[li]
        p_str = f"{s.get('p_value', 'N/A'):.6f}" if "p_value" in s else "  —"
        print(f"  {li:<8} {s['real_f1']:>9.4f} {s['control_f1']:>11.4f} "
              f"{s['selectivity']:>12.4f} {p_str:>9}")


def print_per_class_analysis(results, out_dir):
    """Show which constraint types the probe recovers best."""
    probe_results = results["probe_results"]

    print(f"\n{'='*80}")
    print(f"  PER-CLASS F1 — constraint_type PROBE")
    print(f"  (Which structural labels are linearly decodable?)")
    print(f"{'='*80}")

    layers = sorted(probe_results.keys(), key=lambda x: int(x))

    header = f"  {'Layer':<8}"
    for ct in VALID_TYPES:
        header += f" {ct[:8]:>9}"
    print(header)
    print(f"  {'─'*60}")

    rows = []
    for li in layers:
        lr = probe_results[li]
        ct_result = lr.get("constraint_type", {})
        per_class = ct_result.get("per_class_f1", {})

        row = {"layer": int(li)}
        line = f"  {li:<8}"
        for i, ct in enumerate(VALID_TYPES):
            v = per_class.get(str(i), 0)
            line += f" {v:>9.4f}"
            row[ct] = v
        print(line)
        rows.append(row)

    pd.DataFrame(rows).to_csv(out_dir / "per_class_analysis.csv", index=False)
    print(f"\n  Saved → per_class_analysis.csv")


def print_layer_analysis(results, out_dir):
    """Show how information distributes across layers."""
    probe_results = results["probe_results"]
    layers = sorted(probe_results.keys(), key=lambda x: int(x))

    print(f"\n{'='*80}")
    print(f"  LAYER-WISE ANALYSIS")
    print(f"  (Where does piece vs structure information concentrate?)")
    print(f"{'='*80}")

    rows = []
    for li in layers:
        lr = probe_results[li]

        piece_tasks = {k: v for k, v in lr.items()
                       if isinstance(v, dict) and v.get("is_piece", False)}
        struct = lr.get("constraint_type", {})

        piece_avg = np.mean([v["macro_f1_mean"] for v in piece_tasks.values()
                            if "macro_f1_mean" in v]) if piece_tasks else 0

        rows.append({
            "layer": int(li),
            "piece_avg_f1": round(piece_avg, 4),
            "structure_f1": struct.get("macro_f1_mean", 0),
            "gap": round(piece_avg - struct.get("macro_f1_mean", 0), 4),
            **{k: v.get("macro_f1_mean", 0) for k, v in piece_tasks.items()},
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(out_dir / "layer_analysis.csv", index=False)
    print(f"\n  Saved → layer_analysis.csv")

    # Interpretation
    if len(rows) > 1:
        best_piece_layer = max(rows, key=lambda r: r["piece_avg_f1"])
        best_struct_layer = max(rows, key=lambda r: r["structure_f1"])
        print(f"\n  Best layer for pieces:    {best_piece_layer['layer']} "
              f"(F1={best_piece_layer['piece_avg_f1']:.4f})")
        print(f"  Best layer for structure: {best_struct_layer['layer']} "
              f"(F1={best_struct_layer['structure_f1']:.4f})")


def generate_latex_table(results, out_dir):
    """Generate a LaTeX table for the paper."""
    probe_results = results["probe_results"]
    layers = sorted(probe_results.keys(), key=lambda x: int(x))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Linear probe accuracy (Macro F1) on frozen hidden states. "
        r"\textit{Piece-level} tasks measure extraction of individual constraint "
        r"components; \textit{Structure} measures the 5-class constraint-type "
        r"label. The gap between piece and structure probes indicates the model "
        r"encodes the ingredients but fails to compose them.}",
        r"\label{tab:linear_probe}",
        r"\begin{tabular}{l|cccc|c|c}",
        r"\toprule",
        r"Layer & Is-Con & Thresh & Exc & Ent & \textbf{Piece avg} & \textbf{Structure} \\",
        r"\midrule",
    ]

    for li in layers:
        lr = probe_results[li]

        vals = {}
        for task in ["is_constraint", "has_threshold", "has_exception", "entity_count"]:
            r = lr.get(task, {})
            vals[task] = r.get("macro_f1_mean", 0)

        struct = lr.get("constraint_type", {}).get("macro_f1_mean", 0)
        piece_avg = np.mean(list(vals.values()))

        line = f"  {li}"
        for task in ["is_constraint", "has_threshold", "has_exception", "entity_count"]:
            line += f" & {vals[task]:.3f}"
        line += f" & \\textbf{{{piece_avg:.3f}}} & \\textbf{{{struct:.3f}}} \\\\"
        lines.append(line)

    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{l|}{\textit{Model output (best baseline)}}",
        r"& 0.892 & 0.427 \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex = "\n".join(lines)
    with open(out_dir / "latex_table.tex", "w") as f:
        f.write(latex)
    print(f"\n  LaTeX table saved → latex_table.tex")


def generate_figures(results, out_dir):
    """Generate publication-ready matplotlib figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed. Skipping figures.")
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    probe_results = results["probe_results"]
    layers = sorted(probe_results.keys(), key=lambda x: int(x))

    # ── Figure 1: Probe vs Baseline Composition Gap ──────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline data
    bl_names = list(BASELINE_RESULTS.keys())
    bl_struct = [BASELINE_RESULTS[n]["structure"] for n in bl_names]
    bl_piece = [BASELINE_RESULTS[n]["piece"] for n in bl_names]

    # Probe data (best layer)
    best_layer = max(layers,
                     key=lambda li: probe_results[li].get(
                         "constraint_type", {}).get("macro_f1_mean", 0))
    probe_gap = results.get("probe_composition_gap", {}).get(best_layer, {})

    x = np.arange(len(bl_names) + 1)
    width = 0.35

    bars1 = ax.bar(x[:len(bl_names)] - width/2, bl_piece, width,
                   label="Piece-level", color="#4ECDC4", edgecolor="white")
    bars2 = ax.bar(x[:len(bl_names)] + width/2, bl_struct, width,
                   label="Structure-level", color="#FF6B6B", edgecolor="white")

    # Probe bar
    if probe_gap:
        ax.bar(x[-1] - width/2, probe_gap.get("piece_avg", 0), width,
               color="#4ECDC4", edgecolor="white", hatch="//")
        ax.bar(x[-1] + width/2, probe_gap.get("structure_f1", 0), width,
               color="#FF6B6B", edgecolor="white", hatch="//")

    ax.set_ylabel("Macro F1", fontsize=12)
    ax.set_title("Composition Gap: Model Output vs Linear Probe", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(bl_names + [f"Probe\n(layer {best_layer})"],
                       rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Add gap annotations
    for i, name in enumerate(bl_names):
        gap = BASELINE_RESULTS[name]["gap"]
        mid_y = (bl_piece[i] + bl_struct[i]) / 2
        ax.annotate(f"Δ={gap:.2f}", xy=(i, mid_y),
                    fontsize=8, ha="center", color="#333")

    plt.tight_layout()
    plt.savefig(fig_dir / "probe_vs_baseline.png", dpi=200)
    plt.close()
    log.info(f"  Saved figure: probe_vs_baseline.png")

    # ── Figure 2: Layer-wise F1 ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    layer_ints = [int(li) for li in layers]
    piece_by_layer = []
    struct_by_layer = []

    for li in layers:
        lr = probe_results[li]
        piece_tasks = {k: v for k, v in lr.items()
                       if isinstance(v, dict) and v.get("is_piece", False)}
        piece_avg = np.mean([v["macro_f1_mean"] for v in piece_tasks.values()
                            if "macro_f1_mean" in v]) if piece_tasks else 0
        struct_f1 = lr.get("constraint_type", {}).get("macro_f1_mean", 0)
        piece_by_layer.append(piece_avg)
        struct_by_layer.append(struct_f1)

    ax.plot(range(len(layers)), piece_by_layer, "o-",
            color="#4ECDC4", linewidth=2, markersize=8, label="Piece-level avg")
    ax.plot(range(len(layers)), struct_by_layer, "s-",
            color="#FF6B6B", linewidth=2, markersize=8, label="Structure-level")

    # Fill the gap
    ax.fill_between(range(len(layers)), piece_by_layer, struct_by_layer,
                    alpha=0.15, color="#FF6B6B", label="Composition gap")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Probe Macro F1", fontsize=12)
    ax.set_title("Layer-wise Probe Accuracy: Pieces vs Structure", fontsize=14)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(fig_dir / "layer_wise_f1.png", dpi=200)
    plt.close()
    log.info(f"  Saved figure: layer_wise_f1.png")

    # ── Figure 3: Per-class heatmap ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))

    class_data = []
    for li in layers:
        lr = probe_results[li]
        ct_result = lr.get("constraint_type", {})
        per_class = ct_result.get("per_class_f1", {})
        row = [per_class.get(str(i), 0) for i in range(5)]
        class_data.append(row)

    if class_data:
        data = np.array(class_data)
        im = ax.imshow(data.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

        ax.set_yticks(range(5))
        ax.set_yticklabels(VALID_TYPES, fontsize=10)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers)
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_title("Per-Class Probe F1 by Layer", fontsize=14)

        # Add value annotations
        for i in range(len(layers)):
            for j in range(5):
                color = "white" if data[i, j] < 0.5 else "black"
                ax.text(i, j, f"{data[i, j]:.2f}",
                        ha="center", va="center", fontsize=9, color=color)

        plt.colorbar(im, ax=ax, label="Macro F1")

    plt.tight_layout()
    plt.savefig(fig_dir / "per_class_heatmap.png", dpi=200)
    plt.close()
    log.info(f"  Saved figure: per_class_heatmap.png")

    log.info(f"  All figures saved to {fig_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SOLAR Linear Probe: Analyze results"
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Directory with probe_results.json"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure generation"
    )
    args = parser.parse_args()

    results = load_results(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else \
        Path(args.results_dir).parent / "results" / Path(args.results_dir).name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    print_summary_table(results, out_dir)
    verdict = print_composition_gap_comparison(results, out_dir)
    print_selectivity(results)
    print_per_class_analysis(results, out_dir)
    print_layer_analysis(results, out_dir)
    generate_latex_table(results, out_dir)

    if not args.no_figures:
        generate_figures(results, out_dir)

    # Final summary
    print(f"\n{'='*80}")
    print(f"  FINAL VERDICT FOR CLAIM 3: {verdict}")
    print(f"  Interpretation: {results.get('interpretation', 'N/A')}")
    print(f"{'='*80}")
    print(f"\n  All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
