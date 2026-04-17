"""
SOLAR — Baseline Comparison Script
====================================
Recomputes all metrics from saved prediction files
using the corrected master CSV as ground truth.

Outputs:
  1. Clean comparison table in terminal
  2. Per-class F1 breakdown
  3. Piece-level vs structure-level decomposition
     (for the composition gap analysis)

Usage:
  python scripts/compare_baselines.py
  python scripts/compare_baselines.py --domain transit
  python scripts/compare_baselines.py --split test
"""

import os, re, json, argparse
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
from collections import Counter

PROJECT_ROOT  = Path(__file__).parent.parent
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"
RESULTS_DIR   = PROJECT_ROOT / "data" / "results" / "baselines"

VALID_TYPES = ["HARD","SOFT","HARD-CONDITIONAL","SOFT-CONDITIONAL","NON-CONSTRAINT"]
LABEL_MAP = {
    "NOT_CONSTRAINT":   "NON-CONSTRAINT",
    "NON_CONSTRAINT":   "NON-CONSTRAINT",
    "NONCONSTRAINT":    "NON-CONSTRAINT",
    "HARD_CONDITIONAL": "HARD-CONDITIONAL",
    "SOFT_CONDITIONAL": "SOFT-CONDITIONAL",
    "UNCLEAR":          "NON-CONSTRAINT",
}
EMPTY = {"", "nan", "none", "null", "n/a"}

def norm(t):
    t = str(t).upper().strip()
    return LABEL_MAP.get(t, "NON-CONSTRAINT" if t not in VALID_TYPES else t)

def is_empty(v):
    return str(v).strip().lower() in EMPTY or (
        v != v  # NaN check
    )

def parse_val(raw):
    if is_empty(raw): return None
    for pat in [r'\((\d+\.?\d*)\)', r'=\s*(\d+\.?\d*)', r'\b(\d+\.?\d*)\b']:
        m = re.search(pat, str(raw).lower())
        if m: return float(m.group(1))
    return None

def get_variable_name(exc):
    if isinstance(exc, str): return ""
    if isinstance(exc, dict):
        return (exc.get("variable_name") or
                exc.get("semantic_variable_name") or "")
    return ""

def is_semantic_name(vn):
    """True if variable name looks semantically grounded."""
    if not vn or len(vn) < 4: return False
    if "exception_contract" in vn: return False
    if vn == "exception_unknown": return False
    if re.match(r'^[a-z]+_\d{4,}$', vn): return False  # e.g. exception_0187
    return True

# ── Load ground truth from master CSV ───────────────────────────────────────
def load_ground_truth(domain=None, split=None):
    master = ANNOTATED_DIR / "all_domains_master.csv"
    df = pd.read_csv(master, dtype=str)

    # Remove invalid annotations
    if "annotation_valid" in df.columns:
        df = df[df["annotation_valid"] != "False"]

    df = df[df["is_constraint"].notna()]

    if domain:
        df = df[df["domain"] == domain]
    if split:
        df = df[df["split"] == split]

    # Build sentence_id → ground truth dict
    gt = {}
    for _, row in df.iterrows():
        sid = str(row.get("sentence_id", ""))
        gt[sid] = {
            "constraint_type": norm(row.get("constraint_type", "")),
            "is_constraint":   str(row.get("is_constraint", "No")),
            "threshold":       str(row.get("threshold", "")),
            "exception":       str(row.get("exception", "")),
            "domain":          str(row.get("domain", "")),
        }
    return gt

# ── Compute metrics ──────────────────────────────────────────────────────────
def compute(pred_file, gt_map, domain=None):
    data = json.load(open(pred_file))

    y_true, y_pred = [], []
    domains_true, domains_pred = {}, {}

    thresh_correct = thresh_total = 0
    exc_correct = exc_total = 0
    sem_vars = sem_total = 0
    json_valid = null_preds = 0

    for d in data:
        sid  = str(d.get("sentence_id", ""))
        pred = d.get("prediction") or {}
        gt   = gt_map.get(sid)
        if gt is None:
            continue  # sentence not in filtered set

        gt_type = gt["constraint_type"]
        if gt["is_constraint"] != "Yes":
            gt_type = "NON-CONSTRAINT"

        pt = norm(pred.get("constraint_type", "NON-CONSTRAINT"))

        y_true.append(gt_type)
        y_pred.append(pt)

        dom = gt["domain"]
        domains_true.setdefault(dom, []).append(gt_type)
        domains_pred.setdefault(dom, []).append(pt)

        if pred:
            json_valid += 1
        else:
            null_preds += 1

        # ── Threshold accuracy ───────────────────────────────────────────
        gt_val = parse_val(gt["threshold"])
        if gt_val is not None:
            thresh_total += 1
            vals = []
            for t in (pred.get("thresholds") or []):
                try:   vals.append(float(t.get("value")))
                except: pass
            if vals:
                closest = min(vals, key=lambda v: abs(v - gt_val))
                if abs(closest - gt_val) <= gt_val * 0.1 + 1:
                    thresh_correct += 1

        # ── Exception recall + semantic variable rate ─────────────────────
        if not is_empty(gt["exception"]):
            exc_total += 1
            pred_excs = pred.get("exceptions") or []
            if pred_excs:
                exc_correct += 1
                vn = get_variable_name(pred_excs[0])
                sem_total += 1
                if is_semantic_name(vn):
                    sem_vars += 1

    if not y_true:
        return None

    macro_f1 = f1_score(y_true, y_pred, average="macro",
                         zero_division=0, labels=VALID_TYPES)
    report   = classification_report(y_true, y_pred, labels=VALID_TYPES,
                                      output_dict=True, zero_division=0)
    per_class = {c: round(report[c]["f1-score"], 3)
                 for c in VALID_TYPES if c in report}

    # Per-domain macro F1
    per_domain = {}
    for dom, yt in domains_true.items():
        yp = domains_pred[dom]
        if len(yt) < 10: continue
        per_domain[dom] = round(f1_score(yt, yp, average="macro",
                                          zero_division=0, labels=VALID_TYPES), 4)

    # ── Piece-level score (composition gap analysis) ──────────────────────
    # Weighted average of: threshold_acc, exception_recall, sem_var_rate
    # (type F1 is the structure-level score)
    thresh_acc  = thresh_correct / thresh_total if thresh_total else None
    exc_recall  = exc_correct / exc_total       if exc_total    else None
    sem_rate    = sem_vars / sem_total          if sem_total    else None

    # Simple piece-level proxy: mean of available piece scores
    piece_scores = [x for x in [thresh_acc, exc_recall, sem_rate] if x is not None]
    piece_level  = round(sum(piece_scores) / len(piece_scores), 4) if piece_scores else None
    comp_gap     = round(piece_level - macro_f1, 4) if piece_level else None

    return {
        "n":             len(y_true),
        "macro_f1":      round(macro_f1, 4),
        "per_class_f1":  per_class,
        "thresh_acc":    round(thresh_acc, 4)  if thresh_acc  is not None else None,
        "exc_recall":    round(exc_recall, 4)  if exc_recall  is not None else None,
        "sem_var_rate":  round(sem_rate, 4)    if sem_rate    is not None else None,
        "piece_level":   piece_level,
        "comp_gap":      comp_gap,
        "json_valid":    round(json_valid / len(data), 4) if data else 0,
        "null_preds":    null_preds,
        "per_domain":    per_domain,
        "gt_distribution": dict(Counter(y_true).most_common()),
        "pred_distribution": dict(Counter(y_pred).most_common()),
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default=None)
    parser.add_argument("--split",  default=None,
                        choices=["train","val","test",None])
    parser.add_argument("--save",   default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    print("\n" + "="*72)
    print("  SOLAR — Baseline Comparison (Corrected Annotations)")
    print("="*72)

    gt_map = load_ground_truth(domain=args.domain, split=args.split)
    print(f"\n  Ground truth loaded: {len(gt_map):,} sentences")
    if args.domain: print(f"  Domain filter:  {args.domain}")
    if args.split:  print(f"  Split filter:   {args.split}")

    # Find all prediction files
    pred_files = sorted(RESULTS_DIR.glob("*_predictions.json"))
    if not pred_files:
        print(f"\n  No prediction files found in {RESULTS_DIR}")
        return

    results = {}
    for pf in pred_files:
        model = pf.stem.replace("_predictions", "")
        m = compute(pf, gt_map, domain=args.domain)
        if m is None or m["n"] < 20:
            print(f"  Skipping {model} (n={m['n'] if m else 0})")
            continue
        results[model] = m

    if not results:
        print("  No results to display.")
        return

    # ── Table 1: Main comparison ─────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  TABLE 1 — Main Comparison")
    print(f"{'─'*72}")
    print(f"  {'Model':<18} {'Macro F1':>9} {'Thresh':>8} "
          f"{'Exc Rec':>8} {'Sem Var':>8} {'JSON':>6} {'N':>7}")
    print(f"  {'─'*65}")

    sorted_models = sorted(results.items(),
                           key=lambda x: x[1]["macro_f1"], reverse=True)
    for model, m in sorted_models:
        thresh  = f"{m['thresh_acc']:.4f}"  if m['thresh_acc']  is not None else "  —   "
        exc     = f"{m['exc_recall']:.4f}"  if m['exc_recall']  is not None else "  —   "
        sem     = f"{m['sem_var_rate']:.4f}" if m['sem_var_rate'] is not None else "  —   "
        print(f"  {model:<18} {m['macro_f1']:>9.4f} {thresh:>8} "
              f"{exc:>8} {sem:>8} {m['json_valid']:>6.4f} {m['n']:>7,}")

    # ── Table 2: Composition Gap ─────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  TABLE 2 — Composition Gap Analysis")
    print(f"  (Piece-level = avg of thresh_acc, exc_recall, sem_var_rate)")
    print(f"  (Structure-level = Macro F1 on constraint type)")
    print(f"  (Gap = Piece − Structure  →  higher gap = worse composition)")
    print(f"{'─'*72}")
    print(f"  {'Model':<18} {'Piece-lvl':>10} {'Structure':>10} {'GAP':>8}")
    print(f"  {'─'*50}")

    for model, m in sorted_models:
        if m["piece_level"] is None:
            continue
        gap_str = f"+{m['comp_gap']:.4f}" if m['comp_gap'] > 0 else f"{m['comp_gap']:.4f}"
        print(f"  {model:<18} {m['piece_level']:>10.4f} "
              f"{m['macro_f1']:>10.4f} {gap_str:>8}")

    # ── Table 3: Per-class F1 ────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  TABLE 3 — Per-Class F1 Breakdown")
    print(f"{'─'*72}")
    header = f"  {'Model':<18}"
    for c in VALID_TYPES:
        header += f" {c[:7]:>8}"
    print(header)
    print(f"  {'─'*65}")
    for model, m in sorted_models:
        row = f"  {model:<18}"
        for c in VALID_TYPES:
            v = m["per_class_f1"].get(c, 0)
            row += f" {v:>8.3f}"
        print(row)

    # ── Table 4: Per-domain breakdown ────────────────────────────────────
    if not args.domain:
        all_domains = set()
        for m in results.values():
            all_domains.update(m["per_domain"].keys())
        all_domains = sorted(all_domains)

        if all_domains:
            print(f"\n{'─'*72}")
            print(f"  TABLE 4 — Per-Domain Macro F1")
            print(f"{'─'*72}")
            hdr = f"  {'Model':<18}"
            for d in all_domains:
                hdr += f" {d[:6]:>7}"
            print(hdr)
            print(f"  {'─'*65}")
            for model, m in sorted_models:
                row = f"  {model:<18}"
                for d in all_domains:
                    v = m["per_domain"].get(d)
                    row += f" {v:>7.4f}" if v is not None else f" {'—':>7}"
                print(row)

    # ── Prediction distribution sanity check ────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  SANITY CHECK — Prediction Type Distribution")
    print(f"  (Are models predicting diverse types or collapsing to HARD?)")
    print(f"{'─'*72}")
    for model, m in sorted_models:
        dist = m["pred_distribution"]
        total = sum(dist.values())
        top = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:4]
        dist_str = "  ".join(f"{t}:{c/total:.0%}" for t,c in top)
        print(f"  {model:<18} {dist_str}")

    # ── Key findings ─────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print(f"  KEY FINDINGS")
    print(f"{'─'*72}")

    if results:
        best_model  = max(results, key=lambda x: results[x]["macro_f1"])
        best_f1     = results[best_model]["macro_f1"]
        avg_piece   = sum(m["piece_level"] for m in results.values()
                          if m["piece_level"]) / max(1, sum(
                          1 for m in results.values() if m["piece_level"]))
        avg_struct  = sum(m["macro_f1"] for m in results.values()) / len(results)
        avg_gap     = avg_piece - avg_struct

        print(f"  Best Macro F1:       {best_f1:.4f}  ({best_model})")
        print(f"  Avg piece-level:     {avg_piece:.4f}")
        print(f"  Avg structure-level: {avg_struct:.4f}")
        print(f"  Avg composition gap: {avg_gap:.4f}")
        print()

        # CoT finding
        if "gpt4o" in results and "gpt4o-cot" in results:
            base = results["gpt4o"]["macro_f1"]
            cot  = results["gpt4o-cot"]["macro_f1"]
            print(f"  CoT effect (GPT-4o): base={base:.4f} → cot={cot:.4f} "
                  f"({'WORSE' if cot < base else 'BETTER'} by {abs(cot-base):.4f})")

        # Scale finding
        if "gpt4o" in results and "gpt4o-mini" in results:
            big   = results["gpt4o"]["macro_f1"]
            small = results["gpt4o-mini"]["macro_f1"]
            print(f"  Scale effect:        gpt4o={big:.4f}  gpt4o-mini={small:.4f} "
                  f"({'mini WINS' if small > big else 'big WINS'} by {abs(small-big):.4f})")

        # SOFT/NON-C collapse check
        print()
        print(f"  SOFT + NON-CONSTRAINT F1 (per model):")
        for model, m in sorted_models:
            soft  = m["per_class_f1"].get("SOFT", 0)
            nonc  = m["per_class_f1"].get("NON-CONSTRAINT", 0)
            hardcond = m["per_class_f1"].get("HARD-CONDITIONAL", 0)
            print(f"    {model:<18} SOFT={soft:.3f}  NON-C={nonc:.3f}  "
                  f"HARD-C={hardcond:.3f}")

    # ── Save ─────────────────────────────────────────────────────────────
    save_path = args.save or str(RESULTS_DIR / "comparison_corrected.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {save_path}")
    print("="*72 + "\n")


if __name__ == "__main__":
    main()