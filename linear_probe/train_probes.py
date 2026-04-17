"""
SOLAR — Linear Probe: Step 2 — Train Probes
=============================================
Trains linear classifiers (logistic regression) on frozen hidden states
to determine whether piece-level and structure-level information is
linearly decodable from the model's representations.

Methodology follows:
  - Conneau et al. (2018) — probing sentence embeddings
  - Tenney et al. (2019) — BERT rediscovers the classical NLP pipeline
  - Belinkov (2022) — probing classifiers survey
  - Hewitt & Liang (2019) — control tasks for selectivity

Probes trained:
  STRUCTURE-LEVEL (the hard task):
    1. constraint_type — 5-class classification (HARD/SOFT/HARD-COND/SOFT-COND/NON-C)

  PIECE-LEVEL (the easy tasks):
    2. is_constraint — binary (Yes/No)
    3. has_threshold — binary
    4. has_exception — binary
    5. entity_count — 3-class (0/1/2+)

  CONTROL TASKS (Hewitt & Liang 2019):
    6. control_type — same as constraint_type but with SHUFFLED labels
       If probe accuracy on control ≈ real accuracy, the probe is
       memorizing, not reading genuine information.
       Selectivity = real_accuracy - control_accuracy. Higher = better.

Training protocol:
  - Stratified 5-fold cross-validation (robust for imbalanced labels)
  - L2-regularized logistic regression (sklearn)
  - Hyperparameter sweep: C ∈ {0.001, 0.01, 0.1, 1.0, 10.0}
  - Evaluation: Macro F1 (consistent with baseline comparison script)
  - Per-class F1 for constraint_type (to see which classes the probe
    recovers vs. which the model's own output misses)
  - Statistical significance: paired t-test on fold-level F1 scores
    between piece-probe and structure-probe

Output:
  data/results/linear_probe/probes/
    ├── probe_results.json          # All metrics
    ├── fold_results.json           # Per-fold detail
    ├── best_probes/                # Saved sklearn models
    │   ├── constraint_type_layer_-1.pkl
    │   └── ...
    └── control_labels.json         # Shuffled labels (for reproducibility)

Usage:
  python linear_probe/train_probes.py \
      --hidden-states-dir data/results/linear_probe/hidden_states/llama_3.1_8b/

  # Only specific layers
  python linear_probe/train_probes.py \
      --hidden-states-dir data/results/linear_probe/hidden_states/llama_3.1_8b/ \
      --layers -1,-4,-8

  # Use last_token pooling instead of mean_pool
  python linear_probe/train_probes.py \
      --hidden-states-dir data/results/linear_probe/hidden_states/llama_3.1_8b/ \
      --pooling last_token
"""

import os
import json
import argparse
import logging
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report
)
from sklearn.preprocessing import StandardScaler
from scipy import stats

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
VALID_TYPES = [
    "HARD", "SOFT", "HARD-CONDITIONAL",
    "SOFT-CONDITIONAL", "NON-CONSTRAINT",
]

# Probe tasks: name → (label_key, n_classes, is_piece_level)
PROBE_TASKS = {
    # Structure-level (the composition bottleneck)
    "constraint_type": {
        "label_key": "constraint_type",
        "n_classes": 5,
        "is_piece": False,
        "description": "5-class structural label (HARD/SOFT/HARD-COND/SOFT-COND/NON-C)",
    },
    # Piece-level
    "is_constraint": {
        "label_key": "is_constraint",
        "n_classes": 2,
        "is_piece": True,
        "description": "Binary: is this sentence a constraint?",
    },
    "has_threshold": {
        "label_key": "has_threshold",
        "n_classes": 2,
        "is_piece": True,
        "description": "Binary: does the sentence have a numeric threshold?",
    },
    "has_exception": {
        "label_key": "has_exception",
        "n_classes": 2,
        "is_piece": True,
        "description": "Binary: does the sentence have an exception?",
    },
    "entity_count": {
        "label_key": "entity_count",
        "n_classes": 4,  # 0, 1, 2, 3+
        "is_piece": True,
        "description": "Entity count bin (0/1/2/3+)",
    },
    # Control task (Hewitt & Liang 2019)
    "control_type": {
        "label_key": "control_type",  # will be generated
        "n_classes": 5,
        "is_piece": False,
        "description": "CONTROL: shuffled constraint_type labels (same distribution)",
        "is_control": True,
    },
}

# Regularization strengths to sweep
C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]

# Cross-validation folds
N_FOLDS = 5

# Maximum iterations for logistic regression
MAX_ITER = 2000


# ── Data loading ─────────────────────────────────────────────────────────────
def load_hidden_states_and_labels(hs_dir, pooling="mean_pool"):
    """Load saved hidden states and labels from extraction step."""
    hs_dir = Path(hs_dir)

    # Load metadata
    meta_path = hs_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No meta.json found in {hs_dir}")
    with open(meta_path) as f:
        meta = json.load(f)

    # Load labels
    labels = torch.load(hs_dir / "labels.pt", weights_only=True)
    log.info(f"Labels loaded: {list(labels.keys())}")
    for k, v in labels.items():
        log.info(f"  {k}: shape={v.shape}, unique={v.unique().tolist()}")

    # Find available layers
    pool_dir = hs_dir / pooling
    if not pool_dir.exists():
        raise FileNotFoundError(f"Pooling dir not found: {pool_dir}")

    available_layers = []
    for f in sorted(pool_dir.glob("layer_*.pt")):
        if "partial" not in f.name:
            layer_idx = int(f.stem.split("_")[1])
            available_layers.append(layer_idx)

    log.info(f"Available layers ({pooling}): {available_layers}")

    # Load hidden states
    hidden_states = {}
    for li in available_layers:
        tensor = torch.load(pool_dir / f"layer_{li}.pt", weights_only=True)
        hidden_states[li] = tensor.numpy().astype(np.float32)
        log.info(f"  Layer {li}: shape={tensor.shape}")

    # Load sentence IDs and domains
    with open(hs_dir / "sentence_ids.json") as f:
        sentence_ids = json.load(f)
    domains = None
    if (hs_dir / "domains.json").exists():
        with open(hs_dir / "domains.json") as f:
            domains = json.load(f)

    return hidden_states, labels, meta, sentence_ids, domains, available_layers


def generate_control_labels(labels, seed=42):
    """Generate control task labels (Hewitt & Liang 2019).

    Shuffles the constraint_type labels while preserving the marginal
    distribution. This creates a task of equal difficulty in terms of
    class balance, but the labels are meaningless — they don't correspond
    to any property of the hidden states.

    If a probe achieves high accuracy on the control task, it means the
    probe is memorizing rather than reading genuine information.

    Selectivity = real_accuracy - control_accuracy.
    """
    rng = np.random.RandomState(seed)
    real_labels = labels["constraint_type"].numpy().copy()
    shuffled = real_labels.copy()
    rng.shuffle(shuffled)
    return torch.tensor(shuffled, dtype=torch.long)


# ── Probe training ───────────────────────────────────────────────────────────
def train_probe_cv(X, y, n_classes, n_folds=N_FOLDS, seed=42):
    """Train a linear probe with stratified cross-validation and C sweep.

    Returns:
        dict with:
          - mean/std of macro_f1, accuracy across folds
          - best C value
          - per-fold results
          - per-class F1 from the best C
          - the best trained model (from full data with best C)
    """
    # Filter out any classes with < n_folds examples (can't stratify)
    class_counts = np.bincount(y, minlength=n_classes)
    valid_mask = np.array([class_counts[yi] >= n_folds for yi in y])
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        log.warning(f"Dropping {n_dropped} samples from rare classes "
                    f"(< {n_folds} examples)")
        X = X[valid_mask]
        y = y[valid_mask]

    if len(y) < n_folds * 2:
        log.warning(f"Too few samples ({len(y)}). Skipping this probe.")
        return None

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # C sweep: find best C via mean macro_f1 across folds
    c_results = {}
    for c_val in C_VALUES:
        fold_f1s = []
        fold_accs = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Standardize features (per-fold to avoid data leakage)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = LogisticRegression(
                C=c_val,
                max_iter=MAX_ITER,
                solver="lbfgs",
                multi_class="multinomial" if n_classes > 2 else "auto",
                random_state=seed,
                n_jobs=-1,
            )
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)

            fold_f1s.append(f1_score(y_test, y_pred, average="macro",
                                      zero_division=0))
            fold_accs.append(accuracy_score(y_test, y_pred))

        c_results[c_val] = {
            "macro_f1_mean": np.mean(fold_f1s),
            "macro_f1_std": np.std(fold_f1s),
            "accuracy_mean": np.mean(fold_accs),
            "accuracy_std": np.std(fold_accs),
            "fold_f1s": fold_f1s,
            "fold_accs": fold_accs,
        }

    # Select best C
    best_c = max(c_results, key=lambda c: c_results[c]["macro_f1_mean"])
    best = c_results[best_c]

    # Retrain on full data with best C for per-class report and saving
    scaler_full = StandardScaler()
    X_full = scaler_full.fit_transform(X)
    clf_full = LogisticRegression(
        C=best_c,
        max_iter=MAX_ITER,
        solver="lbfgs",
        multi_class="multinomial" if n_classes > 2 else "auto",
        random_state=seed,
        n_jobs=-1,
    )
    clf_full.fit(X_full, y)
    y_pred_full = clf_full.predict(X_full)

    # Per-class breakdown (on full data — informational, not for main claims)
    labels_present = sorted(set(y))
    report = classification_report(
        y, y_pred_full,
        labels=labels_present,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        str(c): round(report[str(c)]["f1-score"], 4)
        for c in labels_present if str(c) in report
    }

    return {
        "best_c": best_c,
        "macro_f1_mean": round(best["macro_f1_mean"], 4),
        "macro_f1_std": round(best["macro_f1_std"], 4),
        "accuracy_mean": round(best["accuracy_mean"], 4),
        "accuracy_std": round(best["accuracy_std"], 4),
        "fold_f1s": [round(f, 4) for f in best["fold_f1s"]],
        "fold_accs": [round(a, 4) for a in best["fold_accs"]],
        "per_class_f1": per_class,
        "n_samples": len(y),
        "n_classes_actual": len(labels_present),
        "c_sweep": {
            str(c): {
                "macro_f1_mean": round(v["macro_f1_mean"], 4),
                "macro_f1_std": round(v["macro_f1_std"], 4),
            }
            for c, v in c_results.items()
        },
        "model": clf_full,
        "scaler": scaler_full,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SOLAR Linear Probe: Train probes on hidden states"
    )
    parser.add_argument(
        "--hidden-states-dir", required=True,
        help="Directory with extracted hidden states (from extract_hidden_states.py)"
    )
    parser.add_argument(
        "--pooling", default="mean_pool",
        choices=["mean_pool", "last_token"],
        help="Which pooling strategy to use"
    )
    parser.add_argument(
        "--layers", default=None,
        help="Comma-separated layer indices to probe (default: all available)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip-control", action="store_true",
        help="Skip control task (faster but no selectivity measure)"
    )
    args = parser.parse_args()

    # Load data
    hidden_states, labels, meta, sentence_ids, domains, available_layers = \
        load_hidden_states_and_labels(args.hidden_states_dir, args.pooling)

    # Filter layers if specified
    if args.layers:
        requested = [int(x) for x in args.layers.split(",")]
        available_layers = [l for l in available_layers if l in requested]

    # Output directory
    hs_dir = Path(args.hidden_states_dir)
    out_dir = Path(args.output_dir) if args.output_dir else \
        hs_dir.parent.parent / "probes" / hs_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best_probes").mkdir(exist_ok=True)
    log.info(f"Output directory: {out_dir}")

    # Generate control labels
    if not args.skip_control:
        control_labels = generate_control_labels(labels, seed=args.seed)
        labels["control_type"] = control_labels
        # Save for reproducibility
        with open(out_dir / "control_labels.json", "w") as f:
            json.dump(control_labels.tolist(), f)
        log.info(f"Control labels generated (seed={args.seed})")

    # Prepare task list
    tasks = {}
    for name, info in PROBE_TASKS.items():
        if name == "control_type" and args.skip_control:
            continue
        label_key = info["label_key"]
        if label_key not in labels:
            log.warning(f"Skipping task '{name}': label key '{label_key}' not found")
            continue
        tasks[name] = {
            **info,
            "y": labels[label_key].numpy(),
        }

    log.info(f"Tasks to probe: {list(tasks.keys())}")
    log.info(f"Layers to probe: {available_layers}")
    log.info(f"Pooling: {args.pooling}")

    # ── Train probes ─────────────────────────────────────────────────────
    all_results = {}

    for layer_idx in available_layers:
        log.info(f"\n{'='*60}")
        log.info(f"  LAYER {layer_idx}")
        log.info(f"{'='*60}")

        X = hidden_states[layer_idx]
        layer_results = {}

        for task_name, task_info in tasks.items():
            y = task_info["y"]
            n_classes = task_info["n_classes"]

            log.info(f"\n  [{task_name}] n_classes={n_classes}, "
                     f"n_samples={len(y)}")

            result = train_probe_cv(
                X, y, n_classes,
                n_folds=N_FOLDS,
                seed=args.seed,
            )

            if result is None:
                layer_results[task_name] = {"error": "too few samples"}
                continue

            # Save the trained model
            probe_path = out_dir / "best_probes" / \
                f"{task_name}_layer_{layer_idx}.pkl"
            with open(probe_path, "wb") as f:
                pickle.dump({
                    "model": result["model"],
                    "scaler": result["scaler"],
                    "best_c": result["best_c"],
                }, f)

            # Remove non-serializable objects for JSON output
            result_clean = {k: v for k, v in result.items()
                           if k not in ("model", "scaler")}
            result_clean["is_piece"] = task_info["is_piece"]
            result_clean["description"] = task_info["description"]
            layer_results[task_name] = result_clean

            log.info(f"    Macro F1: {result['macro_f1_mean']:.4f} "
                     f"(±{result['macro_f1_std']:.4f})  "
                     f"[best C={result['best_c']}]")

            if task_name == "constraint_type":
                log.info(f"    Per-class F1: {result['per_class_f1']}")

        all_results[str(layer_idx)] = layer_results

    # ── Compute selectivity (if control task was run) ────────────────────
    selectivity = {}
    if not args.skip_control:
        log.info(f"\n{'='*60}")
        log.info(f"  SELECTIVITY ANALYSIS (Hewitt & Liang 2019)")
        log.info(f"{'='*60}")

        for layer_idx in available_layers:
            li_str = str(layer_idx)
            lr = all_results.get(li_str, {})
            real = lr.get("constraint_type", {})
            ctrl = lr.get("control_type", {})

            if "error" in real or "error" in ctrl:
                continue

            real_f1 = real.get("macro_f1_mean", 0)
            ctrl_f1 = ctrl.get("macro_f1_mean", 0)
            sel = round(real_f1 - ctrl_f1, 4)

            selectivity[li_str] = {
                "real_f1": real_f1,
                "control_f1": ctrl_f1,
                "selectivity": sel,
            }

            log.info(f"  Layer {layer_idx}: real={real_f1:.4f} "
                     f"control={ctrl_f1:.4f} selectivity={sel:.4f}")

            # Statistical test: paired t-test on fold F1s
            if "fold_f1s" in real and "fold_f1s" in ctrl:
                t_stat, p_val = stats.ttest_rel(
                    real["fold_f1s"], ctrl["fold_f1s"]
                )
                selectivity[li_str]["t_stat"] = round(t_stat, 4)
                selectivity[li_str]["p_value"] = round(p_val, 6)
                log.info(f"           t={t_stat:.4f}, p={p_val:.6f}")

    # ── Composition Gap from Probes ──────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"  COMPOSITION GAP (Probe-Level)")
    log.info(f"{'='*60}")
    log.info(f"  {'Layer':<8} {'Piece-avg':>10} {'Structure':>10} "
             f"{'Gap':>8} {'Selectivity':>12}")
    log.info(f"  {'─'*52}")

    probe_gaps = {}
    for layer_idx in available_layers:
        li_str = str(layer_idx)
        lr = all_results.get(li_str, {})

        # Piece-level: average of all piece probes
        piece_f1s = []
        for task_name, task_result in lr.items():
            if isinstance(task_result, dict) and task_result.get("is_piece"):
                piece_f1s.append(task_result.get("macro_f1_mean", 0))

        struct_f1 = lr.get("constraint_type", {}).get("macro_f1_mean", 0)
        piece_avg = np.mean(piece_f1s) if piece_f1s else 0
        gap = round(piece_avg - struct_f1, 4)
        sel = selectivity.get(li_str, {}).get("selectivity", None)

        probe_gaps[li_str] = {
            "piece_avg": round(piece_avg, 4),
            "structure_f1": struct_f1,
            "gap": gap,
            "selectivity": sel,
        }

        sel_str = f"{sel:.4f}" if sel is not None else "  —"
        log.info(f"  {layer_idx:<8} {piece_avg:>10.4f} {struct_f1:>10.4f} "
                 f"{gap:>+8.4f} {sel_str:>12}")

    # ── Interpretation ───────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"  INTERPRETATION")
    log.info(f"{'='*60}")

    # Check best structure probe vs best model output
    best_probe_struct = max(
        (all_results[str(li)].get("constraint_type", {}).get("macro_f1_mean", 0)
         for li in available_layers),
        default=0
    )

    log.info(f"  Best probe structure F1:   {best_probe_struct:.4f}")
    log.info(f"  Best baseline structure F1: 0.43 (deepseek-cot from Table 2)")

    if best_probe_struct > 0.55:
        log.info(f"  → OPTION A SUPPORTED: Hidden states contain enough")
        log.info(f"    information for structure classification. The model")
        log.info(f"    'knows' the answer but the output head fails to")
        log.info(f"    compose it. This is a COMPOSITIONAL failure.")
        interpretation = "OPTION_A_COMPOSITIONAL"
    elif best_probe_struct < 0.35:
        log.info(f"  → OPTION B: Hidden states do NOT linearly encode")
        log.info(f"    structural labels. The failure is REPRESENTATIONAL,")
        log.info(f"    not compositional.")
        interpretation = "OPTION_B_REPRESENTATIONAL"
    else:
        log.info(f"  → MIXED: Probe partially recovers structure but not")
        log.info(f"    dramatically better than model output. The composition")
        log.info(f"    gap exists but may be partly representational too.")
        interpretation = "MIXED"

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "meta": {
            "model": meta.get("model"),
            "pooling": args.pooling,
            "layers_probed": available_layers,
            "n_folds": N_FOLDS,
            "c_values": C_VALUES,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        },
        "probe_results": all_results,
        "selectivity": selectivity,
        "probe_composition_gap": probe_gaps,
        "interpretation": interpretation,
        "best_probe_structure_f1": best_probe_struct,
    }

    with open(out_dir / "probe_results.json", "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"\n  Results saved → {out_dir / 'probe_results.json'}")

    log.info(f"\n{'='*60}")
    log.info(f"  DONE. Run analyze_results.py for visualizations.")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
