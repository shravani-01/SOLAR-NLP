"""
SOLAR — Linear Probe: Step 2 — Train Probes (PyTorch GPU)
==========================================================
Trains linear classifiers (nn.Linear) on frozen hidden states using GPU
to determine whether piece-level and structure-level information is
linearly decodable from the model's representations.

This is mathematically equivalent to logistic regression — a single
linear layer + cross-entropy loss learns the same decision boundary.
Using PyTorch on GPU makes it ~100x faster than sklearn on CPU for
large hidden-state matrices (46k × 3584).

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
  - nn.Linear(hidden_dim, n_classes) + CrossEntropyLoss + AdamW
  - Weight decay sweep: {1e-3, 1e-1} (L2 regularization equivalent)
  - Early stopping on validation loss (patience=5)
  - Evaluation: Macro F1 (consistent with baseline comparison script)
  - Per-class F1 for constraint_type
  - Statistical significance: paired t-test on fold-level F1 scores

Output:
  data/results/linear_probe/probes/
    ├── probe_results.json          # All metrics (same format as sklearn version)
    ├── best_probes/                # Saved model state dicts
    │   ├── constraint_type_layer_-1.pt
    │   └── ...
    └── control_labels.json         # Shuffled labels (for reproducibility)

Usage:
  python linear_probe/train_probes.py \\
      --hidden-states-dir data/results/linear_probe/hidden_states/qwen2.5_7b/

  # Only specific layers
  python linear_probe/train_probes.py \\
      --hidden-states-dir data/results/linear_probe/hidden_states/qwen2.5_7b/ \\
      --layers -1,-4,-8

Legacy sklearn version: train_probes_sklearn.py
"""

import os
import json
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report
)
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

# Probe tasks: name → config
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
        "label_key": "control_type",
        "n_classes": 5,
        "is_piece": False,
        "description": "CONTROL: shuffled constraint_type labels (same distribution)",
        "is_control": True,
    },
}

# Weight decay values to sweep (equivalent to L2 regularization)
WD_VALUES = [1e-3, 1e-1]

# Cross-validation folds
N_FOLDS = 5

# Training hyperparameters
LR = 1e-2
BATCH_SIZE = 1024
MAX_EPOCHS = 100
PATIENCE = 5  # early stopping patience


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

    # Load hidden states — keep as tensors for GPU transfer
    hidden_states = {}
    for li in available_layers:
        tensor = torch.load(pool_dir / f"layer_{li}.pt", weights_only=True)
        hidden_states[li] = tensor.float()  # ensure float32
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
    distribution. Creates a task of equal difficulty in terms of class
    balance, but labels are meaningless.

    Selectivity = real_accuracy - control_accuracy.
    """
    rng = np.random.RandomState(seed)
    real_labels = labels["constraint_type"].numpy().copy()
    shuffled = real_labels.copy()
    rng.shuffle(shuffled)
    return torch.tensor(shuffled, dtype=torch.long)


# ── Linear Probe Model ──────────────────────────────────────────────────────
class LinearProbe(nn.Module):
    """Single linear layer — equivalent to logistic regression."""

    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


# ── Training one probe ──────────────────────────────────────────────────────
def train_one_fold(X_train, y_train, X_val, y_val, input_dim, n_classes,
                   weight_decay, device, seed=42):
    """Train a linear probe on one fold, return val metrics and model."""
    torch.manual_seed(seed)

    # Compute class weights for balanced loss
    class_counts = torch.bincount(y_train, minlength=n_classes).float()
    class_weights = (1.0 / (class_counts + 1e-8))
    class_weights = class_weights / class_weights.sum() * n_classes
    class_weights = class_weights.to(device)

    # Move data to GPU
    X_tr = X_train.to(device)
    y_tr = y_train.to(device)
    X_v = X_val.to(device)
    y_v = y_val.to(device)

    # Standardize (compute on train, apply to val)
    mean = X_tr.mean(dim=0)
    std = X_tr.std(dim=0) + 1e-8
    X_tr = (X_tr - mean) / std
    X_v = (X_v - mean) / std

    # Model
    model = LinearProbe(input_dim, n_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # DataLoader for mini-batch training
    train_dataset = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_loss = criterion(val_logits, y_v).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    # Load best model and compute metrics
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds = model(X_v).argmax(dim=1).cpu().numpy()

    y_val_np = y_val.numpy()
    f1 = f1_score(y_val_np, val_preds, average="macro", zero_division=0)
    acc = accuracy_score(y_val_np, val_preds)

    return f1, acc, best_state, mean.cpu(), std.cpu(), epoch + 1


# ── Probe training with CV ──────────────────────────────────────────────────
def train_probe_cv(X, y, n_classes, task_name="", layer_idx=0,
                   n_folds=N_FOLDS, device="cuda", seed=42):
    """Train a linear probe with stratified CV and weight_decay sweep.

    Returns dict compatible with analyze_results.py format.
    """
    y_np = y.numpy() if isinstance(y, torch.Tensor) else y
    y_t = torch.tensor(y_np, dtype=torch.long) if not isinstance(y, torch.Tensor) else y

    # Filter rare classes
    class_counts = np.bincount(y_np, minlength=n_classes)
    valid_mask = np.array([class_counts[yi] >= n_folds for yi in y_np])
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        log.warning(f"Dropping {n_dropped} samples from rare classes "
                    f"(< {n_folds} examples)")
        X = X[valid_mask]
        y_t = y_t[valid_mask]
        y_np = y_np[valid_mask]

    if len(y_np) < n_folds * 2:
        log.warning(f"Too few samples ({len(y_np)}). Skipping this probe.")
        return None

    input_dim = X.shape[1]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # Weight decay sweep
    wd_results = {}
    total_fits = len(WD_VALUES) * n_folds
    fit_count = 0

    for wd in WD_VALUES:
        fold_f1s = []
        fold_accs = []

        for fold_i, (train_idx, test_idx) in enumerate(skf.split(y_np, y_np)):
            fit_count += 1
            t0 = time.time()

            X_train, X_val = X[train_idx], X[test_idx]
            y_train, y_val = y_t[train_idx], y_t[test_idx]

            f1, acc, _, _, _, n_epochs = train_one_fold(
                X_train, y_train, X_val, y_val,
                input_dim, n_classes, wd, device, seed=seed + fold_i
            )
            fold_f1s.append(f1)
            fold_accs.append(acc)

            elapsed = time.time() - t0
            log.info(f"    [{task_name}] Layer {layer_idx} | "
                     f"wd={wd} fold {fold_i+1}/{n_folds}: "
                     f"F1={f1:.4f} acc={acc:.4f} ({elapsed:.1f}s, {n_epochs}ep) "
                     f"[{fit_count}/{total_fits}]")

        mean_f1 = np.mean(fold_f1s)
        log.info(f"    [{task_name}] wd={wd} mean F1: {mean_f1:.4f}")

        wd_results[wd] = {
            "macro_f1_mean": float(np.mean(fold_f1s)),
            "macro_f1_std": float(np.std(fold_f1s)),
            "accuracy_mean": float(np.mean(fold_accs)),
            "accuracy_std": float(np.std(fold_accs)),
            "fold_f1s": [float(f) for f in fold_f1s],
            "fold_accs": [float(a) for a in fold_accs],
        }

    # Select best weight decay
    best_wd = max(wd_results, key=lambda w: wd_results[w]["macro_f1_mean"])
    best = wd_results[best_wd]

    # Retrain on full data with best wd for per-class report
    log.info(f"    [{task_name}] Retraining on full data with best wd={best_wd}...")
    t0 = time.time()

    # Use 90/10 split for early stopping on full retrain
    n_train = int(0.9 * len(y_np))
    perm = np.random.RandomState(seed).permutation(len(y_np))
    train_idx_full = perm[:n_train]
    val_idx_full = perm[n_train:]

    _, _, best_state, mean_full, std_full, _ = train_one_fold(
        X[train_idx_full], y_t[train_idx_full],
        X[val_idx_full], y_t[val_idx_full],
        input_dim, n_classes, best_wd, device, seed=seed
    )

    # Predict on all data for per-class report
    model_full = LinearProbe(input_dim, n_classes).to(device)
    model_full.load_state_dict(best_state)
    model_full.eval()
    X_all = ((X.to(device) - mean_full.to(device)) /
             (std_full.to(device) + 1e-8))
    with torch.no_grad():
        y_pred_full = model_full(X_all).argmax(dim=1).cpu().numpy()
    log.info(f"    [{task_name}] Full retrain done ({time.time()-t0:.1f}s)")

    # Per-class breakdown
    labels_present = sorted(set(y_np))
    report = classification_report(
        y_np, y_pred_full,
        labels=labels_present,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        str(c): round(report[str(c)]["f1-score"], 4)
        for c in labels_present if str(c) in report
    }

    return {
        "best_wd": best_wd,
        "macro_f1_mean": round(best["macro_f1_mean"], 4),
        "macro_f1_std": round(best["macro_f1_std"], 4),
        "accuracy_mean": round(best["accuracy_mean"], 4),
        "accuracy_std": round(best["accuracy_std"], 4),
        "fold_f1s": [round(f, 4) for f in best["fold_f1s"]],
        "fold_accs": [round(a, 4) for a in best["fold_accs"]],
        "per_class_f1": per_class,
        "n_samples": len(y_np),
        "n_classes_actual": len(labels_present),
        "wd_sweep": {
            str(w): {
                "macro_f1_mean": round(v["macro_f1_mean"], 4),
                "macro_f1_std": round(v["macro_f1_std"], 4),
            }
            for w, v in wd_results.items()
        },
        "model_state": best_state,
        "mean": mean_full,
        "std": std_full,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SOLAR Linear Probe: Train probes on hidden states (PyTorch GPU)"
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
    parser.add_argument(
        "--device", default=None,
        help="Device: cuda, cpu, or auto (default: auto)"
    )
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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
            "y": labels[label_key],
        }

    n_total_probes = len(tasks) * len(available_layers)
    log.info(f"Tasks to probe: {list(tasks.keys())}")
    log.info(f"Layers to probe: {available_layers}")
    log.info(f"Pooling: {args.pooling}")
    log.info(f"Total probes to train: {n_total_probes}")
    log.info(f"Fits per probe: {len(WD_VALUES)} wd values x {N_FOLDS} folds "
             f"= {len(WD_VALUES) * N_FOLDS}")
    log.info(f"Weight decay values: {WD_VALUES}")
    log.info(f"Classifier: nn.Linear + CrossEntropyLoss + AdamW (GPU)")
    log.info(f"Training: batch_size={BATCH_SIZE}, lr={LR}, "
             f"max_epochs={MAX_EPOCHS}, patience={PATIENCE}")

    # ── Train probes ─────────────────────────────────────────────────────
    all_results = {}
    probe_count = 0
    overall_start = time.time()

    for layer_idx in available_layers:
        log.info(f"\n{'='*60}")
        log.info(f"  LAYER {layer_idx}")
        log.info(f"{'='*60}")

        X = hidden_states[layer_idx]
        layer_results = {}

        for task_name, task_info in tasks.items():
            probe_count += 1
            y = task_info["y"]
            n_classes = task_info["n_classes"]

            log.info(f"\n  [{task_name}] n_classes={n_classes}, "
                     f"n_samples={len(y)} "
                     f"[probe {probe_count}/{n_total_probes}]")

            result = train_probe_cv(
                X, y, n_classes,
                task_name=task_name,
                layer_idx=layer_idx,
                n_folds=N_FOLDS,
                device=device,
                seed=args.seed,
            )

            if result is None:
                layer_results[task_name] = {"error": "too few samples"}
                continue

            # Save the trained model
            probe_path = out_dir / "best_probes" / \
                f"{task_name}_layer_{layer_idx}.pt"
            torch.save({
                "model_state": result["model_state"],
                "mean": result["mean"],
                "std": result["std"],
                "best_wd": result["best_wd"],
                "input_dim": X.shape[1],
                "n_classes": n_classes,
            }, probe_path)

            # Remove non-serializable objects for JSON output
            result_clean = {k: v for k, v in result.items()
                           if k not in ("model_state", "mean", "std")}
            result_clean["is_piece"] = task_info["is_piece"]
            result_clean["description"] = task_info["description"]
            layer_results[task_name] = result_clean

            log.info(f"  >>> {task_name}: Macro F1 = {result['macro_f1_mean']:.4f} "
                     f"(±{result['macro_f1_std']:.4f})  [best wd={result['best_wd']}]")

            if task_name == "constraint_type":
                log.info(f"    Per-class F1: {result['per_class_f1']}")

            # ETA estimate
            elapsed = time.time() - overall_start
            avg_per_probe = elapsed / probe_count
            remaining = (n_total_probes - probe_count) * avg_per_probe
            log.info(f"    ETA: ~{remaining/60:.0f} min remaining "
                     f"({probe_count}/{n_total_probes} probes done, "
                     f"{elapsed/60:.1f} min elapsed)")

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
                selectivity[li_str]["t_stat"] = round(float(t_stat), 4)
                selectivity[li_str]["p_value"] = round(float(p_val), 6)
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
        piece_avg = float(np.mean(piece_f1s)) if piece_f1s else 0
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

    total_time = time.time() - overall_start
    log.info(f"\n  Total training time: {total_time/60:.1f} minutes")

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "meta": {
            "model": meta.get("model"),
            "pooling": args.pooling,
            "layers_probed": available_layers,
            "n_folds": N_FOLDS,
            "weight_decay_values": WD_VALUES,
            "seed": args.seed,
            "classifier": "nn.Linear + CrossEntropyLoss + AdamW (PyTorch GPU)",
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "device": str(device),
            "total_time_minutes": round(total_time / 60, 1),
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
