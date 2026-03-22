# ============================================================
# SOLAR — Script 08: Evaluate IR Extractor
# ============================================================
# Runs the fine-tuned model on all 211 test sentences and
# reports:
#   - Constraint type F1 (HARD/SOFT)
#   - Threshold extraction accuracy
#   - Exception extraction accuracy
#   - JSON validity rate
#   - Per-class confusion matrix
# ============================================================

import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── Load test data ────────────────────────────────────────────
TEST_FILE = f"{DRIVE_BASE}/data/training/test.jsonl"

print("Loading test data...")
test_records = []
with open(TEST_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            test_records.append(json.loads(line))

print(f"Test records : {len(test_records)}")


# ── Inference function ────────────────────────────────────────
def run_inference(sentence: str) -> tuple[dict | None, str]:
    """
    Run the fine-tuned model on a single sentence.
    Returns (parsed_json, raw_response).
    """
    SYSTEM_PROMPT = """You are a constraint extraction system for transit operations.
Given a sentence from a transit labor agreement or operational policy document,
extract the following structured information in JSON format:

- constraint_type: HARD or SOFT
- hardness_subtype: HARD-CONDITIONAL, SOFT-APPROVAL, SOFT-CONDITIONAL, or same as constraint_type
- entities: {subject: [], authority: [], object: []}
- thresholds: [{variable, value, unit, direction}] or []
- exceptions: [{trigger, type}] or []

Return ONLY valid JSON. No explanation, no preamble, no markdown."""

    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}"
        f"<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Extract constraints from this sentence:\n\n"
        f"\"{sentence.strip()}\""
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 1024,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 512,
            temperature    = 0.1,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens = True,
    ).strip()

    # Try to parse JSON
    try:
        # Handle case where model wraps in ```json ... ```
        clean = response
        if "```" in clean:
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        parsed = json.loads(clean.strip())
        return parsed, response
    except json.JSONDecodeError:
        return None, response


# ── Metrics helpers ───────────────────────────────────────────
def f1_score(tp, fp, fn) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def threshold_match(pred_thresh: list, gold_thresh: list) -> bool:
    """
    Check if predicted thresholds match gold thresholds.
    Matches on value — the most important field.
    """
    if not gold_thresh and not pred_thresh:
        return True
    if not gold_thresh or not pred_thresh:
        return False
    gold_values = {t.get("value") for t in gold_thresh if t.get("value")}
    pred_values = {t.get("value") for t in pred_thresh if t.get("value")}
    return len(gold_values & pred_values) > 0


def exception_match(pred_exc: list, gold_exc: list) -> bool:
    """Check if predicted exceptions match gold exceptions."""
    if not gold_exc and not pred_exc:
        return True
    if not gold_exc or not pred_exc:
        return False
    # Match if any exception trigger overlaps
    gold_triggers = {e.get("trigger", "").lower()[:30] for e in gold_exc}
    pred_triggers = {e.get("trigger", "").lower()[:30] for e in pred_exc}
    return len(gold_triggers & pred_triggers) > 0


# ── Run evaluation ────────────────────────────────────────────
print("\nRunning evaluation on test set...")
print("This takes ~10 minutes for 211 sentences\n")

results = []
confusion = defaultdict(lambda: defaultdict(int))

# Per-class counters
class_tp = defaultdict(int)
class_fp = defaultdict(int)
class_fn = defaultdict(int)

n_valid_json    = 0
n_invalid_json  = 0
n_thresh_correct= 0
n_thresh_total  = 0
n_exc_correct   = 0
n_exc_total     = 0

for i, record in enumerate(test_records):
    if i % 20 == 0:
        print(f"  Progress: {i}/{len(test_records)}")

    # Get gold labels from completion
    try:
        gold = json.loads(record["completion"])
    except json.JSONDecodeError:
        continue

    gold_type  = gold.get("constraint_type", "")
    raw_text   = record.get("prompt", "").split('"')[-2]

    # Run model inference
    pred, raw_response = run_inference(raw_text)

    # JSON validity
    if pred is None:
        n_invalid_json += 1
        pred_type = "INVALID"
        results.append({
            "constraint_id": record.get("constraint_id"),
            "gold_type"    : gold_type,
            "pred_type"    : "INVALID",
            "valid_json"   : False,
            "raw_response" : raw_response[:200],
        })
        confusion[gold_type]["INVALID"] += 1
        class_fn[gold_type] += 1
        continue

    n_valid_json += 1
    pred_type = pred.get("constraint_type", "UNKNOWN")

    # Constraint type classification
    confusion[gold_type][pred_type] += 1
    if pred_type == gold_type:
        class_tp[gold_type] += 1
    else:
        class_fp[pred_type] += 1
        class_fn[gold_type] += 1

    # Threshold extraction
    gold_thresh = gold.get("thresholds", [])
    pred_thresh = pred.get("thresholds", [])
    if gold_thresh:  # only evaluate when gold has thresholds
        n_thresh_total += 1
        if threshold_match(pred_thresh, gold_thresh):
            n_thresh_correct += 1

    # Exception extraction
    gold_exc = gold.get("exceptions", [])
    pred_exc = pred.get("exceptions", [])
    if gold_exc:
        n_exc_total += 1
        if exception_match(pred_exc, gold_exc):
            n_exc_correct += 1

    results.append({
        "constraint_id"  : record.get("constraint_id"),
        "gold_type"      : gold_type,
        "pred_type"      : pred_type,
        "correct"        : pred_type == gold_type,
        "valid_json"     : True,
        "gold_thresholds": gold_thresh,
        "pred_thresholds": pred_thresh,
        "gold_exceptions": gold_exc,
        "pred_exceptions": pred_exc,
    })

print(f"  Progress: {len(test_records)}/{len(test_records)} ✅")


# ── Print results ─────────────────────────────────────────────
total = len(test_records)

print(f"\n{'='*55}")
print(f"  SOLAR IR EXTRACTOR — EVALUATION RESULTS")
print(f"{'='*55}")

# JSON validity
json_pct = 100 * n_valid_json / total
print(f"\n  JSON Validity Rate : {n_valid_json}/{total} ({json_pct:.1f}%)")

# Per-class F1
print(f"\n  Constraint Type Classification:")
print(f"  {'Class':<8} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
print(f"  {'-'*38}")

all_f1 = []
for ctype in ["HARD", "SOFT"]:
    tp = class_tp[ctype]
    fp = class_fp[ctype]
    fn = class_fn[ctype]
    p, r, f1 = f1_score(tp, fp, fn)
    support = tp + fn
    all_f1.append(f1)
    print(f"  {ctype:<8} {p:>6.3f} {r:>6.3f} {f1:>6.3f} {support:>8}")

macro_f1 = np.mean(all_f1)
print(f"  {'Macro F1':<8} {'':>6} {'':>6} {macro_f1:>6.3f}")

# Threshold accuracy
if n_thresh_total > 0:
    thresh_acc = 100 * n_thresh_correct / n_thresh_total
    print(f"\n  Threshold Extraction:")
    print(f"  Accuracy : {n_thresh_correct}/{n_thresh_total} ({thresh_acc:.1f}%)")

# Exception accuracy
if n_exc_total > 0:
    exc_acc = 100 * n_exc_correct / n_exc_total
    print(f"\n  Exception Extraction:")
    print(f"  Accuracy : {n_exc_correct}/{n_exc_total} ({exc_acc:.1f}%)")

# Confusion matrix
print(f"\n  Confusion Matrix (rows=gold, cols=pred):")
pred_classes = ["HARD", "SOFT", "INVALID"]
print(f"  {'':>8}", end="")
for pc in pred_classes:
    print(f" {pc:>8}", end="")
print()
print(f"  {'-'*38}")
for gc in ["HARD", "SOFT"]:
    print(f"  {gc:>8}", end="")
    for pc in pred_classes:
        print(f" {confusion[gc][pc]:>8}", end="")
    print()

print(f"\n{'='*55}")

# Save results
results_path = f"{DRIVE_BASE}/results/evaluation_results.json"
import os
os.makedirs(f"{DRIVE_BASE}/results", exist_ok=True)
with open(results_path, "w") as f:
    json.dump({
        "summary": {
            "total"            : total,
            "valid_json"       : n_valid_json,
            "json_validity_pct": json_pct,
            "macro_f1"         : macro_f1,
            "threshold_acc"    : thresh_acc if n_thresh_total > 0 else None,
            "exception_acc"    : exc_acc if n_exc_total > 0 else None,
        },
        "per_record": results,
    }, f, indent=2)

print(f"  Results saved → {results_path}")
print(f"\n  These numbers go directly into your paper! 🎉")