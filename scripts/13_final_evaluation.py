"""
SOLAR — Script 13: Final Evaluation & Baseline Experiments
===========================================================
Runs all baselines and evaluates SOLAR on the test set.

Baselines:
  - GPT-4o + CoT (1-shot)
  - GPT-4o + RAG (4-shot)
  - DeepSeek-V3 + CoT (1-shot)
  - DeepSeek-V3 + RAG (4-shot)

SOLAR variants:
  - SOLAR w/o CFFT (base CP-SAT generator)
  - SOLAR full (best CFFT model — Round 2)

Usage:
  # Run all baselines
  python scripts/13_final_evaluation.py --mode all

  # Run one baseline only
  python scripts/13_final_evaluation.py --mode gpt4o_cot
  python scripts/13_final_evaluation.py --mode gpt4o_rag
  python scripts/13_final_evaluation.py --mode deepseek_cot
  python scripts/13_final_evaluation.py --mode deepseek_rag

  # Just compute metrics on existing results
  python scripts/13_final_evaluation.py --mode metrics_only
"""

import json
import argparse
import os
from pathlib import Path
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
print("DeepSeek key:", os.environ.get("DEEPSEEK_API_KEY", "NOT FOUND")[:10])

PROJECT_ROOT  = Path(__file__).parent.parent
TRAINING_DIR  = PROJECT_ROOT / "data" / "training" / "transit"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "transit"
RESULTS_DIR   = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load test set ─────────────────────────────────────────────────────────────
def load_test_set() -> list[dict]:
    test_path = TRAINING_DIR / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"Test set not found: {test_path}")
    records = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Test set loaded: {len(records)} records")
    return records

def fmt(v, fmt_str):
    return format(v, fmt_str) if v is not None else "--"

# ── Parse IR JSON from model response ────────────────────────────────────────
def parse_ir_response(response: str) -> dict | None:
    """Extract JSON from model response."""
    try:
        # Try direct parse
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    try:
        # Try extracting JSON block
        start = response.find("{")
        end   = response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass
    return None


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(predictions: list[dict], gold: list[dict]) -> dict:
    """
    Compute evaluation metrics:
    - JSON validity rate
    - Macro F1 (HARD vs SOFT classification)
    - Threshold extraction accuracy
    - Exception extraction accuracy
    """
    assert len(predictions) == len(gold), \
        f"Length mismatch: {len(predictions)} vs {len(gold)}"

    valid_json   = 0
    hard_tp = hard_fp = hard_fn = 0
    soft_tp = soft_fp = soft_fn = 0
    thresh_correct = thresh_total = 0
    except_correct = except_total = 0

    for pred, g in zip(predictions, gold):
        # JSON validity
        if pred is not None:
            valid_json += 1
        else:
            continue

        # Constraint type F1
        pred_type = pred.get("constraint_type", "").upper()
        gold_type = g.get("constraint_type", "").upper()

        if gold_type == "HARD":
            if pred_type == "HARD":
                hard_tp += 1
            else:
                hard_fn += 1
                soft_fp += 1
        elif gold_type == "SOFT":
            if pred_type == "SOFT":
                soft_tp += 1
            else:
                soft_fn += 1
                hard_fp += 1

        # Threshold accuracy
        gold_thresholds = g.get("thresholds", [])
        pred_thresholds = pred.get("thresholds", [])
        if gold_thresholds:
            thresh_total += 1
            if pred_thresholds:
                # Check if any threshold value matches
                gold_vals = {round(float(t.get("value", 0)), 1)
                             for t in gold_thresholds if t.get("value")}
                pred_vals = {round(float(t.get("value", 0)), 1)
                             for t in pred_thresholds if t.get("value")}
                if gold_vals & pred_vals:
                    thresh_correct += 1

        # Exception accuracy
        gold_exceptions = g.get("exceptions", [])
        pred_exceptions = pred.get("exceptions", [])
        if gold_exceptions:
            except_total += 1
            if pred_exceptions:
                except_correct += 1

    # F1 scores
    hard_prec = hard_tp / (hard_tp + hard_fp) if (hard_tp + hard_fp) > 0 else 0
    hard_rec  = hard_tp / (hard_tp + hard_fn) if (hard_tp + hard_fn) > 0 else 0
    hard_f1   = (2 * hard_prec * hard_rec / (hard_prec + hard_rec)
                 if (hard_prec + hard_rec) > 0 else 0)

    soft_prec = soft_tp / (soft_tp + soft_fp) if (soft_tp + soft_fp) > 0 else 0
    soft_rec  = soft_tp / (soft_tp + soft_fn) if (soft_tp + soft_fn) > 0 else 0
    soft_f1   = (2 * soft_prec * soft_rec / (soft_prec + soft_rec)
                 if (soft_prec + soft_rec) > 0 else 0)

    macro_f1      = (hard_f1 + soft_f1) / 2
    json_validity = valid_json / len(gold) if gold else 0
    thresh_acc    = thresh_correct / thresh_total if thresh_total > 0 else 0
    except_acc    = except_correct / except_total if except_total > 0 else 0

    return {
        "macro_f1"       : round(macro_f1, 4),
        "hard_f1"        : round(hard_f1, 4),
        "soft_f1"        : round(soft_f1, 4),
        "json_validity"  : round(json_validity, 4),
        "threshold_acc"  : round(thresh_acc, 4),
        "exception_acc"  : round(except_acc, 4),
        "n_total"        : len(gold),
        "n_valid_json"   : valid_json,
        "n_thresh_total" : thresh_total,
        "n_except_total" : except_total,
    }


# ── Build CoT prompt ──────────────────────────────────────────────────────────
COT_EXAMPLE = """Example:
Sentence: "No operator shall be required to work more than 6 consecutive hours without a meal period of not less than 30 minutes."

Output:
{
  "constraint_type": "HARD",
  "hardness_subtype": "HARD-CONDITIONAL",
  "entities": {"subject": ["operator"], "authority": [], "object": []},
  "thresholds": [
    {"variable": "max_consecutive_hours", "value": 6.0, "unit": "hours", "direction": "max"},
    {"variable": "min_break_minutes", "value": 30.0, "unit": "minutes", "direction": "min"}
  ],
  "exceptions": [{"trigger": "without a meal period", "type": "conditional"}]
}"""

SYSTEM_PROMPT = """You are an expert in extracting operational constraints from transit labor agreements.
Extract the constraint IR JSON from the given sentence.

Output ONLY valid JSON with these fields:
- constraint_type: "HARD" or "SOFT"
- hardness_subtype: "HARD", "HARD-CONDITIONAL", "SOFT-APPROVAL", or "SOFT-CONDITIONAL"
- entities: {subject: [], authority: [], object: []}
- thresholds: [{variable, value, unit, direction}] or []
- exceptions: [{trigger, type}] or []

No explanation, no markdown, just JSON."""


def make_cot_prompt(sentence: str) -> str:
    return f"{COT_EXAMPLE}\n\nNow extract the constraint IR from:\n\"{sentence}\""


def make_rag_prompt(sentence: str, examples: list[dict]) -> str:
    shots = ""
    for ex in examples[:4]:
        shots += f"\nSentence: \"{ex['sentence']}\"\nOutput: {json.dumps(ex['ir'])}\n"
    return f"{shots}\nNow extract:\n\"{sentence}\""


# ── GPT-4o baseline ───────────────────────────────────────────────────────────
def run_gpt4o_baseline(
    test_records: list[dict],
    mode: str,
    rag_examples: list[dict] | None = None,
    model_name: str = "gpt-4o",
) -> list[dict | None]:

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    predictions = []
    errors = 0

    print(f"\n  Running {model_name} {mode.upper()} on {len(test_records)} records...")

    for i, record in enumerate(test_records):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(test_records)}")

        # Get sentence from record
        text = record.get("text", "")
        sentence = ""
        if "<|start_header_id|>user<|end_header_id|>" in text:
            parts = text.split("<|start_header_id|>user<|end_header_id|>")
            if len(parts) > 1:
                sentence = parts[1].split("<|eot_id|>")[0].strip()
                sentence = sentence.replace("Extract constraint IR from this sentence:", "").strip()
                sentence = sentence.strip('"')

        if not sentence:
            predictions.append(None)
            continue

        # Build prompt
        if mode == "cot":
            user_prompt = make_cot_prompt(sentence)
        else:
            user_prompt = make_rag_prompt(sentence, rag_examples or [])

        try:
            response = client.chat.completions.create(
                model    = model_name,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature = 0.0,
                max_tokens  = 512,
            )
            content = response.choices[0].message.content.strip()
            parsed  = parse_ir_response(content)
            predictions.append(parsed)

        except Exception as e:
            errors += 1
            predictions.append(None)
            if errors <= 3:
                print(f"  Error on record {i}: {e}")

    print(f"  Done — {len(predictions)} predictions, {errors} errors")
    return predictions


# ── DeepSeek baseline ─────────────────────────────────────────────────────────
def run_deepseek_baseline(
    test_records: list[dict],
    mode: str,
    rag_examples: list[dict] | None = None,
) -> list[dict | None]:
    from openai import OpenAI as OpenAIClient
    client = OpenAIClient(
        api_key  = os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url = "https://api.deepseek.com",
    )

    predictions = []
    errors = 0

    print(f"\n  Running deepseek-chat {mode.upper()} on {len(test_records)} records...")

    for i, record in enumerate(test_records):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(test_records)}")

        text = record.get("text", "")
        sentence = ""
        if "<|start_header_id|>user<|end_header_id|>" in text:
            parts = text.split("<|start_header_id|>user<|end_header_id|>")
            if len(parts) > 1:
                sentence = parts[1].split("<|eot_id|>")[0].strip()
                sentence = sentence.replace("Extract constraint IR from this sentence:", "").strip()
                sentence = sentence.strip('"')

        if not sentence:
            predictions.append(None)
            continue

        if mode == "cot":
            user_prompt = make_cot_prompt(sentence)
        else:
            user_prompt = make_rag_prompt(sentence, rag_examples or [])

        try:
            response = client.chat.completions.create(
                model       = "deepseek-chat",
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature = 0.0,
                max_tokens  = 512,
            )
            content = response.choices[0].message.content.strip()
            parsed  = parse_ir_response(content)
            predictions.append(parsed)

        except Exception as e:
            errors += 1
            predictions.append(None)
            if errors <= 3:
                print(f"  Error on record {i}: {e}")

    print(f"  Done — {len(predictions)} predictions, {errors} errors")
    return predictions


# ── Load RAG examples from training set ──────────────────────────────────────
def load_rag_examples(n: int = 20) -> list[dict]:
    """Load diverse examples from training set for RAG."""
    train_path = TRAINING_DIR / "train.jsonl"
    examples   = []
    hard_count = soft_count = 0

    with open(train_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text   = record.get("text", "")

            # Extract sentence and IR from training record
            sentence = ""
            ir_str   = ""
            if "<|start_header_id|>user<|end_header_id|>" in text:
                parts    = text.split("<|start_header_id|>user<|end_header_id|>")
                sentence = parts[1].split("<|eot_id|>")[0].strip() if len(parts) > 1 else ""
                sentence = sentence.replace("Extract constraint IR from this sentence:", "").strip().strip('"')
            if "<|start_header_id|>assistant<|end_header_id|>" in text:
                parts  = text.split("<|start_header_id|>assistant<|end_header_id|>")
                ir_str = parts[1].split("<|eot_id|>")[0].strip() if len(parts) > 1 else ""

            if not sentence or not ir_str:
                continue

            try:
                ir = json.loads(ir_str)
            except json.JSONDecodeError:
                continue

            # Balance HARD and SOFT examples
            ct = ir.get("constraint_type", "")
            if ct == "HARD" and hard_count < n // 2:
                examples.append({"sentence": sentence, "ir": ir})
                hard_count += 1
            elif ct == "SOFT" and soft_count < n // 2:
                examples.append({"sentence": sentence, "ir": ir})
                soft_count += 1

            if len(examples) >= n:
                break

    print(f"  RAG examples loaded: {len(examples)} ({hard_count} HARD, {soft_count} SOFT)")
    return examples


# ── Load gold labels from test set ───────────────────────────────────────────
def load_gold_labels(test_records: list[dict]) -> list[dict]:
    """Extract gold IR labels from test JSONL records."""
    gold = []
    for record in test_records:
        text = record.get("text", "")
        ir   = None
        if "<|start_header_id|>assistant<|end_header_id|>" in text:
            parts  = text.split("<|start_header_id|>assistant<|end_header_id|>")
            ir_str = parts[1].split("<|eot_id|>")[0].strip() if len(parts) > 1 else ""
            try:
                ir = json.loads(ir_str)
            except json.JSONDecodeError:
                pass
        gold.append(ir or {})
    return gold


# ── Print results table ───────────────────────────────────────────────────────
def print_results_table(all_results: dict):
    print(f"\n{'='*75}")
    print(f"  SOLAR — FINAL EVALUATION RESULTS")
    print(f"{'='*75}")
    print(f"  {'Model':<25} {'F1':>6} {'Thresh':>8} {'Except':>8} {'JSON':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    order = [
        "gpt4o_cot", "gpt4o_rag",
        "deepseek_cot", "deepseek_rag",
        "solar_no_cfft", "solar_cfft_r1",
        "solar_cfft_r2",
    ]
    labels = {
        "gpt4o_cot"    : "GPT-4o + CoT",
        "gpt4o_rag"    : "GPT-4o + RAG",
        "deepseek_cot" : "DeepSeek + CoT",
        "deepseek_rag" : "DeepSeek + RAG",
        "solar_no_cfft": "SOLAR w/o CFFT",
        "solar_cfft_r1": "SOLAR CFFT R1",
        "solar_cfft_r2": "SOLAR CFFT R2 (best)",
    }

    for key in order:
        if key not in all_results:
            continue
        m   = all_results[key]
        lbl = labels.get(key, key)
        # print(f"  {lbl:<25} {m['macro_f1']:>6.3f} {m['threshold_acc']:>8.1%} {m['exception_acc']:>8.1%} {m['json_validity']:>8.1%}")
        
        print(f"  {lbl:<25} {fmt(m.get('macro_f1'), '.3f'):>6} {fmt(m.get('threshold_acc'), '.1%'):>8} {fmt(m.get('exception_acc'), '.1%'):>8} {fmt(m.get('json_validity'), '.1%'):>8}")


    print(f"{'='*75}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SOLAR Script 13: Final Evaluation")
    parser.add_argument("--mode", default="all",
        choices=["all", "gpt4o_cot", "gpt4o_rag",
                 "deepseek_cot", "deepseek_rag", "metrics_only"],
        help="Which baseline to run")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  SOLAR — Script 13: Final Evaluation")
    print("="*55)

    # Load test set and gold labels
    print("\nLoading test set...")
    test_records = load_test_set()
    gold_labels  = load_gold_labels(test_records)
    print(f"  Gold labels loaded: {len(gold_labels)}")

    # Load existing results
    results_path = RESULTS_DIR / "all_results.json"
    all_results  = {}
    if results_path.exists():
        with open(results_path) as f:
            all_results = json.load(f)
        print(f"  Existing results loaded: {list(all_results.keys())}")

    # Add SOLAR results (already computed)
    all_results["solar_no_cfft"] = {
        "macro_f1"      : 0.858,
        "hard_f1"       : 0.944,
        "soft_f1"       : 0.771,
        "json_validity" : 1.000,
        "threshold_acc" : 0.750,
        "exception_acc" : 0.262,
    }
    all_results["solar_cfft_r1"] = {
        "macro_f1"      : None,
        "threshold_acc" : None,
        "exception_acc" : None,
        "json_validity" : None,
        "feasibility"   : 0.94,
    }
    all_results["solar_cfft_r2"] = {
        "macro_f1"      : None,
        "threshold_acc" : None,
        "exception_acc" : None,
        "json_validity" : None,
        "feasibility"   : 0.94,
    }

    if args.mode == "metrics_only":
        print_results_table(all_results)
        return

    # Load RAG examples for 4-shot baselines
    print("\nLoading RAG examples...")
    rag_examples = load_rag_examples(n=20)

    # Run baselines
    modes_to_run = (
        ["gpt4o_cot", "gpt4o_rag", "deepseek_cot", "deepseek_rag"]
        if args.mode == "all"
        else [args.mode]
    )

    for mode in modes_to_run:
        print(f"\nRunning baseline: {mode}")
        pred_path = RESULTS_DIR / f"{mode}_predictions.json"

        # Skip if already done
        if pred_path.exists() and mode in all_results:
            print(f"  Already done — skipping ({mode})")
            continue

        # Run inference
        if "gpt4o" in mode:
            preds = run_gpt4o_baseline(
                test_records = test_records,
                mode         = mode.replace("gpt4o_", ""),
                rag_examples = rag_examples if "rag" in mode else None,
                model_name   = "gpt-4o",
            )
        elif "deepseek" in mode:
            preds = run_deepseek_baseline(
                test_records = test_records,
                mode         = mode.replace("deepseek_", ""),
                rag_examples = rag_examples if "rag" in mode else None,
            )
        else:
            continue

        # Save predictions
        with open(pred_path, "w") as f:
            json.dump(preds, f, indent=2)

        # Compute metrics
        metrics = compute_metrics(preds, gold_labels)
        all_results[mode] = metrics
        print(f"  {mode}: Macro F1 = {metrics['macro_f1']:.3f}")

        # Save running results
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Final results table
    print_results_table(all_results)

    # Save final results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved → {results_path}")

    # Generate paper table
    print("\n  LaTeX table for paper:")
    print("  \\begin{table*}[t]")
    print("  \\centering")
    print("  \\begin{tabular}{lcccc}")
    print("  \\toprule")
    print("  \\textbf{Model} & \\textbf{Macro F1} & \\textbf{Thresh.} & \\textbf{Except.} & \\textbf{JSON} \\\\")
    print("  \\midrule")

    for key in ["gpt4o_cot", "gpt4o_rag", "deepseek_cot", "deepseek_rag"]:
        if key not in all_results:
            continue
        m   = all_results[key]
        lbl = {"gpt4o_cot": "GPT-4o + CoT", "gpt4o_rag": "GPT-4o + RAG",
               "deepseek_cot": "DeepSeek + CoT", "deepseek_rag": "DeepSeek + RAG"}[key]
        f1  = f"{m['macro_f1']:.3f}" if m.get('macro_f1') else "--"
        th  = f"{m['threshold_acc']:.1%}" if m.get('threshold_acc') else "--"
        ex  = f"{m['exception_acc']:.1%}" if m.get('exception_acc') else "--"
        js  = f"{m['json_validity']:.1%}" if m.get('json_validity') else "--"
        print(f"  {lbl} & {f1} & {th} & {ex} & {js} \\\\")

    print("  \\midrule")
    print(f"  SOLAR w/o CFFT & 0.858 & 75.0\\% & 26.2\\% & 100\\% \\\\")
    print(f"  SOLAR CFFT R2 (best) & -- & -- & -- & -- \\\\")
    print("  \\bottomrule")
    print("  \\end{tabular}")
    print("  \\end{table*}")


if __name__ == "__main__":
    main()