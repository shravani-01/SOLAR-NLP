"""
SOLAR — Script 12a: Collect CFFT Training Pairs
=================================================
Runs the fine-tuned CP-SAT generator model on all IR records,
executes the generated code with OR-Tools, and collects
INFEASIBLE/ERROR cases as contrastive training pairs for CFFT.

This runs on your Mac — no GPU needed.
Uses the Hugging Face API to run inference on the saved model.

Wait — actually we use a smarter approach:
Since loading 8B model on Mac is too slow, we save the
model outputs from Colab first, then run OR-Tools locally.

WORKFLOW:
  Step 1 (Colab)  : Run 12a_generate_outputs.py — generate
                    CP-SAT code for all IR records using
                    the fine-tuned model, save to JSON
  Step 2 (Mac)    : Run 12a_collect_cfft_pairs.py — execute
                    generated code with OR-Tools, collect
                    INFEASIBLE/ERROR pairs

This script = Step 2 (Mac side)

Input:
  data/processed/cpsat_generated_outputs.json  ← from Colab Step 1
  data/processed/constraint_ir.json            ← ground truth

Output:
  data/processed/cfft_training_pairs.json      ← CFFT pairs
  data/processed/cfft_execution_report.json    ← stats

Usage:
  python scripts/12a_collect_cfft_pairs.py
  python scripts/12a_collect_cfft_pairs.py --round 2
"""

import json
import argparse
from pathlib import Path
from collections import Counter

PROJECT_ROOT  = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"/"transit"

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("WARNING: ortools not installed")
    print("Install: pip install ortools --break-system-packages")

# Reuse sandbox from script 09
N_OPERATORS = 5
N_TRIPS     = 10

VARIABLE_REGISTRY = {
    "max_consecutive_hours" : ("consecutive_min",  0, 720,  60),
    "min_break_minutes"     : ("break_min",         0, 480,   1),
    "min_break_hours"       : ("break_min",         0, 480,  60),
    "min_layover_minutes"   : ("layover_min",       0, 120,   1),
    "max_deadhead_minutes"  : ("deadhead_min",      0, 120,   1),
    "max_holdover_hours"    : ("holdover_min",      0, 480,  60),
    "max_shift_hours"       : ("shift_min",         0, 960,  60),
    "max_work_hours"        : ("work_min",          0, 960,  60),
    "min_rest_hours"        : ("rest_min",          0, 960,  60),
    "min_rest_minutes"      : ("rest_min",          0, 960,   1),
    "max_overtime_hours"    : ("overtime_min",      0, 480,  60),
    "max_suspension_days"   : ("suspension_days",   0, 365,   1),
    "max_absence_days"      : ("absence_days",      0, 365,   1),
    "max_vacation_days"     : ("vacation_days",     0,  60,   1),
    "max_hearing_days"      : ("hearing_days",      0,  90,   1),
    "max_reschedule_days"   : ("reschedule_days",   0,  90,   1),
    "min_notice_days"       : ("notice_days",       0,  30,   1),
    "duration_days"         : ("duration_days",     0, 365,   1),
    "duration_months"       : ("duration_months",   0,  60,   1),
    "duration_years"        : ("duration_years",    0,  10,   1),
    "frequency_per_week"    : ("frequency_week",    0,   7,   1),
    "pay_rate_usd_per_hour" : ("pay_rate_cents",    0, 10000, 1),
    "duration_unknown"      : ("generic_value",     0, 10000, 1),
    "unknown"               : ("generic_value",     0, 10000, 1),
}


# ============================================================
# SANDBOX EXECUTOR — same as script 09
# ============================================================
def build_sandbox_context(model) -> dict:
    ctx = {
        "model"       : model,
        "N_OPERATORS" : N_OPERATORS,
        "N_TRIPS"     : N_TRIPS,
    }
    created = set()
    for ir_var, (cp_var, lo, hi, scale) in VARIABLE_REGISTRY.items():
        if cp_var not in created:
            ctx[cp_var] = [
                model.NewIntVar(lo, hi * scale, f"{cp_var}_{i}")
                for i in range(N_OPERATORS)
            ]
            created.add(cp_var)
    ctx["generic_value"] = [
        model.NewIntVar(0, 10000, f"generic_{i}")
        for i in range(N_OPERATORS)
    ]
    return ctx


def execute_code(code: str) -> str:
    """Execute CP-SAT code in sandbox. Returns FEASIBLE/INFEASIBLE/ERROR."""
    if not ORTOOLS_AVAILABLE:
        return "SKIPPED:ortools_not_available"
    try:
        model  = cp_model.CpModel()
        solver = cp_model.CpSolver()
        ctx    = build_sandbox_context(model)
        exec(compile(code, "<generated>", "exec"), ctx)
        solver.parameters.max_time_in_seconds = 5.0
        status = solver.Solve(model)
        if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            return "FEASIBLE"
        elif status == cp_model.INFEASIBLE:
            return "INFEASIBLE"
        else:
            return f"INFEASIBLE:status_{status}"
    except SyntaxError as e:
        return f"ERROR:SyntaxError:{str(e)[:100]}"
    except Exception as e:
        return f"ERROR:{type(e).__name__}:{str(e)[:100]}"


# ============================================================
# CFFT PAIR BUILDER
# ============================================================
def build_cfft_pair(
    ir: dict,
    generated_code: str,
    solver_status: str,
    correct_code: str,
    round_num: int,
) -> dict:
    """
    Build a CFFT contrastive training pair.

    Each pair contains:
      - wrong_code    : what the model generated (INFEASIBLE/ERROR)
      - correct_code  : the template-generated verified code
      - cfft_signal   : INFEASIBLE or ERROR
      - contrastive_prompt : training prompt showing both
    """
    # Build contrastive prompt for fine-tuning
    contrastive_prompt = (
        f"# CFFT Round {round_num} — Learn from solver failure\n"
        f"# Constraint: {ir['constraint_id']}\n"
        f"# Text: {ir['raw_text'][:100]}...\n"
        f"# Solver signal: {solver_status}\n\n"
        f"# WRONG (solver rejected):\n"
        f"{generated_code}\n\n"
        f"# CORRECT (solver accepted):\n"
        f"{correct_code}"
    )

    return {
        "constraint_id"    : ir["constraint_id"],
        "round"            : round_num,
        "raw_text"         : ir["raw_text"],
        "constraint_type"  : ir["constraint_type"],
        "hardness_subtype" : ir["hardness_subtype"],
        "wrong_code"       : generated_code,
        "correct_code"     : correct_code,
        "solver_status"    : solver_status,
        "cfft_signal"      : solver_status.split(":")[0],
        "contrastive_prompt": contrastive_prompt,
        # Training record for fine-tuning
        "text": (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a CP-SAT constraint code generator. "
            f"Generate correct, solver-verified CP-SAT code."
            f"<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Generate CP-SAT code for:\n{json.dumps({'constraint_id': ir['constraint_id'], 'raw_text': ir['raw_text'], 'constraint_type': ir['constraint_type'], 'thresholds': ir['thresholds'], 'exceptions': ir['exceptions']}, indent=2)}"
            f"<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{correct_code}"
            f"<|eot_id|>"
        )
    }


# ============================================================
# MAIN PIPELINE
# ============================================================
def collect_cfft_pairs(
    generated_outputs: list[dict],
    ir_lookup: dict[str, dict],
    round_num: int,
) -> tuple[list, dict]:
    """
    Execute all generated outputs, collect failures as CFFT pairs.
    Returns (cfft_pairs, stats).
    """
    cfft_pairs = []
    stats      = Counter()

    print(f"\n  Executing {len(generated_outputs)} generated snippets...")
    print(f"  Running OR-Tools solver on each...\n")

    for i, output in enumerate(generated_outputs):
        if i % 200 == 0:
            print(f"  Progress: {i}/{len(generated_outputs)}")

        cid            = output["constraint_id"]
        generated_code = output["generated_code"]
        ir             = ir_lookup.get(cid)

        if ir is None:
            stats["missing_ir"] += 1
            continue

        # Execute generated code
        solver_status = execute_code(generated_code)

        if solver_status == "FEASIBLE":
            stats["feasible"] += 1
        elif solver_status.startswith("INFEASIBLE"):
            stats["infeasible"] += 1
            cfft_pairs.append(build_cfft_pair(
                ir             = ir,
                generated_code = generated_code,
                solver_status  = solver_status,
                correct_code   = ir.get("cpsat_code", ""),
                round_num      = round_num,
            ))
        elif solver_status.startswith("ERROR"):
            stats["error"] += 1
            err_type = solver_status.split(":")[1] if ":" in solver_status else "unknown"
            stats[f"error_{err_type}"] += 1
            cfft_pairs.append(build_cfft_pair(
                ir             = ir,
                generated_code = generated_code,
                solver_status  = solver_status,
                correct_code   = ir.get("cpsat_code", ""),
                round_num      = round_num,
            ))
        else:
            stats["skipped"] += 1

    print(f"  Progress: {len(generated_outputs)}/{len(generated_outputs)} ✅")
    return cfft_pairs, dict(stats)


def print_report(stats: dict, cfft_pairs: list, total: int, round_num: int):
    """Print execution and CFFT collection summary."""
    print(f"\n{'='*55}")
    print(f"  CFFT ROUND {round_num} — EXECUTION REPORT")
    print(f"{'='*55}")
    print(f"  Total executed      : {total}")
    feasible = stats.get('feasible', 0)
    infeasible = stats.get('infeasible', 0)
    error = stats.get('error', 0)
    print(f"\n  Solver results:")
    print(f"    FEASIBLE          : {feasible}  "
          f"({100*feasible//total if total else 0}%)")
    print(f"    INFEASIBLE        : {infeasible}")
    print(f"    ERROR             : {error}")
    print(f"\n  CFFT pairs collected: {len(cfft_pairs)}")
    print(f"    → These become Round {round_num} training signal")

    if error > 0:
        print(f"\n  Error breakdown:")
        for k, v in stats.items():
            if k.startswith("error_"):
                print(f"    {v:>4}  {k[6:]}")

    feasibility_rate = 100 * feasible // total if total else 0
    print(f"\n  Feasibility rate    : {feasibility_rate}%")
    print(f"  Target after CFFT   : {min(feasibility_rate + 15, 95)}%+")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser(
        description="SOLAR Script 12a: Collect CFFT Pairs")
    parser.add_argument("--round", type=int, default=1,
        help="CFFT round number (default: 1)")
    parser.add_argument("--outputs", default=None,
        help="Path to generated outputs JSON")
    args = parser.parse_args()

    print("\n" + "="*55)
    print(f"SOLAR — Script 12a: Collect CFFT Pairs (Round {args.round})")
    print("="*55)

    # Load generated outputs from Colab
    outputs_path = (
        Path(args.outputs) if args.outputs
        else PROCESSED_DIR / "cpsat_generated_outputs_round3.json"
    )

    if not outputs_path.exists():
        print(f"\nERROR: {outputs_path} not found.")
        print("\nYou need to run the Colab inference step first:")
        print("  1. Open Colab")
        print("  2. Run 12a_generate_outputs (Colab cell)")
        print("  3. Download cpsat_generated_outputs.json to data/processed/")
        print("  4. Run this script again")
        print("\nSee scripts/12a_generate_outputs_colab.py for the Colab cell")
        return

    with open(outputs_path) as f:
        generated_outputs = json.load(f)

    print(f"\n  Generated outputs loaded : {len(generated_outputs)}")

    # Load IR for ground truth correct code
    ir_path = PROCESSED_DIR / "constraint_ir.json"
    with open(ir_path) as f:
        ir_records = json.load(f)

    ir_lookup = {r["constraint_id"]: r for r in ir_records}
    print(f"  IR records loaded        : {len(ir_records)}")
    print(f"  OR-Tools available       : {ORTOOLS_AVAILABLE}")

    # Collect CFFT pairs
    cfft_pairs, stats = collect_cfft_pairs(
        generated_outputs = generated_outputs,
        ir_lookup         = ir_lookup,
        round_num         = args.round,
    )

    # Save CFFT pairs
    cfft_path = PROCESSED_DIR / f"cfft_pairs_round{args.round}.json"
    with open(cfft_path, "w") as f:
        json.dump(cfft_pairs, f, indent=2)
    print(f"\n  CFFT pairs saved → {cfft_path.relative_to(PROJECT_ROOT)}")

    # Save report
    report = {
        "round"       : args.round,
        "total"       : len(generated_outputs),
        "cfft_pairs"  : len(cfft_pairs),
        "stats"       : stats,
    }
    report_path = PROCESSED_DIR / f"cfft_report_round{args.round}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print_report(stats, cfft_pairs, len(generated_outputs), args.round)

    # Show sample CFFT pairs
    if cfft_pairs:
        print(f"\n  Sample CFFT pairs (first 2):")
        for pair in cfft_pairs[:2]:
            print(f"\n  [{pair['constraint_type']}] {pair['constraint_id']}")
            print(f"  Signal : {pair['cfft_signal']}")
            print(f"  Wrong code (first 3 lines):")
            for line in pair['wrong_code'].split('\n')[:3]:
                print(f"    {line}")
            print(f"  Correct code (first 3 lines):")
            for line in pair['correct_code'].split('\n')[:3]:
                print(f"    {line}")

    print(f"\n  Next steps:")
    print(f"  1. Upload cfft_pairs_round{args.round}.json to Google Drive")
    print(f"  2. Run 12b_cfft_finetune.py on Colab")
    print(f"  3. Download updated model")
    print(f"  4. Run 12a again with --round {args.round + 1}\n")


if __name__ == "__main__":
    main()