"""
SOLAR — Script 09: Generate CP-SAT Code
=========================================
Converts Constraint IR records into executable CP-SAT Python code.

Approach: Hybrid
  - Type 1 (numeric threshold) → template-based generation
  - Type 2 (boolean/procedural) → template-based generation
  - Type 3 (conditional) → template + OnlyEnforceIf

Then executes each generated snippet in a sandboxed CP-SAT
environment and records: FEASIBLE / INFEASIBLE / ERROR

Updates constraint_ir.json with:
  - cpsat_code     : generated code string
  - solver_status  : FEASIBLE | INFEASIBLE | ERROR | SKIPPED
  - verified       : True if FEASIBLE

Output:
  data/processed/constraint_ir.json          — updated with solver_status
  data/processed/cpsat_generation_report.json — full generation report
  data/processed/cfft_pairs.json             — INFEASIBLE/ERROR pairs for Phase 5

Usage:
  python scripts/09_generate_cpsat.py
  python scripts/09_generate_cpsat.py --limit 100
"""

import re
import json
import argparse
from pathlib import Path
from collections import Counter

PROJECT_ROOT  = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("WARNING: ortools not installed — solver verification disabled")
    print("Install: pip install ortools --break-system-packages")


# ============================================================
# VARIABLE REGISTRY
# Maps IR variable names → (cp_var_name, lo, hi, scale)
# scale: multiplier to convert to integers (e.g. hours → minutes)
# ============================================================
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
    "max_penalty_days"      : ("penalty_days",      0,  30,   1),
    "max_probation_days"    : ("probation_days",    0, 365,   1),
    "duration_days"         : ("duration_days",     0, 365,   1),
    "max_absence_months"    : ("absence_months",    0,  24,   1),
    "max_absence_years"     : ("absence_years",     0,   5,   1),
    "max_probation_months"  : ("probation_months",  0,  24,   1),
    "min_leave_months"      : ("leave_months",      0,  24,   1),
    "duration_months"       : ("duration_months",   0,  60,   1),
    "duration_years"        : ("duration_years",    0,  10,   1),
    "frequency_per_week"    : ("frequency_week",    0,   7,   1),
    "pay_rate_usd_per_hour" : ("pay_rate_cents",    0, 10000, 1),
    "max_rate_usd_per_hour" : ("pay_rate_cents",    0, 10000, 1),
    "min_pay_percentage"    : ("pay_pct",           0, 100,   1),
    "max_pay_percentage"    : ("pay_pct",           0, 100,   1),
    "duration_unknown"      : ("generic_value",     0, 10000, 1),
    "rate_usd"              : ("pay_rate_cents",    0, 10000, 1),
    "unknown"               : ("generic_value",     0, 10000, 1),
}

DEFAULT_VAR    = ("generic_value", 0, 10000, 1)
N_OPERATORS    = 5
N_TRIPS        = 10


# ============================================================
# HELPERS
# ============================================================
def get_var_info(variable: str) -> tuple:
    """Look up variable info, fall back to generic."""
    return VARIABLE_REGISTRY.get(variable, DEFAULT_VAR)


def _get_subject(ir: dict) -> str:
    """Extract primary subject type from entity dict."""
    subjects = ir.get("entities", {}).get("subject", [])
    if not subjects:
        return "generic"
    s = subjects[0].lower()
    if any(kw in s for kw in ["operator", "driver", "employee"]):
        return "operator"
    return "generic"


def _make_exception_var(cid: str) -> str:
    """Generate a safe exception variable name."""
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', cid.lower())
    return f"exception_{safe}"


# ============================================================
# CODE TEMPLATES
# ============================================================
def template_numeric_threshold(ir: dict) -> str:
    """
    Generate CP-SAT code for constraints with numeric thresholds.
    Uses [0] index for all list variables — works for both
    per-operator and scalar cases in the sandbox.
    """
    lines = []
    lines.append(f"# Constraint: {ir['constraint_id']}")
    lines.append(f"# Type: {ir['hardness_subtype']}")
    lines.append(f"# Text: {ir['raw_text'][:80]}...")
    lines.append("")

    subject     = _get_subject(ir)
    has_exception = len(ir.get("exceptions", [])) > 0
    exception_var = None

    if has_exception:
        exc = ir["exceptions"][0]
        exception_var = _make_exception_var(ir["constraint_id"])
        lines.append(f"# Exception: {exc.get('trigger', '')[:60]}")
        lines.append(
            f"{exception_var} = model.NewBoolVar('{exception_var}')"
        )
        lines.append("")

    for threshold in ir["thresholds"]:
        value    = threshold.get("value")
        variable = threshold.get("variable", "unknown")
        unit     = threshold.get("unit", "") or ""
        direction= threshold.get("direction")

        if value is None:
            lines.append(
                f"# SKIPPED: threshold value is None for {variable}"
            )
            continue

        var_info    = get_var_info(variable)
        cp_var      = var_info[0]
        scale       = var_info[3]
        scaled_value= int(value * scale)
        config_name = re.sub(r'[^A-Z0-9_]', '_', variable.upper())

        lines.append(f"# Config value")
        lines.append(
            f"{config_name} = {scaled_value}  "
            f"# {value} {unit} × {scale} scale factor"
        )
        lines.append("")
        lines.append(f"# CP-SAT constraint")

        if subject == "operator":
            # Per-operator loop
            lines.append(f"for op in range(N_OPERATORS):")
            if direction == "max":
                c = f"    model.Add({cp_var}[op] <= {config_name})"
            elif direction == "min":
                c = f"    model.Add({cp_var}[op] >= {config_name})"
            else:
                c = (f"    model.Add({cp_var}[op] <= {config_name})"
                     f"  # direction inferred max")
            if exception_var:
                c += f".OnlyEnforceIf({exception_var}.Not())"
            lines.append(c)
        else:
            # ── FIX: always index [0] to avoid list vs int TypeError ──
            if direction == "max":
                c = f"model.Add({cp_var}[0] <= {config_name})"
            elif direction == "min":
                c = f"model.Add({cp_var}[0] >= {config_name})"
            else:
                c = (f"model.Add({cp_var}[0] <= {config_name})"
                     f"  # direction inferred max")
            if exception_var:
                c += f".OnlyEnforceIf({exception_var}.Not())"
            lines.append(c)

        lines.append("")

    return "\n".join(lines)


def template_boolean_constraint(ir: dict) -> str:
    """
    Generate CP-SAT code for boolean/procedural constraints.
    No numeric threshold — mandatory condition enforced as BoolVar.
    """
    cid   = re.sub(r'[^a-zA-Z0-9_]', '_', ir["constraint_id"].lower())
    lines = []
    lines.append(f"# Constraint: {ir['constraint_id']}")
    lines.append(f"# Type: {ir['hardness_subtype']} (boolean/procedural)")
    lines.append(f"# Text: {ir['raw_text'][:80]}...")
    lines.append("")

    bool_var = f"condition_{cid}"
    lines.append(f"# Boolean constraint — mandatory condition")
    lines.append(f"{bool_var} = model.NewBoolVar('{bool_var}')")

    has_exception = len(ir.get("exceptions", [])) > 0
    if has_exception:
        exc     = ir["exceptions"][0]
        exc_type= exc.get("type", "conditional")
        exc_var = _make_exception_var(ir["constraint_id"])
        lines.append(
            f"# Exception ({exc_type}): {exc.get('trigger','')[:60]}"
        )
        lines.append(f"{exc_var} = model.NewBoolVar('{exc_var}')")
        lines.append(
            f"model.Add({bool_var} == 1)"
            f".OnlyEnforceIf({exc_var}.Not())"
        )
    else:
        lines.append(f"model.Add({bool_var} == 1)")

    return "\n".join(lines)


def template_soft_constraint(ir: dict) -> str:
    """
    Generate CP-SAT code for SOFT constraints.
    Uses slack variable — violation penalized, not prohibited.
    Always uses [0] index on list variables.
    """
    cid   = re.sub(r'[^a-zA-Z0-9_]', '_', ir["constraint_id"].lower())
    lines = []
    lines.append(f"# Constraint: {ir['constraint_id']}")
    lines.append(f"# Type: {ir['hardness_subtype']} (soft — penalized)")
    lines.append(f"# Text: {ir['raw_text'][:80]}...")
    lines.append("")

    thresholds = ir.get("thresholds", [])
    if thresholds and thresholds[0].get("value") is not None:
        t         = thresholds[0]
        value     = t.get("value")
        variable  = t.get("variable", "unknown")
        unit      = t.get("unit", "") or ""
        direction = t.get("direction")
        var_info  = get_var_info(variable)
        cp_var    = var_info[0]
        scale     = var_info[3]
        scaled    = int(value * scale)

        slack_var = f"slack_{cid}"
        lines.append(f"# Soft constraint with slack variable")
        lines.append(
            f"{slack_var} = model.NewIntVar(0, {max(scaled, 1)}, "
            f"'{slack_var}')"
        )
        # ── FIX: always use [0] index for list variables ──
        if direction == "max":
            lines.append(
                f"model.Add({cp_var}[0] <= {scaled} + {slack_var})"
            )
        else:
            lines.append(
                f"model.Add({cp_var}[0] >= {scaled} - {slack_var})"
            )
        lines.append(f"# Minimize slack in objective (penalize violation)")
        lines.append(f"# objective_terms.append({slack_var})")
    else:
        # Boolean soft preference
        pref_var = f"preference_{cid}"
        lines.append(f"# Soft boolean preference")
        lines.append(f"{pref_var} = model.NewBoolVar('{pref_var}')")
        lines.append(f"# objective_terms.append({pref_var}.Not())")

    return "\n".join(lines)


# ============================================================
# CODE ROUTER
# ============================================================
def generate_code(ir: dict) -> tuple[str, str]:
    """
    Route IR record to the appropriate code template.
    Returns (code_string, template_type).
    """
    constraint_type = ir.get("constraint_type", "")
    thresholds      = ir.get("thresholds", [])
    has_real_thresh = any(
        t.get("value") is not None for t in thresholds
    )

    if constraint_type == "SOFT":
        return template_soft_constraint(ir), "soft"
    elif has_real_thresh:
        return template_numeric_threshold(ir), "numeric"
    else:
        return template_boolean_constraint(ir), "boolean"


# ============================================================
# CP-SAT SANDBOX
# ============================================================
def build_sandbox_context(model: "cp_model.CpModel") -> dict:
    """
    Build sandboxed variable context for code execution.
    All variables are lists of length N_OPERATORS so both
    per-operator loops and [0] indexing work correctly.
    """
    ctx = {
        "model"      : model,
        "N_OPERATORS": N_OPERATORS,
        "N_TRIPS"    : N_TRIPS,
    }

    created = set()
    for ir_var, (cp_var, lo, hi, scale) in VARIABLE_REGISTRY.items():
        if cp_var not in created:
            ctx[cp_var] = [
                model.NewIntVar(lo, hi * scale, f"{cp_var}_{i}")
                for i in range(N_OPERATORS)
            ]
            created.add(cp_var)

    # Generic fallback
    ctx["generic_value"] = [
        model.NewIntVar(0, 10000, f"generic_{i}")
        for i in range(N_OPERATORS)
    ]

    return ctx


def execute_code(code: str) -> str:
    """
    Execute generated CP-SAT code in a fresh sandboxed model.
    Returns: FEASIBLE | INFEASIBLE | ERROR:<type>:<message>
    """
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
# MAIN PIPELINE
# ============================================================
def process_ir(
    ir_records: list[dict],
    limit: int = None,
) -> tuple[list, list, dict]:
    """
    Generate code and verify all IR records.
    Returns (updated_records, cfft_pairs, stats).
    """
    if limit:
        ir_records = ir_records[:limit]

    updated    = []
    cfft_pairs = []
    stats      = Counter()

    for i, ir in enumerate(ir_records):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(ir_records)}")

        code, template_type = generate_code(ir)
        stats[f"template_{template_type}"] += 1

        solver_status = execute_code(code)

        if solver_status == "FEASIBLE":
            stats["feasible"] += 1
            verified = True
        elif solver_status.startswith("INFEASIBLE"):
            stats["infeasible"] += 1
            verified = False
            cfft_pairs.append({
                "constraint_id"  : ir["constraint_id"],
                "raw_text"       : ir["raw_text"],
                "constraint_type": ir["constraint_type"],
                "wrong_ir"       : ir,
                "wrong_code"     : code,
                "solver_status"  : solver_status,
                "cfft_signal"    : "INFEASIBLE",
            })
        elif solver_status.startswith("ERROR"):
            stats["error"] += 1
            verified = False
            cfft_pairs.append({
                "constraint_id"  : ir["constraint_id"],
                "raw_text"       : ir["raw_text"],
                "constraint_type": ir["constraint_type"],
                "wrong_ir"       : ir,
                "wrong_code"     : code,
                "solver_status"  : solver_status,
                "cfft_signal"    : "ERROR",
            })
        else:
            stats["skipped"] += 1
            verified = False

        updated.append({**ir,
            "cpsat_code"   : code,
            "solver_status": solver_status,
            "verified"     : verified,
            "template_type": template_type,
        })

    return updated, cfft_pairs, dict(stats)


def print_report(stats: dict, cfft_pairs: list, total: int):
    """Print generation and verification summary."""
    print(f"\n{'='*55}")
    print(f"  CP-SAT CODE GENERATION REPORT")
    print(f"{'='*55}")
    print(f"  Total processed     : {total}")
    print(f"\n  Template usage:")
    print(f"    Numeric threshold  : {stats.get('template_numeric', 0)}")
    print(f"    Boolean/procedural : {stats.get('template_boolean', 0)}")
    print(f"    Soft constraint    : {stats.get('template_soft', 0)}")
    print(f"\n  Solver verification:")
    feasible = stats.get('feasible', 0)
    pct = 100 * feasible // total if total > 0 else 0
    print(f"    FEASIBLE           : {feasible}  ({pct}%)")
    print(f"    INFEASIBLE         : {stats.get('infeasible', 0)}")
    print(f"    ERROR              : {stats.get('error', 0)}")
    print(f"    SKIPPED            : {stats.get('skipped', 0)}")
    print(f"\n  CFFT training pairs : {len(cfft_pairs)}")
    print(f"    (INFEASIBLE+ERROR → Phase 5 training signal)")
    print(f"{'='*55}")

    if stats.get("error", 0) > 0:
        error_types = Counter()
        for pair in cfft_pairs:
            if pair["cfft_signal"] == "ERROR":
                parts = pair["solver_status"].split(":")
                error_types[parts[1] if len(parts) > 1 else "unknown"] += 1
        print(f"\n  Error types:")
        for err, count in error_types.most_common(5):
            print(f"    {count:>4}  {err}")


def main():
    parser = argparse.ArgumentParser(
        description="SOLAR Step 9: Generate CP-SAT Code from IR")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N records (for testing)")
    parser.add_argument(
        "--ir", default=None,
        help="Path to IR JSON (default: data/processed/constraint_ir.json)")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("SOLAR — Step 09: CP-SAT Code Generator")
    print("="*55)

    if not ORTOOLS_AVAILABLE:
        print("\nWARNING: ortools not found.")
        print("Install: pip install ortools --break-system-packages\n")

    ir_path = (
        Path(args.ir) if args.ir
        else PROCESSED_DIR / "constraint_ir.json"
    )
    if not ir_path.exists():
        print(f"ERROR: {ir_path} not found. Run 05_build_ir.py first.")
        return

    with open(ir_path) as f:
        ir_records = json.load(f)

    total = min(args.limit, len(ir_records)) if args.limit else len(ir_records)
    print(f"\n  IR records loaded  : {len(ir_records)}")
    if args.limit:
        print(f"  Processing limit   : {args.limit}")
    print(f"  OR-Tools available : {ORTOOLS_AVAILABLE}")
    print(f"\n  Generating and verifying {total} constraints...")

    updated, cfft_pairs, stats = process_ir(ir_records, args.limit)
    print(f"  Progress: {total}/{total} ✅")

    # Save updated IR
    with open(PROCESSED_DIR / "constraint_ir.json", "w") as f:
        json.dump(updated, f, indent=2)
    print(f"\n  Updated IR → data/processed/constraint_ir.json")

    # Save CFFT pairs
    with open(PROCESSED_DIR / "cfft_pairs.json", "w") as f:
        json.dump(cfft_pairs, f, indent=2)
    print(f"  CFFT pairs → data/processed/cfft_pairs.json")

    # Save report
    with open(PROCESSED_DIR / "cpsat_generation_report.json", "w") as f:
        json.dump({"stats": stats, "total": total,
                   "cfft_pairs": len(cfft_pairs)}, f, indent=2)

    print_report(stats, cfft_pairs, total)

    # Show 3 FEASIBLE samples
    print(f"\n  Sample generated codes (FEASIBLE):")
    shown = 0
    for r in updated:
        if r.get("solver_status") == "FEASIBLE" and shown < 3:
            print(f"\n  [{r['constraint_type']}] {r['constraint_id']}"
                  f" — {r['solver_status']}")
            print(f"  Template: {r.get('template_type')}")
            for line in r['cpsat_code'].split('\n')[:7]:
                print(f"  {line}")
            print(f"  ...")
            shown += 1

    print(f"\n  Next: python scripts/10_cfft_training_loop.py")
    print(f"  cfft_pairs.json → Phase 5 training signal\n")


if __name__ == "__main__":
    main()