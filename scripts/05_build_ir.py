"""
SOLAR — Script 05: Build Constraint IR
========================================
Converts annotated CSV into a structured, validated
Constraint IR (Intermediate Representation) in JSON format.

Standardizes:
  - threshold format  → {variable, value, unit, direction}
  - entity format     → {subject, authority, object}
  - exception format  → {trigger, type}
  - flags quality issues for review

Output:
  data/processed/constraint_ir.json        — full IR dataset
  data/processed/constraint_ir_issues.csv  — quality flags

Usage:
  python scripts/05_build_ir.py
  python scripts/05_build_ir.py --source data/annotated/contract_v5_annotated.csv
"""

import re
import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT  = Path(__file__).parent.parent
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Threshold parsing ─────────────────────────────────────────────────────────
UNIT_MAP = {
    "hour"  : "hours",   "hours"  : "hours",  "hr"     : "hours",
    "minute": "minutes", "minutes": "minutes", "min"    : "minutes",
    "day"   : "days",    "days"   : "days",
    "week"  : "weeks",   "weeks"  : "weeks",
    "month" : "months",  "months" : "months",
    "year"  : "years",   "years"  : "years",
}

DIR_MAP = {
    "max"          : "max", "maximum"       : "max",
    "not more than": "max", "not to exceed" : "max",
    "up to"        : "max", "at most"       : "max",
    "min"          : "min", "minimum"       : "min",
    "not less than": "min", "at least"      : "min",
}

VAR_TEMPLATES = {
    ("consecutive", "hour")  : "max_consecutive_hours",
    ("break", "minute")      : "min_break_minutes",
    ("break", "hour")        : "min_break_hours",
    ("layover", "minute")    : "min_layover_minutes",
    ("deadhead", "minute")   : "max_deadhead_minutes",
    ("holdover", "hour")     : "max_holdover_hours",
    ("suspension", "day")    : "max_suspension_days",
    ("reschedule", "day")    : "max_reschedule_days",
    ("absence", "day")       : "max_absence_days",
    ("absence", "month")     : "max_absence_months",
    ("absence", "year")      : "max_absence_years",
    ("leave", "month")       : "min_leave_months",
    ("vacation", "day")      : "max_vacation_days",
    ("hearing", "day")       : "max_hearing_days",
    ("notice", "day")        : "min_notice_days",
    ("sick", "day")          : "max_sick_days",
    ("month", "month")       : "duration_months",
    ("day", "day")           : "duration_days",
    ("week", "week")         : "frequency_per_week",
    ("year", "year")         : "duration_years",
}

EMPTY_THRESHOLD_VALUES = {"", "nan", "none", "null", "n/a", "na","empty"}


def is_empty_threshold(raw: str) -> bool:
    """Return True if the threshold field is effectively empty."""
    return str(raw).strip().lower() in EMPTY_THRESHOLD_VALUES


def parse_threshold(raw_threshold: str, raw_text: str) -> list[dict]:
    """
    Convert a raw threshold string into a list of structured threshold dicts.
    Returns empty list if threshold field is empty or nan.
    """
    if not raw_threshold or pd.isna(raw_threshold):
        return []

    raw = str(raw_threshold).strip()

    # Return empty list for nan/empty — not an error, just no threshold
    if is_empty_threshold(raw):
        return []

    raw_lower = raw.lower()
    thresholds = []

    # Pattern 1: key=value format (e.g. "max_hours=4")
    kv_match = re.match(r'([a-z_]+)\s*=\s*(\d+\.?\d*)\s*([a-z]*)', raw_lower)
    if kv_match:
        key, val, unit_raw = kv_match.groups()
        unit = UNIT_MAP.get(unit_raw.rstrip('s'), None) if unit_raw else None

        # Infer unit from variable name if not in raw
        if unit is None:
            for u_key, u_val in UNIT_MAP.items():
                if u_key in key:
                    unit = u_val
                    break

        direction = None
        if key.startswith("max"):
            direction = "max"
        elif key.startswith("min"):
            direction = "min"

        thresholds.append({
            "variable"  : key,
            "value"     : float(val),
            "unit"      : unit,
            "direction" : direction,
            "raw"       : raw,
        })
        return thresholds

    # Pattern 2: written/numeric value + unit
    # First try parenthesised number e.g. "twenty (20) days"
    paren_match = re.search(r'\((\d+)\)', raw)
    if paren_match:
        value = float(paren_match.group(1))
    else:
        num_match = re.search(r'\b(\d+\.?\d*)\b', raw)
        if num_match:
            value = float(num_match.group(1))
        else:
            word_nums = {
                "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "fifteen": 15, "twenty": 20, "thirty": 30, "sixty": 60,
                "ninety": 90, "twelve": 12, "twenty-four": 24,
            }
            value = None
            for word, num in word_nums.items():
                if word in raw_lower:
                    value = float(num)
                    break

    if value is None:
        return [{
            "variable"  : "unknown",
            "value"     : None,
            "unit"      : None,
            "direction" : None,
            "raw"       : raw,
        }]

    # Extract unit
    unit = None
    for u_key, u_val in UNIT_MAP.items():
        if u_key in raw_lower:
            unit = u_val
            break

    # Infer direction from combined context
    direction = None
    raw_combined = (raw_lower + " " + raw_text.lower()) if raw_text else raw_lower
    for d_key, d_val in DIR_MAP.items():
        if d_key in raw_combined:
            direction = d_val
            break

    # Infer variable name from context keywords
    variable = "duration_" + (unit or "unknown")
    raw_text_lower = raw_text.lower() if raw_text else ""
    for (ctx1, ctx2), var_name in VAR_TEMPLATES.items():
        if ctx1 in raw_text_lower and unit and unit.startswith(ctx2[:3]):
            variable = var_name
            break

    thresholds.append({
        "variable"  : variable,
        "value"     : value,
        "unit"      : unit,
        "direction" : direction,
        "raw"       : raw,
    })
    return thresholds


def parse_entities(raw_entities: str) -> dict:
    """Convert comma-separated entity string into typed entity dict."""
    if not raw_entities or pd.isna(raw_entities):
        return {"subject": [], "authority": [], "object": []}

    AUTHORITY_KEYWORDS = {
        "authority", "board", "manager", "director", "gm",
        "union", "arbitrator", "department", "operating authority",
        "transit authority", "employer", "management"
    }
    SUBJECT_KEYWORDS = {
        "employee", "operator", "driver", "worker", "cleaner",
        "member", "personnel", "staff", "employees", "operators"
    }

    entities_raw = [e.strip() for e in str(raw_entities).split(",") if e.strip()]
    result = {"subject": [], "authority": [], "object": []}

    for ent in entities_raw:
        ent_lower = ent.lower()
        if any(kw in ent_lower for kw in SUBJECT_KEYWORDS):
            result["subject"].append(ent)
        elif any(kw in ent_lower for kw in AUTHORITY_KEYWORDS):
            result["authority"].append(ent)
        else:
            result["object"].append(ent)

    return result


def parse_exception(raw_exception: str) -> list[dict]:
    """Convert raw exception string into structured exception dict."""
    if not raw_exception or pd.isna(raw_exception):
        return []

    raw = str(raw_exception).strip()
    if is_empty_threshold(raw):
        return []

    raw_lower = raw.lower()

    exc_type = "conditional"
    if any(kw in raw_lower for kw in
           ["subject to approval", "upon approval", "with approval"]):
        exc_type = "approval_based"
    elif any(kw in raw_lower for kw in ["emergency", "extraordinary"]):
        exc_type = "emergency"
    elif any(kw in raw_lower for kw in
             ["section", "subsection", "article", "paragraph"]):
        exc_type = "reference"

    return [{"trigger": raw, "type": exc_type}]


def build_ir_record(row: pd.Series) -> tuple[dict, list[str]]:
    """Convert one annotated CSV row into a Constraint IR dict."""
    issues = []

    raw_text   = str(row.get("raw_text", "") or "")
    thresholds = parse_threshold(
        str(row.get("threshold", "") or ""), raw_text
    )
    entities   = parse_entities(str(row.get("entities", "") or ""))
    exceptions = parse_exception(str(row.get("exception", "") or ""))

    constraint_type = str(row.get("constraint_type", "")).upper().strip()

    # ── Quality checks — only flag genuinely fixable issues ──────────────────

    # # Flag HARD constraints with no exception AND no threshold
    # # (may be procedural boilerplate)
    # if constraint_type == "HARD" and not thresholds and not exceptions:
    #     issues.append("HARD with no threshold or exception — review if procedural")

    # Flag missing entities
    if constraint_type in ("HARD", "SOFT"):
        if not entities["subject"] and not entities["authority"]:
            issues.append("No recognizable entities — check annotation")

    # Flag thresholds that had a real value but couldn't be parsed
    for t in thresholds:
        raw_val = str(t.get("raw", "")).strip()
        # Skip nan/empty — not an error
        if is_empty_threshold(raw_val):
            continue
        if t["value"] is None:
            issues.append(f"Could not parse numeric value: {raw_val}")
        if t["unit"] is None and t["variable"] == "unknown":
            issues.append(f"Could not determine unit: {raw_val}")

    # Determine hardness subtype
    hardness_subtype = constraint_type
    if constraint_type == "HARD" and exceptions:
        hardness_subtype = "HARD-CONDITIONAL"
    elif constraint_type == "SOFT":
        if exceptions and exceptions[0]["type"] == "approval_based":
            hardness_subtype = "SOFT-APPROVAL"
        elif exceptions:
            hardness_subtype = "SOFT-CONDITIONAL"

    ir = {
        "constraint_id"    : str(row.get("sentence_id", "")),
        "source_doc"       : str(row.get("source_doc", "")),
        "page_num"         : int(row["page_num"])
                             if pd.notna(row.get("page_num")) else None,
        "raw_text"         : raw_text,
        "constraint_type"  : constraint_type,
        "hardness_subtype" : hardness_subtype,
        "entities"         : entities,
        "thresholds"       : thresholds,
        "exceptions"       : exceptions,
        "quality_issues"   : issues,
        "solver_status"    : None,
        "cpsat_code"       : None,
        "verified"         : False,
        "source_signals"   : str(row.get("signals", "")),
        "confidence_score" : int(row.get("confidence_score", 0)),
        "annotation_source": "human"
                             if "human" in str(row.get("notes", "")).lower()
                             else "auto-gpt4o",
        "created_at"       : datetime.now().strftime("%Y-%m-%d"),
    }

    return ir, issues


def build_ir(csv_path: Path) -> tuple[list[dict], pd.DataFrame]:
    """Load annotated CSV and convert all rows to IR format."""
    df = pd.read_csv(csv_path, dtype={
        "is_constraint"  : "object",
        "constraint_type": "object",
        "entities"       : "object",
        "threshold"      : "object",
        "exception"      : "object",
        "notes"          : "object",
    })

    # Only process annotated constraints
    df = df[
        df["is_constraint"].notna() &
        (df["is_constraint"] == "Yes") &
        df["constraint_type"].notna() &
        (df["constraint_type"] != "")
    ].copy()

    print(f"\n  Processing {len(df)} annotated constraints...")

    ir_records = []
    issue_rows = []

    for _, row in df.iterrows():
        ir, issues = build_ir_record(row)
        ir_records.append(ir)
        if issues:
            for issue in issues:
                issue_rows.append({
                    "constraint_id"  : ir["constraint_id"],
                    "constraint_type": ir["constraint_type"],
                    "issue"          : issue,
                    "raw_text"       : ir["raw_text"][:120],
                })

    issues_df = pd.DataFrame(issue_rows)
    return ir_records, issues_df


def print_stats(ir_records: list[dict], issues_df: pd.DataFrame):
    """Print summary statistics."""
    total     = len(ir_records)
    hard      = sum(1 for r in ir_records if r["constraint_type"] == "HARD")
    soft      = sum(1 for r in ir_records if r["constraint_type"] == "SOFT")
    hard_cond = sum(1 for r in ir_records
                    if r["hardness_subtype"] == "HARD-CONDITIONAL")
    soft_appr = sum(1 for r in ir_records
                    if r["hardness_subtype"] == "SOFT-APPROVAL")
    soft_cond = sum(1 for r in ir_records
                    if r["hardness_subtype"] == "SOFT-CONDITIONAL")

    # Only count records with real parsed numeric thresholds
    with_thresh = sum(
        1 for r in ir_records
        if r["thresholds"] and
        any(t["value"] is not None for t in r["thresholds"])
    )
    with_exc    = sum(1 for r in ir_records if r["exceptions"])
    with_issues = sum(1 for r in ir_records if r["quality_issues"])

    print(f"\n{'='*55}")
    print(f"  CONSTRAINT IR SUMMARY")
    print(f"{'='*55}")
    print(f"  Total IR records       : {total}")
    print(f"  HARD                   : {hard}  ({100*hard//total}%)")
    print(f"    HARD-CONDITIONAL     : {hard_cond}")
    print(f"  SOFT                   : {soft}  ({100*soft//total}%)")
    print(f"    SOFT-APPROVAL        : {soft_appr}")
    print(f"    SOFT-CONDITIONAL     : {soft_cond}")
    print(f"  With thresholds        : {with_thresh}  ({100*with_thresh//total}%)")
    print(f"  With exceptions        : {with_exc}  ({100*with_exc//total}%)")
    print(f"  Quality issues flagged : {with_issues}  ({100*with_issues//total}%)")
    print(f"{'='*55}")

    if not issues_df.empty:
        print(f"\n  Top issue types:")
        for issue, count in issues_df["issue"].value_counts().head(8).items():
            print(f"    {count:>4}  {issue[:70]}")


def main():
    parser = argparse.ArgumentParser(
        description="SOLAR Step 5: Build Constraint IR")
    parser.add_argument(
        "--source",
        help="Path to annotated CSV (default: contract_v5_annotated.csv)")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("SOLAR — Step 5: Build Constraint IR")
    print("="*55)

    csv_path = (Path(args.source) if args.source
                else ANNOTATED_DIR / "contract_v5_annotated.csv")

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    print(f"\n  Source: {csv_path.name}")

    ir_records, issues_df = build_ir(csv_path)

    # Save IR JSON
    ir_path = PROCESSED_DIR / "constraint_ir.json"
    with open(ir_path, "w") as f:
        json.dump(ir_records, f, indent=2)
    print(f"\n  IR saved   → {ir_path.relative_to(PROJECT_ROOT)}")

    # Save issues CSV
    if not issues_df.empty:
        issues_path = PROCESSED_DIR / "constraint_ir_issues.csv"
        issues_df.to_csv(issues_path, index=False)
        print(f"  Issues     → {issues_path.relative_to(PROJECT_ROOT)}")

    print_stats(ir_records, issues_df)

    # Show 3 sample IR records
    print(f"\n  Sample IR records:")
    for ir in ir_records[:3]:
        print(f"\n  [{ir['constraint_type']}] {ir['constraint_id']}")
        print(f"  Text      : {ir['raw_text'][:80]}...")
        print(f"  Subtype   : {ir['hardness_subtype']}")
        print(f"  Entities  : {ir['entities']}")
        print(f"  Thresholds: {ir['thresholds']}")
        print(f"  Exceptions: {ir['exceptions']}")
        if ir["quality_issues"]:
            print(f"  Issues    : {ir['quality_issues']}")

    print(f"\n  Next: python scripts/06_generate_cpsat.py")
    print(f"  Converts IR → CP-SAT code for solver verification\n")


if __name__ == "__main__":
    main()