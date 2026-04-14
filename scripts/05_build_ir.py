"""
SOLAR — Script 05: Build Constraint IR
========================================
Converts annotated CSV rows into structured Constraint IR JSON.

Approach: GPT-4o-mini for structured extraction (no hardcoding).
  - Reads raw_text + annotation fields as hints
  - GPT extracts entities, thresholds, exceptions
  - GPT generates semantic exception variable name from trigger
  - Annotations gate quality: threshold value + type must match
  - Falls back gracefully if GPT call fails

Works across all 8 domains without domain-specific rules.

Output:
  data/processed/{domain}/constraint_ir.json
  data/processed/{domain}/constraint_ir_issues.csv
  data/processed/all_domains_ir.json  (if --domain all)

Usage:
  python scripts/05_build_ir.py --domain transit
  python scripts/05_build_ir.py --domain healthcare
  python scripts/05_build_ir.py --domain all
  python scripts/05_build_ir.py --domain all --limit 100
  python scripts/05_build_ir.py --domain all --dry-run
"""

import re
import os
import json
import time
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from openai import OpenAI

PROJECT_ROOT  = Path(__file__).parent.parent
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DOMAINS = [
    "transit", "healthcare", "education", "municipal",
    "construction", "aviation", "building_services", "hospitality",
]

VALID_TYPES  = {"HARD", "SOFT", "HARD-CONDITIONAL",
                "SOFT-CONDITIONAL", "NON-CONSTRAINT"}
EMPTY_VALUES = {"", "nan", "none", "null", "n/a", "na",
                "empty", "empty string"}

_api_key = os.environ.get("OPENAI_API_KEY", "")
client   = OpenAI(api_key=_api_key) if _api_key else None
GPT_AVAILABLE = bool(_api_key)


def is_empty(val) -> bool:
    return str(val).strip().lower() in EMPTY_VALUES or pd.isna(val)


# ── GPT-4o-mini structured extraction ────────────────────────────────────────

GPT_SYSTEM = """You are a constraint extraction assistant for labor contract analysis.
Given a contract sentence and its annotation hints, extract structured information.
Return only valid JSON. No explanation, no markdown, no backticks."""

GPT_PROMPT = """Extract structured constraint information from this labor contract sentence.

Sentence: "{raw_text}"

Annotation hints (may be incomplete or imprecise — use as guidance only):
  constraint_type: {constraint_type}
  raw_threshold:   {threshold}
  raw_entities:    {entities}
  raw_exception:   {exception}

Return JSON with exactly these fields:
{{
  "entities": {{
    "subject":   ["who must follow this rule — workers, nurses, pilots etc"],
    "authority": ["who enforces or grants exceptions — employer, management etc"],
    "object":    ["what is being regulated — trips, shifts, hours etc"]
  }},
  "thresholds": [
    {{
      "variable":  "descriptive_snake_case_name reflecting the constraint",
      "value":     <number or null>,
      "unit":      "hours/minutes/days/months/percent/usd or null",
      "direction": "max or min or null"
    }}
  ],
  "exceptions": [
    {{
      "trigger":       "exact text of the exception condition from the sentence",
      "type":          "conditional or approval_based or emergency or reference",
      "variable_name": "short semantic snake_case boolean name for this exception"
    }}
  ]
}}

Rules for variable_name (most important):
- Must reflect the MEANING of the exception, not the contract ID
- Must be a boolean condition that is True when exception applies
- 2-4 words max in snake_case
- Examples:
    "except in case of emergency"              → emergency_declared
    "if the employee has no other discipline"  → no_prior_discipline
    "upon the approval of management"          → management_approved
    "unless parties mutually agree"            → mutual_agreement_active
    "where it imperils health or safety"       → health_safety_risk
    "subject to operational requirements"      → operational_requirements_apply
    "beyond the termination of this agreement" → agreement_terminated
- NEVER use: exception_contract_XXXX or any contract ID based name

If no threshold exists return [].
If no exception exists return [].
Return only valid JSON."""


def gpt_extract(raw_text: str, constraint_type: str,
                threshold: str, entities: str,
                exception: str) -> dict | None:
    """Call GPT-4o-mini for structured extraction. Returns dict or None."""
    if not GPT_AVAILABLE or client is None:
        return None
    try:
        prompt = GPT_PROMPT.format(
            raw_text        = raw_text[:500],
            constraint_type = constraint_type,
            threshold       = threshold,
            entities        = entities,
            exception       = exception,
        )
        response = client.chat.completions.create(
            model       = "gpt-4o-mini",
            messages    = [
                {"role": "system", "content": GPT_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature = 0,
            max_tokens  = 600,
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$",     "", raw)
        return json.loads(raw)
    except Exception as e:
        return None


# ── Annotation quality gate ───────────────────────────────────────────────────

def _parse_annotated_value(raw: str) -> float | None:
    """Extract numeric value from raw annotation threshold string."""
    if is_empty(raw):
        return None
    s = str(raw).lower()
    m = re.search(r'\((\d+)\)', s)
    if m:
        return float(m.group(1))
    m = re.search(r'=\s*(\d+\.?\d*)', s)
    if m:
        return float(m.group(1))
    m = re.search(r'\b(\d+\.?\d*)\b', s)
    if m:
        return float(m.group(1))
    return None


def validate_against_annotation(gpt_result: dict,
                                 ann_threshold: str,
                                 ann_type: str) -> tuple[bool, list[str]]:
    """
    Check GPT extraction against annotation ground truth.
    Returns (is_valid, list_of_issues).
    Annotations are the quality gate — if GPT gets the threshold
    value or constraint type wrong, flag it.
    """
    issues = []

    ann_value = _parse_annotated_value(ann_threshold)
    if ann_value is not None and gpt_result.get("thresholds"):
        gpt_values = []
        for t in gpt_result["thresholds"]:
            v = t.get("value")
            if v is None:
                continue
            try:
                gpt_values.append(float(v))
            except (TypeError, ValueError):
                continue
        if gpt_values:
            closest = min(gpt_values,
                          key=lambda v: abs(v - ann_value))
            if abs(closest - ann_value) > ann_value * 0.1 + 1:
                issues.append(
                    f"Threshold mismatch: annotation={ann_value}, "
                    f"GPT={closest}"
                )

    if not is_empty(ann_threshold) and ann_value is not None:
        if not gpt_result.get("thresholds"):
            issues.append(
                f"Annotation has threshold '{ann_threshold}' "
                f"but GPT returned none"
            )

    ann_type_clean = str(ann_type).upper().strip()
    if ann_type_clean in ("HARD", "HARD-CONDITIONAL"):
        gpt_thresholds = gpt_result.get("thresholds", [])
        if gpt_thresholds:
            for t in gpt_thresholds:
                if "slack" in str(t.get("variable", "")).lower():
                    issues.append(
                        "HARD constraint but GPT generated soft/slack variable"
                    )

    is_valid = len(issues) == 0
    return is_valid, issues


# ── Fallback parser (used when GPT unavailable) ───────────────────────────────

def fallback_extract(raw_text: str, constraint_type: str,
                     threshold: str, entities: str,
                     exception: str) -> dict:
    """
    Minimal fallback when GPT is unavailable.
    Produces basic structure — better than nothing,
    but not used for training data.
    """
    exc_list = []
    if not is_empty(exception):
        exc_str = str(exception).strip()
        exc_list = [{
            "trigger":       exc_str,
            "type":          "conditional",
            "variable_name": "exception_unknown",
        }]

    thresh_list = []
    if not is_empty(threshold):
        ann_val = _parse_annotated_value(str(threshold))
        if ann_val is not None:
            thresh_list = [{
                "variable":  "duration_unknown",
                "value":     ann_val,
                "unit":      None,
                "direction": None,
            }]

    return {
        "entities":   {"subject": [], "authority": [], "object": []},
        "thresholds": thresh_list,
        "exceptions": exc_list,
    }


# ── IR record builder ─────────────────────────────────────────────────────────

def build_ir_record(row: pd.Series,
                    use_gpt: bool = True,
                    dry_run: bool = False) -> tuple[dict, list[str]]:
    """Convert one annotated CSV row into a Constraint IR dict."""
    issues      = []
    raw_text    = str(row.get("raw_text", "") or "")
    ctype       = str(row.get("constraint_type", "")).upper().strip()
    threshold   = str(row.get("threshold", "") or "")
    entities    = str(row.get("entities", "") or "")
    exception   = str(row.get("exception", "") or "")

    hardness_subtype = ctype
    if ctype == "HARD" and not is_empty(exception):
        hardness_subtype = "HARD-CONDITIONAL"
    elif ctype == "SOFT":
        exc_lower = exception.lower()
        if any(kw in exc_lower for kw in
               ["approval", "approved", "authorize"]):
            hardness_subtype = "SOFT-APPROVAL"
        elif not is_empty(exception):
            hardness_subtype = "SOFT-CONDITIONAL"

    if dry_run:
        extracted = fallback_extract(
            raw_text, ctype, threshold, entities, exception
        )
        gpt_used = False
    elif use_gpt and GPT_AVAILABLE:
        extracted = gpt_extract(
            raw_text, ctype, threshold, entities, exception
        )
        gpt_used  = extracted is not None
        if extracted is None:
            extracted = fallback_extract(
                raw_text, ctype, threshold, entities, exception
            )
            issues.append("GPT extraction failed — used fallback")
    else:
        extracted = fallback_extract(
            raw_text, ctype, threshold, entities, exception
        )
        gpt_used = False

    if gpt_used:
        is_valid, gate_issues = validate_against_annotation(
            extracted, threshold, ctype
        )
        if not is_valid:
            issues.extend(gate_issues)

    ir = {
        "constraint_id"     : str(row.get("sentence_id", "")),
        "source_doc"        : str(row.get("source_doc", "")),
        "domain"            : str(row.get("domain", "")),
        "page_num"          : (int(row["page_num"])
                               if pd.notna(row.get("page_num")) else None),
        "raw_text"          : raw_text,
        "constraint_type"   : ctype,
        "hardness_subtype"  : hardness_subtype,
        "entities"          : extracted.get("entities",
                                            {"subject": [],
                                             "authority": [],
                                             "object": []}),
        "thresholds"        : extracted.get("thresholds", []),
        "exceptions"        : extracted.get("exceptions", []),
        "quality_issues"    : issues,
        "annotation_valid"  : bool(row.get("annotation_valid", True)),
        "gpt_extracted"     : gpt_used,
        "solver_status"     : None,
        "cpsat_code"        : None,
        "verified"          : False,
        "source_signals"    : str(row.get("signals", "")),
        "confidence_score"  : int(row.get("confidence_score", 0)
                                  if pd.notna(
                                      row.get("confidence_score")) else 0),
        "split"             : str(row.get("split", "train")),
        "created_at"        : datetime.now().strftime("%Y-%m-%d"),
    }

    return ir, issues


# ── Domain processing ─────────────────────────────────────────────────────────

def process_domain(domain: str,
                   limit: int | None = None,
                   use_gpt: bool = True,
                   dry_run: bool = False,
                   valid_only: bool = True) -> list[dict]:
    """Load annotated CSV for a domain and build IR records."""

    ann_dir = ANNOTATED_DIR / domain
    if not ann_dir.exists():
        print(f"  SKIP {domain} — no annotated directory")
        return []

    ann_files = sorted(ann_dir.glob("*_annotated.csv"))
    if not ann_files:
        print(f"  SKIP {domain} — no annotated files")
        return []

    frames = []
    for f in ann_files:
        try:
            df = pd.read_csv(f, dtype={
                "is_constraint":   "object",
                "constraint_type": "object",
                "entities":        "object",
                "threshold":       "object",
                "exception":       "object",
                "notes":           "object",
            })
            df["domain"] = domain
            frames.append(df)
        except Exception as e:
            print(f"  ERROR {f.name}: {e}")

    if not frames:
        return []

    df = pd.concat(frames, ignore_index=True)

    df = df[
        df["is_constraint"].notna() &
        (df["is_constraint"] == "Yes") &
        df["constraint_type"].notna() &
        (df["constraint_type"].str.upper().isin(
            {"HARD", "SOFT", "HARD-CONDITIONAL",
             "SOFT-CONDITIONAL", "NON-CONSTRAINT"}))
    ].copy()

    if valid_only and "annotation_valid" in df.columns:
        before = len(df)
        df = df[df["annotation_valid"] != False].copy()
        removed = before - len(df)
        if removed > 0:
            print(f"  {domain}: removed {removed} "
                  f"invalid annotations")

    if limit:
        df = df.head(limit)

    print(f"  {domain:<20} processing {len(df)} constraints...")

    ir_records = []
    issue_rows = []
    gpt_fails  = 0

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 100 == 0 and i > 0:
            print(f"    {i}/{len(df)}...")

        ir, issues = build_ir_record(row, use_gpt=use_gpt,
                                      dry_run=dry_run)

        if not ir["gpt_extracted"] and use_gpt and not dry_run:
            gpt_fails += 1

        ir_records.append(ir)

        for issue in issues:
            issue_rows.append({
                "constraint_id"  : ir["constraint_id"],
                "domain"         : domain,
                "constraint_type": ir["constraint_type"],
                "issue"          : issue,
                "raw_text"       : ir["raw_text"][:120],
            })

        if use_gpt and not dry_run and i % 50 == 49:
            time.sleep(0.5)

    out_dir = PROCESSED_DIR / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    ir_path = out_dir / "constraint_ir.json"
    with open(ir_path, "w") as f:
        json.dump(ir_records, f, indent=2)

    if issue_rows:
        issues_df  = pd.DataFrame(issue_rows)
        issues_path = out_dir / "constraint_ir_issues.csv"
        issues_df.to_csv(issues_path, index=False)

    gpt_count = sum(1 for r in ir_records if r["gpt_extracted"])
    exc_count = sum(1 for r in ir_records if r["exceptions"])
    thr_count = sum(1 for r in ir_records if r["thresholds"])
    sem_count = sum(
        1 for r in ir_records
        if r["exceptions"] and
        r["exceptions"][0].get("variable_name", "")
        not in ("exception_unknown", "", None) and
        "exception_contract" not in
        r["exceptions"][0].get("variable_name", "")
    )

    print(f"  {domain:<20} done  "
          f"{len(ir_records)} IR records  "
          f"GPT:{gpt_count}  "
          f"thresh:{thr_count}  "
          f"exc:{exc_count}  "
          f"semantic_vars:{sem_count}")

    return ir_records


# ── Stats ─────────────────────────────────────────────────────────────────────

def print_stats(all_records: list[dict]):
    total     = len(all_records)
    gpt       = sum(1 for r in all_records if r["gpt_extracted"])
    with_thr  = sum(1 for r in all_records if r["thresholds"])
    with_exc  = sum(1 for r in all_records if r["exceptions"])
    with_sem  = sum(
        1 for r in all_records
        if r["exceptions"] and
        r["exceptions"][0].get("variable_name", "")
        not in ("exception_unknown", "", None) and
        "exception_contract" not in
        r["exceptions"][0].get("variable_name", "")
    )
    with_issues = sum(1 for r in all_records if r["quality_issues"])

    print(f"\n{'='*60}")
    print(f"  CONSTRAINT IR SUMMARY")
    print(f"{'='*60}")
    print(f"  Total IR records      : {total:,}")
    print(f"  GPT extracted         : {gpt:,}  ({100*gpt//total}%)")
    print(f"  With thresholds       : {with_thr:,}  ({100*with_thr//total}%)")
    print(f"  With exceptions       : {with_exc:,}  ({100*with_exc//total}%)")
    print(f"  Semantic var names    : {with_sem:,}  ({100*with_sem//with_exc if with_exc else 0}% of exc)")
    print(f"  Quality issues        : {with_issues:,}")
    print(f"\n  By domain:")

    domain_counts: dict = {}
    for r in all_records:
        d = r["domain"]
        if d not in domain_counts:
            domain_counts[d] = {"total": 0, "exc": 0, "sem": 0}
        domain_counts[d]["total"] += 1
        if r["exceptions"]:
            domain_counts[d]["exc"] += 1
            vn = r["exceptions"][0].get("variable_name", "")
            if vn and "exception_contract" not in vn \
                    and vn != "exception_unknown":
                domain_counts[d]["sem"] += 1

    for domain, counts in domain_counts.items():
        sem_pct = (100 * counts["sem"] // counts["exc"]
                   if counts["exc"] else 0)
        print(f"    {domain:<20} {counts['total']:>5}  "
              f"exc:{counts['exc']:>4}  "
              f"semantic:{counts['sem']:>4} ({sem_pct}%)")

    print(f"{'='*60}")

    print(f"\n  Sample semantic variable names:")
    shown = 0
    for r in all_records:
        if not r["exceptions"]:
            continue
        vn = r["exceptions"][0].get("variable_name", "")
        if vn and "exception_contract" not in vn \
                and vn != "exception_unknown":
            trigger = r["exceptions"][0].get("trigger", "")[:60]
            print(f"    '{trigger}'")
            print(f"    → {vn}\n")
            shown += 1
            if shown >= 5:
                break


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SOLAR Step 05: Build Constraint IR")
    parser.add_argument(
        "--domain", default="transit",
        help="Domain to process, or 'all' for all domains")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N records per domain (for testing)")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip GPT calls, use fallback parser (fast, for testing)")
    parser.add_argument(
        "--no-gpt", action="store_true",
        help="Disable GPT, use fallback parser only")
    args = parser.parse_args()

    use_gpt = not args.no_gpt and not args.dry_run

    print("\n" + "="*60)
    print("SOLAR — Step 05: Build Constraint IR")
    print("="*60)
    print(f"\n  Domain  : {args.domain}")
    print(f"  GPT     : {'enabled (gpt-4o-mini)' if use_gpt else 'disabled (fallback)'}")
    print(f"  Dry run : {args.dry_run}")
    if args.limit:
        print(f"  Limit   : {args.limit} per domain")

    if use_gpt and not GPT_AVAILABLE:
        print("\n  WARNING: OPENAI_API_KEY not set.")
        print("  Set it with: export OPENAI_API_KEY=your_key")
        print("  Falling back to rule-based parser.\n")
        use_gpt = False

    if args.domain == "all":
        domains = DOMAINS
    else:
        domains = [args.domain]

    all_records = []
    for domain in domains:
        print(f"\n  [{domain}]")
        records = process_domain(
            domain,
            limit   = args.limit,
            use_gpt = use_gpt,
            dry_run = args.dry_run,
        )
        all_records.extend(records)

    if args.domain == "all" and all_records:
        out_path = PROCESSED_DIR / "all_domains_ir.json"
        with open(out_path, "w") as f:
            json.dump(all_records, f, indent=2)
        print(f"\n  Combined IR → data/processed/all_domains_ir.json")

    if all_records:
        print_stats(all_records)

    print(f"\n  Next: python scripts/06_gpt_enrich_ir.py --domain all")
    print(f"        (enriches exception variable names)\n")


if __name__ == "__main__":
    main()