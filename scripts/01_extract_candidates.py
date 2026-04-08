"""
SOLAR — Script 01: Extract Candidate Constraint Sentences
==========================================================
Reads all PDFs in data/raw/ subfolders and extracts sentences
that are likely to be operational constraints, based on:
  - Deontic modal verbs (shall, must, may, will not...)
  - Quantitative threshold patterns (X hours, Y minutes, N days...)
  - Exception markers (except, unless, provided that...)
  - Prohibition markers (no employee, not permitted...)

Output per PDF:
  data/processed/<doc_name>_candidates.csv   — candidate sentences
  data/processed/<doc_name>_stats.txt        — extraction summary

Usage:
  python scripts/01_extract_candidates.py
  python scripts/01_extract_candidates.py --source fta_regulations
  python scripts/01_extract_candidates.py --file "TWU_2023.pdf"
"""

import os
import re
import csv
import argparse
from pathlib import Path
from datetime import datetime

# ── Try importing pdfplumber, give clear install message if missing ──────────
try:
    import pdfplumber
except ImportError:
    print("ERROR: pdfplumber not installed.")
    print("Run: pip install pdfplumber pandas openpyxl tqdm")
    exit(1)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed.")
    print("Run: pip install pandas")
    exit(1)

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw" 
PROCESSED_DIR= PROJECT_ROOT / "data" / "processed" 
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Constraint detection patterns ─────────────────────────────────────────────
# Each pattern has a name, regex, and predicted constraint type

DEONTIC_PATTERNS = [
    # Hard mandatory
    (r'\bshall\s+not\b',           "HARD",   "prohibition"),
    (r'\bmust\s+not\b',            "HARD",   "prohibition"),
    (r'\bshall\b',                 "HARD",   "obligation"),
    (r'\bmust\b',                  "HARD",   "obligation"),
    (r'\bis\s+required\s+to\b',    "HARD",   "obligation"),
    (r'\bare\s+required\s+to\b',   "HARD",   "obligation"),
    (r'\bwill\s+not\s+be\s+permitted\b', "HARD", "prohibition"),
    (r'\bno\s+(?:employee|operator|driver|worker|member)\b', "HARD", "prohibition"),
    # Soft preference
    (r'\bshould\b',                "SOFT",   "preference"),
    (r'\bmay\s+not\b',             "SOFT",   "restriction"),
    (r'\bmay\b',                   "SOFT",   "permission"),
    (r'\bwhere\s+possible\b',      "SOFT",   "preference"),
    (r'\bwhenever\s+practicable\b',"SOFT",   "preference"),
    (r'\bas\s+far\s+as\s+possible\b',"SOFT", "preference"),
]

THRESHOLD_PATTERNS = [
    r'\b\d+\s*(?:hours?|hrs?)\b',
    r'\b\d+\s*(?:minutes?|mins?)\b',
    r'\b\d+\s*(?:days?)\b',
    r'\b\d+\s*(?:weeks?)\b',
    r'\b\d+\s*(?:months?)\b',
    r'\b\d+\s*(?:percent|%)\b',
    r'\bno\s+(?:more|less)\s+than\s+\d+\b',
    r'\bat\s+least\s+\d+\b',
    r'\bnot\s+(?:more|less)\s+than\s+\d+\b',
    r'\bmaximum\s+of\s+\d+\b',
    r'\bminimum\s+of\s+\d+\b',
    r'\bnot\s+to\s+exceed\s+\d+\b',
]

EXCEPTION_PATTERNS = [
    r'\bexcept\b',
    r'\bunless\b',
    r'\bprovided\s+that\b',
    r'\bnotwithstanding\b',
    r'\bin\s+cases?\s+of\b',
    r'\bin\s+the\s+event\s+of\b',
    r'\bsubject\s+to\b',
]

def clean_text(text):
    """Clean raw PDF text — remove extra whitespace, fix hyphenation."""
    text = re.sub(r'-\n', '', text)          # fix hyphenated line breaks
    text = re.sub(r'\n+', ' ', text)         # collapse newlines
    text = re.sub(r'\s{2,}', ' ', text)      # collapse multiple spaces
    return text.strip()

def split_sentences(text):
    """Simple sentence splitter — handles legal text better than nltk."""
    # Split on period/semicolon followed by space and capital letter
    # but preserve abbreviations like "Section 2.3" or "No. 4"
    sentences = re.split(r'(?<=[.;])\s+(?=[A-Z\(])', text)
    # Also split on numbered list items: "1. No employee..."
    expanded = []
    for s in sentences:
        parts = re.split(r'(?<=\s)(?=\(\w\)\s)', s)  # split on "(a) (b)..."
        expanded.extend(parts)
    return [s.strip() for s in expanded if len(s.strip()) > 20]

def detect_constraints(sentence):
    """
    Check a sentence for constraint signals.
    Returns: (is_constraint, predicted_type, signals_found)
    """
    sentence_lower = sentence.lower()
    signals = []
    predicted_type = None

    # Check deontic patterns
    for pattern, ctype, label in DEONTIC_PATTERNS:
        if re.search(pattern, sentence_lower):
            signals.append(f"deontic:{label}")
            if predicted_type is None:
                predicted_type = ctype

    # Check threshold patterns
    for pattern in THRESHOLD_PATTERNS:
        if re.search(pattern, sentence_lower):
            signals.append("threshold")
            break

    # Check exception patterns
    for pattern in EXCEPTION_PATTERNS:
        if re.search(pattern, sentence_lower):
            signals.append("exception")
            break

    is_constraint = len(signals) > 0

    # If we have both deontic AND threshold — stronger signal
    has_deontic   = any(s.startswith("deontic") for s in signals)
    has_threshold = "threshold" in signals
    has_exception = "exception" in signals

    # Confidence score (rough)
    score = 0
    if has_deontic:   score += 2
    if has_threshold: score += 1
    if has_exception: score += 1

    return is_constraint, predicted_type or "UNCLEAR", signals, score

def extract_from_pdf(pdf_path):
    """Extract candidate constraint sentences from a single PDF."""
    print(f"\n  Processing: {pdf_path.name}")
    candidates = []
    total_sentences = 0
    total_pages = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"  Pages: {total_pages}")

            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if not text:
                    continue

                text = clean_text(text)
                sentences = split_sentences(text)
                total_sentences += len(sentences)

                for sent in sentences:
                    is_constraint, pred_type, signals, score = detect_constraints(sent)
                    if is_constraint:
                        candidates.append({
                            "sentence_id":       "",          # filled after dedup
                            "source_doc":        pdf_path.stem,
                            "page_num":          page_num,
                            "raw_text":          sent,
                            "predicted_type":    pred_type,   # model's guess
                            "signals":           "|".join(signals),
                            "confidence_score":  score,
                            # ── Human annotation columns (fill these in) ──
                            "is_constraint":     "",          # Yes / No
                            "constraint_type":   "",          # HARD / SOFT / UNCLEAR
                            "entities":          "",          # e.g. operator, vehicle
                            "threshold":         "",          # e.g. max_hours=6
                            "exception":         "",          # exception text if any
                            "notes":             "",          # your notes
                        })

    except Exception as e:
        print(f"  ERROR reading {pdf_path.name}: {e}")
        return [], 0, 0

    # Remove duplicates (same sentence appearing on multiple pages)
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = re.sub(r'\s+', ' ', c["raw_text"].lower()[:80])
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    # Assign sentence IDs
    doc_prefix = pdf_path.stem[:8].upper().replace(" ", "_")
    for i, c in enumerate(unique_candidates):
        c["sentence_id"] = f"{doc_prefix}_{i+1:04d}"

    print(f"  Total sentences scanned : {total_sentences}")
    print(f"  Candidate constraints   : {len(unique_candidates)}")

    return unique_candidates, total_sentences, total_pages

def save_candidates(candidates, doc_name):
    """Save candidates to CSV for annotation."""
    out_path = PROCESSED_DIR / f"{doc_name}_candidates.csv"

    if not candidates:
        print(f"  No candidates found for {doc_name}")
        return

    df = pd.DataFrame(candidates)

    # Sort by confidence score descending — review best candidates first
    df = df.sort_values("confidence_score", ascending=False)

    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  Saved → {out_path.relative_to(PROJECT_ROOT)}")
    return out_path

def save_stats(doc_name, candidates, total_sentences, total_pages):
    """Save extraction statistics."""
    stats_path = PROCESSED_DIR / f"{doc_name}_stats.txt"
    hard_count = sum(1 for c in candidates if c["predicted_type"] == "HARD")
    soft_count = sum(1 for c in candidates if c["predicted_type"] == "SOFT")
    high_conf  = sum(1 for c in candidates if c["confidence_score"] >= 3)

    with open(stats_path, "w") as f:
        f.write(f"SOLAR Extraction Stats — {doc_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Pages processed       : {total_pages}\n")
        f.write(f"Total sentences       : {total_sentences}\n")
        f.write(f"Candidate constraints : {len(candidates)}\n")
        f.write(f"  → Predicted HARD    : {hard_count}\n")
        f.write(f"  → Predicted SOFT    : {soft_count}\n")
        f.write(f"High confidence (≥3)  : {high_conf}\n\n")
        f.write("Top 10 highest confidence candidates:\n")
        f.write("-" * 50 + "\n")
        top10 = sorted(candidates, key=lambda x: -x["confidence_score"])[:10]
        for i, c in enumerate(top10, 1):
            f.write(f"\n{i}. [{c['predicted_type']}] score={c['confidence_score']}\n")
            f.write(f"   {c['raw_text'][:120]}...\n" if len(c['raw_text']) > 120
                    else f"   {c['raw_text']}\n")

    print(f"  Stats  → {stats_path.relative_to(PROJECT_ROOT)}")

def get_pdf_files(source_filter=None, file_filter=None):
    """Get list of PDFs to process."""
    pdf_files = []

    if file_filter:
        # Search across all raw subdirs
        for subdir in RAW_DIR.iterdir():
            if subdir.is_dir():
                match = subdir / file_filter
                if match.exists():
                    pdf_files.append(match)
    elif source_filter:
        subdir = RAW_DIR / source_filter
        if subdir.exists():
            pdf_files = list(subdir.glob("*.pdf"))
        else:
            print(f"ERROR: Source folder not found: {subdir}")
    else:
        # Process all PDFs in all raw subdirs
        for subdir in RAW_DIR.iterdir():
            if subdir.is_dir():
                pdf_files.extend(subdir.glob("*.pdf"))

    return sorted(pdf_files)

def main():
    parser = argparse.ArgumentParser(description="SOLAR Step 1: Extract constraint candidates from PDFs")
    parser.add_argument("--source", help="Subfolder to process: contracts / fta_regulations / gtfs")
    parser.add_argument("--file",   help="Process a single file by name, e.g. 'TWU_2023.pdf'")
    parser.add_argument("--domain", default="transit", help="Domain: transit/healthcare/education/building_services/hospitality/municipal")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("SOLAR — Step 1: Constraint Candidate Extraction")
    print("="*60)

    global RAW_DIR, PROCESSED_DIR
    RAW_DIR = PROJECT_ROOT / "data" / "raw" / args.domain
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / args.domain
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = get_pdf_files(args.source, args.file)

    if not pdf_files:
        print(f"\nNo PDFs found.")
        print(f"Place your PDF files in:")
        print(f"  {RAW_DIR / 'contracts'}      ← CBA documents")
        print(f"  {RAW_DIR / 'fta_regulations'} ← FTA regulatory docs")
        print(f"  {RAW_DIR / 'gtfs'}            ← GTFS spec files")
        return

    print(f"\nFound {len(pdf_files)} PDF(s) to process\n")

    all_stats = []
    for pdf_path in pdf_files:
        candidates, total_sents, total_pages = extract_from_pdf(pdf_path)
        if candidates:
            save_candidates(candidates, pdf_path.stem)
            save_stats(pdf_path.stem, candidates, total_sents, total_pages)
            all_stats.append({
                "document": pdf_path.name,
                "pages": total_pages,
                "sentences": total_sents,
                "candidates": len(candidates)
            })

    # Summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    if all_stats:
        total_cands = sum(s["candidates"] for s in all_stats)
        print(f"\nDocuments processed : {len(all_stats)}")
        print(f"Total candidates    : {total_cands}")
        print(f"\nNext step:")
        print(f"  Open data/processed/*_candidates.csv")
        print(f"  Fill in the annotation columns:")
        print(f"    is_constraint   → Yes / No")
        print(f"    constraint_type → HARD / SOFT / UNCLEAR")
        print(f"    entities        → e.g. 'operator, shift'")
        print(f"    threshold       → e.g. 'max_hours=6'")
        print(f"    exception       → exception text if any")
        print(f"\nOr run: python scripts/02_annotate.py  (coming next)")

if __name__ == "__main__":
    main()
