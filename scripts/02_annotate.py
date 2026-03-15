"""
SOLAR — Script 02: Annotation Helper
=====================================
Loads a candidates CSV and walks you through each sentence
one by one, letting you confirm/correct the predicted label.

This replaces manual spreadsheet work — you just press keys
to annotate and the script saves your progress automatically.

Usage:
  python scripts/02_annotate.py
  python scripts/02_annotate.py --file data/processed/TWU_2023_candidates.csv
  python scripts/02_annotate.py --resume   ← continue where you left off
"""

import os
import re
import csv
import argparse
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    print("ERROR: pip install pandas")
    exit(1)

PROJECT_ROOT  = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def wrap(text, width=80):
    words = text.split()
    lines, line = [], []
    for w in words:
        if sum(len(x)+1 for x in line) + len(w) > width:
            lines.append(' '.join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(' '.join(line))
    return '\n'.join(lines)

def pick_file():
    """Let user choose which candidates file to annotate."""
    files = sorted(PROCESSED_DIR.glob("*_candidates.csv"))
    if not files:
        print(f"\nNo candidate files found in {PROCESSED_DIR}")
        print("Run python scripts/01_extract_candidates.py first")
        return None
    print("\nAvailable candidate files:")
    for i, f in enumerate(files, 1):
        df = pd.read_csv(f)
        done = df['is_constraint'].notna().sum()
        total = len(df)
        print(f"  {i}. {f.name}  [{done}/{total} annotated]")
    choice = input("\nEnter number to annotate: ").strip()
    try:
        return files[int(choice)-1]
    except:
        print("Invalid choice")
        return None

def annotate(csv_path, resume=False):
    """Main annotation loop."""
    df = pd.read_csv(csv_path)
    total = len(df)
    out_path = ANNOTATED_DIR / csv_path.name.replace("_candidates", "_annotated")

    # Load existing annotations if resuming
    if resume and out_path.exists():
        existing = pd.read_csv(out_path)
        df.update(existing)
        print(f"\nResuming from existing annotations in {out_path.name}")

    # Find first unannotated row
    start_idx = 0
    if resume:
        unannotated = df[df['is_constraint'].isna() | (df['is_constraint'] == '')].index
        start_idx = unannotated[0] if len(unannotated) > 0 else total

    annotated_count = df['is_constraint'].notna().sum()

    print(f"\n{'='*60}")
    print(f"SOLAR Annotation Tool")
    print(f"File   : {csv_path.name}")
    print(f"Total  : {total} candidates")
    print(f"Done   : {annotated_count}")
    print(f"{'='*60}")
    print("\nControls:")
    print("  h = HARD constraint   s = SOFT constraint")
    print("  n = Not a constraint  u = UNCLEAR")
    print("  b = Go back           q = Save & quit")
    print("\nPress Enter to start...", end="")
    input()

    i = start_idx
    while i < total:
        row = df.iloc[i]
        clear()

        # Progress bar
        pct = int((i / total) * 40)
        bar = "█" * pct + "░" * (40 - pct)
        print(f"\n[{bar}] {i+1}/{total}  |  Annotated: {annotated_count}")
        print("="*60)

        # Show sentence
        print(f"\nSource  : {row['source_doc']}  (page {row.get('page_num', '?')})")
        print(f"ID      : {row['sentence_id']}")
        print(f"\nSentence:\n")
        print(f"  {wrap(row['raw_text'], 72)}")
        print(f"\nPredicted type : {row['predicted_type']}")
        print(f"Signals found  : {row.get('signals', '')}")
        print(f"Confidence     : {row.get('confidence_score', '?')}/4")

        # Show existing annotation if any
        if pd.notna(row['is_constraint']) and row['is_constraint'] != '':
            print(f"\nCurrent label  : [{row['constraint_type']}] {row['is_constraint']}")

        print(f"\n{'─'*60}")
        print("  h=HARD  s=SOFT  n=Not constraint  u=UNCLEAR  b=Back  q=Quit")
        print("  > ", end="", flush=True)

        key = input().strip().lower()

        if key == 'q':
            break
        elif key == 'b':
            i = max(0, i - 1)
            continue
        elif key in ('h', 's', 'n', 'u'):
            # Map key to values
            mapping = {
                'h': ('Yes', 'HARD'),
                's': ('Yes', 'SOFT'),
                'n': ('No',  'NOT_CONSTRAINT'),
                'u': ('Yes', 'UNCLEAR'),
            }
            is_c, ctype = mapping[key]
            df.at[df.index[i], 'is_constraint']   = is_c
            df.at[df.index[i], 'constraint_type'] = ctype

            # For hard constraints, prompt for quick details
            if key == 'h':
                print(f"\n  Entities (e.g. 'operator, shift') or Enter to skip: ", end="")
                ents = input().strip()
                if ents:
                    df.at[df.index[i], 'entities'] = ents

                print(f"  Threshold (e.g. 'max_hours=6') or Enter to skip: ", end="")
                thresh = input().strip()
                if thresh:
                    df.at[df.index[i], 'threshold'] = thresh

                print(f"  Exception clause? (y/n): ", end="")
                exc = input().strip().lower()
                if exc == 'y':
                    print(f"  Exception text: ", end="")
                    exc_text = input().strip()
                    df.at[df.index[i], 'exception'] = exc_text

            annotated_count += 1

            # Autosave every 10 annotations
            if annotated_count % 10 == 0:
                df.to_csv(out_path, index=False)
                print(f"\n  ✓ Autosaved ({annotated_count} annotations)")

            i += 1
        else:
            # Invalid key — stay on same sentence
            continue

    # Final save
    df.to_csv(out_path, index=False)
    done_final = df[df['is_constraint'].isin(['Yes','No'])].shape[0]
    hard_final = df[df['constraint_type'] == 'HARD'].shape[0]
    soft_final = df[df['constraint_type'] == 'SOFT'].shape[0]

    print(f"\n{'='*60}")
    print(f"Session saved → {out_path.relative_to(PROJECT_ROOT)}")
    print(f"Annotated this session : {annotated_count}")
    print(f"Total annotated        : {done_final} / {total}")
    print(f"  HARD   : {hard_final}")
    print(f"  SOFT   : {soft_final}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser(description="SOLAR Step 2: Annotation helper")
    parser.add_argument("--file",   help="Path to a specific candidates CSV")
    parser.add_argument("--resume", action="store_true", help="Resume from last saved position")
    args = parser.parse_args()

    if args.file:
        csv_path = Path(args.file)
    else:
        csv_path = pick_file()

    if csv_path and csv_path.exists():
        annotate(csv_path, resume=args.resume)
    else:
        print("File not found.")

if __name__ == "__main__":
    main()
