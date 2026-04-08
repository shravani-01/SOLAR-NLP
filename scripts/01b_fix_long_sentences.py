"""
SOLAR — Script 01b: Fix Long Sentences
========================================
Takes the existing candidates CSV and splits sentences
that are over 80 words into individual constraint sentences.

Handles legal text patterns like:
  "1) ... 2) ... 3) ..."
  "a) ... b) ... c) ..."
  "; provided that ..."
  "; except that ..."

Usage:
  python scripts/01b_fix_long_sentences.py
"""

import re
import pandas as pd
from pathlib import Path

PROJECT_ROOT  = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" /"municipal"

WORD_LIMIT = 80  # sentences over this get split

def split_legal_sentence(text):
    """
    Split a long legal sentence into individual constraint sentences.
    Handles numbered/lettered lists and semicolon-separated clauses.
    """
    # Pattern 1: numbered items "1) ... 2) ... 3) ..."
    numbered = re.split(r'(?=\d+\)\s)', text)
    if len(numbered) > 1:
        return [s.strip() for s in numbered if len(s.strip()) > 15]

    # Pattern 2: lettered items "a) ... b) ... c) ..."
    lettered = re.split(r'(?=[a-z]\)\s)', text)
    if len(lettered) > 1:
        return [s.strip() for s in lettered if len(s.strip()) > 15]

    # Pattern 3: bullet points "• ..."
    bulleted = re.split(r'\s*[•·]\s*', text)
    if len(bulleted) > 1:
        return [s.strip() for s in bulleted if len(s.strip()) > 15]

    # Pattern 4: semicolons before key legal words
    semi_split = re.split(
        r';\s*(?=(?:provided|except|unless|and\s+(?:shall|must|may)|'
        r'such\s+employee|the\s+(?:authority|union|employee)))',
        text, flags=re.IGNORECASE
    )
    if len(semi_split) > 1:
        return [s.strip() for s in semi_split if len(s.strip()) > 15]

    # Pattern 5: period followed by capital letter
    period_split = re.split(r'\.\s+(?=[A-Z\(])', text)
    if len(period_split) > 1:
        splits = [s.strip() for s in period_split if len(s.strip()) > 15]
        if all(len(s.split()) < WORD_LIMIT for s in splits):
            return splits

    # Can't split further — return as-is
    return [text]

def fix_candidates(csv_path):
    """Load candidates CSV, split long sentences, save updated version."""
    print(f"\nProcessing: {csv_path.name}")
    df = pd.read_csv(csv_path)
    original_count = len(df)

    df['word_count'] = df['raw_text'].str.split().str.len()

    short_df = df[df['word_count'] <= WORD_LIMIT].copy()
    long_df  = df[df['word_count'] >  WORD_LIMIT].copy()

    print(f"  Total candidates    : {original_count}")
    print(f"  Already short (≤{WORD_LIMIT}w): {len(short_df)}")
    print(f"  Need splitting (>{WORD_LIMIT}w): {len(long_df)}")

    if len(long_df) == 0:
        print("  Nothing to fix!")
        return

    # Split long sentences
    new_rows = []
    split_count = 0

    for _, row in long_df.iterrows():
        parts = split_legal_sentence(row['raw_text'])
        if len(parts) > 1:
            split_count += 1
            for j, part in enumerate(parts):
                wc = len(part.split())
                if wc < 8:
                    continue
                new_row = row.copy()
                new_row['raw_text']    = part
                new_row['word_count']  = wc
                new_row['sentence_id'] = f"{row['sentence_id']}_p{j+1}"
                new_rows.append(new_row)
        else:
            row['notes'] = 'LONG_UNSPLIT'
            new_rows.append(row)

    split_df = pd.DataFrame(new_rows)

    # Combine short (unchanged) + newly split
    final_df = pd.concat([short_df, split_df], ignore_index=True)
    final_df = final_df.drop(columns=['word_count'])

    # Re-check lengths
    final_df['word_count'] = final_df['raw_text'].str.split().str.len()
    still_long = (final_df['word_count'] > WORD_LIMIT).sum()

    print(f"\n  Sentences split     : {split_count}")
    print(f"  New total           : {len(final_df)}")
    print(f"  Still over {WORD_LIMIT}w     : {still_long}")

    print(f"\n  New length distribution:")
    print(f"    Under 50 words  : {(final_df.word_count < 50).sum()}")
    print(f"    50-80 words     : {((final_df.word_count >= 50) & (final_df.word_count <= 80)).sum()}")
    print(f"    80-150 words    : {((final_df.word_count > 80) & (final_df.word_count <= 150)).sum()}")
    print(f"    Over 150 words  : {(final_df.word_count > 150).sum()}")

    final_df = final_df.drop(columns=['word_count'])
    final_df.to_csv(csv_path, index=False)
    print(f"\n  Saved → {csv_path.relative_to(PROJECT_ROOT)}")

def main():
    csv_files = list(PROCESSED_DIR.glob("*_candidates.csv"))

    if not csv_files:
        print("No candidate files found. Run 01_extract_candidates.py first.")
        return

    print("=" * 55)
    print("SOLAR — Step 1b: Fix Long Sentences")
    print("=" * 55)

    for csv_path in csv_files:
        fix_candidates(csv_path)

    print("\n" + "=" * 55)
    print("Done. Now run: python scripts/02_annotate.py")
    print("=" * 55)

if __name__ == "__main__":
    main()