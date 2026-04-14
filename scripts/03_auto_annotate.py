"""
SOLAR — Script 03: Auto-Annotate using GPT-4o or Groq
=======================================================
Uses OpenAI GPT-4o (preferred) or Groq Llama-3 (free fallback)
to auto-label candidate constraint sentences.

Usage:
  python scripts/03_auto_annotate.py                  # uses GPT-4o
  python scripts/03_auto_annotate.py --model groq     # uses Groq
  python scripts/03_auto_annotate.py --dry-run        # preview only
  python scripts/03_auto_annotate.py --confidence 2   # include score-2
"""

import os
import re
import time
import json
import argparse
import pandas as pd
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("ERROR: pip install python-dotenv")
    exit(1)

PROJECT_ROOT  = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "manufacturing"
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated" / "manufacturing"
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """You are an expert in transit labor law and operational constraints.
You are annotating sentences from transit collective bargaining agreements (CBAs)
and regulatory documents for an NLP research project.

Your job is to classify each sentence and extract constraint information.

Respond ONLY with a valid JSON object — no explanation, no markdown, no extra text.
"""

def build_prompt(sentence):
    return f"""Classify this sentence from a transit labor contract:

SENTENCE: "{sentence}"

Respond with exactly this JSON structure:
{{
  "is_constraint": "Yes" or "No",
  "constraint_type": "HARD" or "SOFT" or "NOT_CONSTRAINT",
  "entities": "comma-separated entities e.g. operator, authority, union",
  "threshold": "any numeric thresholds e.g. max_hours=6, min_break=30min, or empty string",
  "exception": "any exception or conditional clause, or empty string",
  "reasoning": "one sentence explaining your classification"
}}

Rules:
- HARD = legally mandatory, uses shall/must/will not, no discretion allowed
- SOFT = discretionary, uses may/should/subject to approval, can be negotiated
- NOT_CONSTRAINT = administrative text, definitions, preamble, not a rule
- Always fill entities field
- Fill threshold for BOTH hard and soft constraints if a number exists
- Fill exception for BOTH hard and soft if except/unless/subject to/provided that exists"""

def call_openai(sentence, client, retries=3):
    """Call OpenAI GPT-4o API."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # cheaper, nearly as good as gpt-4o
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_prompt(sentence)}
                ],
                temperature=0.1,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'^```\s*',     '', raw)
            raw = re.sub(r'\s*```$',     '', raw)
            return json.loads(raw)

        except json.JSONDecodeError:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            return None
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                print(f"\n  Rate limit — waiting 30s...")
                time.sleep(30)
                continue
            if attempt < retries - 1:
                time.sleep(2)
                continue
            print(f"    API error: {e}")
            return None
    return None

def call_groq(sentence, client, retries=3):
    """Call Groq Llama-3 API."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_prompt(sentence)}
                ],
                temperature=0.1,
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'^```\s*',     '', raw)
            raw = re.sub(r'\s*```$',     '', raw)
            return json.loads(raw)

        except json.JSONDecodeError:
            if attempt < retries - 1:
                time.sleep(1)
                continue
            return None
        except Exception as e:
            err = str(e)
            if "rate_limit" in err.lower() or "429" in err:
                wait = 60
                print(f"\n  Rate limit — waiting {wait}s...")
                time.sleep(wait)
                continue
            if attempt < retries - 1:
                time.sleep(2)
                continue
            print(f"    API error: {e}")
            return None
    return None

def auto_annotate(csv_path, use_model="openai", min_confidence=2, dry_run=False):
    """Auto-annotate candidates using chosen model."""

    # Load CSV with string dtypes for annotation columns
    df = pd.read_csv(csv_path, dtype={
        'is_constraint':   'object',
        'constraint_type': 'object',
        'entities':        'object',
        'threshold':       'object',
        'exception':       'object',
        'notes':           'object',
    })

    # Load existing annotated file if it exists to skip already done rows
    out_path = ANNOTATED_DIR / csv_path.name.replace("_candidates", "_annotated")
    done_ids = set()
    if out_path.exists():
        existing = pd.read_csv(out_path, dtype={'is_constraint': 'object'})
        done_ids = set(
            existing[
                existing['is_constraint'].notna() &
                (existing['is_constraint'] != '')
            ]['sentence_id']
        )
        # Merge existing annotations back into df
        existing_indexed = existing.set_index('sentence_id')
        for sid in done_ids:
            if sid in df['sentence_id'].values:
                idx = df[df['sentence_id'] == sid].index[0]
                for col in ['is_constraint','constraint_type','entities',
                            'threshold','exception','notes']:
                    if col in existing_indexed.columns:
                        df.at[idx, col] = existing_indexed.loc[sid, col]

    # Select rows to annotate
    mask = (
        (df['confidence_score'] >= min_confidence) &
        (~df['sentence_id'].isin(done_ids))
    )
    to_annotate = df[mask].copy()

    total        = len(df)
    already_done = len(done_ids)
    to_do        = len(to_annotate)
    skipped      = total - already_done - to_do

    print(f"\n  Total candidates     : {total}")
    print(f"  Already annotated    : {already_done}")
    print(f"  Skipped (low conf.)  : {skipped}")
    print(f"  To auto-annotate     : {to_do}")

    if dry_run:
        print(f"\n  DRY RUN — no API calls made")
        print(f"  Model selected : {use_model}")
        print(f"  Sample sentences:")
        for _, row in to_annotate.head(5).iterrows():
            print(f"    [{row['confidence_score']}] {row['raw_text'][:80]}...")
        return

    if to_do == 0:
        print("  Nothing left to annotate!")
        return

    # Initialize client
    if use_model == "openai":
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("ERROR: OPENAI_API_KEY not found in .env")
                exit(1)
            client = OpenAI(api_key=api_key)
            call_fn = lambda s: call_openai(s, client)
            est_mins = to_do // 60
            print(f"\n  Model  : GPT-4o-mini")
            print(f"  Cost   : ~${to_do * 0.0002:.2f} estimated")
            print(f"  Time   : ~{max(1, est_mins)} minutes\n")
        except ImportError:
            print("ERROR: pip install openai")
            exit(1)
    else:
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("ERROR: GROQ_API_KEY not found in .env")
                exit(1)
            client = Groq(api_key=api_key)
            call_fn = lambda s: call_groq(s, client)
            print(f"\n  Model  : Groq Llama-3.1-8b\n")
        except ImportError:
            print("ERROR: pip install groq")
            exit(1)

    success = 0
    failed  = 0
    stats   = {"HARD": 0, "SOFT": 0, "NOT_CONSTRAINT": 0}

    for i, (idx, row) in enumerate(to_annotate.iterrows()):
        if i % 50 == 0 and i > 0:
            pct = int((i / to_do) * 30)
            bar = "█" * pct + "░" * (30 - pct)
            print(f"  [{bar}] {i}/{to_do} | ✓{success} ✗{failed}")

        result = call_fn(row['raw_text'])

        if result:
            df.at[idx, 'is_constraint']   = result.get('is_constraint', '')
            df.at[idx, 'constraint_type'] = result.get('constraint_type', '')
            df.at[idx, 'entities']        = result.get('entities', '')
            df.at[idx, 'threshold']       = result.get('threshold', '')
            df.at[idx, 'exception']       = result.get('exception', '')
            df.at[idx, 'notes']           = f"auto-{use_model}:{result.get('reasoning','')}"
            ctype = result.get('constraint_type', '')
            if ctype in stats:
                stats[ctype] += 1
            success += 1
        else:
            df.at[idx, 'notes'] = f'AUTO_FAILED_{use_model.upper()}'
            failed += 1

        # Autosave every 100 rows
        if (i + 1) % 100 == 0:
            df.to_csv(out_path, index=False)
            print(f"  Autosaved at {i+1} rows")

        time.sleep(0.05)  # small delay

    # Final save
    df.to_csv(out_path, index=False)

    print(f"\n  {'='*45}")
    print(f"  Auto-annotation complete!")
    print(f"  Successfully labeled : {success}")
    print(f"  Failed               : {failed}")
    print(f"  HARD                 : {stats['HARD']}")
    print(f"  SOFT                 : {stats['SOFT']}")
    print(f"  NOT_CONSTRAINT       : {stats['NOT_CONSTRAINT']}")
    print(f"  Saved → {out_path.relative_to(PROJECT_ROOT)}")
    print(f"\n  Next: run quality check")
    print(f"  Then: python scripts/02_annotate.py --resume for manual review")

def pick_file():
    files = sorted(PROCESSED_DIR.glob("*_candidates.csv"))
    if not files:
        print("No candidate files found.")
        return None
    print("\nAvailable files:")
    for i, f in enumerate(files, 1):
        out = ANNOTATED_DIR / f.name.replace("_candidates", "_annotated")
        done = 0
        if out.exists():
            df = pd.read_csv(out, dtype={'is_constraint': 'object'})
            done = (df['is_constraint'].notna() & (df['is_constraint'] != '')).sum()
        print(f"  {i}. {f.name}  [{done}/{len(pd.read_csv(f))} annotated]")
    choice = input("\nEnter number: ").strip()
    try:
        return files[int(choice) - 1]
    except:
        print("Invalid choice")
        return None

def main():
    parser = argparse.ArgumentParser(description="SOLAR Step 3: Auto-annotate")
    parser.add_argument("--model",      default="openai", choices=["openai","groq"])
    parser.add_argument("--confidence", type=int, default=2,
                        help="Min confidence to auto-annotate (default 2 = all)")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--file",       help="Specific candidates CSV path")
    args = parser.parse_args()

    print("\n" + "="*55)
    print("SOLAR — Step 3: Auto-Annotation")
    print("="*55)

    csv_path = Path(args.file) if args.file else pick_file()
    if not csv_path or not csv_path.exists():
        print("File not found.")
        return

    auto_annotate(csv_path, args.model, args.confidence, args.dry_run)

if __name__ == "__main__":
    main()