import pandas as pd
from pathlib import Path

domains = ["transit","healthcare","education","municipal","construction","aviation","building_services","hospitality"]

print(f"{'Domain':<22} {'Candidates':>12} {'Confirmed':>10} {'Ann Files':>10} {'Total Ann Rows':>15}")
print("-" * 75)

grand_candidates = 0
grand_confirmed = 0
grand_ann_rows = 0

for domain in domains:
    proc = Path(f"data/processed/{domain}")
    ann  = Path(f"data/annotated/{domain}")
    candidates = 0
    confirmed = 0
    ann_files = 0
    ann_rows = 0

    if proc.exists():
        for f in proc.glob("*_candidates.csv"):
            try:
                candidates += sum(1 for _ in open(f)) - 1
            except:
                pass

    if ann.exists():
        ann_file_list = list(ann.glob("*_annotated.csv"))
        ann_files = len(ann_file_list)
        for f in ann_file_list:
            try:
                df = pd.read_csv(f, dtype={'is_constraint':'object'})
                ann_rows += len(df)
                confirmed += (df['is_constraint'] == 'Yes').sum()
            except:
                pass

    grand_candidates += candidates
    grand_confirmed += confirmed
    grand_ann_rows += ann_rows
    print(f"  {domain:<20} {candidates:>12,} {confirmed:>10,} {ann_files:>10} {ann_rows:>15,}")

print("-" * 75)
print(f"  {'TOTAL':<20} {grand_candidates:>12,} {grand_confirmed:>10,} {'':>10} {grand_ann_rows:>15,}")
print(f"\n  Target: 50,000 | Gap: {50000 - grand_confirmed:,}")
print(f"  Total annotated rows (Yes+No): {grand_ann_rows:,}")
print(f"  Constraint rate: {grand_confirmed/grand_ann_rows*100:.1f}%")
