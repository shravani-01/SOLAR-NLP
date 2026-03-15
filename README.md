# SOLAR — Solver-in-the-Loop Constraint Acquisition and Refinement

## Project Structure

```
SOLAR/
│
├── data/
│   ├── raw/
│   │   ├── contracts/          ← CBA PDFs go here (TWU, ATU, WMATA etc.)
│   │   ├── fta_regulations/    ← FTA regulatory PDFs go here
│   │   └── gtfs/               ← GTFS spec documents go here
│   │
│   ├── processed/              ← Extracted text + candidate sentences (auto-generated)
│   └── annotated/              ← Your annotation CSVs live here
│
├── scripts/
│   ├── 01_extract_candidates.py   ← Step 1: PDF → candidate constraint sentences
│   ├── 02_annotate.py             ← Step 2: Annotation helper
│   ├── 03_build_ir.py             ← Step 3: Build Constraint IR from annotations
│   ├── 04_generate_cpsat.py       ← Step 4: IR → CP-SAT code
│   └── 05_cfft_train.py           ← Step 5: CFFT training loop
│
├── notebooks/                  ← Jupyter notebooks for exploration & analysis
├── models/                     ← Fine-tuned model checkpoints
├── results/                    ← Evaluation outputs, tables, plots
└── paper/                      ← Draft paper sections

## Setup

pip install pdfplumber pandas openpyxl tqdm spacy transformers ortools

## Usage — Step 1 (Data Collection)

# Place your PDFs in data/raw/contracts/
# Then run:
python scripts/01_extract_candidates.py

# This will produce:
# data/processed/<filename>_candidates.csv   ← candidate constraint sentences
# data/processed/<filename>_stats.txt        ← extraction statistics
```
