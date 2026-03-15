"""
SOLAR — Setup & Installation Check
====================================
Run this first to verify your environment is ready.

Usage:
  python setup.py
"""

import sys
import subprocess
from pathlib import Path

REQUIRED = [
    ("pdfplumber",    "pdfplumber"),
    ("pandas",        "pandas"),
    ("openpyxl",      "openpyxl"),
    ("tqdm",          "tqdm"),
    ("transformers",  "transformers"),
    ("torch",         "torch"),
    ("ortools",       "ortools"),
]

OPTIONAL = [
    ("spacy",         "spacy"),
    ("sklearn",       "scikit-learn"),
    ("matplotlib",    "matplotlib"),
]

def check_package(import_name, install_name):
    try:
        __import__(import_name)
        return True, None
    except ImportError:
        return False, install_name

def main():
    print("\n" + "="*55)
    print("  SOLAR — Environment Setup Check")
    print("="*55)

    print(f"\nPython version: {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        print("WARNING: Python 3.9+ recommended")

    print("\nRequired packages:")
    missing_required = []
    for import_name, install_name in REQUIRED:
        ok, pkg = check_package(import_name, install_name)
        status = "✓" if ok else "✗ MISSING"
        print(f"  {status:12} {import_name}")
        if not ok:
            missing_required.append(pkg)

    print("\nOptional packages:")
    missing_optional = []
    for import_name, install_name in OPTIONAL:
        ok, pkg = check_package(import_name, install_name)
        status = "✓" if ok else "○ optional"
        print(f"  {status:12} {import_name}")
        if not ok:
            missing_optional.append(pkg)

    print("\nProject folders:")
    root = Path(__file__).parent
    folders = [
        "data/raw/contracts",
        "data/raw/fta_regulations",
        "data/raw/gtfs",
        "data/processed",
        "data/annotated",
        "scripts",
        "models",
        "results",
    ]
    for f in folders:
        path = root / f
        exists = path.exists()
        status = "✓" if exists else "✗ missing"
        print(f"  {status:12} {f}")

    print("\n" + "="*55)
    if missing_required:
        print("\nInstall missing packages with:")
        pkgs = " ".join(missing_required)
        print(f"\n  pip install {pkgs}\n")
        print("Or install everything at once:")
        all_pkgs = " ".join([p for _, p in REQUIRED + OPTIONAL])
        print(f"\n  pip install {all_pkgs}\n")
    else:
        print("\nAll required packages installed!")
        print("You're ready to run:")
        print("\n  python scripts/01_extract_candidates.py\n")

if __name__ == "__main__":
    main()
