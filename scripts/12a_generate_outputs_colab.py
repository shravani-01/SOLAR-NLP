"""
SOLAR — Script 12a (Colab side): Generate CP-SAT Outputs
==========================================================
Run this on Colab AFTER training Script 11.
Generates CP-SAT code for all IR records using the
fine-tuned CP-SAT generator model.

Saves output to Google Drive for Mac-side execution.

Copy this into a Colab cell and run it.
"""

import json
import torch
from pathlib import Path

DRIVE_BASE   = "/content/drive/MyDrive/Research/SOLAR"
IR_FILE      = f"{DRIVE_BASE}/data/processed/constraint_ir.json"
MODEL_DIR    = f"{DRIVE_BASE}/models/cpsat_generator"
OUTPUT_FILE  = f"{DRIVE_BASE}/data/processed/cpsat_generated_outputs.json"

SYSTEM_PROMPT = """You are a CP-SAT constraint code generator for transit operations.
Given a structured constraint IR (JSON), generate executable Python CP-SAT code
using Google OR-Tools that implements the constraint.

Rules:
- Use model.Add() for hard constraints
- Use model.NewBoolVar() for boolean conditions
- Use OnlyEnforceIf() for conditional constraints
- Use slack variables for soft constraints
- Always include a comment with the constraint ID and source text
- Return ONLY executable Python code. No explanation, no markdown."""


def generate_cpsat_code(ir: dict) -> str:
    """Generate CP-SAT code for one IR record."""
    prompt_ir = {
        "constraint_id"   : ir["constraint_id"],
        "raw_text"        : ir["raw_text"],
        "constraint_type" : ir["constraint_type"],
        "hardness_subtype": ir["hardness_subtype"],
        "entities"        : ir["entities"],
        "thresholds"      : ir["thresholds"],
        "exceptions"      : ir["exceptions"],
    }

    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}"
        f"<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Generate CP-SAT code for this constraint:\n\n"
        f"{json.dumps(prompt_ir, indent=2)}"
        f"<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 1536,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 512,
            temperature    = 0.1,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens = True,
    ).strip()

    return response


# ── Load IR records ───────────────────────────────────────────────────────────
print("Loading IR records...")
with open(IR_FILE) as f:
    ir_records = json.load(f)
print(f"IR records: {len(ir_records)}")

# ── Generate outputs ──────────────────────────────────────────────────────────
print(f"\nGenerating CP-SAT code for {len(ir_records)} constraints...")
print("This takes ~20-30 minutes on A100...\n")

generated_outputs = []
errors = 0

for i, ir in enumerate(ir_records):
    if i % 100 == 0:
        print(f"  Progress: {i}/{len(ir_records)}")
    try:
        code = generate_cpsat_code(ir)
        generated_outputs.append({
            "constraint_id"  : ir["constraint_id"],
            "constraint_type": ir["constraint_type"],
            "generated_code" : code,
        })
    except Exception as e:
        errors += 1
        generated_outputs.append({
            "constraint_id"  : ir["constraint_id"],
            "constraint_type": ir["constraint_type"],
            "generated_code" : f"# ERROR: {str(e)}",
        })

print(f"  Progress: {len(ir_records)}/{len(ir_records)} ✅")
print(f"  Errors   : {errors}")

# ── Save to Drive ─────────────────────────────────────────────────────────────
import os
os.makedirs(f"{DRIVE_BASE}/data/processed", exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(generated_outputs, f, indent=2)

print(f"\n✅ Generated outputs saved → {OUTPUT_FILE}")
print(f"   {len(generated_outputs)} records")
print(f"\nNext:")
print(f"  Download cpsat_generated_outputs.json to your Mac")
print(f"  Place in SOLAR/data/processed/")
print(f"  Run: python scripts/12a_collect_cfft_pairs.py")
