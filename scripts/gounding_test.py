import json
import random
from pathlib import Path

PROCESSED = Path("data/processed/transit")

with open(PROCESSED / "constraint_ir.json") as f:
    ir_records = json.load(f)

with open(PROCESSED / "cpsat_generated_outputs.json") as f:
    generated = json.load(f)

gen_lookup = {r["constraint_id"]: r for r in generated}

# Get constraints with exceptions where SOLAR used OnlyEnforceIf
sample_cases = []
for ir in ir_records:
    if not ir.get("exceptions"):
        continue
    cid = ir["constraint_id"]
    if cid not in gen_lookup:
        continue
    code = gen_lookup[cid].get("generated_code", "")
    if "OnlyEnforceIf" in code:
        sample_cases.append({
            "constraint_id" : cid,
            "raw_text"      : ir["raw_text"],
            "exceptions"    : ir["exceptions"],
            "generated_code": code
        })

# Sample 50
random.seed(42)
sample = random.sample(sample_cases, min(50, len(sample_cases)))

# Save for manual review
with open("data/results/grounding_validation_sample.json", "w") as f:
    json.dump(sample, f, indent=2)

print(f"Total with OnlyEnforceIf: {len(sample_cases)}")
print(f"Sample saved: {len(sample)} cases")
print(f"→ data/results/grounding_validation_sample.json")
