import json

with open("data/processed/cfft_pairs.json") as f:
    pairs = json.load(f)

for p in pairs:
    print(f"\n{p['constraint_id']}")
    print(f"Status  : {p['solver_status']}")
    print(f"IR      : type={p['constraint_type']}, "
          f"thresh={p['wrong_ir']['thresholds']}")
    print(f"Code (first 8 lines):")
    for line in p['wrong_code'].split('\n')[:8]:
        print(f"  {line}")
        