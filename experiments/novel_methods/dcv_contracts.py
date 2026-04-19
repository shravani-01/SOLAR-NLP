#!/usr/bin/env python3
"""
Decompose-Compose-Verify (DCV) Loop for Contract Classification.

A novel 3-step prompting strategy that explicitly bridges piece extraction
and structural classification, forcing the model to compose pieces into
structure as a distinct reasoning step.

Steps:
  1. DECOMPOSE — Extract constraint pieces (entities, thresholds, exceptions, conditions)
  2. COMPOSE   — Given extracted pieces, determine structural type
  3. VERIFY    — Cross-check: does the classification match the extracted pieces?

Usage:
    python dcv_contracts.py --model gpt4o --limit 50
    python dcv_contracts.py --model deepseek --limit 50
    python dcv_contracts.py --model all
"""

import csv
import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "annotated" / "splits"
RESULTS_DIR = Path(__file__).parent / "results"

# Load .env
for _env_candidate in [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / "spider_composition_gap" / ".env",
]:
    if _env_candidate.exists():
        with open(_env_candidate) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _key, _val = _line.split("=", 1)
                    os.environ.setdefault(_key.strip(), _val.strip())
        break

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

MODELS = {
    "gpt4o":        {"provider": "openai",   "model": "gpt-4o"},
    "deepseek":     {"provider": "deepseek", "model": "deepseek-chat"},
}

CONSTRAINT_TYPES = ["HARD", "SOFT", "HARD-CONDITIONAL", "SOFT-CONDITIONAL", "NON-CONSTRAINT"]

REQUESTS_PER_MINUTE = 20  # lower than usual since 3 calls per example
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


# ─── Prompts ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a constraint analysis expert for labor contracts. Follow instructions precisely."

DECOMPOSE_PROMPT = """Analyze this labor contract sentence and extract all constraint PIECES:

Sentence: "{text}"

Extract the following pieces:
1. ENTITIES: Who/what is constrained? (subject, authority, affected parties)
2. THRESHOLDS: Any numerical limits, durations, percentages, amounts?
3. EXCEPTIONS: Any conditions that modify or exempt the constraint?
4. MODALITY: Is the language mandatory (shall, must, required) or discretionary (may, can, should)?
5. CONDITIONS: Are there if/when/unless clauses that trigger the constraint?

Return ONLY a JSON object:
{{"entities": [...], "thresholds": [...], "exceptions": [...], "modality": "mandatory|discretionary|none", "conditions": [...]}}"""

COMPOSE_PROMPT = """Given the following extracted pieces from a labor contract sentence, determine the STRUCTURAL TYPE of this constraint.

Original sentence: "{text}"

Extracted pieces:
{pieces_json}

Classification rules:
- HARD: Mandatory language (shall/must) + no conditions. Firm requirement.
- SOFT: Discretionary language (may/can/should) + no conditions. Flexible guideline.
- HARD-CONDITIONAL: Mandatory language + conditions (if/when/unless). Firm but triggered by condition.
- SOFT-CONDITIONAL: Discretionary language + conditions. Flexible and triggered by condition.
- NON-CONSTRAINT: Not a constraint at all. Definitions, descriptions, informational statements.

Based on the extracted pieces, what structural type is this?

Return ONLY a JSON object:
{{"structural_type": "<TYPE>", "reasoning": "<1-2 sentence explanation>"}}"""

VERIFY_PROMPT = """Verify this constraint classification by cross-checking the pieces against the type.

Original sentence: "{text}"

Extracted pieces: {pieces_json}
Proposed classification: {proposed_type}
Reasoning: {reasoning}

Verification checklist:
1. If classified as HARD — is the language truly mandatory (shall/must)? Are there really NO conditions?
2. If classified as SOFT — is the language truly discretionary (may/can)? Are there really NO conditions?
3. If classified as HARD-CONDITIONAL — is there BOTH mandatory language AND a condition?
4. If classified as SOFT-CONDITIONAL — is there BOTH discretionary language AND a condition?
5. If classified as NON-CONSTRAINT — is this truly NOT prescribing any behavior?

If the classification is correct, confirm it. If not, provide the corrected type.

Return ONLY a JSON object:
{{"final_type": "<TYPE>", "changed": true/false, "correction_reason": "<reason if changed, else empty>"}}"""


# ─── API calls ──────────────────────────────────────────────────────────────

def call_model(prompt: str, provider: str, model: str) -> str:
    from openai import OpenAI
    if provider == "openai":
        client = OpenAI(api_key=OPENAI_API_KEY)
    elif provider == "deepseek":
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def parse_json_response(response: str) -> dict:
    """Extract JSON from response, handling markdown blocks."""
    # Try markdown code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if match:
        response = match.group(1).strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


# ─── Data loading ───────────────────────────────────────────────────────────

def load_test_data(limit: int = None) -> list:
    """Load contract test data."""
    test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        print(f"[ERROR] Test data not found at {test_path}")
        sys.exit(1)

    examples = []
    with open(test_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("constraint_type") in CONSTRAINT_TYPES:
                examples.append({
                    "text": row["raw_text"],
                    "gold_type": row["constraint_type"],
                    "domain": row.get("domain", ""),
                    "sentence_id": row.get("sentence_id", ""),
                })

    if limit:
        examples = examples[:limit]
    return examples


# ─── DCV Pipeline ───────────────────────────────────────────────────────────

def run_dcv(model_key: str, test_data: list):
    """Run the Decompose-Compose-Verify loop."""
    config = MODELS[model_key]
    provider = config["provider"]
    model = config["model"]

    print(f"\n{'='*60}")
    print(f"  DCV Loop: {model_key} ({model})")
    print(f"{'='*60}")

    if provider == "openai" and not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY not set.")
        return None
    if provider == "deepseek" and not DEEPSEEK_API_KEY:
        print("[ERROR] DEEPSEEK_API_KEY not set.")
        return None

    print(f"[INFO] Processing {len(test_data)} examples (3 calls each)...")

    predictions = []
    errors = 0
    start_time = time.time()

    for i, example in enumerate(test_data):
        text = example["text"]
        gold_type = example["gold_type"]

        try:
            # Step 1: DECOMPOSE
            decompose_prompt = DECOMPOSE_PROMPT.format(text=text)
            decompose_resp = call_model(decompose_prompt, provider, model)
            pieces = parse_json_response(decompose_resp)
            time.sleep(REQUEST_DELAY)

            # Step 2: COMPOSE
            pieces_json = json.dumps(pieces, indent=2) if pieces else decompose_resp
            compose_prompt = COMPOSE_PROMPT.format(text=text, pieces_json=pieces_json)
            compose_resp = call_model(compose_prompt, provider, model)
            compose_result = parse_json_response(compose_resp)
            proposed_type = compose_result.get("structural_type", "HARD")
            reasoning = compose_result.get("reasoning", "")
            time.sleep(REQUEST_DELAY)

            # Step 3: VERIFY
            verify_prompt = VERIFY_PROMPT.format(
                text=text,
                pieces_json=pieces_json,
                proposed_type=proposed_type,
                reasoning=reasoning,
            )
            verify_resp = call_model(verify_prompt, provider, model)
            verify_result = parse_json_response(verify_resp)
            final_type = verify_result.get("final_type", proposed_type)
            changed = verify_result.get("changed", False)
            time.sleep(REQUEST_DELAY)

            # Normalize type
            final_type = final_type.upper().strip()
            if final_type not in CONSTRAINT_TYPES:
                final_type = proposed_type.upper().strip()
            if final_type not in CONSTRAINT_TYPES:
                final_type = "HARD"

            predictions.append({
                "idx": i,
                "sentence_id": example["sentence_id"],
                "text": text[:200],
                "gold_type": gold_type,
                "pieces": pieces,
                "proposed_type": proposed_type,
                "final_type": final_type,
                "verified_changed": changed,
                "reasoning": reasoning,
            })

        except Exception as e:
            errors += 1
            predictions.append({
                "idx": i,
                "sentence_id": example["sentence_id"],
                "text": text[:200],
                "gold_type": gold_type,
                "pieces": {},
                "proposed_type": "HARD",
                "final_type": "HARD",
                "verified_changed": False,
                "reasoning": "",
                "error": str(e),
            })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(test_data) - i - 1) / rate if rate > 0 else 0
            correct = sum(1 for p in predictions if p["final_type"] == p["gold_type"])
            print(f"[INFO] Progress: {i+1}/{len(test_data)} "
                  f"(errors: {errors}, acc: {correct/(i+1):.3f}, ETA: {eta/60:.1f} min)")

    elapsed = time.time() - start_time
    print(f"[INFO] Done. {len(predictions)} predictions, {errors} errors, {elapsed/60:.1f} min")

    # Evaluate
    correct = sum(1 for p in predictions if p["final_type"] == p["gold_type"])
    accuracy = correct / len(predictions)

    # Per-class F1
    tp, fp, fn = Counter(), Counter(), Counter()
    for p in predictions:
        if p["final_type"] == p["gold_type"]:
            tp[p["gold_type"]] += 1
        else:
            fn[p["gold_type"]] += 1
            fp[p["final_type"]] += 1

    f1s = {}
    for ct in CONSTRAINT_TYPES:
        prec = tp[ct] / (tp[ct] + fp[ct]) if (tp[ct] + fp[ct]) > 0 else 0
        rec = tp[ct] / (tp[ct] + fn[ct]) if (tp[ct] + fn[ct]) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s[ct] = round(f1, 4)

    macro_f1 = sum(f1s.values()) / len(f1s)

    # Verification stats
    n_changed = sum(1 for p in predictions if p.get("verified_changed"))

    print(f"\n  Results:")
    print(f"  {'Type':<20} {'F1':>8}")
    print(f"  {'-'*30}")
    for ct in CONSTRAINT_TYPES:
        print(f"  {ct:<20} {f1s[ct]:>8.4f}")
    print(f"  {'-'*30}")
    print(f"  {'MACRO F1':<20} {macro_f1:>8.4f}")
    print(f"  {'Accuracy':<20} {accuracy:>8.4f}")
    print(f"  {'Verify changed':<20} {n_changed:>8} ({n_changed/len(predictions)*100:.1f}%)")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "method": "DCV",
        "domain": "contracts",
        "model": model_key,
        "n_examples": len(predictions),
        "n_errors": errors,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": f1s,
        "n_verify_changed": n_changed,
        "timestamp": datetime.now().isoformat(),
    }
    results_path = RESULTS_DIR / f"dcv_contracts_{model_key}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    pred_path = RESULTS_DIR / f"dcv_contracts_predictions_{model_key}.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"[INFO] Results saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="DCV Loop for Contract Classification")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    test_data = load_test_data(args.limit)
    print(f"[INFO] Loaded {len(test_data)} test examples")

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        run_dcv(model_key, test_data)


if __name__ == "__main__":
    main()
