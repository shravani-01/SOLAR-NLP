#!/usr/bin/env python3
"""
Probe-and-Prompt (PaP) for Contract Classification.

Novel inference-time method: uses a trained linear probe on Qwen2.5-7B
hidden states to extract the model's INTERNAL structural prediction,
then feeds it back as a hint for re-classification.

This is "self-knowledge distillation at inference time" — the model's
own internal representations guide its output.

Steps:
  1. PROBE — Run input through Qwen2.5-7B, extract layer -8 hidden states,
             run trained probe to get predicted structural type
  2. PROMPT — Re-prompt any model with the probe's prediction as a hint

Requires GPU for probe extraction. Re-prompting can use API or OSS models.

Usage (on GPU):
    python pap_contracts.py --model gpt4o --limit 50
    python pap_contracts.py --model deepseek --limit 50
    python pap_contracts.py --model qwen7b --limit 50
    python pap_contracts.py --model all
"""

import csv
import json
import os
import re
import sys
import time
import pickle
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

import torch
import numpy as np

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "annotated" / "splits"
PROBE_DIR = Path(__file__).parent.parent.parent / "data" / "results" / "linear_probe" / "probes" / "qwen2.5_7b" / "best_probes"
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
    "gpt4o":    {"backend": "api", "provider": "openai",   "model": "gpt-4o"},
    "deepseek": {"backend": "api", "provider": "deepseek", "model": "deepseek-chat"},
    "qwen7b":   {"backend": "oss", "hf_name": "Qwen/Qwen2.5-7B-Instruct"},
}

CONSTRAINT_TYPES = ["HARD", "SOFT", "HARD-CONDITIONAL", "SOFT-CONDITIONAL", "NON-CONSTRAINT"]

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE

# Probe layer
PROBE_LAYER = -8  # Best performing layer from our probe analysis


# ─── Prompts ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a constraint extraction system for labor contract analysis."

PAP_PROMPT = """Classify this labor contract sentence into one of 5 constraint types.

Sentence: "{text}"

IMPORTANT HINT: Internal model analysis suggests this is most likely a **{probe_prediction}** constraint (confidence: {confidence:.0%}).

The 5 types are:
- HARD: Mandatory requirement (shall/must), no conditions
- SOFT: Discretionary guideline (may/can/should), no conditions
- HARD-CONDITIONAL: Mandatory requirement WITH conditions (if/when/unless triggers)
- SOFT-CONDITIONAL: Discretionary guideline WITH conditions
- NON-CONSTRAINT: Not a constraint (definition, description, informational)

Consider the hint but make your OWN judgment based on the text. The hint is from an internal analysis and may not be perfect.

Return ONLY a JSON object:
{{"constraint_type": "<TYPE>", "reasoning": "<brief explanation>"}}"""


# ─── Probe extraction ───────────────────────────────────────────────────────

_qwen_model = None
_qwen_tokenizer = None


def load_qwen():
    """Load Qwen2.5-7B for hidden state extraction."""
    global _qwen_model, _qwen_tokenizer

    if _qwen_model is not None:
        return _qwen_model, _qwen_tokenizer

    from transformers import AutoTokenizer, AutoModelForCausalLM

    hf_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"[INFO] Loading {hf_name} for probe extraction...")

    _qwen_tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if _qwen_tokenizer.pad_token is None:
        _qwen_tokenizer.pad_token = _qwen_tokenizer.eos_token

    _qwen_model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
    )
    _qwen_model.eval()
    print(f"[INFO] Qwen loaded. Device: {_qwen_model.device}")
    return _qwen_model, _qwen_tokenizer


def load_probe():
    """Load the trained constraint_type probe."""
    probe_path = PROBE_DIR / "constraint_type_layer_-1.pkl"
    if not probe_path.exists():
        # Try other paths
        for candidate in PROBE_DIR.glob("constraint_type*.pkl"):
            probe_path = candidate
            break

    if not probe_path.exists():
        print(f"[ERROR] No probe found in {PROBE_DIR}")
        sys.exit(1)

    print(f"[INFO] Loading probe from {probe_path}")
    with open(probe_path, 'rb') as f:
        probe_data = pickle.load(f)
    if isinstance(probe_data, dict):
        print(f"[INFO] Probe keys: {list(probe_data.keys())}")
        return probe_data
    return {"model": probe_data, "scaler": None, "pca": None}


def extract_hidden_state(model, tokenizer, text: str, layer: int = -8) -> np.ndarray:
    """Extract hidden state at specified layer for the last token."""
    messages = [
        {"role": "system", "content": "You are a constraint analysis system."},
        {"role": "user", "content": f"Classify this constraint: {text}"},
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get hidden state at specified layer, last token
    hidden = outputs.hidden_states[layer][0, -1, :].float().cpu().numpy()
    return hidden


def probe_predict(probe_data, hidden_state: np.ndarray) -> tuple:
    """Run probe on hidden state, return (prediction, confidence)."""
    hidden_2d = hidden_state.reshape(1, -1)
    if isinstance(probe_data, dict):
        if probe_data.get("scaler") is not None:
            hidden_2d = probe_data["scaler"].transform(hidden_2d)
        if probe_data.get("pca") is not None:
            hidden_2d = probe_data["pca"].transform(hidden_2d)
        model = probe_data["model"]
    else:
        model = probe_data
    pred = model.predict(hidden_2d)[0]
    proba = model.predict_proba(hidden_2d)[0]
    confidence = max(proba)
    return pred, confidence


# ─── API / OSS calls ────────────────────────────────────────────────────────

def call_api_model(prompt: str, provider: str, model: str) -> str:
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
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


def call_oss_model(prompt: str, model, tokenizer) -> str:
    """Generate response from loaded OSS model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=None,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def parse_json_response(response: str) -> dict:
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if match:
        response = match.group(1).strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


# ─── Data loading ───────────────────────────────────────────────────────────

def load_test_data(limit: int = None) -> list:
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


# ─── PaP Pipeline ───────────────────────────────────────────────────────────

def run_pap(model_key: str, test_data: list):
    config = MODELS[model_key]
    backend = config["backend"]

    print(f"\n{'='*60}")
    print(f"  PaP (Probe-and-Prompt): {model_key}")
    print(f"{'='*60}")

    # Load Qwen + probe (always needed for hidden state extraction)
    qwen_model, qwen_tokenizer = load_qwen()
    probe = load_probe()

    # For OSS re-prompting, we reuse the same Qwen model
    # For API re-prompting, we call the API

    print(f"[INFO] Processing {len(test_data)} examples...")

    predictions = []
    errors = 0
    start_time = time.time()

    for i, example in enumerate(test_data):
        text = example["text"]
        gold_type = example["gold_type"]

        try:
            # Step 1: PROBE — extract hidden state and run probe
            hidden = extract_hidden_state(qwen_model, qwen_tokenizer, text, PROBE_LAYER)
            probe_pred, probe_conf = probe_predict(probe, hidden)

            # Step 2: PROMPT — re-prompt with probe hint
            pap_prompt = PAP_PROMPT.format(
                text=text,
                probe_prediction=probe_pred,
                confidence=probe_conf,
            )

            if backend == "api":
                response = call_api_model(pap_prompt, config["provider"], config["model"])
                time.sleep(REQUEST_DELAY)
            else:
                response = call_oss_model(pap_prompt, qwen_model, qwen_tokenizer)

            result = parse_json_response(response)
            pred_type = result.get("constraint_type", probe_pred).upper().strip()

            if pred_type not in CONSTRAINT_TYPES:
                pred_type = probe_pred  # fall back to probe

            predictions.append({
                "idx": i,
                "sentence_id": example["sentence_id"],
                "text": text[:200],
                "gold_type": gold_type,
                "probe_prediction": probe_pred,
                "probe_confidence": round(probe_conf, 4),
                "final_type": pred_type,
                "followed_probe": pred_type == probe_pred,
            })

        except Exception as e:
            errors += 1
            predictions.append({
                "idx": i,
                "sentence_id": example["sentence_id"],
                "text": text[:200],
                "gold_type": gold_type,
                "probe_prediction": "HARD",
                "probe_confidence": 0.0,
                "final_type": "HARD",
                "followed_probe": False,
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

    # Probe agreement stats
    followed = sum(1 for p in predictions if p.get("followed_probe"))
    probe_correct = sum(1 for p in predictions if p["probe_prediction"] == p["gold_type"])

    print(f"\n  Results:")
    print(f"  {'Type':<20} {'F1':>8}")
    print(f"  {'-'*30}")
    for ct in CONSTRAINT_TYPES:
        print(f"  {ct:<20} {f1s[ct]:>8.4f}")
    print(f"  {'-'*30}")
    print(f"  {'MACRO F1':<20} {macro_f1:>8.4f}")
    print(f"  {'Probe alone acc':<20} {probe_correct/len(predictions):>8.3f}")
    print(f"  {'Followed probe':<20} {followed}/{len(predictions)} ({followed/len(predictions)*100:.1f}%)")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "method": "PaP",
        "domain": "contracts",
        "model": model_key,
        "n_examples": len(predictions),
        "n_errors": errors,
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": f1s,
        "probe_alone_accuracy": round(probe_correct / len(predictions), 4),
        "followed_probe_pct": round(followed / len(predictions) * 100, 1),
        "timestamp": datetime.now().isoformat(),
    }
    results_path = RESULTS_DIR / f"pap_contracts_{model_key}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    pred_path = RESULTS_DIR / f"pap_contracts_predictions_{model_key}.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)

    print(f"[INFO] Results saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Probe-and-Prompt for Contracts (GPU required)")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODELS.keys()) + ["all"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    test_data = load_test_data(args.limit)
    print(f"[INFO] Loaded {len(test_data)} test examples")

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    for model_key in models_to_run:
        run_pap(model_key, test_data)


if __name__ == "__main__":
    main()
