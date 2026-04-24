#!/usr/bin/env python3
"""
SOLAR-NLP v2 — Core module for 2-Pass Composition Gap evaluation.

The composition gap (following Press et al., 2023) is defined as:

    Composition Gap = P(composed_wrong | all_pieces_correct)

Pass 1: Ask the model piece sub-questions SEPARATELY.
         The model proves it knows each piece by answering correctly.
Pass 2: Ask the model the COMPOSED question.
         Check if it gets the structural composition right.

Gap = examples where Pass 1 is ALL correct but Pass 2 is wrong.
"""

import json
import os
import re
import time
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime


# ─── Load .env ──────────────────────────────────────────────────────────────

def load_env():
    """Load API keys from .env files."""
    for candidate in [
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / "experiments" / "spider_composition_gap" / ".env",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ.setdefault(key.strip(), val.strip())
            break

load_env()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# ─── Model configs ──────────────────────────────────────────────────────────

API_MODELS = {
    "gpt4o":        {"provider": "openai",   "model": "gpt-4o",        "cot": False},
    "gpt4o-cot":    {"provider": "openai",   "model": "gpt-4o",        "cot": True},
    "deepseek":     {"provider": "deepseek", "model": "deepseek-chat", "cot": False},
    "deepseek-cot": {"provider": "deepseek", "model": "deepseek-chat", "cot": True},
}

OSS_MODELS = {
    "qwen7b":       {"hf_name": "Qwen/Qwen2.5-7B-Instruct",        "cot": False},
    "qwen7b-cot":   {"hf_name": "Qwen/Qwen2.5-7B-Instruct",        "cot": True},
    "llama8b":      {"hf_name": "meta-llama/Llama-3.1-8B-Instruct", "cot": False},
    "llama8b-cot":  {"hf_name": "meta-llama/Llama-3.1-8B-Instruct", "cot": True},
}

REQUESTS_PER_MINUTE = 30
REQUEST_DELAY = 60.0 / REQUESTS_PER_MINUTE


# ─── API call ──────────────────────────────────────────────────────────────

def call_api(prompt: str, provider: str, model: str, system: str = "You are a helpful assistant.", max_tokens: int = 512) -> str:
    """Call an API model."""
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
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ─── OSS model loading ─────────────────────────────────────────────────────

_loaded_models = {}

def load_oss_model(hf_name: str):
    """Load an OSS model for inference."""
    if hf_name in _loaded_models:
        return _loaded_models[hf_name]

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[INFO] Loading {hf_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        hf_name, device_map="auto", trust_remote_code=True,
        torch_dtype=__import__('torch').bfloat16,
    )
    model.eval()
    print(f"[INFO] Loaded. Device: {model.device}")
    _loaded_models[hf_name] = (model, tokenizer)
    return model, tokenizer


def call_oss(prompt: str, model, tokenizer, system: str = "You are a helpful assistant.", max_tokens: int = 512) -> str:
    """Generate response from OSS model."""
    import torch

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=None,
            do_sample=False, pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ─── Evaluation ────────────────────────────────────────────────────────────

def compute_composition_gap(predictions: list) -> dict:
    """Compute the per-example composition gap.

    Each prediction must have:
      - pieces_all_correct: bool (did model get ALL piece sub-questions right?)
      - composed_correct: bool (did model get the composed question right?)
    """
    total = 0
    pieces_correct = 0
    composed_correct = 0
    both_correct = 0
    pieces_right_composed_wrong = 0

    per_type = defaultdict(lambda: {"pieces_ok": 0, "composed_fail": 0})

    for pred in predictions:
        if pred.get("error"):
            continue
        total += 1

        p_correct = pred["pieces_all_correct"]
        c_correct = pred["composed_correct"]
        gold_type = pred.get("gold_type", "")

        if p_correct:
            pieces_correct += 1
            per_type[gold_type]["pieces_ok"] += 1
            if c_correct:
                both_correct += 1
            else:
                pieces_right_composed_wrong += 1
                per_type[gold_type]["composed_fail"] += 1

        if c_correct:
            composed_correct += 1

    gap = pieces_right_composed_wrong / pieces_correct if pieces_correct > 0 else 0.0

    per_type_gap = {}
    for stype, stats in per_type.items():
        if stats["pieces_ok"] > 0:
            per_type_gap[stype] = {
                "pieces_ok": stats["pieces_ok"],
                "composed_fail": stats["composed_fail"],
                "gap": round(stats["composed_fail"] / stats["pieces_ok"], 4),
            }

    return {
        "total": total,
        "pieces_correct": pieces_correct,
        "pieces_correct_pct": round(pieces_correct / total * 100, 1) if total > 0 else 0,
        "composed_correct": composed_correct,
        "composed_accuracy": round(composed_correct / total, 4) if total > 0 else 0,
        "both_correct": both_correct,
        "pieces_right_composed_wrong": pieces_right_composed_wrong,
        "composition_gap": round(gap, 4),
        "composition_gap_pct": round(gap * 100, 1),
        "per_type_gap": per_type_gap,
    }


def print_results(results: dict, model: str, domain: str):
    """Pretty-print results."""
    print(f"\n{'='*60}")
    print(f"  {domain} | {model}")
    print(f"{'='*60}")
    print(f"  Examples:            {results['total']}")
    print(f"  Pieces all correct:  {results['pieces_correct']} ({results['pieces_correct_pct']}%)")
    print(f"  Composed correct:    {results['composed_correct']} ({results['composed_accuracy']:.1%})")
    print(f"  Both correct:        {results['both_correct']}")
    print(f"  Pieces OK + Comp WRONG: {results['pieces_right_composed_wrong']}")
    print(f"\n  COMPOSITION GAP = {results['composition_gap']:.1%}")
    print(f"  (Model knew all pieces but failed composition {results['composition_gap_pct']}% of the time)")

    if results["per_type_gap"]:
        print(f"\n  Per-Type Breakdown:")
        print(f"  {'Type':<22} {'Pieces OK':>10} {'Comp Fail':>10} {'Gap':>8}")
        print(f"  {'-'*52}")
        for stype, stats in sorted(results["per_type_gap"].items(), key=lambda x: -x[1]["gap"]):
            print(f"  {stype:<22} {stats['pieces_ok']:>10} {stats['composed_fail']:>10} {stats['gap']:>7.1%}")


def save_results(predictions: list, results: dict, model: str, domain: str, results_dir: Path):
    """Save predictions and results."""
    results_dir.mkdir(parents=True, exist_ok=True)

    pred_path = results_dir / f"predictions_{model}.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2, default=str)

    summary = {
        "model": model, "domain": domain,
        "metric": "P(composed_wrong | all_pieces_correct)",
        **results,
        "timestamp": datetime.now().isoformat(),
    }
    res_path = results_dir / f"results_{model}.json"
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Saved to {res_path}")
