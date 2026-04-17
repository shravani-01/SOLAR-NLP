"""
SOLAR — Script: Run IR Extraction Baselines
=============================================
Runs baseline models on the full annotated dataset.

INPUT TO EVERY MODEL: raw contract sentence ONLY.
Annotation fields (constraint_type, threshold, entities, exception)
are used ONLY as ground truth for evaluation — NEVER passed to model.
This simulates real-world usage where only raw text is available.

Baselines:
  gpt4o          — GPT-4o zero-shot
  gpt4o-mini     — GPT-4o-mini zero-shot
  gpt4o-cot      — GPT-4o with chain-of-thought
  gpt4o-rag      — GPT-4o with retrieved examples
  deepseek       — DeepSeek-V3 zero-shot
  deepseek-cot   — DeepSeek-V3 with chain-of-thought
  qwen2.5-72b    — Qwen2.5-72B zero-shot via Together.ai  ← NEW
  llama3.3-70b   — Llama-3.3-70B zero-shot via Together.ai ← NEW
  qwen           — Qwen2.5-7B zero-shot via Ollama (free, local)
  llama          — Llama-3.1-8B zero-shot via Ollama (free, local)

Usage:
  python scripts/run_baselines.py --model qwen2.5-72b --limit 5
  python scripts/run_baselines.py --model qwen2.5-72b --split test
  python scripts/run_baselines.py --model llama3.3-70b --split test
"""

import os
import re
import json
import time
import argparse
import requests
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, classification_report

# ── Load .env file automatically ─────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT  = Path(__file__).parent.parent
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"
RESULTS_DIR   = PROJECT_ROOT / "data" / "results" / "baselines"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VALID_TYPES = [
    "HARD", "SOFT", "HARD-CONDITIONAL",
    "SOFT-CONDITIONAL", "NON-CONSTRAINT"
]

DOMAINS = [
    "transit", "healthcare", "education", "municipal",
    "construction", "aviation", "building_services", "hospitality",
]

# ── API clients ───────────────────────────────────────────────────────────────
try:
    from openai import OpenAI

    _oai_key = os.environ.get("OPENAI_API_KEY", "")
    oai_client = OpenAI(api_key=_oai_key) if _oai_key else None
    GPT_AVAILABLE = bool(_oai_key)

    _ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
    ds_client = OpenAI(
        api_key  = _ds_key,
        base_url = "https://api.deepseek.com"
    ) if _ds_key else None
    DS_AVAILABLE = bool(_ds_key)

    # Together.ai — OpenAI-compatible, hosts Qwen + Llama 70B
    _together_key = os.environ.get("TOGETHER_API_KEY", "")
    together_client = OpenAI(
        api_key  = _together_key,
        base_url = "https://api.together.xyz/v1"
    ) if _together_key else None
    TOGETHER_AVAILABLE = bool(_together_key)

except ImportError:
    oai_client      = None
    ds_client       = None
    together_client = None
    GPT_AVAILABLE      = False
    DS_AVAILABLE       = False
    TOGETHER_AVAILABLE = False


# ── Retry helper ──────────────────────────────────────────────────────────────
def _call_with_retry(fn, max_retries=5, base_wait=10):
    """Retry on network errors. Fatal billing/auth errors raise immediately."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in ["402", "401", "403",
                                        "insufficient balance",
                                        "invalid api key"]):
                print(f"\n  FATAL API ERROR: {e}")
                print(f"  Check your API key and account balance.")
                raise
            is_network = any(x in err for x in [
                "connection", "timeout", "network", "reset",
                "unavailable", "502", "503", "504", "rate"
            ])
            if is_network and attempt < max_retries - 1:
                wait = base_wait * (2 ** attempt)
                print(f"    Network error. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return None
    return None


# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a constraint extraction system for labor contract analysis. "
    "Given a contract sentence, extract structured constraint information. "
    "Return only valid JSON. No explanation, no markdown, no backticks."
)

BASE_PROMPT = (
    'Extract constraint information from this labor contract sentence.\n\n'
    'Sentence: "{sentence}"\n\n'
    'Return JSON with exactly these fields:\n'
    '{{\n'
    '  "is_constraint": "Yes or No",\n'
    '  "constraint_type": "HARD or SOFT or HARD-CONDITIONAL or SOFT-CONDITIONAL or NON-CONSTRAINT",\n'
    '  "hardness_subtype": "same as constraint_type unless conditional",\n'
    '  "entities": {{"subject": [], "authority": [], "object": []}},\n'
    '  "thresholds": [{{"variable": "name", "value": null, "unit": null, "direction": null}}],\n'
    '  "exceptions": [{{"trigger": "text", "type": "conditional", "variable_name": "semantic_name"}}]\n'
    '}}\n\n'
    'Constraint types:\n'
    '  HARD             — must always be followed, no exceptions\n'
    '  HARD-CONDITIONAL — must be followed unless a specific condition applies\n'
    '  SOFT             — should be followed but violation incurs a penalty\n'
    '  SOFT-CONDITIONAL — should be followed unless a condition applies\n'
    '  NON-CONSTRAINT   — informational text, not a rule\n\n'
    'variable_name must reflect exception MEANING not a contract ID.\n'
    'Examples: emergency_declared, no_prior_discipline, management_approved\n\n'
    'Return only valid JSON.'
)

COT_PROMPT = (
    'Analyze this labor contract sentence step by step.\n\n'
    'Sentence: "{sentence}"\n\n'
    'Step 1: Is this a constraint (a rule, obligation, or prohibition)? Yes or No.\n'
    'Step 2: Pick constraint_type from EXACTLY these 5 options only:\n'
    '  HARD             — mandatory rule, no exceptions\n'
    '  HARD-CONDITIONAL — mandatory rule with a conditional exception\n'
    '  SOFT             — preferred rule, violation incurs penalty\n'
    '  SOFT-CONDITIONAL — preferred rule with a conditional exception\n'
    '  NON-CONSTRAINT   — informational text, not a rule\n'
    'Step 3: Who are the entities? (subject, authority, object)\n'
    'Step 4: Are there numeric thresholds? (value, unit, direction=max/min)\n'
    'Step 5: Are there conditional exceptions? (trigger text, semantic variable_name)\n\n'
    'IMPORTANT: constraint_type MUST be exactly one of:\n'
    'HARD, HARD-CONDITIONAL, SOFT, SOFT-CONDITIONAL, NON-CONSTRAINT\n'
    'Do NOT use words like prohibition, mandatory, optional, etc.\n\n'
    'After your reasoning, return ONLY this JSON on the final line:\n'
    '{{"is_constraint":"Yes or No","constraint_type":"HARD or HARD-CONDITIONAL or SOFT or SOFT-CONDITIONAL or NON-CONSTRAINT",'
    '"hardness_subtype":"same as constraint_type",'
    '"entities":{{"subject":[],"authority":[],"object":[]}},'
    '"thresholds":[],"exceptions":[]}}'
)

RAG_PROMPT = (
    'Extract constraint information from this labor contract sentence.\n\n'
    'The examples below show the REQUIRED OUTPUT FORMAT only.\n'
    'Do NOT copy constraint_type from examples — classify the new sentence independently.\n\n'
    '--- FORMAT EXAMPLES ---\n'
    '{examples}\n'
    '--- END EXAMPLES ---\n\n'
    'Now classify this new sentence:\n'
    'Sentence: "{sentence}"\n\n'
    'IMPORTANT: constraint_type MUST be exactly one of these 5 values:\n'
    '  HARD             — must always be followed, no exceptions\n'
    '  HARD-CONDITIONAL — must be followed unless a specific condition applies\n'
    '  SOFT             — should be followed but violation incurs a penalty\n'
    '  SOFT-CONDITIONAL — should be followed unless a condition applies\n'
    '  NON-CONSTRAINT   — informational text, not a rule\n\n'
    'Do NOT use topic words like compensation, assignment, obligation, payment, etc.\n'
    'constraint_type must be one of the 5 values above — nothing else.\n\n'
    'Return JSON with exactly these fields:\n'
    '{{\n'
    '  "is_constraint": "Yes or No",\n'
    '  "constraint_type": "HARD or SOFT or HARD-CONDITIONAL or SOFT-CONDITIONAL or NON-CONSTRAINT",\n'
    '  "hardness_subtype": "same as constraint_type",\n'
    '  "entities": {{"subject": [], "authority": [], "object": []}},\n'
    '  "thresholds": [{{"variable": "snake_case_name", "value": <number>, "unit": "hours/days/months/percent", "direction": "max or min"}}],\n'
    '  "exceptions": [{{"trigger": "exact text from sentence", "type": "conditional", "variable_name": "semantic_snake_case_name"}}]\n'
    '}}\n\n'
    'For exceptions, variable_name must be semantic: emergency_declared, suspension_active, mutual_agreement\n'
    'For thresholds, use separate value and unit fields — never combine them.'
)


# ── RAG examples ──────────────────────────────────────────────────────────────
_rag_cache = None

def load_rag_examples(n_per_type=2):
    global _rag_cache
    if _rag_cache is not None:
        return _rag_cache
    master = ANNOTATED_DIR / "all_domains_master.csv"
    if not master.exists():
        return []
    df = pd.read_csv(master, dtype={"is_constraint": "object",
                                     "constraint_type": "object"})
    train = df[df["split"] == "train"]
    examples = []
    for ctype in VALID_TYPES:
        subset = train[
            (train["is_constraint"] == "Yes") &
            (train["constraint_type"] == ctype)
        ].head(n_per_type)
        for _, row in subset.iterrows():
            examples.append({
                "sentence":        str(row["raw_text"])[:200],
                "constraint_type": ctype,
                "threshold":       str(row.get("threshold", "")),
                "exception":       str(row.get("exception", "")),
            })
    _rag_cache = examples
    return examples

def format_rag_examples(examples):
    lines = []
    for i, ex in enumerate(examples[:10]):
        lines.append(f"Example {i+1}:")
        lines.append(f"  Sentence: {ex['sentence']}")
        lines.append(f"  constraint_type: {ex['constraint_type']}")
        if ex["threshold"] not in ("nan", "none", ""):
            lines.append(f"  threshold: {ex['threshold']}")
        if ex["exception"] not in ("nan", "none", ""):
            lines.append(f"  exception: {ex['exception']}")
        lines.append("")
    return "\n".join(lines)


# ── JSON parser ───────────────────────────────────────────────────────────────
def parse_json_response(raw: str):
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```", "", raw)
    raw = raw.strip()

    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            try:
                result = json.loads(line)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                continue

    for match in re.finditer(r"\{[^{}]{5,800}\}", raw, re.DOTALL):
        try:
            result = json.loads(match.group())
            if isinstance(result, dict) and "constraint_type" in result:
                return result
        except json.JSONDecodeError:
            continue

    return None


# ── Model callers ─────────────────────────────────────────────────────────────
def call_openai(sentence: str, model: str, mode: str = "base"):
    if not GPT_AVAILABLE or oai_client is None:
        return None
    if mode == "cot":
        content = COT_PROMPT.format(sentence=sentence[:500])
    elif mode == "rag":
        ex_str  = format_rag_examples(load_rag_examples())
        content = RAG_PROMPT.format(examples=ex_str, sentence=sentence[:500])
    else:
        content = BASE_PROMPT.format(sentence=sentence[:500])

    def _call():
        r = oai_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",   "content": content}],
            temperature=0, max_tokens=800,
        )
        return parse_json_response(r.choices[0].message.content.strip())
    return _call_with_retry(_call)


def call_deepseek(sentence: str, mode: str = "base"):
    if not DS_AVAILABLE or ds_client is None:
        return None
    content = (COT_PROMPT if mode == "cot" else BASE_PROMPT).format(
        sentence=sentence[:500]
    )
    def _call():
        r = ds_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",   "content": content}],
            temperature=0, max_tokens=800,
        )
        return parse_json_response(r.choices[0].message.content.strip())
    return _call_with_retry(_call)


def call_together(sentence: str, model: str, mode: str = "base"):
    """Call Together.ai API — hosts Qwen2.5-72B and Llama-3.3-70B."""
    if not TOGETHER_AVAILABLE or together_client is None:
        print("  [together] SKIPPED — TOGETHER_API_KEY not set or client not init")
        return None
    content = (COT_PROMPT if mode == "cot" else BASE_PROMPT).format(
        sentence=sentence[:500]
    )
    def _call():
        r = together_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",   "content": content}],
            temperature=0, max_tokens=1200,  # increased to avoid truncation
        )
        raw = r.choices[0].message.content.strip()
        parsed = parse_json_response(raw)
        if parsed is None:
            # Only log first 100 chars to avoid console spam
            print(f"  [together] PARSE FAILED. Raw: {raw[:100]}")
        return parsed
    return _call_with_retry(_call)


def call_ollama(sentence: str, model: str = "qwen2.5:7b"):
    prompt = BASE_PROMPT.format(sentence=sentence[:500])
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model,
                  "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                  "stream": False,
                  "options": {"temperature": 0}},
            timeout=60,
        )
        if resp.status_code == 200:
            return parse_json_response(resp.json().get("response", ""))
    except Exception:
        pass
    return None


# ── Metrics ───────────────────────────────────────────────────────────────────
LABEL_MAP = {
    "NOT_CONSTRAINT":   "NON-CONSTRAINT",
    "NON_CONSTRAINT":   "NON-CONSTRAINT",
    "NONCONSTRAINT":    "NON-CONSTRAINT",
    "HARD_CONDITIONAL": "HARD-CONDITIONAL",
    "SOFT_CONDITIONAL": "SOFT-CONDITIONAL",
    "UNCLEAR":          "NON-CONSTRAINT",
}
IS_YES = {"yes", "true", "1", "y"}


def normalize_prediction(pred: dict) -> dict:
    if not pred:
        return pred
    excs = pred.get("exceptions") or []
    normalized_excs = []
    for exc in excs:
        if not isinstance(exc, dict):
            continue
        normalized_excs.append({
            "trigger": (exc.get("trigger") or exc.get("trigger_text") or
                        exc.get("exception_trigger") or exc.get("condition") or ""),
            "type": exc.get("type") or exc.get("exception_type") or "conditional",
            "variable_name": (exc.get("variable_name") or
                              exc.get("semantic_variable_name") or
                              exc.get("var_name") or exc.get("boolean_name") or ""),
        })
    if normalized_excs:
        pred["exceptions"] = normalized_excs

    threshs = pred.get("thresholds") or []
    normalized_threshs = []
    for t in threshs:
        if not isinstance(t, dict):
            continue
        normalized_threshs.append({
            "variable":  (t.get("variable") or t.get("variable_name") or
                          t.get("var_name") or t.get("name") or "unknown"),
            "value":     t.get("value"),
            "unit":      t.get("unit"),
            "direction": t.get("direction"),
        })
    if normalized_threshs:
        pred["thresholds"] = normalized_threshs
    return pred


def compute_metrics(predictions, ground_truth):
    y_true, y_pred = [], []
    thresh_correct = thresh_total = 0
    exc_correct = exc_total = 0
    json_valid = semantic_vars = exc_with_vars = 0

    for pred, gt in zip(predictions, ground_truth):
        gt = gt if isinstance(gt, dict) else {}
        if pred is None:
            pred = {"is_constraint": "No", "constraint_type": "NON-CONSTRAINT"}
        else:
            pred = normalize_prediction(pred)

        gt_is_con = str(gt.get("is_constraint", "No"))
        gt_type   = LABEL_MAP.get(
            str(gt.get("constraint_type", "")).upper().strip(),
            str(gt.get("constraint_type", "")).upper().strip()
        )
        if gt_is_con != "Yes":
            gt_type = "NON-CONSTRAINT"
        if gt_type not in VALID_TYPES:
            gt_type = "NON-CONSTRAINT"

        pred_type = LABEL_MAP.get(
            str(pred.get("constraint_type", "NON-CONSTRAINT")).upper().strip(),
            str(pred.get("constraint_type", "NON-CONSTRAINT")).upper().strip()
        )
        if pred_type not in VALID_TYPES:
            pred_type = "NON-CONSTRAINT"

        y_true.append(gt_type)
        y_pred.append(pred_type)

        if pred is not None and isinstance(pred, dict) and len(pred) > 0:
            json_valid += 1

        gt_val = _parse_value(str(gt.get("threshold", "")))
        if gt_val is not None:
            thresh_total += 1
            vals = []
            for t in (pred.get("thresholds") or []):
                try:
                    vals.append(float(t.get("value")))
                except (TypeError, ValueError):
                    pass
            if vals:
                closest = min(vals, key=lambda v: abs(v - gt_val))
                if abs(closest - gt_val) <= gt_val * 0.1 + 1:
                    thresh_correct += 1

        gt_exc = str(gt.get("exception", "")).strip().lower()
        if gt_exc not in ("", "nan", "none", "null", "n/a"):
            exc_total += 1
            pred_excs = pred.get("exceptions") or []
            if pred_excs:
                exc_correct += 1
                exc_with_vars += 1
                first_exc = pred_excs[0]
                if isinstance(first_exc, str):
                    first_exc = {"trigger": first_exc, "variable_name": ""}
                vn = first_exc.get("variable_name", "") if isinstance(first_exc, dict) else ""
                if (vn and len(vn) > 3 and
                        "exception_contract" not in vn and
                        vn != "exception_unknown"):
                    semantic_vars += 1

    macro_f1 = f1_score(y_true, y_pred, average="macro",
                         zero_division=0, labels=VALID_TYPES)
    report   = classification_report(y_true, y_pred, labels=VALID_TYPES,
                                      output_dict=True, zero_division=0)
    per_class = {c: round(report[c]["f1-score"], 3)
                 for c in VALID_TYPES if c in report}

    return {
        "macro_f1":          round(macro_f1, 4),
        "per_class_f1":      per_class,
        "threshold_acc":     round(thresh_correct/thresh_total, 4) if thresh_total else 0.0,
        "exception_recall":  round(exc_correct/exc_total, 4)       if exc_total    else 0.0,
        "semantic_var_rate": round(semantic_vars/exc_with_vars, 4) if exc_with_vars else 0.0,
        "json_valid_rate":   round(json_valid/len(predictions), 4) if predictions  else 0.0,
        "n_total":           len(predictions),
        "thresh_total":      thresh_total,
        "exc_total":         exc_total,
    }


def compute_per_domain_metrics(predictions, ground_truth, domains):
    results = {}
    for domain in DOMAINS:
        idx = [i for i, d in enumerate(domains) if d == domain]
        if not idx:
            continue
        m = compute_metrics([predictions[i] for i in idx],
                             [ground_truth[i]  for i in idx])
        results[domain] = m["macro_f1"]
    return results


def _parse_value(raw):
    if not raw or str(raw).strip().lower() in ("", "nan", "none", "null", "n/a"):
        return None
    s = str(raw).lower()
    for pat in [r'\((\d+)\)', r'=\s*(\d+\.?\d*)', r'\b(\d+\.?\d*)\b']:
        m = re.search(pat, s)
        if m:
            return float(m.group(1))
    return None


# ── Model configs ─────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    # OpenAI
    "gpt4o":        {"fn": call_openai,   "kwargs": {"model": "gpt-4o",      "mode": "base"}},
    "gpt4o-mini":   {"fn": call_openai,   "kwargs": {"model": "gpt-4o-mini", "mode": "base"}},
    "gpt4o-cot":    {"fn": call_openai,   "kwargs": {"model": "gpt-4o",      "mode": "cot"}},
    "gpt4o-rag":    {"fn": call_openai,   "kwargs": {"model": "gpt-4o",      "mode": "rag"}},
    # DeepSeek
    "deepseek":     {"fn": call_deepseek, "kwargs": {"mode": "base"}},
    "deepseek-cot": {"fn": call_deepseek, "kwargs": {"mode": "cot"}},
    # Together.ai — large OSS models (zero-shot, Aman's recommendation)
    # Only llama3.3-70b is serverless on this account — Qwen requires dedicated endpoint
    "llama3.3-70b": {"fn": call_together, "kwargs": {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "mode": "base"}},
    # qwen2.5-72b kept here but requires dedicated endpoint — enable if account upgraded
    # "qwen2.5-72b": {"fn": call_together, "kwargs": {
    #     "model": "Qwen/Qwen2.5-72B-Instruct", "mode": "base"}},
    # Ollama — small local models
    "qwen":         {"fn": call_ollama,   "kwargs": {"model": "qwen2.5:7b"}},
    "llama":        {"fn": call_ollama,   "kwargs": {"model": "llama3.1:8b"}},
}


# ── Runner ────────────────────────────────────────────────────────────────────
def run_baseline(model_name, df, limit=None):
    if limit:
        df = df.head(limit)

    fn     = MODEL_CONFIGS[model_name]["fn"]
    kwargs = MODEL_CONFIGS[model_name]["kwargs"]

    predictions       = []
    ground_truth_rows = []
    domains_list      = []

    ckpt_path = RESULTS_DIR / f"{model_name}_checkpoint.json"
    start_idx = 0
    if ckpt_path.exists():
        try:
            ckpt              = json.load(open(ckpt_path))
            predictions       = ckpt["predictions"]
            ground_truth_rows = ckpt["ground_truth_rows"]
            domains_list      = ckpt["domains_list"]
            start_idx         = ckpt["next_idx"]
            print(f"  Resuming from checkpoint at row {start_idx}")
        except Exception:
            start_idx = 0

    rows = list(df.iterrows())
    print(f"\n  Running {model_name} on {len(df)} sentences "
          f"(starting from {start_idx})...")

    for i, (_, row) in enumerate(rows):
        if i < start_idx:
            continue
        if i % 100 == 0 and i > 0:
            print(f"    {i}/{len(df)}...")

        pred = fn(str(row.get("raw_text", ""))[:500], **kwargs)
        predictions.append(pred)
        ground_truth_rows.append(row.to_dict())
        domains_list.append(str(row.get("domain", "")))

        # Rate limiting
        is_api_model = any(x in model_name for x in
                           ["gpt", "deepseek", "qwen2.5-72b", "llama3.3-70b"])
        if is_api_model and i % 50 == 49:
            time.sleep(0.3)

        # Checkpoint every 500 rows
        if i % 500 == 499:
            json.dump({"predictions": predictions,
                       "ground_truth_rows": ground_truth_rows,
                       "domains_list": domains_list,
                       "next_idx": i + 1},
                      open(ckpt_path, "w"))

    # Save predictions FIRST before metrics
    out_path = RESULTS_DIR / f"{model_name}_predictions.json"
    with open(out_path, "w") as f:
        json.dump([
            {"sentence_id":               str(gt.get("sentence_id", "") if isinstance(gt, dict) else ""),
             "domain":                    str(gt.get("domain", "")      if isinstance(gt, dict) else ""),
             "ground_truth_type":         str(gt.get("constraint_type", "") if isinstance(gt, dict) else ""),
             "ground_truth_is_constraint":str(gt.get("is_constraint", "")   if isinstance(gt, dict) else ""),
             "prediction":                pred}
            for pred, gt in zip(predictions, ground_truth_rows)
        ], f, indent=2)
    print(f"  Predictions saved → {out_path.name}")

    if ckpt_path.exists():
        ckpt_path.unlink()

    gt_safe        = [gt if isinstance(gt, dict) else {} for gt in ground_truth_rows]
    metrics        = compute_metrics(predictions, gt_safe)
    domain_metrics = compute_per_domain_metrics(predictions, gt_safe, domains_list)
    metrics["per_domain_f1"] = domain_metrics

    json.dump(metrics, open(RESULTS_DIR / f"{model_name}_metrics.json", "w"), indent=2)
    return predictions, metrics


# ── Comparison table ──────────────────────────────────────────────────────────
def print_comparison_table(all_metrics):
    print(f"\n{'='*80}")
    print(f"  BASELINE COMPARISON TABLE")
    print(f"{'='*80}")
    print(f"  {'Model':<18} {'Macro F1':>9} {'Thresh':>8} {'Exc':>7} {'Sem var':>9} {'JSON':>7} {'N':>7}")
    print(f"  {'-'*70}")
    for model, m in sorted(all_metrics.items(),
                            key=lambda x: x[1].get("macro_f1", 0), reverse=True):
        print(f"  {model:<18} "
              f"{m.get('macro_f1',0):>9.4f} "
              f"{m.get('threshold_acc',0):>8.4f} "
              f"{m.get('exception_recall',0):>7.4f} "
              f"{m.get('semantic_var_rate',0):>9.4f} "
              f"{m.get('json_valid_rate',0):>7.4f} "
              f"{m.get('n_total',0):>7}")

    print(f"\n  Per-domain Macro F1:")
    domains_seen = set()
    for m in all_metrics.values():
        domains_seen.update(m.get("per_domain_f1", {}).keys())
    header = f"  {'Model':<18}"
    for d in sorted(domains_seen):
        header += f" {d[:8]:>9}"
    print(header)
    print(f"  {'-'*80}")
    for model, m in sorted(all_metrics.items(),
                            key=lambda x: x[1].get("macro_f1", 0), reverse=True):
        row = f"  {model:<18}"
        for d in sorted(domains_seen):
            row += f" {m.get('per_domain_f1', {}).get(d, 0.0):>9.4f}"
        print(row)
    print(f"{'='*80}\n")

    rows = []
    for model, m in all_metrics.items():
        r = {"model": model,
             **{k: v for k, v in m.items()
                if k not in ("per_class_f1", "per_domain_f1")}}
        r.update({f"domain_{d}": v
                  for d, v in m.get("per_domain_f1", {}).items()})
        rows.append(r)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    print(f"  Saved → data/results/baselines/comparison_table.csv")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SOLAR: Run IR extraction baselines")
    parser.add_argument("--model", default="gpt4o-mini",
                        choices=list(MODEL_CONFIGS.keys()) + ["all"])
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--split",  default=None,
                        choices=["train", "val", "test", None])
    args = parser.parse_args()

    print("\n" + "="*60)
    print("SOLAR — Baseline IR Extraction Evaluation")
    print("="*60)

    print(f"\n  API keys loaded:")
    print(f"    OPENAI_API_KEY:    {'set' if GPT_AVAILABLE      else 'NOT SET'}")
    print(f"    DEEPSEEK_API_KEY:  {'set' if DS_AVAILABLE       else 'NOT SET'}")
    print(f"    TOGETHER_API_KEY:  {'set' if TOGETHER_AVAILABLE else 'NOT SET'}")

    master_path = ANNOTATED_DIR / "all_domains_master.csv"
    if not master_path.exists():
        print(f"ERROR: {master_path} not found.")
        return

    df = pd.read_csv(master_path, dtype={
        "is_constraint":    "object",
        "constraint_type":  "object",
        "threshold":        "object",
        "exception":        "object",
        "annotation_valid": "object",
    })

    if "annotation_valid" in df.columns:
        before = len(df)
        df = df[df["annotation_valid"] != "False"].copy()
        print(f"  Removed {before - len(df)} invalid annotations")

    df = df[df["is_constraint"].notna()].copy()

    if args.domain:
        df = df[df["domain"] == args.domain].copy()
        print(f"  Filtered to domain: {args.domain}")
    if args.split:
        df = df[df["split"] == args.split].copy()
        print(f"  Filtered to split: {args.split}")

    print(f"  Total sentences: {len(df):,}")
    print(f"  Constraints:     {(df['is_constraint']=='Yes').sum():,}")
    print(f"  Domains:         {df['domain'].nunique()}")

    models_to_run = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]

    # Availability checks
    if not GPT_AVAILABLE and any("gpt" in m for m in models_to_run):
        print("\n  WARNING: OPENAI_API_KEY not set — GPT models skipped.")
        models_to_run = [m for m in models_to_run if "gpt" not in m]
    if not DS_AVAILABLE and any("deepseek" in m for m in models_to_run):
        print("\n  WARNING: DEEPSEEK_API_KEY not set — DeepSeek models skipped.")
        models_to_run = [m for m in models_to_run if "deepseek" not in m]
    if not TOGETHER_AVAILABLE and any(
            m in ("qwen2.5-72b", "llama3.3-70b") for m in models_to_run):
        print("\n  WARNING: TOGETHER_API_KEY not set — Together.ai models skipped.")
        print("  Get a free key at together.ai (includes $25 free credits)")
        models_to_run = [m for m in models_to_run
                         if m not in ("qwen2.5-72b", "llama3.3-70b")]

    all_metrics = {}
    for f in RESULTS_DIR.glob("*_metrics.json"):
        name = f.stem.replace("_metrics", "")
        if name not in models_to_run:
            all_metrics[name] = json.load(open(f))
            print(f"  Loaded existing: {name}")

    for model_name in models_to_run:
        print(f"\n  [{model_name}]")
        _, metrics = run_baseline(model_name, df.copy(), args.limit)
        all_metrics[model_name] = metrics
        print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
        print(f"  Threshold acc:    {metrics['threshold_acc']:.4f}")
        print(f"  Exception recall: {metrics['exception_recall']:.4f}")
        print(f"  Semantic vars:    {metrics['semantic_var_rate']:.4f}")
        print(f"  JSON valid:       {metrics['json_valid_rate']:.4f}")

    if len(all_metrics) > 1:
        print_comparison_table(all_metrics)


if __name__ == "__main__":
    main()