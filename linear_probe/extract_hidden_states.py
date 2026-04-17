"""
SOLAR — Linear Probe: Step 1 — Extract Hidden States
=====================================================
Runs a causal-LM (e.g. Llama-3.1-8B, DeepSeek-V3, Qwen-2.5-72B)
on every sentence in the SOLAR dataset and saves the hidden-state
vectors at specified layers.

For each sentence the script stores:
  - The mean-pooled hidden state across non-padding tokens (standard
    practice for probing — Conneau et al. 2018, Tenney et al. 2019)
  - The last-token hidden state (useful for decoder-only models where
    the final position carries the most task-relevant information)

Pooling strategies are both saved so the probe training script can
compare them. This avoids having to re-run expensive GPU inference.

Output:
  data/results/linear_probe/hidden_states/
    ├── meta.json            # model name, layers, dataset info
    ├── labels.pt            # ground-truth labels for every sample
    ├── sentence_ids.json    # sentence_id list (aligned with labels)
    ├── mean_pool/
    │   ├── layer_-1.pt      # (N, hidden_dim) tensor
    │   ├── layer_-4.pt
    │   └── ...
    └── last_token/
        ├── layer_-1.pt
        └── ...

Usage:
  # Llama-3.1-8B (fits on single 24GB GPU with 4-bit)
  python linear_probe/extract_hidden_states.py \
      --model meta-llama/Llama-3.1-8B \
      --layers -1,-4,-8,-16,-24,-32 \
      --batch-size 8 \
      --quantize 4bit

  # DeepSeek-V3 via API hidden states (if supported)
  python linear_probe/extract_hidden_states.py \
      --model deepseek-ai/DeepSeek-V3 \
      --layers -1,-4,-8,-16 \
      --batch-size 4

  # Use a specific split only
  python linear_probe/extract_hidden_states.py \
      --model meta-llama/Llama-3.1-8B \
      --split test \
      --layers -1,-4,-8,-16 \
      --batch-size 8

  # Limit to N examples (for debugging)
  python linear_probe/extract_hidden_states.py \
      --model meta-llama/Llama-3.1-8B \
      --limit 500 \
      --layers -1 \
      --batch-size 16

Notes:
  - Uses the SAME prompt format as run_baselines.py (BASE_PROMPT) so
    the hidden states correspond to the same inference conditions that
    produced the baseline results.
  - 4-bit quantization (bitsandbytes) is recommended for models > 7B
    to fit on consumer GPUs. Probing literature shows quantization has
    minimal effect on probe accuracy (Puccetti et al. 2022).
  - Mean-pooling over input tokens only (not generated tokens) — we
    want the representation of the sentence, not the output.
"""

import os
import re
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "linear_probe" / "hidden_states"

# ── Constants ────────────────────────────────────────────────────────────────
VALID_TYPES = [
    "HARD", "SOFT", "HARD-CONDITIONAL",
    "SOFT-CONDITIONAL", "NON-CONSTRAINT",
]
LABEL_MAP = {
    "NOT_CONSTRAINT":   "NON-CONSTRAINT",
    "NON_CONSTRAINT":   "NON-CONSTRAINT",
    "NONCONSTRAINT":    "NON-CONSTRAINT",
    "HARD_CONDITIONAL": "HARD-CONDITIONAL",
    "SOFT_CONDITIONAL": "SOFT-CONDITIONAL",
    "UNCLEAR":          "NON-CONSTRAINT",
}

# Match the prompt from run_baselines.py exactly
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
    '  "constraint_type": "HARD or SOFT or HARD-CONDITIONAL or '
    'SOFT-CONDITIONAL or NON-CONSTRAINT",\n'
    '  "hardness_subtype": "same as constraint_type unless conditional",\n'
    '  "entities": {{"subject": [], "authority": [], "object": []}},\n'
    '  "thresholds": [{{"variable": "name", "value": null, "unit": null, '
    '"direction": null}}],\n'
    '  "exceptions": [{{"trigger": "text", "type": "conditional", '
    '"variable_name": "semantic_name"}}]\n'
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


def norm_type(t):
    """Normalize constraint type label."""
    t = str(t).upper().strip()
    return LABEL_MAP.get(t, t if t in VALID_TYPES else "NON-CONSTRAINT")


# ── Data loading ─────────────────────────────────────────────────────────────
def load_dataset(split=None, domain=None, limit=None):
    """Load the master CSV and return a list of dicts with sentence + labels.

    Labels extracted:
      - constraint_type (5-class)         → structural label
      - has_threshold (binary)            → piece-level
      - has_exception (binary)            → piece-level
      - is_constraint (binary)            → piece-level
      - entity_count (int, binned 0/1/2+) → piece-level
    """
    master = ANNOTATED_DIR / "all_domains_master.csv"
    if not master.exists():
        raise FileNotFoundError(f"Master CSV not found at {master}")

    df = pd.read_csv(master, dtype=str)

    # Filter invalid annotations
    if "annotation_valid" in df.columns:
        df = df[df["annotation_valid"] != "False"]
    df = df[df["is_constraint"].notna()]

    if split:
        df = df[df["split"] == split]
    if domain:
        df = df[df["domain"] == domain]
    if limit:
        df = df.head(limit)

    records = []
    for _, row in df.iterrows():
        raw_text = str(row.get("raw_text", ""))
        if not raw_text.strip():
            continue

        # Structural label (the hard one)
        ct = norm_type(row.get("constraint_type", "NON-CONSTRAINT"))
        if str(row.get("is_constraint", "No")) != "Yes":
            ct = "NON-CONSTRAINT"

        # Piece-level labels
        is_constraint = 1 if str(row.get("is_constraint", "No")) == "Yes" else 0

        thresh_raw = str(row.get("threshold", "")).strip().lower()
        has_threshold = 0 if thresh_raw in ("", "nan", "none", "null", "n/a") else 1

        exc_raw = str(row.get("exception", "")).strip().lower()
        has_exception = 0 if exc_raw in ("", "nan", "none", "null", "n/a") else 1

        entities_raw = str(row.get("entities", "")).strip()
        if entities_raw in ("", "nan", "none"):
            entity_count = 0
        else:
            entity_count = min(len(entities_raw.split(",")), 3)

        records.append({
            "sentence_id": str(row.get("sentence_id", "")),
            "domain": str(row.get("domain", "")),
            "raw_text": raw_text[:512],  # truncate for tokenizer
            "constraint_type": ct,
            "is_constraint": is_constraint,
            "has_threshold": has_threshold,
            "has_exception": has_exception,
            "entity_count": entity_count,
        })

    log.info(f"Loaded {len(records)} records"
             f"{f' (split={split})' if split else ''}"
             f"{f' (domain={domain})' if domain else ''}")
    return records


# ── Model loading ────────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_name, quantize=None, device_map="auto"):
    """Load a HuggingFace causal LM with optional quantization.

    Args:
        model_name: HuggingFace model ID or local path
        quantize: None, "4bit", or "8bit"
        device_map: "auto" for multi-GPU, "cuda:0" for single GPU

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    log.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",  # for batch generation with causal LM
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantization config
    bnb_config = None
    torch_dtype = torch.float16
    if quantize == "4bit":
        log.info("Using 4-bit quantization (bitsandbytes NF4)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantize == "8bit":
        log.info("Using 8-bit quantization (bitsandbytes)")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    log.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        output_hidden_states=True,  # CRITICAL: we need all layer activations
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    log.info(f"Model loaded: {n_layers} layers, hidden_dim={hidden_dim}, "
             f"dtype={model.dtype}")

    return model, tokenizer


# ── Prompt formatting ────────────────────────────────────────────────────────
def format_prompt(raw_text, tokenizer):
    """Format the sentence into the same prompt used by run_baselines.py.

    Uses the model's chat template if available (Llama, Qwen, etc.),
    otherwise falls back to plain concatenation.
    """
    user_content = BASE_PROMPT.format(sentence=raw_text[:500])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for models without chat template
        prompt = f"{SYSTEM_PROMPT}\n\n{user_content}"

    return prompt


# ── Hidden state extraction ──────────────────────────────────────────────────
@torch.no_grad()
def extract_batch(model, tokenizer, sentences, layer_indices, device):
    """Extract hidden states for a batch of sentences.

    For each sentence, we:
      1. Tokenize with the same prompt format as baselines
      2. Run forward pass with output_hidden_states=True
      3. Extract hidden states at specified layers
      4. Compute mean-pool (over non-padding tokens) and last-token

    Args:
        model: HuggingFace causal LM
        tokenizer: Corresponding tokenizer
        sentences: list of raw_text strings
        layer_indices: list of ints (negative = from end, e.g. -1 = last layer)
        device: torch device

    Returns:
        dict of {layer_idx: {"mean_pool": tensor, "last_token": tensor}}
        Each tensor is (batch_size, hidden_dim)
    """
    # Format prompts
    prompts = [format_prompt(s, tokenizer) for s in sentences]

    # Tokenize
    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)

    # Forward pass
    outputs = model(**encodings, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)

    # Resolve negative layer indices
    n_layers = len(hidden_states) - 1  # -1 because index 0 is the embedding layer
    resolved = {}
    for li in layer_indices:
        # hidden_states[0] = embedding layer
        # hidden_states[1] = layer 1 output
        # hidden_states[-1] = last layer output
        actual_idx = li if li >= 0 else len(hidden_states) + li
        if 0 <= actual_idx < len(hidden_states):
            resolved[li] = actual_idx

    # Attention mask for pooling (1 = real token, 0 = padding)
    mask = encodings["attention_mask"].unsqueeze(-1).float()  # (B, S, 1)

    results = {}
    for layer_idx, actual_idx in resolved.items():
        hs = hidden_states[actual_idx].float()  # (B, S, D)

        # Mean pool: average over non-padding tokens
        masked_hs = hs * mask
        sum_hs = masked_hs.sum(dim=1)  # (B, D)
        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        mean_pool = (sum_hs / count).cpu()

        # Last real token (not padding): find last non-pad position per row
        lengths = encodings["attention_mask"].sum(dim=1) - 1  # (B,)
        last_tok = hs[torch.arange(hs.size(0)), lengths].cpu()

        results[layer_idx] = {
            "mean_pool": mean_pool,
            "last_token": last_tok,
        }

    return results


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SOLAR Linear Probe: Extract hidden states from LLM"
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model ID or local path (e.g. meta-llama/Llama-3.1-8B)"
    )
    parser.add_argument(
        "--layers", default="-1,-4,-8,-16,-24,-32",
        help="Comma-separated layer indices to extract (negative = from end)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for inference (reduce if OOM)"
    )
    parser.add_argument(
        "--quantize", choices=["4bit", "8bit"], default=None,
        help="Quantization for large models"
    )
    parser.add_argument(
        "--split", default=None, choices=["train", "val", "test"],
        help="Dataset split to process (default: all)"
    )
    parser.add_argument(
        "--domain", default=None,
        help="Filter to specific domain"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of examples (for debugging)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device map: 'auto', 'cuda:0', 'cpu'"
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=1000,
        help="Save checkpoint every N examples"
    )
    args = parser.parse_args()

    # Parse layer indices
    layer_indices = [int(x.strip()) for x in args.layers.split(",")]
    log.info(f"Layers to extract: {layer_indices}")

    # Output directory
    model_short = args.model.split("/")[-1].lower().replace("-", "_")
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR / model_short
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mean_pool").mkdir(exist_ok=True)
    (out_dir / "last_token").mkdir(exist_ok=True)
    log.info(f"Output directory: {out_dir}")

    # Load dataset
    records = load_dataset(
        split=args.split,
        domain=args.domain,
        limit=args.limit,
    )
    if not records:
        log.error("No records loaded. Check filters.")
        return

    # Prepare labels
    type2id = {t: i for i, t in enumerate(VALID_TYPES)}
    labels = {
        "constraint_type": torch.tensor(
            [type2id[r["constraint_type"]] for r in records], dtype=torch.long
        ),
        "is_constraint": torch.tensor(
            [r["is_constraint"] for r in records], dtype=torch.long
        ),
        "has_threshold": torch.tensor(
            [r["has_threshold"] for r in records], dtype=torch.long
        ),
        "has_exception": torch.tensor(
            [r["has_exception"] for r in records], dtype=torch.long
        ),
        "entity_count": torch.tensor(
            [r["entity_count"] for r in records], dtype=torch.long
        ),
    }
    sentence_ids = [r["sentence_id"] for r in records]
    domains = [r["domain"] for r in records]
    raw_texts = [r["raw_text"] for r in records]

    # Save labels and metadata
    torch.save(labels, out_dir / "labels.pt")
    with open(out_dir / "sentence_ids.json", "w") as f:
        json.dump(sentence_ids, f)
    with open(out_dir / "domains.json", "w") as f:
        json.dump(domains, f)

    # Distribution check
    type_dist = {t: 0 for t in VALID_TYPES}
    for r in records:
        type_dist[r["constraint_type"]] += 1
    log.info(f"Label distribution: {type_dist}")

    # Check for existing checkpoint
    ckpt_path = out_dir / "extraction_checkpoint.json"
    start_idx = 0
    accumulated = {li: {"mean_pool": [], "last_token": []} for li in layer_indices}

    if ckpt_path.exists():
        try:
            ckpt = json.load(open(ckpt_path))
            start_idx = ckpt["next_idx"]
            # Load partial tensors
            for li in layer_indices:
                mp_path = out_dir / "mean_pool" / f"layer_{li}_partial.pt"
                lt_path = out_dir / "last_token" / f"layer_{li}_partial.pt"
                if mp_path.exists() and lt_path.exists():
                    accumulated[li]["mean_pool"].append(torch.load(mp_path))
                    accumulated[li]["last_token"].append(torch.load(lt_path))
            log.info(f"Resuming from checkpoint at index {start_idx}")
        except Exception as e:
            log.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
            start_idx = 0
            accumulated = {li: {"mean_pool": [], "last_token": []} for li in layer_indices}

    # Load model
    device_map = args.device if args.device != "auto" else "auto"
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        quantize=args.quantize,
        device_map=device_map,
    )

    # Determine device for input tensors
    if hasattr(model, "device"):
        device = model.device
    elif hasattr(model, "hf_device_map"):
        # Multi-GPU: use first device
        device = next(iter(model.hf_device_map.values()))
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f"cuda:{device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Input tensors device: {device}")

    # Extract hidden states in batches
    n_total = len(raw_texts)
    batch_size = args.batch_size
    n_batches = (n_total - start_idx + batch_size - 1) // batch_size

    log.info(f"Extracting hidden states: {n_total} examples, "
             f"batch_size={batch_size}, starting from {start_idx}")

    pbar = tqdm(total=n_total - start_idx, desc="Extracting", unit="sent")

    for batch_start in range(start_idx, n_total, batch_size):
        batch_end = min(batch_start + batch_size, n_total)
        batch_texts = raw_texts[batch_start:batch_end]

        try:
            batch_results = extract_batch(
                model, tokenizer, batch_texts, layer_indices, device
            )
        except torch.cuda.OutOfMemoryError:
            log.warning(f"OOM at batch {batch_start}. Trying batch_size=1.")
            torch.cuda.empty_cache()
            # Fallback: process one at a time
            batch_results = {li: {"mean_pool": [], "last_token": []}
                            for li in layer_indices}
            for text in batch_texts:
                try:
                    single = extract_batch(
                        model, tokenizer, [text], layer_indices, device
                    )
                    for li in layer_indices:
                        batch_results[li]["mean_pool"].append(
                            single[li]["mean_pool"])
                        batch_results[li]["last_token"].append(
                            single[li]["last_token"])
                except torch.cuda.OutOfMemoryError:
                    log.error(f"OOM even with batch_size=1. Skipping.")
                    torch.cuda.empty_cache()
                    hidden_dim = model.config.hidden_size
                    for li in layer_indices:
                        batch_results[li]["mean_pool"].append(
                            torch.zeros(1, hidden_dim))
                        batch_results[li]["last_token"].append(
                            torch.zeros(1, hidden_dim))

            # Stack single results
            for li in layer_indices:
                batch_results[li] = {
                    "mean_pool": torch.cat(batch_results[li]["mean_pool"], dim=0),
                    "last_token": torch.cat(batch_results[li]["last_token"], dim=0),
                }

        # Accumulate
        for li in layer_indices:
            accumulated[li]["mean_pool"].append(batch_results[li]["mean_pool"])
            accumulated[li]["last_token"].append(batch_results[li]["last_token"])

        pbar.update(batch_end - batch_start)

        # Periodic checkpoint
        if (batch_end % args.checkpoint_every < batch_size) and batch_end < n_total:
            log.info(f"Saving checkpoint at {batch_end}/{n_total}")
            for li in layer_indices:
                torch.save(
                    torch.cat(accumulated[li]["mean_pool"], dim=0),
                    out_dir / "mean_pool" / f"layer_{li}_partial.pt"
                )
                torch.save(
                    torch.cat(accumulated[li]["last_token"], dim=0),
                    out_dir / "last_token" / f"layer_{li}_partial.pt"
                )
            json.dump({"next_idx": batch_end},
                      open(ckpt_path, "w"))

    pbar.close()

    # Save final tensors
    log.info("Saving final hidden state tensors...")
    for li in layer_indices:
        if accumulated[li]["mean_pool"]:
            mp = torch.cat(accumulated[li]["mean_pool"], dim=0)
            lt = torch.cat(accumulated[li]["last_token"], dim=0)
            torch.save(mp, out_dir / "mean_pool" / f"layer_{li}.pt")
            torch.save(lt, out_dir / "last_token" / f"layer_{li}.pt")
            log.info(f"  Layer {li}: mean_pool={mp.shape}, last_token={lt.shape}")

    # Clean up partials and checkpoint
    for li in layer_indices:
        for suffix in ["_partial.pt"]:
            for pool_type in ["mean_pool", "last_token"]:
                p = out_dir / pool_type / f"layer_{li}{suffix}"
                if p.exists():
                    p.unlink()
    if ckpt_path.exists():
        ckpt_path.unlink()

    # Save metadata
    meta = {
        "model": args.model,
        "model_short": model_short,
        "layers_extracted": layer_indices,
        "n_layers_total": model.config.num_hidden_layers,
        "hidden_dim": model.config.hidden_size,
        "n_examples": n_total,
        "split": args.split,
        "domain": args.domain,
        "quantization": args.quantize,
        "batch_size": args.batch_size,
        "max_seq_length": 1024,
        "pooling_strategies": ["mean_pool", "last_token"],
        "label_names": list(labels.keys()),
        "type_to_id": type2id,
        "type_distribution": type_dist,
        "timestamp": datetime.now().isoformat(),
        "prompt_template": "BASE_PROMPT (same as run_baselines.py)",
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("="*60)
    log.info("  Extraction complete!")
    log.info(f"  Model:    {args.model}")
    log.info(f"  Examples: {n_total}")
    log.info(f"  Layers:   {layer_indices}")
    log.info(f"  Output:   {out_dir}")
    log.info("="*60)


if __name__ == "__main__":
    main()
