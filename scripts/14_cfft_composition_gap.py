"""
SOLAR — Script 14: CFFT for the Composition Gap (Claim 4)
==========================================================
Error-targeted Contrastive SFT to close the Composition Gap.

The Composition Gap: baselines extract pieces well (0.88-0.92 F1)
but fail at structural classification (0.30-0.43 Macro F1 on 5-class
constraint_type). Linear probes show the model encodes the correct
structure internally (0.68 F1) but can't output it.

This script:
  1. Loads the master annotated dataset (all 8 domains, ~46k examples)
  2. Splits into train/val/test
  3. Identifies "composition gap" training examples — focusing on
     the hard structural classification cases
  4. Builds Llama-3 instruct-format training prompts
  5. LoRA fine-tunes on the structure-focused training set
  6. Evaluates on the held-out test split

The key insight: we don't just SFT on all data. We oversample the
hard structural cases (HARD-CONDITIONAL, SOFT-CONDITIONAL, NON-CONSTRAINT)
that baselines consistently fail on, while keeping piece-level extraction
in the prompt to maintain those capabilities.

Paper narrative:
  "The probe proves the model KNOWS the answer (Claim 3).
   Error-targeted CFFT teaches it to SAY it (Claim 4)."

Prerequisites:
  pip install torch transformers peft trl bitsandbytes accelerate datasets

Usage:
  # Step 1: Prepare data (no GPU needed)
  python scripts/14_cfft_composition_gap.py --mode prepare

  # Step 2: Fine-tune (needs GPU — RunPod RTX 3090)
  python scripts/14_cfft_composition_gap.py --mode train

  # Step 3: Evaluate (needs GPU)
  python scripts/14_cfft_composition_gap.py --mode evaluate

  # All three steps sequentially
  python scripts/14_cfft_composition_gap.py --mode all

  # Quick test run (100 examples)
  python scripts/14_cfft_composition_gap.py --mode all --limit 100
"""

import os
import re
import json
import argparse
import logging
import time
from pathlib import Path
from collections import Counter
from datetime import datetime

import pandas as pd
import numpy as np

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "training" / "composition_gap"
MODEL_DIR     = PROJECT_ROOT / "models" / "composition_gap"
RESULTS_DIR   = PROJECT_ROOT / "data" / "results" / "composition_gap"

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

# Hard structural classes — these are what baselines fail on
HARD_CLASSES = {"HARD-CONDITIONAL", "SOFT-CONDITIONAL", "NON-CONSTRAINT"}

# Model config
BASE_MODEL = "Qwen/Qwen2.5-7B"  # ungated, no license needed
HF_TOKEN   = os.environ.get("HF_TOKEN", None) or None  # Qwen2.5 is ungated, no token needed

# LoRA config
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training hyperparameters
LEARNING_RATE  = 2e-4
BATCH_SIZE     = 4
GRAD_ACCUM     = 4   # effective batch = 16
NUM_EPOCHS     = 5    # more epochs for smaller balanced dataset
MAX_SEQ_LENGTH = 768
WARMUP_RATIO   = 0.05


# ── Data helpers ─────────────────────────────────────────────────────────────
def norm_type(t):
    """Normalize constraint type to one of 5 valid labels."""
    t = str(t).upper().strip()
    return LABEL_MAP.get(t, "NON-CONSTRAINT" if t not in VALID_TYPES else t)


def load_master_csv():
    """Load the master annotated CSV with all domains."""
    master = ANNOTATED_DIR / "all_domains_master.csv"
    if not master.exists():
        raise FileNotFoundError(f"Master CSV not found: {master}")
    df = pd.read_csv(master, dtype=str)

    # Filter invalid annotations
    if "annotation_valid" in df.columns:
        df = df[df["annotation_valid"] != "False"]
    df = df[df["is_constraint"].notna()]

    # Normalize constraint_type
    df["constraint_type_norm"] = df["constraint_type"].apply(norm_type)
    # Set NON-CONSTRAINT for non-constraints
    df.loc[df["is_constraint"] != "Yes", "constraint_type_norm"] = "NON-CONSTRAINT"

    log.info(f"Master CSV loaded: {len(df)} rows")
    log.info(f"Domains: {df['domain'].value_counts().to_dict()}")
    log.info(f"Types: {df['constraint_type_norm'].value_counts().to_dict()}")
    log.info(f"Splits: {df['split'].value_counts().to_dict()}")
    return df


# ── Prompt building ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert at classifying constraints in labor contracts. "
    "Given a sentence from a labor agreement, extract structured constraint "
    "information and return ONLY valid JSON."
)

def build_user_prompt(sentence):
    """Build the user prompt — same structure as run_baselines.py."""
    return (
        f'Extract constraint information from this labor contract sentence.\n\n'
        f'Sentence: "{sentence}"\n\n'
        f'Return JSON with exactly these fields:\n'
        f'{{\n'
        f'  "is_constraint": "Yes or No",\n'
        f'  "constraint_type": "HARD or SOFT or HARD-CONDITIONAL or '
        f'SOFT-CONDITIONAL or NON-CONSTRAINT",\n'
        f'  "entities": {{"subject": [], "authority": [], "object": []}},\n'
        f'  "thresholds": [{{"variable": "name", "value": null, "unit": null, '
        f'"direction": null}}],\n'
        f'  "exceptions": [{{"trigger": "text", "type": "conditional"}}]\n'
        f'}}\n\n'
        f'Constraint types:\n'
        f'  HARD             — must always be followed, no exceptions\n'
        f'  HARD-CONDITIONAL — must be followed unless a specific condition applies\n'
        f'  SOFT             — should be followed but violation incurs a penalty\n'
        f'  SOFT-CONDITIONAL — should be followed unless a condition applies\n'
        f'  NON-CONSTRAINT   — informational text, not a rule\n\n'
        f'Return only valid JSON.'
    )


def build_completion(row):
    """Build the gold completion JSON from the annotated row."""
    entities = {}
    try:
        entities = json.loads(str(row.get("entities", "{}")))
    except (json.JSONDecodeError, TypeError):
        entities = {"subject": [], "authority": [], "object": []}

    thresholds = []
    thresh_val = row.get("threshold_value", "")
    if thresh_val and str(thresh_val).strip() not in ("", "nan", "None"):
        thresholds = [{
            "variable": str(row.get("semantic_variable_name", "unknown")),
            "value": thresh_val,
            "unit": str(row.get("threshold_unit", "")),
            "direction": str(row.get("threshold_direction", "")),
        }]

    exceptions = []
    exc = row.get("exception", "")
    if exc and str(exc).strip() not in ("", "nan", "None"):
        exceptions = [{
            "trigger": str(exc)[:200],
            "type": str(row.get("exception_type_normalized", "conditional")),
        }]

    completion = {
        "is_constraint": str(row.get("is_constraint", "No")),
        "constraint_type": row["constraint_type_norm"],
        "entities": entities if isinstance(entities, dict) else {"subject": [], "authority": [], "object": []},
        "thresholds": thresholds,
        "exceptions": exceptions,
    }
    return json.dumps(completion, indent=2)


def build_training_text(row):
    """Build a complete instruct-format training example.

    Uses ChatML format for Qwen2.5:
      <|im_start|>system\n...<|im_end|>
      <|im_start|>user\n...<|im_end|>
      <|im_start|>assistant\n...<|im_end|>
    """
    sentence = str(row.get("raw_text", ""))[:500]
    user_prompt = build_user_prompt(sentence)
    completion = build_completion(row)

    text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{completion}<|im_end|>"
    )
    return text


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: PREPARE DATA
# ═══════════════════════════════════════════════════════════════════════════
def prepare_data(limit=None):
    """Prepare error-targeted training data for the composition gap.

    Strategy:
      - Use ALL domains (not just transit) for maximum diversity
      - Oversample the hard structural classes that baselines fail on:
        HARD-CONDITIONAL, SOFT-CONDITIONAL, NON-CONSTRAINT
      - Include balanced representation of easy classes (HARD, SOFT)
        to prevent catastrophic forgetting
      - Save as train.jsonl and val.jsonl
    """
    log.info("\n" + "="*60)
    log.info("  STEP 1: PREPARE TRAINING DATA")
    log.info("="*60)

    df = load_master_csv()

    # Split by existing split column
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if limit:
        train_df = train_df.head(limit)
        val_df = val_df.head(min(limit // 5, len(val_df)))

    log.info(f"\nRaw split sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # ── Error-targeted oversampling ──────────────────────────────────────
    # The key insight: oversample the hard classes to focus training
    # on the composition gap cases. Without this, HARD dominates (53%)
    # and the model just predicts HARD for everything.

    type_counts = train_df["constraint_type_norm"].value_counts()
    log.info(f"\nOriginal class distribution:")
    for t, c in type_counts.items():
        log.info(f"  {t:<22} {c:>6} ({100*c/len(train_df):.1f}%)")

    # Target: balanced classes, capped at MAX_PER_CLASS to keep
    # training fast (~1-2 hours on A100) while still learning
    # the structural distinctions. 1500 per class × 5 = 7500 total.
    MAX_PER_CLASS = 1500
    max_count = type_counts.max()
    target_per_class = min(int(max_count * 0.8), MAX_PER_CLASS)

    balanced_dfs = []
    for ctype in VALID_TYPES:
        class_df = train_df[train_df["constraint_type_norm"] == ctype]
        n = len(class_df)
        if n == 0:
            continue
        if n < target_per_class:
            # Oversample with replacement
            oversampled = class_df.sample(
                n=target_per_class, replace=True, random_state=42
            )
            balanced_dfs.append(oversampled)
            log.info(f"  {ctype}: {n} → {target_per_class} (oversampled {target_per_class/n:.1f}x)")
        else:
            # Downsample majority class slightly
            downsampled = class_df.sample(
                n=target_per_class, replace=False, random_state=42
            )
            balanced_dfs.append(downsampled)
            log.info(f"  {ctype}: {n} → {target_per_class} (downsampled)")

    train_balanced = pd.concat(balanced_dfs, ignore_index=True)
    train_balanced = train_balanced.sample(frac=1, random_state=42)  # shuffle

    log.info(f"\nBalanced training set: {len(train_balanced)} examples")
    log.info(f"Class distribution after balancing:")
    for t, c in train_balanced["constraint_type_norm"].value_counts().items():
        log.info(f"  {t:<22} {c:>6} ({100*c/len(train_balanced):.1f}%)")

    # ── Build training texts ─────────────────────────────────────────────
    log.info(f"\nBuilding training prompts...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Training set
    train_records = []
    for _, row in train_balanced.iterrows():
        text = build_training_text(row)
        train_records.append({
            "text": text,
            "sentence_id": row.get("sentence_id", ""),
            "constraint_type": row["constraint_type_norm"],
            "domain": row.get("domain", ""),
        })

    train_path = OUTPUT_DIR / "train.jsonl"
    with open(train_path, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
    log.info(f"  Train: {len(train_records)} → {train_path}")

    # Validation set (unbalanced — real distribution for proper eval)
    val_records = []
    for _, row in val_df.iterrows():
        text = build_training_text(row)
        val_records.append({
            "text": text,
            "sentence_id": row.get("sentence_id", ""),
            "constraint_type": row["constraint_type_norm"],
            "domain": row.get("domain", ""),
        })

    val_path = OUTPUT_DIR / "val.jsonl"
    with open(val_path, "w") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")
    log.info(f"  Val: {len(val_records)} → {val_path}")

    # Save test sentence_ids for evaluation
    test_ids = test_df["sentence_id"].tolist()
    with open(OUTPUT_DIR / "test_sentence_ids.json", "w") as f:
        json.dump(test_ids, f)
    log.info(f"  Test IDs: {len(test_ids)} → test_sentence_ids.json")

    # Save stats
    stats = {
        "timestamp": datetime.now().isoformat(),
        "train_total": len(train_records),
        "val_total": len(val_records),
        "test_total": len(test_ids),
        "train_class_dist": train_balanced["constraint_type_norm"].value_counts().to_dict(),
        "val_class_dist": val_df["constraint_type_norm"].value_counts().to_dict(),
        "domains": train_balanced["domain"].value_counts().to_dict(),
        "base_model": BASE_MODEL,
    }
    with open(OUTPUT_DIR / "data_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"\n  Data preparation complete!")
    log.info(f"  Sample training text (first 400 chars):")
    log.info(f"  {train_records[0]['text'][:400]}...")

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: FINE-TUNE
# ═══════════════════════════════════════════════════════════════════════════
def train_model():
    """LoRA fine-tune on the error-targeted training set."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    from trl import SFTTrainer

    log.info("\n" + "="*60)
    log.info("  STEP 2: LoRA FINE-TUNING")
    log.info("="*60)

    # ── Load training data ───────────────────────────────────────────────
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_path}\n"
            f"Run: python scripts/14_cfft_composition_gap.py --mode prepare"
        )

    train_records = []
    with open(train_path) as f:
        for line in f:
            if line.strip():
                train_records.append(json.loads(line))

    val_records = []
    if val_path.exists():
        with open(val_path) as f:
            for line in f:
                if line.strip():
                    val_records.append(json.loads(line))

    train_dataset = Dataset.from_list([{"text": r["text"]} for r in train_records])
    val_dataset = Dataset.from_list([{"text": r["text"]} for r in val_records]) if val_records else None

    log.info(f"  Train: {len(train_dataset)} examples")
    log.info(f"  Val:   {len(val_dataset) if val_dataset else 0} examples")

    # ── Load model ───────────────────────────────────────────────────────
    log.info(f"\n  Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    log.info(f"  Loading model in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    log.info(f"  Model loaded. Parameters: "
             f"{sum(p.numel() for p in model.parameters())/1e9:.1f}B")
    if torch.cuda.is_available():
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Apply LoRA ───────────────────────────────────────────────────────
    log.info(f"\n  Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training ─────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = str(MODEL_DIR / "qwen2.5_7b_composition_gap")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=200 if val_dataset else None,
        load_best_model_at_end=True if val_dataset else False,
        report_to="none",
        optim="paged_adamw_8bit",
        seed=42,
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=False,
    )

    log.info(f"\n  Starting training...")
    log.info(f"  Epochs:         {NUM_EPOCHS}")
    log.info(f"  Batch size:     {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM}")
    log.info(f"  Learning rate:  {LEARNING_RATE}")
    log.info(f"  Max seq length: {MAX_SEQ_LENGTH}")
    log.info(f"  Output:         {output_dir}")

    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0

    log.info(f"\n  Training complete! ({train_time/60:.1f} minutes)")

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"  Model saved → {output_dir}")

    # Save metrics
    metrics = {
        "train_time_minutes": round(train_time / 60, 1),
        "train_loss": trainer.state.log_history[-1].get("train_loss"),
        "epochs": NUM_EPOCHS,
        "n_train": len(train_dataset),
        "base_model": BASE_MODEL,
        "lora_r": LORA_R,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: EVALUATE
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_model(limit=None):
    """Run the fine-tuned model on the test set and compare to baselines."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from sklearn.metrics import f1_score, classification_report

    log.info("\n" + "="*60)
    log.info("  STEP 3: EVALUATE FINE-TUNED MODEL")
    log.info("="*60)

    adapter_dir = str(MODEL_DIR / "qwen2.5_7b_composition_gap")
    if not Path(adapter_dir).exists():
        raise FileNotFoundError(
            f"Model not found: {adapter_dir}\n"
            f"Run: python scripts/14_cfft_composition_gap.py --mode train"
        )

    # ── Load model ───────────────────────────────────────────────────────
    log.info(f"  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info(f"  Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    log.info(f"  Loading LoRA adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    log.info(f"  Model loaded!")

    # ── Load test data ───────────────────────────────────────────────────
    df = load_master_csv()
    test_df = df[df["split"] == "test"].copy()

    if limit:
        test_df = test_df.head(limit)

    log.info(f"  Test set: {len(test_df)} examples")

    # ── Run inference ────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = []
    errors = 0

    log.info(f"\n  Running inference...")
    t0 = time.time()

    for i, (_, row) in enumerate(test_df.iterrows()):
        if i % 100 == 0:
            log.info(f"  Progress: {i}/{len(test_df)}")

        sentence = str(row.get("raw_text", ""))[:500]
        user_prompt = build_user_prompt(sentence)

        # Build prompt in ChatML format
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        ).to(model.device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            # Parse JSON from response
            parsed = None
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                # Try extracting JSON block
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    try:
                        parsed = json.loads(response[start:end])
                    except json.JSONDecodeError:
                        pass

            pred_type = norm_type(
                parsed.get("constraint_type", "NON-CONSTRAINT") if parsed else "NON-CONSTRAINT"
            )

        except Exception as e:
            errors += 1
            parsed = None
            pred_type = "NON-CONSTRAINT"
            if errors <= 5:
                log.warning(f"  Error on row {i}: {e}")

        predictions.append({
            "sentence_id": row.get("sentence_id", ""),
            "domain": row.get("domain", ""),
            "gold_type": row["constraint_type_norm"],
            "pred_type": pred_type,
            "prediction": parsed,
            "raw_response": response[:500] if 'response' in dir() else "",
        })

    infer_time = time.time() - t0
    log.info(f"  Inference done: {len(predictions)} predictions, "
             f"{errors} errors, {infer_time/60:.1f} min")

    # ── Compute metrics ──────────────────────────────────────────────────
    y_true = [p["gold_type"] for p in predictions]
    y_pred = [p["pred_type"] for p in predictions]

    macro_f1 = f1_score(y_true, y_pred, average="macro",
                        labels=VALID_TYPES, zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=VALID_TYPES,
        output_dict=True, zero_division=0
    )
    per_class = {c: round(report[c]["f1-score"], 4) for c in VALID_TYPES if c in report}

    # Per-domain F1
    per_domain = {}
    domain_groups = {}
    for p in predictions:
        domain_groups.setdefault(p["domain"], {"true": [], "pred": []})
        domain_groups[p["domain"]]["true"].append(p["gold_type"])
        domain_groups[p["domain"]]["pred"].append(p["pred_type"])
    for dom, group in domain_groups.items():
        if len(group["true"]) >= 10:
            per_domain[dom] = round(f1_score(
                group["true"], group["pred"],
                average="macro", labels=VALID_TYPES, zero_division=0
            ), 4)

    # Piece-level proxy (JSON validity as indicator)
    json_valid = sum(1 for p in predictions if p["prediction"] is not None)
    json_rate = json_valid / len(predictions) if predictions else 0

    # ── Compare to baselines ─────────────────────────────────────────────
    BASELINE_RESULTS = {
        "deepseek-cot":  0.4268,
        "gpt4o-rag":     0.3986,
        "gpt4o-mini":    0.3935,
        "gpt4o":         0.3696,
        "gpt4o-cot":     0.3434,
        "llama3.3-70b":  0.3145,
        "deepseek":      0.3070,
    }
    PROBE_BEST = 0.6832  # from probe results

    log.info(f"\n{'='*60}")
    log.info(f"  RESULTS: COMPOSITION GAP EVALUATION")
    log.info(f"{'='*60}")
    log.info(f"  CFFT fine-tuned model: {macro_f1:.4f} Macro F1")
    log.info(f"  Per-class F1: {per_class}")
    log.info(f"  JSON validity: {json_rate:.1%}")
    log.info(f"\n  Comparison:")
    log.info(f"  {'Model':<25} {'Structure F1':>12}")
    log.info(f"  {'-'*40}")
    log.info(f"  {'CFFT (ours)':<25} {macro_f1:>12.4f}  ← NEW")
    log.info(f"  {'Probe ceiling':<25} {PROBE_BEST:>12.4f}")
    for name, f1 in sorted(BASELINE_RESULTS.items(), key=lambda x: -x[1]):
        log.info(f"  {name:<25} {f1:>12.4f}")

    improvement = macro_f1 - max(BASELINE_RESULTS.values())
    log.info(f"\n  Improvement over best baseline: {improvement:+.4f}")
    log.info(f"  Gap closed: {improvement / (PROBE_BEST - max(BASELINE_RESULTS.values())) * 100:.1f}% "
             f"of probe ceiling")

    if per_domain:
        log.info(f"\n  Per-domain F1:")
        for dom, f1 in sorted(per_domain.items(), key=lambda x: -x[1]):
            log.info(f"    {dom:<20} {f1:.4f}")

    # ── Save results ─────────────────────────────────────────────────────
    results = {
        "model": BASE_MODEL,
        "adapter": "composition_gap_cfft",
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": per_class,
        "per_domain_f1": per_domain,
        "json_validity": round(json_rate, 4),
        "n_test": len(predictions),
        "n_errors": errors,
        "inference_time_min": round(infer_time / 60, 1),
        "baseline_comparison": BASELINE_RESULTS,
        "probe_ceiling": PROBE_BEST,
        "improvement_over_best_baseline": round(improvement, 4),
        "timestamp": datetime.now().isoformat(),
    }

    with open(RESULTS_DIR / "cfft_composition_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\n  Results saved → {RESULTS_DIR / 'cfft_composition_results.json'}")

    # Save predictions
    with open(RESULTS_DIR / "cfft_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)
    log.info(f"  Predictions saved → {RESULTS_DIR / 'cfft_predictions.json'}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="SOLAR Script 14: CFFT for the Composition Gap"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["prepare", "train", "evaluate", "all"],
        help="Which step to run"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit examples (for testing)"
    )
    args = parser.parse_args()

    log.info("\n" + "="*60)
    log.info("  SOLAR — Script 14: CFFT for Composition Gap")
    log.info("="*60)
    log.info(f"  Mode:  {args.mode}")
    log.info(f"  Model: {BASE_MODEL}")
    log.info(f"  Limit: {args.limit or 'all'}")

    if args.mode in ("prepare", "all"):
        prepare_data(limit=args.limit)

    if args.mode in ("train", "all"):
        train_model()

    if args.mode in ("evaluate", "all"):
        evaluate_model(limit=args.limit)

    log.info("\n  Done!")


if __name__ == "__main__":
    main()
