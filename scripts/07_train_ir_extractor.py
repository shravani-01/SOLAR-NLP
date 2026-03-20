"""
SOLAR — Script 07: Train IR Extractor
=======================================
Fine-tunes Llama-3.1-8B-Instruct on TransitOpsBench
using LoRA (Low-Rank Adaptation) via Hugging Face PEFT.

Run this on Google Colab Pro (A100 GPU).

Steps:
  1. Install dependencies
  2. Load Llama-3.1-8B-Instruct in 4-bit quantization
  3. Apply LoRA adapters
  4. Train on train.jsonl, evaluate on val.jsonl
  5. Save fine-tuned model to Google Drive

Usage on Colab:
  - Upload train.jsonl, val.jsonl to Colab or mount Google Drive
  - Run each cell in order
  - Training takes ~2-3 hours on A100
"""

# ============================================================
# CELL 1 — Install dependencies
# ============================================================
# Run this cell first — takes ~3 minutes
"""
!pip install -q transformers==4.44.0
!pip install -q peft==0.12.0
!pip install -q bitsandbytes==0.43.3
!pip install -q trl==0.10.1
!pip install -q accelerate==0.34.0
!pip install -q datasets==2.21.0
!pip install -q wandb
"""

# ============================================================
# CELL 2 — Imports and config
# ============================================================
import os
import json
import torch
import wandb
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR     = "/content/drive/MyDrive/SOLAR/models/ir_extractor"
TRAIN_FILE     = "/content/drive/MyDrive/SOLAR/data/training/train.jsonl"
VAL_FILE       = "/content/drive/MyDrive/SOLAR/data/training/val.jsonl"
HF_TOKEN       = ""   # ← paste your Hugging Face token here

# Training hyperparameters
BATCH_SIZE          = 4
GRAD_ACCUM_STEPS    = 4    # effective batch size = 4 * 4 = 16
LEARNING_RATE       = 2e-4
NUM_EPOCHS          = 3
MAX_SEQ_LENGTH      = 1024
WARMUP_RATIO        = 0.05
LR_SCHEDULER        = "cosine"
SAVE_STEPS          = 100
EVAL_STEPS          = 100
LOGGING_STEPS       = 10

# LoRA hyperparameters
LORA_R          = 16    # rank — higher = more capacity, more memory
LORA_ALPHA      = 32    # scaling factor — typically 2x rank
LORA_DROPOUT    = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

print(f"Model     : {MODEL_ID}")
print(f"Output    : {OUTPUT_DIR}")
print(f"Device    : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"VRAM      : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
      if torch.cuda.is_available() else "")


# ============================================================
# CELL 3 — Mount Google Drive
# ============================================================
"""
from google.colab import drive
drive.mount('/content/drive')

# Verify files exist
import os
print("train.jsonl exists:", os.path.exists(TRAIN_FILE))
print("val.jsonl exists  :", os.path.exists(VAL_FILE))
"""


# ============================================================
# CELL 4 — Load training data
# ============================================================
def load_jsonl(path: str) -> Dataset:
    """Load a .jsonl file into a Hugging Face Dataset."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Only keep the 'text' field for training
    # 'text' = full prompt + completion + <|eot_id|>
    dataset = Dataset.from_list([{"text": r["text"]} for r in records])
    return dataset


def load_jsonl_with_meta(path: str) -> list[dict]:
    """Load jsonl keeping all fields — used for evaluation."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# Load datasets
print("Loading training data...")
train_dataset = load_jsonl(TRAIN_FILE)
val_dataset   = load_jsonl(VAL_FILE)

print(f"Train records : {len(train_dataset)}")
print(f"Val records   : {len(val_dataset)}")
print(f"\nSample text (first 300 chars):")
print(train_dataset[0]["text"][:300])


# ============================================================
# CELL 5 — Load model in 4-bit quantization
# ============================================================
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token          = HF_TOKEN,
    trust_remote_code = True,
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4-bit quantization config — reduces memory from ~16GB to ~6GB
bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
)

print("Loading model in 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config = bnb_config,
    device_map          = "auto",
    token               = HF_TOKEN,
    trust_remote_code   = True,
)
model.config.use_cache = False

print(f"Model loaded — parameters: "
      f"{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")


# ============================================================
# CELL 6 — Apply LoRA
# ============================================================
print("\nApplying LoRA adapters...")

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r                = LORA_R,
    lora_alpha       = LORA_ALPHA,
    lora_dropout     = LORA_DROPOUT,
    bias             = "none",
    task_type        = TaskType.CAUSAL_LM,
    target_modules   = LORA_TARGET_MODULES,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: ~0.7% of parameters trainable


# ============================================================
# CELL 7 — Training arguments
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM_STEPS,
    learning_rate               = LEARNING_RATE,
    lr_scheduler_type           = LR_SCHEDULER,
    warmup_ratio                = WARMUP_RATIO,
    evaluation_strategy         = "steps",
    eval_steps                  = EVAL_STEPS,
    save_strategy               = "steps",
    save_steps                  = SAVE_STEPS,
    save_total_limit            = 3,
    load_best_model_at_end      = True,
    logging_steps               = LOGGING_STEPS,
    fp16                        = False,
    bf16                        = True,     # A100 supports bfloat16
    report_to                   = "none",   # set to "wandb" if you want logging
    dataloader_num_workers      = 2,
    group_by_length             = True,     # speeds up training
    seed                        = 42,
)


# ============================================================
# CELL 8 — Train
# ============================================================
print("\nStarting training...")
print(f"Epochs          : {NUM_EPOCHS}")
print(f"Batch size      : {BATCH_SIZE} × {GRAD_ACCUM_STEPS} = "
      f"{BATCH_SIZE * GRAD_ACCUM_STEPS} effective")
print(f"Learning rate   : {LEARNING_RATE}")
print(f"Max seq length  : {MAX_SEQ_LENGTH}")
print(f"Estimated time  : ~2-3 hours on A100\n")

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = train_dataset,
    eval_dataset    = val_dataset,
    dataset_text_field = "text",
    max_seq_length  = MAX_SEQ_LENGTH,
    args            = training_args,
    packing         = False,
)

trainer.train()
print("\n✅ Training complete!")


# ============================================================
# CELL 9 — Save model
# ============================================================
print("\nSaving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Model saved → {OUTPUT_DIR}")

# Save training metrics
metrics = trainer.state.log_history
metrics_path = os.path.join(OUTPUT_DIR, "training_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Metrics saved → {metrics_path}")


# ============================================================
# CELL 10 — Quick inference test
# ============================================================
print("\nRunning inference test...")

TEST_SENTENCE = (
    "No operator shall be required to work more than "
    "6 consecutive hours without a meal period of not "
    "less than 30 minutes."
)

SYSTEM_PROMPT = """You are a constraint extraction system for transit operations.
Given a sentence from a transit labor agreement or operational policy document,
extract the following structured information in JSON format:

- constraint_type: HARD or SOFT
- hardness_subtype: HARD-CONDITIONAL, SOFT-APPROVAL, SOFT-CONDITIONAL, or same as constraint_type
- entities: {subject: [], authority: [], object: []}
- thresholds: [{variable, value, unit, direction}] or []
- exceptions: [{trigger, type}] or []

Return ONLY valid JSON. No explanation, no preamble, no markdown."""

prompt = (
    f"<|begin_of_text|>"
    f"<|start_header_id|>system<|end_header_id|>\n\n"
    f"{SYSTEM_PROMPT}"
    f"<|eot_id|>"
    f"<|start_header_id|>user<|end_header_id|>\n\n"
    f"Extract constraints from this sentence:\n\n"
    f"\"{TEST_SENTENCE}\""
    f"<|eot_id|>"
    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
)

inputs = tokenizer(
    prompt,
    return_tensors = "pt",
    truncation     = True,
    max_length     = MAX_SEQ_LENGTH,
).to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens  = 512,
        temperature     = 0.1,
        do_sample       = True,
        pad_token_id    = tokenizer.eos_token_id,
    )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens = True,
)

print(f"\nTest sentence:\n{TEST_SENTENCE}")
print(f"\nModel output:\n{response}")

# Validate JSON
try:
    parsed = json.loads(response.strip())
    print("\n✅ Valid JSON output")
    print(f"  constraint_type  : {parsed.get('constraint_type')}")
    print(f"  hardness_subtype : {parsed.get('hardness_subtype')}")
    print(f"  thresholds       : {parsed.get('thresholds')}")
    print(f"  exceptions       : {parsed.get('exceptions')}")
except json.JSONDecodeError:
    print("\n⚠ Output is not valid JSON — may need more training epochs")

print("\n✅ Script complete. Next: run 08_evaluate_ir_extractor.py")