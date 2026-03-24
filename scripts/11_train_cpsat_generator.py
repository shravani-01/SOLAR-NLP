"""
SOLAR — Script 11: Train CP-SAT Code Generator
================================================
Fine-tunes Llama-3.1-8B-Instruct to generate CP-SAT code
directly from Constraint IR JSON using LoRA.

This is Phase 5 Part A — teaching the model to write
CP-SAT code. The model will make mistakes — those mistakes
become the CFFT training signal in Script 12.

Run this on Google Colab Pro (A100 GPU).

Input:
  Google Drive: SOLAR/data/training/cpsat_train.jsonl
  Google Drive: SOLAR/data/training/cpsat_val.jsonl

Output:
  Google Drive: SOLAR/models/cpsat_generator/

Training time: ~15-20 minutes on A100
"""

# ============================================================
# CELL 1 — Install dependencies
# ============================================================
# Run once at the start of your Colab session
"""
!pip install -q -U bitsandbytes
!pip install -q transformers==4.44.0
!pip install -q peft==0.12.0
!pip install -q trl==0.10.1
!pip install -q accelerate==0.34.0
!pip install -q datasets==2.21.0
"""

# ============================================================
# CELL 2 — Imports and config
# ============================================================
import os
import json
import torch
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
MODEL_ID   = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN   = ""  # ← paste your HF token here

DRIVE_BASE  = "/content/drive/MyDrive/Research/SOLAR"
TRAIN_FILE  = f"{DRIVE_BASE}/data/training/cpsat_train.jsonl"
VAL_FILE    = f"{DRIVE_BASE}/data/training/cpsat_val.jsonl"
OUTPUT_DIR  = f"{DRIVE_BASE}/models/cpsat_generator"

# ── Hyperparameters ───────────────────────────────────────────────────────────
BATCH_SIZE       = 4
GRAD_ACCUM_STEPS = 4      # effective batch = 16
LEARNING_RATE    = 2e-4
NUM_EPOCHS       = 3
MAX_SEQ_LENGTH   = 1536   # longer than IR extractor — CP-SAT code is verbose
LORA_R           = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

print(f"Device : {torch.cuda.get_device_name(0)}")
print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"Model  : {MODEL_ID}")


# ============================================================
# CELL 3 — Mount Google Drive
# ============================================================
"""
from google.colab import drive
drive.mount('/content/drive')

print("cpsat_train.jsonl:", os.path.exists(TRAIN_FILE))
print("cpsat_val.jsonl  :", os.path.exists(VAL_FILE))
os.makedirs(OUTPUT_DIR, exist_ok=True)
"""


# ============================================================
# CELL 4 — Load training data
# ============================================================
def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list([{"text": r["text"]} for r in records])

def load_jsonl_full(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

print("Loading training data...")
train_dataset = load_jsonl(TRAIN_FILE)
val_dataset   = load_jsonl(VAL_FILE)

print(f"Train : {len(train_dataset)} records")
print(f"Val   : {len(val_dataset)} records")
print(f"\nSample text (first 300 chars):")
print(train_dataset[0]["text"][:300])


# ============================================================
# CELL 5 — Load tokenizer and model in 4-bit
# ============================================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, token=HF_TOKEN, trust_remote_code=True
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"
print("Tokenizer loaded ✅")

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

print(f"Model loaded ✅")
print(f"Params : {sum(p.numel() for p in model.parameters())/1e9:.1f}B")
print(f"VRAM   : {torch.cuda.memory_allocated()/1e9:.1f} GB")


# ============================================================
# CELL 6 — Apply LoRA
# ============================================================
print("Applying LoRA...")
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    bias           = "none",
    task_type      = TaskType.CAUSAL_LM,
    target_modules = LORA_TARGET_MODULES,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ============================================================
# CELL 7 — Train
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM_STEPS,
    learning_rate               = LEARNING_RATE,
    lr_scheduler_type           = "cosine",
    warmup_ratio                = 0.05,
    evaluation_strategy         = "steps",
    eval_steps                  = 100,
    save_strategy               = "steps",
    save_steps                  = 100,
    save_total_limit            = 3,
    load_best_model_at_end      = True,
    logging_steps               = 10,
    bf16                        = True,
    report_to                   = "none",
    group_by_length             = True,
    seed                        = 42,
)

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    train_dataset      = train_dataset,
    eval_dataset       = val_dataset,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LENGTH,
    args               = training_args,
    packing            = False,
)

print("\nStarting CP-SAT generator training...")
print(f"Task           : IR JSON → CP-SAT code")
print(f"Train records  : {len(train_dataset)}")
print(f"Epochs         : {NUM_EPOCHS}")
print(f"Effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"Max seq length : {MAX_SEQ_LENGTH}")
print(f"Estimated time : ~20 minutes on A100\n")

trainer.train()
print("\n✅ Training complete!")


# ============================================================
# CELL 8 — Save model
# ============================================================
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

metrics = trainer.state.log_history
with open(os.path.join(OUTPUT_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Model saved → {OUTPUT_DIR}")
print(f"Saved files: {os.listdir(OUTPUT_DIR)}")


# ============================================================
# CELL 9 — Quick inference test
# ============================================================
print("\nRunning inference test...")

TEST_IR = {
    "constraint_id"   : "TEST_001",
    "raw_text"        : "No operator shall work more than 8 consecutive hours without a break of at least 30 minutes.",
    "constraint_type" : "HARD",
    "hardness_subtype": "HARD",
    "entities"        : {"subject": ["operator"], "authority": [], "object": []},
    "thresholds"      : [
        {"variable": "max_consecutive_hours", "value": 8.0,
         "unit": "hours", "direction": "max"},
        {"variable": "min_break_minutes", "value": 30.0,
         "unit": "minutes", "direction": "min"}
    ],
    "exceptions"      : []
}

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

prompt = (
    f"<|begin_of_text|>"
    f"<|start_header_id|>system<|end_header_id|>\n\n"
    f"{SYSTEM_PROMPT}"
    f"<|eot_id|>"
    f"<|start_header_id|>user<|end_header_id|>\n\n"
    f"Generate CP-SAT code for this constraint:\n\n"
    f"{json.dumps(TEST_IR, indent=2)}"
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
        max_new_tokens = 512,
        temperature    = 0.1,
        do_sample      = True,
        pad_token_id   = tokenizer.eos_token_id,
    )

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens = True,
).strip()

print(f"\nTest IR:")
print(f"  max 8 consecutive hours + min 30 min break")
print(f"\nGenerated CP-SAT code:")
print(response)

# Check if it looks like valid CP-SAT code
has_model_add  = "model.Add" in response
has_for_loop   = "for op" in response or "for i" in response
has_comment    = response.startswith("#")

print(f"\nCode quality checks:")
print(f"  Contains model.Add() : {has_model_add}")
print(f"  Contains loop        : {has_for_loop}")
print(f"  Starts with comment  : {has_comment}")

if has_model_add:
    print("\n✅ Model generating valid CP-SAT structure!")
else:
    print("\n⚠ Model output may need more training epochs")

print(f"\n✅ Script 11 complete.")
print(f"Next: Run scripts/12_cfft_loop.py on Mac")
print(f"      Uses generated code + OR-Tools to build CFFT pairs\n")