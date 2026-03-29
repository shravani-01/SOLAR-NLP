"""
SOLAR — Script 12b: CFFT Fine-Tuning
=====================================
Constraint Faithfulness Fine-Tuning (CFFT) — the core novel
contribution of SOLAR.

Fine-tunes the CP-SAT generator on contrastive pairs where
the model's own generated code was rejected by OR-Tools
(INFEASIBLE or ERROR), using the verified template code
as the correct supervision target.

Run this on Vast.ai after collecting CFFT pairs with 12a.

Usage:
  # Round 1
  python scripts/12b_cfft_finetune.py --round 1

  # Round 2 (after collecting round 2 pairs)
  python scripts/12b_cfft_finetune.py --round 2

  # Round 3
  python scripts/12b_cfft_finetune.py --round 3
"""

import json
import argparse
import os
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer

# ── Paths ─────────────────────────────────────────────────────────────────────
DRIVE_BASE   = "/root/SOLAR"
MODEL_DIR    = f"{DRIVE_BASE}/models/cpsat_generator"
DATA_DIR     = f"{DRIVE_BASE}/data/processed/transit"
OUTPUT_BASE  = f"{DRIVE_BASE}/models"
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
BASE_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"

# ── LoRA config ───────────────────────────────────────────────────────────────
LORA_CONFIG = LoraConfig(
    r                   = 16,
    lora_alpha          = 32,
    lora_dropout        = 0.05,
    bias                = "none",
    task_type           = TaskType.CAUSAL_LM,
    target_modules      = ["q_proj", "v_proj",
                           "k_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
)


# ── Load CFFT pairs ───────────────────────────────────────────────────────────
def load_cfft_pairs(round_num: int) -> list[dict]:
    path = Path(DATA_DIR) / f"cfft_pairs_round{round_num}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"\nCFFT pairs not found: {path}"
            f"\nRun first: python scripts/12a_collect_cfft_pairs.py --round {round_num}"
        )
    with open(path) as f:
        pairs = json.load(f)
    print(f"  Loaded {len(pairs)} CFFT pairs from round {round_num}")
    return pairs


# ── Build training dataset ────────────────────────────────────────────────────
def build_dataset(pairs: list[dict]) -> Dataset:
    """
    Each CFFT pair has a 'text' field already formatted as a
    Llama-3 instruct prompt with the CORRECT code as completion.
    This is the supervision signal — model learns correct code.
    """
    records = []
    skipped = 0

    for pair in pairs:
        text = pair.get("text", "")
        if not text or len(text) < 50:
            skipped += 1
            continue
        records.append({"text": text})

    print(f"  Training records : {len(records)}")
    print(f"  Skipped          : {skipped}")
    return Dataset.from_list(records)


# ── Load model ────────────────────────────────────────────────────────────────
def load_model_and_tokenizer(adapter_dir: str):
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token           = HF_TOKEN,
        trust_remote_code = True,
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("Tokenizer ✅")

    print("Loading base model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit              = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type       = "nf4",
        bnb_4bit_compute_dtype    = torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config = bnb_config,
        device_map          = "auto",
        token               = HF_TOKEN,
        trust_remote_code   = True,
    )
    print("Base model ✅")

    print(f"Loading LoRA adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        is_trainable = True,
    )
    print(f"Adapter loaded ✅")
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    return model, tokenizer


# ── Training ──────────────────────────────────────────────────────────────────
def run_cfft_training(
    model,
    tokenizer,
    dataset: Dataset,
    output_dir: str,
    round_num: int,
):
    n_pairs = len(dataset)

    # Scale epochs based on dataset size
    # Small dataset (187 pairs) → more epochs for signal
    if n_pairs < 100:
        epochs = 5
    elif n_pairs < 300:
        epochs = 4
    else:
        epochs = 3

    training_args = TrainingArguments(
        output_dir              = output_dir,
        num_train_epochs        = epochs,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        learning_rate           = 1e-4,
        lr_scheduler_type       = "cosine",
        warmup_ratio            = 0.1,
        fp16                    = False,
        bf16                    = True,
        logging_steps           = 5,
        save_steps              = 50,
        save_total_limit        = 2,
        report_to               = "none",
        optim                   = "paged_adamw_8bit",
        dataloader_pin_memory   = False,
        run_name                = f"cfft_round{round_num}",
    )

    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = dataset,
        dataset_text_field = "text",
        max_seq_length  = 1024,
        args            = training_args,
        packing         = False,
    )

    print(f"\n  Starting CFFT Round {round_num} fine-tuning...")
    print(f"  Pairs    : {n_pairs}")
    print(f"  Epochs   : {epochs}")
    print(f"  Output   : {output_dir}\n")

    trainer.train()

    print(f"\n  Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Model saved ✅")

    # Save training metrics
    metrics = {
        "round"         : round_num,
        "n_pairs"       : n_pairs,
        "epochs"        : epochs,
        "train_loss"    : trainer.state.log_history[-1].get("train_loss", None),
        "output_dir"    : output_dir,
    }
    metrics_path = os.path.join(output_dir, "cfft_training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SOLAR Script 12b: CFFT Fine-Tuning")
    parser.add_argument("--round", type=int, default=1,
        help="CFFT round number (1, 2, or 3)")
    parser.add_argument("--base_adapter", default=None,
        help="Path to adapter to start from (defaults to cpsat_generator)")
    args = parser.parse_args()

    print("\n" + "="*55)
    print(f"  SOLAR — Script 12b: CFFT Fine-Tuning Round {args.round}")
    print("="*55)

    # Determine which adapter to start from
    if args.base_adapter:
        adapter_dir = args.base_adapter
    elif args.round == 1:
        # Round 1 starts from the base CP-SAT generator
        adapter_dir = MODEL_DIR
    else:
        # Subsequent rounds start from previous CFFT round
        adapter_dir = f"{OUTPUT_BASE}/cpsat_cfft_round{args.round - 1}"

    output_dir = f"{OUTPUT_BASE}/cpsat_cfft_round{args.round}"

    print(f"\n  Round         : {args.round}")
    print(f"  Base adapter  : {adapter_dir}")
    print(f"  Output        : {output_dir}")

    # Check adapter exists
    if not Path(adapter_dir).exists():
        print(f"\nERROR: Adapter not found: {adapter_dir}")
        if args.round > 1:
            print(f"Run round {args.round-1} first.")
        return

    # Load CFFT pairs
    print(f"\nLoading CFFT pairs...")
    pairs = load_cfft_pairs(args.round)

    if len(pairs) == 0:
        print(f"\nNo CFFT pairs found for round {args.round}.")
        print("This means the model achieved 100% feasibility — done!")
        return

    # Build dataset
    print(f"\nBuilding training dataset...")
    dataset = build_dataset(pairs)

    # Load model
    model, tokenizer = load_model_and_tokenizer(adapter_dir)

    # Run CFFT fine-tuning
    metrics = run_cfft_training(
        model      = model,
        tokenizer  = tokenizer,
        dataset    = dataset,
        output_dir = output_dir,
        round_num  = args.round,
    )

    # Summary
    print(f"\n{'='*55}")
    print(f"  CFFT Round {args.round} Complete!")
    print(f"{'='*55}")
    print(f"  Pairs trained on : {metrics['n_pairs']}")
    print(f"  Epochs           : {metrics['epochs']}")
    print(f"  Final train loss : {metrics['train_loss']}")
    print(f"  Model saved to   : {metrics['output_dir']}")
    print(f"\n  Next steps:")
    if args.round < 3:
        print(f"  1. Download model from {output_dir}")
        print(f"  2. Run inference: python scripts/12a_generate_outputs_colab.py")
        print(f"     (loads from cpsat_cfft_round{args.round})")
        print(f"  3. Collect pairs: python scripts/12a_collect_cfft_pairs.py")
        print(f"     --round {args.round + 1}")
        print(f"  4. Upload cfft_pairs_round{args.round+1}.json to Vast.ai")
        print(f"  5. Run: python scripts/12b_cfft_finetune.py --round {args.round+1}")
    else:
        print(f"  All 3 CFFT rounds complete!")
        print(f"  Run final evaluation: python scripts/13_final_evaluation.py")
    print()


if __name__ == "__main__":
    main()