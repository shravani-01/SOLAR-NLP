# SOLAR — Solver-in-the-Loop Constraint Acquisition and Refinement

**Paper:** SOLAR: Solver-in-the-Loop Constraint Acquisition and Refinement for Operational Policy Documents  
**Authors:** Shravani Hariprasad, Kapil, Vinija Jain, Aman Chadha  
**Target:** ARR April 2026 → EMNLP 2026  
**Status:** Phase 6 complete — all baselines done, paper draft live

---

## What is SOLAR?

SOLAR is a neuro-symbolic NLP pipeline that automatically extracts, formalizes, and verifies operational constraints from real policy documents (CBAs, federal regulations, safety specs) using Google OR-Tools CP-SAT as a verification oracle.

SOLAR bridges two communities:
- **Regulatory NLP** — extracts obligations from real documents (Macro F1)
- **Neuro-symbolic CP modeling** — generates verified solver code (feasibility %)

Core contribution: **CFFT (Constraint Faithfulness Fine-Tuning)** — uses CP-SAT solver feasibility signals as training supervision, not just inference-time correction.

---

## Key Results

| Model | Macro F1 | Thresh. | Except. | JSON |
|---|---|---|---|---|
| GPT-4o + CoT | 0.480 | 66.7% | 95.3% | 98.6% |
| GPT-4o + RAG | 0.788 | 71.4% | 72.3% | 100% |
| DeepSeek + CoT | 0.742 | 76.8% | 93.8% | 100% |
| DeepSeek + RAG | 0.697 | 78.6% | 89.2% | 100% |
| **SOLAR (ours)** | **0.858** | **75.0%** | 26.2% | **100%** |

**CFFT feasibility improvement:**

| Round | Feasibility | Pairs | Train Loss |
|---|---|---|---|
| Baseline | 91.2% | 187 | — |
| Round 1 | 94.0% | 108 | 0.251 |
| Round 2 (best) | 94.0% | 152 | 0.190 |

**Exception grounding analysis:**

| Metric | Value |
|---|---|
| Exception extraction — GPT-4o | 95.3% |
| Exception extraction — SOLAR | 26.2% |
| Conditional realization — SOLAR | 94.8% |
| Semantic grounding strict | ~20% |
| Semantic grounding partial | ~55% |

---

## Project Structure
```
SOLAR/
├── data/
│   ├── raw/transit/contracts/     ← MTA CBA PDFs
│   ├── processed/transit/         ← IR JSON, CFFT pairs, outputs
│   ├── annotated/transit/         ← Annotation CSVs
│   ├── training/transit/          ← JSONL training files (gitignored)
│   └── results/                   ← all_results.json, predictions, grounding sample
│
├── scripts/
│   ├── 01_extract_candidates.py       ← PDF → candidate sentences
│   ├── 01b_fix_long_sentences.py      ← Split >80 word sentences
│   ├── 02_annotate.py                 ← Interactive annotation tool
│   ├── 03_auto_annotate.py            ← GPT-4o-mini auto-annotation
│   ├── 04_evaluate_annotations.py     ← Human vs GPT quality eval
│   ├── 05_build_ir.py                 ← CSV → Constraint IR JSON
│   ├── 06_prepare_training_data.py    ← IR → train/val/test JSONL
│   ├── 07_train_ir_extractor.py       ← Llama LoRA: sentence → IR
│   ├── 08_evaluate_ir_extractor.py    ← F1 evaluation (n=211)
│   ├── 09_generate_cpsat.py           ← IR → CP-SAT templates
│   ├── 10_prepare_cpsat_training_data.py ← IR → CP-SAT training JSONL
│   ├── 11_train_cpsat_generator.py    ← Llama LoRA: IR → CP-SAT
│   ├── 12a_generate_outputs_colab.py  ← Run generator (Vast.ai)
│   ├── 12a_collect_cfft_pairs.py      ← OR-Tools execution → CFFT pairs
│   ├── 12b_cfft_finetune.py           ← CFFT fine-tuning Rounds 1-3
│   ├── 13_final_evaluation.py         ← Full baseline comparison
│   ├── test.py                        ← Exception grounding analysis
│   ├── gounding_test.py               ← Grounding validation sample
│   ├── plot1.py                       ← Macro F1 bar chart
│   └── plot2.py                       ← Exception gap figure
│
├── figures/
│   ├── figure2_macro_f1.pdf           ← Macro F1 comparison plot
│   └── figure3_exception_gap.pdf      ← Exception handling 3-level plot
│
└── data/results/
    ├── all_results.json               ← All baseline + SOLAR results
    ├── grounding_validation_sample.json ← 50 manual validation cases
    └── *_predictions.json             ← Per-model predictions
```

---

## Pipeline — Phase Status

| Phase | Description | Status | Key Output |
|---|---|---|---|
| 1 | TransitOpsBench dataset | ✅ | 2,118 annotated sentences |
| 2 | Constraint IR schema | ✅ | constraint_ir.json |
| 3 | IR extractor fine-tuning | ✅ | Macro F1 0.858, JSON 100% |
| 4 | CP-SAT template generator | ✅ | 2,118 snippets, 100% feasible |
| 5a | CP-SAT LLM generator | ✅ | Val loss 0.297 |
| 5b | CFFT inference + pairs | ✅ | 187 pairs, 91.2% feasibility |
| 5c | CFFT fine-tuning R1-R2 | ✅ | 94.0% feasibility (best) |
| 6 | Full baseline experiments | ✅ | All 4 baselines complete |
| 7 | Paper writing | 🔄 | Draft live on Overleaf |

---

## Setup
```bash
pip install pdfplumber pandas openpyxl tqdm spacy \
            transformers ortools openai python-dotenv
```

Create `.env` file:
```
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
```

---

## Running Baselines
```bash
# GPT-4o CoT
python scripts/13_final_evaluation.py --mode gpt4o_cot

# GPT-4o RAG
python scripts/13_final_evaluation.py --mode gpt4o_rag

# DeepSeek CoT
python scripts/13_final_evaluation.py --mode deepseek_cot

# DeepSeek RAG
python scripts/13_final_evaluation.py --mode deepseek_rag
```

---

## OpsBench

**Transit split (complete):** 2,118 annotated sentences from MTA TWU Local 100 CBAs  
**Healthcare split:** In progress (target 500+ sentences)  
**Aviation split:** Planned — FAA 14 CFR Part 117, ALPA CBAs  
**Construction split:** Planned — Carpenters union, IBEW, OSHA 29 CFR 1926  

---

## Infrastructure

| Tool | Purpose | Cost |
|---|---|---|
| Vast.ai A100-SXM4-80GB | Training + inference | $0.94/hr |
| Mac (Apple Silicon) | Data prep, OR-Tools, baselines | Free |
| Google Drive | Model checkpoint storage | Free |
| OpenAI API | GPT-4o baselines | ~$1.30 used |
| DeepSeek API | DeepSeek baselines | ~$0.05 used |

---

## Citation
```bibtex
@article{hariprasad2026solar,
  title   = {SOLAR: Solver-in-the-Loop Constraint 
             Acquisition and Refinement for 
             Operational Policy Documents},
  author  = {Hariprasad, Shravani and Kapil and 
             Jain, Vinija and Chadha, Aman},
  journal = {arXiv preprint},
  year    = {2026}
}
```
