# When Structure Is Hidden: Measuring and Diagnosing the Compositional Gap in LLMs

<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/shravani-01/composition-gap/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/EMNLP-2026-green.svg" alt="EMNLP 2026">
  <img src="https://img.shields.io/badge/python-3.9+-orange.svg" alt="Python">
</p>

<p align="center">
  <b>Shravani Hariprasad</b><sup>1</sup> &nbsp;&nbsp;
  <b>Mohsin Ali Mohammed</b><sup>2</sup> &nbsp;&nbsp;
  <b>Vinija Jain</b><sup>3</sup> &nbsp;&nbsp;
  <b>Aman Chadha</b><sup>4</sup>
  <br><br>
  <sup>1</sup>Hatch &nbsp;&nbsp;
  <sup>2</sup>Telaeris &nbsp;&nbsp;
  <sup>3</sup>Google &nbsp;&nbsp;
  <sup>4</sup>Google DeepMind
  <br><br>
  Correspondence: <a href="mailto:hariprasadshravani@gmail.com">hariprasadshravani@gmail.com</a>
</p>

---

## Abstract

Large language models (LLMs) often recover the local pieces of a problem yet still fail to assemble them into the correct global solution. Prior work studies this *compositionality gap* mainly in settings with explicit sub-question decompositions. We extend that analysis to tasks where the underlying structure is **implicit** and must be inferred from the input itself.

We introduce a **two-pass evaluation framework** that first verifies piece-level understanding and then measures composition on the same example using the conditional metric P(composed wrong | all pieces correct).

Across **five domains** (contracts, math, SQL, code, and logic) and **eight model configurations** spanning four model families, we find that the gap grows systematically with structural implicitness. The pattern holds across GPT-4o, DeepSeek, Qwen2.5-7B, and Llama-3.1-8B, and Chain-of-Thought prompting yields negligible improvement.

A controlled three-condition mechanism study shows that correct structural hints substantially reduce the gap in contracts and SQL while wrong hints increase failure, consistent with a structure-inference bottleneck. Structure-aware prompting is capability-dependent: it helps GPT-4o but worsens smaller open-source models on the same domain.

---

## Key Results

### Composition Gap Across Domains

| Domain | Implicitness | GPT-4o | DeepSeek | Qwen-7B | Llama-8B |
|--------|-------------|--------|----------|---------|----------|
| Contracts | Low | 5.4% | 0.0% | 1.7% | 2.2% |
| Math | Med-Low | 8.1% | 4.1% | 10.4% | 17.3% |
| SQL | Medium | 21.1% | 17.9% | 25.6% | 31.1% |
| Code | High | 90.2%* | 36.0% | 40.9% | 57.9% |
| Logic | Very High | 50.0% | 60.0% | 57.5% | 46.5% |

*GPT-4o code value is an outlier; see paper for discussion.

Gap = P(composed wrong | all pieces correct). Higher values indicate greater composition failure.

### Mechanism Study

| Domain | Original | Correct Hint | Wrong Hint | Verdict |
|--------|----------|-------------|-----------|---------|
| Contracts | 2.3% | 0.6% (-1.7pp) | 39.5% (+37.2pp) | Closed |
| SQL | 22.1% | 5.9% (-16.2pp) | 56.8% (+34.7pp) | Closed |
| Math | 9.2% | 9.1% (-0.1pp) | 10.7% (+1.5pp) | No effect |
| Code | 55.7% | 41.3% (-14.4pp) | 71.3% (+15.6pp) | Mixed |
| Logic | 54.2% | 49.9% (-4.3pp) | 52.6% (-1.6pp) | Weak |

### Error Taxonomy (916 gap cases)

| Failure Type | Count | Share |
|-------------|-------|-------|
| Partial composition | 370 | 40.4% |
| Correct structure, wrong execution | 217 | 23.7% |
| Hallucinated structure | 173 | 18.9% |
| Wrong structure selection | 156 | 17.0% |

Inter-annotator agreement: κ = 0.74 (Cohen's kappa, 200 cases).

---

## Framework

```
Input Example
      │
      ▼
┌─────────────────────────────────────┐
│           PASS 1 (Pieces)           │
│  Q1: piece question 1               │
│  Q2: piece question 2               │
│  Q3: piece question 3 (if needed)   │
└─────────────────────────────────────┘
      │
      ├── Any wrong → NOT gap-eligible (excluded from metric)
      │
      └── All correct → gap-eligible
                              │
                              ▼
                  ┌───────────────────────┐
                  │    PASS 2 (Composed)  │
                  │  Full task answer     │
                  └───────────────────────┘
                              │
                              ├── Correct → not a gap case
                              │
                              └── Wrong → GAP CASE
                                            │
                                  ┌─────────┴──────────┐
                                  ▼                    ▼
                           Mechanism Study       Error Analysis
                           (correct/wrong hint)  (failure taxonomy)
```

**Metric:** Gap = P(composed wrong | all pieces correct)

This conditional definition removes the confound that composed tasks may simply be harder overall. Only examples where the model demonstrably understands all constituent pieces enter the denominator.

---

## Domains

| Domain | Dataset | N | Structure Type | Implicitness |
|--------|---------|---|---------------|-------------|
| Contracts | CUAD | 510 | Implicit language cues | Low |
| Math | GSM8K | 500 | Explicit equations | Med-Low |
| SQL | Spider | 500 | Semi-structured queries | Medium |
| Code | HumanEval | 164 | Compositional execution | High |
| Logic | FOLIO | 204 | Abstract formal rules | Very High |

---

## Repository Structure

```
composition-gap/
├── README.md
├── requirements.txt
│
├── data/
│   ├── contracts/          # Sampled CUAD clauses + Pass-1/2 annotations
│   ├── math/               # Sampled GSM8K problems
│   ├── sql/                # Sampled Spider examples
│   ├── code/               # HumanEval problems + structure annotations
│   └── logic/              # FOLIO examples
│
├── evaluation/
│   ├── pass1_scoring.py    # Domain-specific Pass-1 scoring heuristics
│   ├── pass2_scoring.py    # Pass-2 evaluation per domain
│   ├── gap_metric.py       # Conditional composition gap computation
│   └── bootstrap_ci.py     # Bootstrap confidence intervals (10k resamples)
│
├── experiments/
│   ├── run_baselines.py    # Run all 8 model configurations
│   ├── run_mechanism.py    # Three-condition hint study
│   ├── run_mitigation.py   # Structure-aware prompting experiments
│   └── error_analysis/
│       ├── annotate.py     # Gap case annotation tool
│       └── taxonomy.py     # Four-category error classifier
│
├── models/
│   ├── api_callers.py      # GPT-4o, DeepSeek API wrappers
│   └── local_models.py     # Qwen, Llama inference (HuggingFace)
│
├── prompts/
│   ├── contracts/          # Pass-1 and Pass-2 prompts for contracts
│   ├── math/               # Pass-1 and Pass-2 prompts for math
│   ├── sql/                # Pass-1 and Pass-2 prompts for SQL
│   ├── code/               # Pass-1 and Pass-2 prompts for code
│   ├── logic/              # Pass-1 and Pass-2 prompts for logic
│   └── mechanism/          # Correct-hint and wrong-hint prompt templates
│
├── results/
│   ├── baselines/          # Raw model outputs per domain
│   ├── mechanism/          # Hint study outputs
│   └── mitigation/         # Structure-aware prompting outputs
│
├── analysis/
│   ├── compute_gap.py      # Main gap computation script
│   ├── sensitivity.py      # Pass-1 noise sensitivity analysis
│   └── figures/            # Figure generation scripts
│
└── paper/
    ├── main.tex
    └── figures/
```

---

## Installation

```bash
git clone https://github.com/shravani-01/composition-gap.git
cd composition-gap
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- openai >= 1.0.0
- transformers >= 4.40.0
- torch >= 2.1.0
- scipy, numpy, pandas
- tqdm

**API keys** — create a `.env` file:

```
OPENAI_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
```

---

## Reproducing Results

### Step 1 — Run baselines (all domains, all models)

```bash
# GPT-4o on contracts
python experiments/run_baselines.py \
  --model gpt4o \
  --domain contracts \
  --pass1_prompt prompts/contracts/pass1.txt \
  --pass2_prompt prompts/contracts/pass2.txt

# All models, all domains
python experiments/run_baselines.py --model all --domain all
```

### Step 2 — Compute the composition gap

```bash
python analysis/compute_gap.py \
  --results_dir results/baselines/ \
  --output gap_table.csv
```

### Step 3 — Run the mechanism study

```bash
python experiments/run_mechanism.py \
  --domain contracts \
  --conditions original correct_hint wrong_hint \
  --model gpt4o
```

### Step 4 — Run structure-aware prompting

```bash
python experiments/run_mitigation.py \
  --domain sql \
  --strategy self_structure \
  --model gpt4o
```

### Step 5 — Sensitivity analysis

```bash
python analysis/sensitivity.py \
  --fpr_rates 0.0 0.05 0.10 0.15 \
  --gap_results results/baselines/gap_table.csv
```

---

## Per-Domain Pass Design

**Contracts (CUAD)**
- Pass 1 Q1: Is the obligation mandatory, discretionary, or neither?
- Pass 1 Q2: Does this clause contain a condition?
- Pass 2: What is the full constraint type? (HARD / SOFT / HARD-CONDITIONAL / SOFT-CONDITIONAL / NON-CONSTRAINT)

**Math (GSM8K)**
- Pass 1 Q1: What quantities are given?
- Pass 1 Q2: What operations are needed?
- Pass 2: Solve the problem. Final answer after ####.

**SQL (Spider)**
- Pass 1 Q1: Which tables are needed?
- Pass 1 Q2: What filter conditions are required?
- Pass 1 Q3: What aggregation is needed?
- Pass 2: Write the complete SQL query.

**Code (HumanEval)**
- Pass 1 Q1: Does this require iteration, recursion, or both?
- Pass 1 Q2: What data structure or structural pattern is central?
- Pass 2: Write the complete function.

**Logic (FOLIO)**
- Pass 1 Q1: What entities are involved?
- Pass 1 Q2: What is the coarse logical form?
- Pass 2: Is the conclusion true, false, or unknown?

---

## Models

| Model | Size | Provider | Access |
|-------|------|----------|--------|
| GPT-4o | Proprietary | OpenAI | API |
| GPT-4o + CoT | Proprietary | OpenAI | API |
| DeepSeek-V3 | Proprietary | DeepSeek | API |
| DeepSeek-V3 + CoT | Proprietary | DeepSeek | API |
| Qwen2.5-7B-Instruct | 7B | Alibaba | HuggingFace |
| Qwen2.5-7B + CoT | 7B | Alibaba | HuggingFace |
| Llama-3.1-8B-Instruct | 8B | Meta | HuggingFace |
| Llama-3.1-8B + CoT | 8B | Meta | HuggingFace |

All experiments use deterministic decoding (temperature = 0) where supported.

---

## Statistical Reporting

All gap rates are paired with 95% bootstrap confidence intervals (10,000 resamples). Cross-domain and CoT comparisons use 10,000-permutation tests.

- 0 / 20 CoT comparisons reach significance
- 41 / 48 cross-domain comparisons reach significance (p < 0.05)

---

## Datasets and Licenses

| Dataset | License | Source |
|---------|---------|--------|
| CUAD | CC BY 4.0 | Hendrycks et al., 2021 |
| Spider | CC BY-SA 4.0 | Yu et al., 2018 |
| GSM8K | MIT | Cobbe et al., 2021 |
| HumanEval | MIT | Chen et al., 2021 |
| FOLIO | CC BY 4.0 | Han et al., 2022 |

No new datasets are introduced. All sampled example IDs are provided in `data/`.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{hariprasad2026composition,
  title     = {When Structure Is Hidden: Measuring and Diagnosing the Compositional Gap in LLMs},
  author    = {Hariprasad, Shravani and Mohammed, Mohsin Ali and Jain, Vinija and Chadha, Aman},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

---

## Contact

For questions about the code or paper, please open a GitHub issue or contact [hariprasadshravani@gmail.com](mailto:hariprasadshravani@gmail.com).