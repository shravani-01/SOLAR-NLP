# SOLAR — Experiment Tracker & Paper Planning Document

**Target:** EMNLP 2026 (May 25 ARR deadline)
**Last updated:** April 18, 2026

---

## Paper Title (Working)

**"The Composition Gap: Why Large Language Models Extract Constraint Pieces but Fail at Structural Classification"**

---

## Core Thesis

LLMs can extract individual constraint components (thresholds, entities, exceptions) at high accuracy (0.88–0.92 F1) but fail catastrophically at composing them into the correct structural type (0.30–0.43 Macro F1 on 5-class classification). We call this the **Composition Gap**. Linear probes show the model internally encodes the correct structure (0.68 F1) but cannot output it. We show that **any SFT — even with natural class distributions — closes the gap** (0.72+ F1), and all SFT variants surpass the probe ceiling. The probes serve as a **diagnostic framework** that explains *why* SFT works: it aligns the decoder with structural representations the model already possesses.

---

## Claims & Experimental Evidence

### Claim 1: The Composition Gap Exists (COMPLETE)

**Finding:** All 7 baselines show a 40–60 point F1 gap between piece-level extraction and structure-level classification.

| Model | Piece-Level F1 | Structure F1 | Gap |
|-------|----------------|-------------|-----|
| GPT-4o | 0.89 | 0.37 | 0.52 |
| GPT-4o-CoT | 0.91 | 0.34 | 0.57 |
| GPT-4o-RAG | 0.92 | 0.40 | 0.52 |
| DeepSeek-V3 | 0.88 | 0.31 | 0.57 |
| DeepSeek-CoT | 0.90 | 0.43 | 0.47 |
| Llama-3.3-70B | 0.89 | 0.31 | 0.58 |
| Qwen2.5-72B | 0.88 | 0.35 | 0.53 |

**Data:** 46,566 annotated examples across 8 domains (healthcare, aviation, education, transit, municipal, construction, building services, hospitality).

**5-class taxonomy:** HARD, SOFT, HARD-CONDITIONAL, SOFT-CONDITIONAL, NON-CONSTRAINT.

**Status:** Complete. Results in `data/results/all_results.json` and `scripts/compare_baselines.py`.

---

### Claim 2: The Gap is Universal (IN PROGRESS)

**Goal:** Show the Composition Gap appears in at least one other domain beyond labor contracts, proving it's a fundamental LLM limitation.

**Candidate domain:** Text-to-SQL (Spider dataset) — SQL queries have pieces (tables, columns, conditions) and structural types (JOIN types, subquery nesting).

**Status:** Week 3 if time permits. Would significantly strengthen the paper.

---

### Claim 3: The Model Knows the Answer Internally (COMPLETE)

**Finding:** Linear probes on Qwen2.5-7B hidden states recover the correct structural type at 0.68 F1 — far above the model's own output (0.35 F1) and approaching the best baseline (0.43 F1).

**Methodology:**
- Linear probes (nn.Linear) on frozen hidden states at 6 layers (-1, -2, -4, -8, -16, -32)
- Stratified 5-fold cross-validation with weight decay sweep [1e-3, 1e-1]
- Control tasks (shuffled labels, Hewitt & Liang 2019) to verify signal vs memorization
- PyTorch GPU implementation (7s/fold on RTX 3090)

**Key Results:**

| Layer | Piece F1 | Structure F1 | Selectivity |
|-------|----------|-------------|------------|
| -1 | 0.7254 | 0.6562 | — |
| -2 | 0.7215 | 0.6644 | — |
| -4 | 0.7234 | 0.6691 | — |
| -8 | 0.7299 | 0.6832 | — |
| -16 | 0.7181 | 0.6587 | — |
| -32 | 0.7029 | 0.6335 | — |

**Probe Composition Gap:** 0.047 at layer -8 (vs 0.40–0.62 for baselines). The model's internal representations nearly close the gap.

**Interpretation:** Structure peaks at layer -8 (early-to-mid), then declines at layer -1 (output). The model computes the correct answer but the generation head suppresses it.

**Status:** Complete. Results in `data/results/linear_probe/`. Figures ready for paper.

---

### Claim 4: SFT Closes the Composition Gap (COMPLETE)

**Finding:** All three SFT strategies close the Composition Gap, jumping from 0.43 to 0.72+ Macro F1. All surpass the probe ceiling (0.6832), indicating SFT not only unlocks existing representations but improves decoder alignment beyond what probes can recover from frozen states.

**Method:** LoRA SFT with three-way ablation study.

**Technical details:**
- Base model: Qwen2.5-7B (ungated, no license needed)
- LoRA: r=16, alpha=32, dropout=0.05, all projection layers
- 4-bit quantization via bitsandbytes (NF4)
- Prompt format: ChatML (Qwen2.5 native)
- Training: lr=2e-4, batch=4x4=16 effective, 5 epochs, cosine scheduler
- Optimizer: paged_adamw_8bit
- Hardware: NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM), RunPod

#### Ablation Results (THREE training strategies)

**A. Vanilla SFT** (`--strategy vanilla`)
- Natural distribution (57% HARD, 18% SOFT, 12% HARD-COND, 7% SOFT-COND, 5% NON-CONST)
- ~7,500 examples, no balancing

**B. Random Balanced SFT** (`--strategy random_balanced`)
- Uniform oversampling: 1,500 per class = 7,500 total

**C. Probe-Guided Error-Targeted SFT** (`--strategy probe_guided`)
- Hard classes (HARD-CONDITIONAL, SOFT-CONDITIONAL, NON-CONSTRAINT): 1,500 each
- Easy classes (HARD, SOFT): 750 each
- Total: ~6,000 examples

#### Final Results

| Strategy | Macro F1 | Improvement over Best Baseline |
|----------|----------|-------------------------------|
| Best baseline (DeepSeek-CoT) | 0.4268 | — |
| Probe ceiling (layer -8) | 0.6832 | +0.2564 |
| **Vanilla SFT** | **0.7164** | **+0.2896** |
| **Probe-guided SFT** | **0.7264** | **+0.2996** |
| **Random balanced SFT** | **0.7311** | **+0.3043** |

#### Per-Class F1 Breakdown

| Class | Baselines (best) | Vanilla | Probe-guided | Random Balanced |
|-------|-----------------|---------|--------------|-----------------|
| HARD | ~0.55 | 0.8993 | 0.8776 | 0.8804 |
| SOFT | ~0.40 | 0.7717 | 0.7711 | 0.7845 |
| HARD-CONDITIONAL | ~0.30 | 0.6743 | 0.6727 | 0.6809 |
| SOFT-CONDITIONAL | ~0.20 | 0.6003 | 0.6541 | 0.6585 |
| NON-CONSTRAINT | ~0.15 | 0.6362 | 0.6567 | 0.6514 |

#### Per-Domain F1

| Domain | Vanilla | Probe-guided | Random Balanced |
|--------|---------|--------------|-----------------|
| Healthcare | 0.7287 | 0.7821 | 0.7819 |
| Aviation | 0.7183 | 0.7289 | 0.7319 |
| Building Services | 0.7017 | 0.6831 | 0.7137 |
| Transit | 0.6846 | 0.7248 | 0.6976 |

#### Key Insights

1. **SFT robustly closes the gap regardless of class distribution.** Even vanilla SFT with heavily imbalanced data achieves 0.7164 — a 30-point jump over the best zero-shot baseline.

2. **All SFT variants surpass the probe ceiling (0.6832).** This means SFT doesn't merely unlock frozen representations — it restructures the decoder-representation interface. The probe ceiling measured what a linear readout could extract from frozen hidden states; SFT changes both the representations and the generation head.

3. **Probe-guided targeting shows modest gains on hard classes.** SOFT-CONDITIONAL improves from 0.6003 (vanilla) to 0.6541 (probe-guided), and NON-CONSTRAINT from 0.6362 to 0.6567. The error-targeting works as intended but the effect is incremental, not transformative.

4. **The Composition Gap is primarily a decoder alignment problem.** The probes correctly diagnosed that the model already encodes structural information internally. SFT aligns the generation head with these representations — any amount of supervised signal suffices.

**Status:** Complete. Results in `data/results/composition_gap/`. All three result JSONs and prediction files saved.

---

## Paper Structure (Revised)

1. **Introduction** — The Composition Gap phenomenon: LLMs extract pieces but fail at structure
2. **Related Work** — Probing (Conneau et al., Hewitt & Liang), constraint extraction, compositional generalization, representation-behavior gaps
3. **The Composition Gap** — Formal definition, measurement methodology, taxonomy
4. **Experimental Setup** — Dataset (8 domains, 46k examples), 7 baselines, 5-class taxonomy, evaluation metrics
5. **Results: The Gap is Real** — Claim 1 (all 7 baselines show 40–60 pt gap across all domains)
6. **Results: The Model Knows** — Claim 3 (linear probes at 0.68 F1, layer-wise analysis, control tasks)
7. **Results: SFT Closes the Gap** — Claim 4 (3-way ablation: all strategies achieve 0.72+, all surpass probe ceiling)
8. **Analysis** — Why all SFT strategies work (decoder alignment hypothesis), per-class gains on minority types, per-domain consistency, probe ceiling surpassed
9. **Discussion** — The Composition Gap as a decoder alignment problem, implications for compositional reasoning, when probing reveals actionable bottlenecks
10. **Conclusion**

**Revised narrative:** The paper's contribution is the *diagnostic framework* — probes identify *where* and *why* LLMs fail at composition, and this diagnosis explains why even simple SFT succeeds. The probes don't just guide the fix; they explain the mechanism.

---

## Dataset Summary

**Name:** TransitOpsBench (working name)

| Domain | Examples | Source |
|--------|----------|--------|
| Healthcare | 8,516 | Labor agreements |
| Aviation | 8,377 | Labor agreements |
| Education | 7,798 | Labor agreements |
| Transit | 6,898 | Labor agreements |
| Municipal | 5,998 | Labor agreements |
| Construction | 4,249 | Labor agreements |
| Building Services | 2,456 | Labor agreements |
| Hospitality | 2,274 | Labor agreements |
| **Total** | **46,566** | |

**Splits:** Train 39,545 / Val 2,791 / Test 4,230

**Class distribution:**
- HARD: 26,760 (57.5%)
- SOFT: 8,526 (18.3%)
- HARD-CONDITIONAL: 5,597 (12.0%)
- SOFT-CONDITIONAL: 3,446 (7.4%)
- NON-CONSTRAINT: 2,237 (4.8%)

---

## Key Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/run_baselines.py` | Run 7 LLM baselines | Complete |
| `scripts/compare_baselines.py` | Compute Composition Gap | Complete |
| `linear_probe/train_probes.py` | PyTorch GPU linear probes | Complete |
| `linear_probe/analyze_results.py` | Probe analysis & figures | Complete |
| `scripts/14_cfft_composition_gap.py` | Ablation study (3 strategies) | Complete |
| `scripts/13_final_evaluation.py` | Legacy 2-class evaluation | Complete (legacy) |

---

## Timeline to Submission

| Week | Task | Status |
|------|------|--------|
| Week 1 (Apr 14–18) | Probe experiment, CFFT training, 3-way ablation | **Complete** |
| Week 2 (Apr 21–25) | Spider SQL replication (Claim 2), post-SFT probing, start paper draft | Planned |
| Week 3 (Apr 28–May 2) | IAA annotation (500 examples), figures & tables, paper writing | Planned |
| Week 4 (May 5–9) | Paper writing, internal review with advisors | Planned |
| Week 5 (May 12–16) | Revisions based on feedback, camera-ready prep | Planned |
| Week 6 (May 19–25) | Final polish, submit by May 25 | Planned |

---

## Compute Budget

| Experiment | GPU | Hours | Cost |
|-----------|-----|-------|------|
| Baselines (7 models) | API calls | — | ~$50 |
| Linear probes | RTX 3090 | 1 hr | $0.46 |
| Probe-guided SFT | RTX PRO 6000 | ~2.5 hr | $4.73 |
| Vanilla SFT ablation | RTX PRO 6000 | ~2.5 hr | $4.73 |
| Random balanced ablation | RTX PRO 6000 | ~2.5 hr | $4.73 |
| Evaluation (3 models) | RTX PRO 6000 | ~3 hr | $5.67 |
| **Total GPU** | | **~11.5 hr** | **~$21** |

---

## Open Issues

1. **Inter-annotator agreement:** No IAA scores yet. Need someone to annotate 500 examples; compute Cohen's kappa. Target: Week 3.
2. **Spider SQL replication (Claim 2):** Critical for EMNLP main. Must show Composition Gap in at least one other domain (Text-to-SQL) to prevent "domain artifact" critique. Target: Week 2.
3. **Post-SFT probing:** Run linear probes on fine-tuned model hidden states to test whether SFT changed internal representations or only the decoder. Strengthens "decoder alignment" narrative. Target: Week 2.
4. **Literature review URLs:** Papers [27]–[30] have placeholder arXiv IDs.
5. **GitHub token exposure:** Token has been exposed — **must be revoked immediately** via GitHub Settings → Developer settings → Personal access tokens.
6. **Reframe probe ceiling narrative:** SFT surpassing probe ceiling is expected — SFT changes both representations and decoder, while probes only read frozen representations. Need careful framing in paper.

---

## Completed Experiments Summary

| Experiment | Key Result | Date |
|-----------|-----------|------|
| 7 LLM baselines | 0.30–0.43 Structure F1 (40–60 pt gap) | Apr 15 |
| Linear probes (6 layers) | 0.6832 F1 at layer -8 | Apr 16 |
| Probe-guided SFT | 0.7264 Macro F1 | Apr 17 |
| Vanilla SFT ablation | 0.7164 Macro F1 | Apr 18 |
| Random balanced SFT ablation | 0.7311 Macro F1 | Apr 18 |
                                         