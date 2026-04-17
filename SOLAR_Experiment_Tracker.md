# SOLAR — Experiment Tracker & Paper Planning Document

**Target:** EMNLP 2026 (May 25 ARR deadline)
**Last updated:** April 17, 2026

---

## Paper Title (Working)

**"The Composition Gap: Why Large Language Models Extract Constraint Pieces but Fail at Structural Classification"**

---

## Core Thesis

LLMs can extract individual constraint components (thresholds, entities, exceptions) at high accuracy (0.88–0.92 F1) but fail catastrophically at composing them into the correct structural type (0.30–0.43 Macro F1 on 5-class classification). We call this the **Composition Gap**. Linear probes show the model internally encodes the correct structure (0.68 F1) but cannot output it. We introduce a **probe-guided diagnostic-to-intervention framework** that closes the gap through error-targeted fine-tuning.

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

### Claim 4: Probe-Guided Intervention Closes the Gap (RUNNING)

**Goal:** Show that targeted fine-tuning, guided by the probe's error analysis, can close the Composition Gap where naive approaches fail.

**Method:** Error-Targeted Contrastive SFT with ablation study.

**Technical details:**
- Base model: Qwen2.5-7B (ungated, no license needed)
- LoRA: r=16, alpha=32, dropout=0.05, all projection layers
- 4-bit quantization via bitsandbytes (NF4)
- Prompt format: ChatML (Qwen2.5 native)
- Training: lr=2e-4, batch=4×4=16 effective, 5 epochs, cosine scheduler
- Optimizer: paged_adamw_8bit

#### Ablation Study (THREE training strategies)

This is the key methodological contribution. We run three variants to prove that the probe diagnosis is essential:

**A. Vanilla SFT** (`--strategy vanilla`)
- Natural distribution (57% HARD, 18% SOFT, 12% HARD-COND, 7% SOFT-COND, 5% NON-CONST)
- ~7,500 examples, no balancing
- **Expected:** Model predicts HARD for everything. Minimal improvement.
- **Purpose:** Shows that naive SFT doesn't solve the problem.

**B. Random Balanced SFT** (`--strategy random_balanced`)
- Uniform oversampling: 1,500 per class = 7,500 total
- All classes treated equally, no error analysis
- **Expected:** Marginal improvement on minority classes, but not targeted.
- **Purpose:** Shows that generic class balancing isn't enough.

**C. Probe-Guided Error-Targeted SFT** (`--strategy probe_guided`)
- Uses probe analysis to identify which classes have the worst Composition Gap
- Hard classes (HARD-CONDITIONAL, SOFT-CONDITIONAL, NON-CONSTRAINT): 1,500 each
- Easy classes (HARD, SOFT): 750 each
- Total: ~6,000 examples, deliberately skewed toward failure modes
- **Expected:** Significant improvement, especially on the three hard classes.
- **Purpose:** Shows that the probe diagnosis → targeted intervention pipeline works.

**Running on:** NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM), RunPod

**Current status:** probe_guided strategy training (~2,345 steps, loss dropped to 0.20). Ablation variants ready to run.

**Target results:**

| Strategy | Expected Macro F1 | Paper Narrative |
|----------|-------------------|-----------------|
| Vanilla SFT | ~0.35–0.40 | "Naive SFT fails" |
| Random Balanced | ~0.42–0.48 | "Generic balancing helps marginally" |
| Probe-Guided | ~0.55–0.65 | "Targeted intervention closes the gap" |
| Probe ceiling | 0.6832 | Upper bound from Claim 3 |
| Best baseline | 0.4268 | DeepSeek-CoT (no fine-tuning) |

---

## Paper Structure (Planned)

1. **Introduction** — The Composition Gap phenomenon
2. **Related Work** — Probing (Conneau et al.), constraint extraction, compositional generalization
3. **The Composition Gap** — Definition, measurement methodology
4. **Experimental Setup** — Dataset (8 domains, 46k examples), baselines, evaluation
5. **Results: The Gap is Real** — Claim 1 (7 baselines all show 40–60 pt gap)
6. **Results: The Model Knows** — Claim 3 (linear probes at 0.68 F1)
7. **Results: Closing the Gap** — Claim 4 (ablation study, probe-guided intervention)
8. **Analysis** — Per-class breakdown, per-domain generalization, layer-wise patterns
9. **Discussion** — Implications for compositional reasoning in LLMs
10. **Conclusion**

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
| `scripts/14_cfft_composition_gap.py` | Ablation study (3 strategies) | Running |
| `scripts/13_final_evaluation.py` | Legacy 2-class evaluation | Complete (legacy) |

---

## Timeline to Submission

| Week | Task | Status |
|------|------|--------|
| Week 1 (Apr 14–18) | Probe experiment, CFFT training, ablation setup | In Progress |
| Week 2 (Apr 21–25) | Ablation results, start paper draft | Planned |
| Week 3 (Apr 28–May 2) | Spider SQL replication (Claim 2), figures | Planned |
| Week 4 (May 5–9) | Paper writing, internal review | Planned |
| Week 5 (May 12–16) | Revisions, camera-ready prep | Planned |
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

1. **Inter-annotator agreement:** No IAA scores yet. Need at least a sample validation.
2. **Spider SQL replication:** Not started. Would strengthen universality claim.
3. **Literature review URLs:** Papers [27]–[30] have placeholder arXiv IDs.
4. **GitHub token exposure:** needs to be revoked.
