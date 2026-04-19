# GSM8K — Composition Gap Replication Experiment

**Purpose:** Third domain replication proving the Composition Gap is universal — across NLP classification (contracts), formal language generation (SQL), AND mathematical reasoning (math word problems).

**Dataset:** GSM8K (Cobbe et al., 2021) — 8,792 grade-school math problems with step-by-step solutions. MIT License.

---

## The Analogy

| Concept | Labor Contracts | Text-to-SQL | Math (GSM8K) |
|---------|----------------|-------------|--------------|
| **Task** | Classify constraint | Generate SQL | Solve math problem |
| **Pieces** | Threshold, entity, exception | Tables, columns, conditions | Numbers, operations, units |
| **Structure** | HARD / SOFT / COND / etc. | SIMPLE / JOIN / NESTED / etc. | LINEAR / RATIO / MULTI-STEP / etc. |
| **Piece metric** | Component extraction F1 | Per-clause SQL accuracy | Operand & operator extraction |
| **Structure metric** | 5-class Macro F1 | Structural type Macro F1 | Solution structure type Macro F1 |

---

## Math Structural Type Taxonomy (5 classes)

We classify each GSM8K problem by the **structural pattern** of its solution:

### 1. SINGLE-OP (analogous to SIMPLE/HARD)
- Solution requires exactly one arithmetic operation
- Example: "John has 5 apples. He buys 3 more. How many?" → 5 + 3 = 8

### 2. MULTI-STEP (analogous to JOIN)
- Solution requires 2-3 sequential operations (no branching)
- Example: "John earns $10/hr for 8 hours, then spends $30. How much left?" → (10 × 8) - 30 = 50

### 3. RATIO-PROPORTION (analogous to HARD-CONDITIONAL)
- Solution involves ratios, percentages, fractions, or proportional reasoning
- Example: "A shirt costs $40. It's 25% off. What's the price?" → 40 × (1 - 0.25) = 30

### 4. COMPARISON (analogous to SOFT-CONDITIONAL)
- Solution requires computing values for multiple entities then comparing
- Example: "Alice has 3× as many as Bob. Bob has 5. Carol has 2 more than Alice. Who has most?"

### 5. SYSTEM (analogous to NON-CONSTRAINT)
- Solution requires tracking multiple interacting quantities or iterative/recursive logic
- Example: "Each day John saves $5 more than yesterday. Day 1 he saves $10. How much after 5 days?"

### Classification Method
We classify based on the **gold solution steps** (GSM8K provides step-by-step answers):
- Count distinct operations in solution
- Detect ratio/percentage keywords
- Detect comparison patterns
- Detect iterative/accumulation patterns

---

## Measurement Approach

### Piece-Level Accuracy (what models get right)
Extract from the model's response:
- **Number extraction**: Did the model identify the correct numbers from the problem?
- **Operation identification**: Did it use the right operations (+, -, ×, ÷)?
- **Unit tracking**: Did it track units correctly (dollars, hours, items)?
- **Final answer**: Is the numerical answer correct?

Compute average across these components → **Piece F1**

### Structure-Level Accuracy (what models get wrong)
- Classify the model's solution approach into our 5 structural types
- Compare against the gold solution's structural type
- Compute **Structure Macro F1**

### The Composition Gap
```
Composition Gap = Piece F1 - Structure Macro F1
```

---

## Baselines

Same models as Spider for direct comparison:
1. GPT-4o (zero-shot)
2. GPT-4o + CoT
3. DeepSeek-V3 (zero-shot)
4. DeepSeek-V3 + CoT

Plus OSS (on RunPod):
5. Qwen2.5-7B
6. Qwen2.5-7B + CoT

---

## Cost Estimate

- GSM8K test set: 1,319 examples
- 4-6 baselines × 1,319 = ~6,000 API calls
- GPT-4o: ~$8-10
- DeepSeek: ~$1-2
- OSS: ~$2 RunPod time
- **Total: ~$12-15**

---

## Files

```
experiments/gsm8k_composition_gap/
├── DESIGN.md                       # This file
├── download_gsm8k.py               # Download GSM8K from HuggingFace
├── classify_math_structure.py      # Classify solution structural types
├── run_baselines.py                # Run API baselines (GPT-4o, DeepSeek)
├── run_baselines_oss.py            # Run OSS baselines (Qwen, Llama)
├── evaluate_composition_gap.py     # Compute piece-level, structure-level, gap
└── results/
```
