# Spider SQL — Composition Gap Replication Experiment

**Purpose:** Prove the Composition Gap is universal (Claim 2) by showing it exists in Text-to-SQL, not just labor contract classification.

**Dataset:** Spider 1.0 (Yu et al., EMNLP 2018) — 10,181 NL-to-SQL pairs across 200 databases. CC BY-SA 4.0.

---

## The Analogy

| Concept | Labor Contracts (Ours) | Text-to-SQL (Spider) |
|---------|----------------------|---------------------|
| **Task** | Classify constraint type | Generate SQL query |
| **Pieces** | Threshold, entity, exception, temporal scope | Table selection, column selection, WHERE conditions, aggregation functions |
| **Structure** | HARD / SOFT / HARD-COND / SOFT-COND / NON-CONST | SIMPLE / JOIN / NESTED / SET-OP / MULTI-AGG |
| **Piece-level metric** | Per-component extraction F1 | Per-clause accuracy (SELECT acc, WHERE acc, etc.) |
| **Structure-level metric** | 5-class Macro F1 | Structural type classification accuracy |
| **Expected gap** | 40–60 points | 20–40 points (hypothesis) |

---

## SQL Structural Type Taxonomy (5 classes)

We classify each Spider gold SQL query into one of 5 structural types, analogous to our 5 constraint types:

### 1. SIMPLE (analogous to HARD — most frequent, easiest)
- Single table, no JOIN, no subquery, no set operation
- May have WHERE, ORDER BY, LIMIT
- Example: `SELECT name FROM students WHERE age > 20`

### 2. JOIN (analogous to SOFT — requires relational reasoning)
- Contains one or more JOIN clauses
- No nested subqueries or set operations
- Example: `SELECT T1.name FROM students AS T1 JOIN enrollments AS T2 ON T1.id = T2.student_id WHERE T2.grade = 'A'`

### 3. NESTED (analogous to HARD-CONDITIONAL — requires recursive structure)
- Contains at least one nested subquery (in WHERE, HAVING, or FROM)
- Example: `SELECT name FROM students WHERE age > (SELECT AVG(age) FROM students)`

### 4. SET-OP (analogous to SOFT-CONDITIONAL — requires compositional set reasoning)
- Contains UNION, INTERSECT, or EXCEPT
- Example: `SELECT name FROM students WHERE major = 'CS' INTERSECT SELECT name FROM students WHERE gpa > 3.5`

### 5. MULTI-AGG (analogous to NON-CONSTRAINT — requires complex aggregation composition)
- Contains GROUP BY + HAVING, or multiple aggregation functions, or ORDER BY with aggregation
- No subqueries or set operations (those take priority)
- Example: `SELECT department, COUNT(*) FROM employees GROUP BY department HAVING COUNT(*) > 10`

### Classification Priority (for queries matching multiple types):
SET-OP > NESTED > JOIN > MULTI-AGG > SIMPLE

---

## Measurement Approach

### Piece-Level Accuracy (what models get right)
Using Spider's official component evaluation:
- **SELECT accuracy**: Correct columns and aggregation functions
- **WHERE accuracy**: Correct conditions (column, operator, value)
- **GROUP BY accuracy**: Correct grouping columns
- **ORDER BY accuracy**: Correct ordering columns and direction
- **Keyword accuracy**: Correct SQL keywords (JOIN, GROUP BY, HAVING, etc.)

Average these into a single **Piece F1** score.

### Structure-Level Accuracy (what models get wrong)
- Given the generated SQL, classify it into one of our 5 structural types
- Compare against the gold SQL's structural type
- Compute **Structure Macro F1** across the 5 types

### The Composition Gap
```
Composition Gap = Piece F1 - Structure Macro F1
```

If this gap is 20+ points (similar direction to our 40-60 point gap in contracts), we've proven universality.

---

## Baselines to Test

We test 3-4 LLMs (subset of our contract baselines for direct comparison):

1. **GPT-4o** — zero-shot text-to-SQL
2. **GPT-4o + CoT** — chain-of-thought prompting
3. **DeepSeek-V3** — zero-shot
4. **DeepSeek-V3 + CoT** — chain-of-thought

### Prompt Template
```
Given the following database schema:
{schema}

Generate a SQL query that answers:
{question}

Return ONLY the SQL query, no explanation.
```

### CoT Prompt Template
```
Given the following database schema:
{schema}

Generate a SQL query that answers:
{question}

Think step by step:
1. What tables are needed?
2. What columns should be selected?
3. What conditions apply?
4. What is the structural pattern (simple select, join, subquery, set operation, aggregation)?

Then write the final SQL query.
```

---

## Evaluation Pipeline

1. **Download Spider dev set** (1,034 examples with gold SQL)
2. **Classify gold SQL** into 5 structural types using rule-based parser
3. **Run each baseline** on all 1,034 examples
4. **Compute piece-level** accuracy using Spider's official component evaluation
5. **Classify predicted SQL** into structural types
6. **Compute structure-level** Macro F1
7. **Report the Composition Gap** per model

---

## Expected Results

| Model | Piece F1 | Structure F1 | Gap |
|-------|----------|-------------|-----|
| GPT-4o | ~0.80 | ~0.55 | ~25 |
| GPT-4o-CoT | ~0.82 | ~0.58 | ~24 |
| DeepSeek-V3 | ~0.78 | ~0.50 | ~28 |
| DeepSeek-CoT | ~0.80 | ~0.55 | ~25 |

These are conservative estimates. The key claim is that the gap exists and is consistent across models — the exact numbers matter less than the pattern.

---

## Cost Estimate

- Spider dev set: 1,034 examples
- 4 baselines × 1,034 = 4,136 API calls
- Average ~500 tokens input + 100 tokens output per call
- GPT-4o: ~$5-8 total
- DeepSeek: ~$1-2 total
- **Total: ~$10-15**

No GPU needed — this is API-only.

---

## Files to Create

```
experiments/spider_composition_gap/
├── DESIGN.md                    # This file
├── download_spider.py           # Download and prepare Spider data
├── classify_sql_structure.py    # Rule-based SQL → structural type classifier
├── run_baselines.py             # Run LLM baselines on Spider
├── evaluate_composition_gap.py  # Compute piece-level, structure-level, and gap
└── results/                     # Output directory
```

---

## Timeline

- Day 1: Download Spider, write SQL structural classifier, verify taxonomy distribution
- Day 2: Write baseline prompts, run GPT-4o and DeepSeek
- Day 3: Evaluate results, compute Composition Gap, create comparison table
- **Total: 3 days of work, ~$10-15 in API costs**
