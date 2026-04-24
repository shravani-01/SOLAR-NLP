#!/usr/bin/env python3
"""
Rule-based logical reasoning structural type classifier.

Classifies each FOLIO problem by its reasoning structure:
  1. MODUS-PONENS     — If P then Q; P is true; therefore Q (direct implication)
  2. SYLLOGISM        — All A are B; X is A; therefore X is B (categorical reasoning)
  3. DISJUNCTIVE      — P or Q; not P; therefore Q (elimination reasoning)
  4. CONDITIONAL-CHAIN — If A then B; if B then C; A; therefore C (transitive)
  5. NEGATION         — Proof by contradiction, "not all", double negation, contrapositive

Priority: CONDITIONAL-CHAIN > NEGATION > DISJUNCTIVE > SYLLOGISM > MODUS-PONENS

Works on both premise text and FOL annotations when available.
"""

import re
import json
import sys
from pathlib import Path
from collections import Counter


# ─── Structural Type Constants ───────────────────────────────────────────────

MODUS_PONENS = "MODUS-PONENS"
SYLLOGISM = "SYLLOGISM"
DISJUNCTIVE = "DISJUNCTIVE"
CONDITIONAL_CHAIN = "CONDITIONAL-CHAIN"
NEGATION = "NEGATION"

STRUCTURAL_TYPES = [MODUS_PONENS, SYLLOGISM, DISJUNCTIVE, CONDITIONAL_CHAIN, NEGATION]


# ─── Classification helpers ─────────────────────────────────────────────────

def _count_conditionals(premises: list) -> int:
    """Count premises that express conditional (if-then) relationships."""
    count = 0
    for p in premises:
        p_lower = p.lower().strip()
        if re.search(r'\bif\b.*\bthen\b', p_lower):
            count += 1
        elif re.search(r'\bwhenever\b|\bwhen\b.*,', p_lower):
            count += 1
        elif re.search(r'\bimplies\b|\bentails\b', p_lower):
            count += 1
    return count


def _has_universal_quantifier(premises: list) -> bool:
    """Check for universal statements (All X are Y, Every X is Y)."""
    for p in premises:
        p_lower = p.lower().strip()
        if re.search(r'\b(all|every|each|any)\b\s+\w+\s+(are|is|have|has|can|will|must)\b', p_lower):
            return True
        if re.search(r'\bno\s+\w+\s+(are|is|have|has|can|will)\b', p_lower):
            return True  # "No X are Y" is also universal
    return False


def _has_disjunction(premises: list) -> bool:
    """Check for disjunctive premises (P or Q, either...or)."""
    for p in premises:
        p_lower = p.lower().strip()
        if re.search(r'\beither\b.*\bor\b', p_lower):
            return True
        if re.search(r'\bor\b', p_lower) and not re.search(r'\bfor\b|\bmore\b|\bbefore\b', p_lower):
            # "or" that's not part of "for", "more", "before"
            # Check it's a logical disjunction, not just casual "or"
            if re.search(r'[,;]\s*or\b|\bor\s+\w+\s+(is|are|will|can|has)\b', p_lower):
                return True
    return False


def _has_negation_pattern(premises: list, conclusion: str) -> bool:
    """Check for negation-heavy reasoning patterns."""
    neg_count = 0
    for p in premises:
        p_lower = p.lower()
        if re.search(r'\bnot\b|\bno\b|\bnever\b|\bnor\b|\bneither\b|\bcannot\b|\bdon\'t\b|\bdoesn\'t\b', p_lower):
            neg_count += 1
        if re.search(r'\bnot all\b|\bnot every\b', p_lower):
            return True  # Explicit "not all" is strong negation signal

    conc_lower = conclusion.lower() if conclusion else ""
    if re.search(r'\bnot\b|\bno\b|\bnever\b|\bcannot\b', conc_lower):
        neg_count += 1

    # Strong negation pattern: multiple negations or conclusion is negative
    return neg_count >= 2


def _count_chained_conditions(premises: list) -> int:
    """Count potential chain links: if A then B, if B then C forms a chain."""
    conditionals = []
    for p in premises:
        p_lower = p.lower().strip()
        match = re.search(r'if\s+(.+?)\s*,?\s*then\s+(.+?)\.?$', p_lower)
        if match:
            conditionals.append((match.group(1).strip(), match.group(2).strip()))

    if len(conditionals) < 2:
        return 0

    # Check for chains: consequent of one matches antecedent of another
    chains = 0
    for i, (a1, c1) in enumerate(conditionals):
        for j, (a2, c2) in enumerate(conditionals):
            if i != j:
                # Check if c1 overlaps with a2 (loose word matching)
                c1_words = set(c1.split())
                a2_words = set(a2.split())
                overlap = len(c1_words & a2_words)
                if overlap >= 2 or (overlap >= 1 and len(c1_words) <= 3):
                    chains += 1

    return chains


def _has_individual_instance(premises: list) -> bool:
    """Check if premises contain specific individual instances (proper nouns, 'X is a Y')."""
    for p in premises:
        # Proper nouns (capitalized words not at start of sentence)
        words = p.split()
        for i, w in enumerate(words):
            if i > 0 and w[0].isupper() and w.lower() not in {
                'the', 'a', 'an', 'if', 'then', 'and', 'or', 'not', 'all', 'every',
                'some', 'no', 'is', 'are', 'has', 'have', 'will', 'can', 'but', 'either',
                'neither', 'nor', 'when', 'whenever', 'therefore', 'however', 'also',
                'both', 'each', 'any', 'this', 'that', 'these', 'those',
            }:
                return True
        # "X is a Y" pattern
        if re.search(r'\b\w+\s+is\s+a\s+\w+', p):
            return True
    return False


# ─── Main classifier ───────────────────────────────────────────────────────

def classify_logic(premises: list, conclusion: str = "", fol_premises: list = None) -> str:
    """Classify a logical reasoning problem into a structural type.

    Args:
        premises: List of natural language premise strings.
        conclusion: The conclusion string.
        fol_premises: Optional FOL annotations (for additional signal).

    Returns:
        One of STRUCTURAL_TYPES.
    """
    if not premises:
        return MODUS_PONENS

    n_conditionals = _count_conditionals(premises)
    has_universal = _has_universal_quantifier(premises)
    has_disjunction = _has_disjunction(premises)
    has_negation = _has_negation_pattern(premises, conclusion)
    chain_count = _count_chained_conditions(premises)
    has_instance = _has_individual_instance(premises)

    # CONDITIONAL-CHAIN: 2+ conditionals that form a chain
    if chain_count >= 1 and n_conditionals >= 2:
        return CONDITIONAL_CHAIN

    # NEGATION: strong negation patterns (not all, proof by contradiction)
    if has_negation and not has_disjunction:
        return NEGATION

    # DISJUNCTIVE: explicit either/or with elimination
    if has_disjunction:
        return DISJUNCTIVE

    # SYLLOGISM: universal quantifier + specific instance
    if has_universal and has_instance:
        return SYLLOGISM

    # SYLLOGISM: universal quantifier + another universal (All A are B, All B are C)
    if has_universal and n_conditionals == 0:
        return SYLLOGISM

    # MODUS-PONENS: simple if-then with instance
    if n_conditionals >= 1:
        return MODUS_PONENS

    # Default based on premise count
    if len(premises) >= 4:
        return CONDITIONAL_CHAIN  # Many premises suggest chaining
    if has_universal:
        return SYLLOGISM

    return MODUS_PONENS


def classify_response(premises: list, conclusion: str, response: str) -> str:
    """Classify a model's reasoning response.

    Analyzes the response text for structural patterns used in the reasoning.
    """
    if not response:
        return classify_logic(premises, conclusion)

    resp_lower = response.lower()

    # Check for explicit structural patterns in the response
    chain_signals = len(re.findall(r'\btherefore\b|\bthus\b|\bso\b|\bhence\b', resp_lower))
    neg_signals = len(re.findall(r'\bnot\b|\bno\b|\bcontradiction\b|\bcontrapositive\b', resp_lower))
    disj_signals = len(re.findall(r'\beither\b|\bor\b.*\beliminate\b|\bdisjunct', resp_lower))

    # If the response explicitly uses chaining (multiple therefore/thus)
    if chain_signals >= 3:
        return CONDITIONAL_CHAIN

    # If the response uses contradiction/contrapositive reasoning
    if re.search(r'\bcontradiction\b|\bcontrapositive\b|\bproof by\b', resp_lower):
        return NEGATION

    # Fall back to premise-based classification
    return classify_logic(premises, conclusion)


# ─── Batch classification ───────────────────────────────────────────────────

def classify_dataset(data_path: str, output_path: str = None):
    """Classify all problems in a FOLIO dataset file."""
    with open(data_path) as f:
        data = json.load(f)

    type_counts = Counter()

    for item in data:
        premises = item.get('premises', [])
        conclusion = item.get('conclusion', '')
        fol_premises = item.get('premises-FOL', None)

        stype = classify_logic(premises, conclusion, fol_premises)
        item['structural_type'] = stype
        type_counts[stype] += 1

    total = len(data)
    print(f"\nStructural Type Distribution ({total} problems):")
    print(f"{'Type':<20} {'Count':>6} {'Pct':>8}")
    print(f"{'-'*35}")
    for stype in STRUCTURAL_TYPES:
        count = type_counts[stype]
        pct = count / total * 100 if total > 0 else 0
        print(f"{stype:<20} {count:>6} {pct:>7.1f}%")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved to {output_path}")

    return data, type_counts


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify_logic_structure.py <data.json> [output.json]")
        sys.exit(1)

    data_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    classify_dataset(data_path, output_path)
