#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# SOLAR-NLP — Mechanism Experiment: OSS Models (Sequential + Auto-Commit)
# ═══════════════════════════════════════════════════════════════════════
#
# Runs mechanism experiments for each OSS model one at a time.
# After each model finishes ALL 5 domains, results are committed & pushed.
#
# Requires: GPU (RunPod or similar)
#
# Usage:
#   bash run_mechanism_oss.sh              # All OSS models
#   bash run_mechanism_oss.sh 10           # All OSS models, limit 10
#   bash run_mechanism_oss.sh 0 qwen7b    # Single model, no limit
#
# ═══════════════════════════════════════════════════════════════════════

set -e

LIMIT=${1:-""}
SINGLE_MODEL=${2:-""}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MECH_DIR="$SCRIPT_DIR/mechanism"

cd "$MECH_DIR"

LIMIT_ARG=""
if [ -n "$LIMIT" ] && [ "$LIMIT" != "0" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

# OSS models in order
OSS_MODELS=("qwen7b" "qwen7b-cot" "llama8b" "llama8b-cot")

if [ -n "$SINGLE_MODEL" ]; then
    OSS_MODELS=("$SINGLE_MODEL")
fi

TOTAL=${#OSS_MODELS[@]}
COUNT=0

echo "═══════════════════════════════════════════════════════════════"
echo "  SOLAR-NLP Mechanism — OSS Models (Sequential + Auto-Commit)"
echo "  Models: ${OSS_MODELS[*]}"
echo "  Limit: ${LIMIT:-none}"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "  GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  (could not query GPU)"
else
    echo ""
    echo "  ⚠ WARNING: nvidia-smi not found. OSS models require GPU!"
    echo "  Continuing anyway (will fail at model load if no GPU)..."
fi

for MODEL in "${OSS_MODELS[@]}"; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║  Model $COUNT/$TOTAL: $MODEL"
    echo "║  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "╚═══════════════════════════════════════════════════════════╝"

    for DOMAIN in contracts sql math logic code; do
        echo ""
        echo "  ▶ [$MODEL] $DOMAIN"
        echo "  ─────────────────────────────────────────────"
        python "run_${DOMAIN}.py" --model "$MODEL" --backend oss $LIMIT_ARG
        echo "  ✓ [$MODEL] $DOMAIN complete"
    done

    echo ""
    echo "  ▶ [$MODEL] Running cross-domain analysis..."
    python analyze_mechanism.py --model "$MODEL" 2>/dev/null || true
    echo "  ✓ [$MODEL] Analysis complete"

    # ── Git commit & push (with retry for parallel safety) ─────────
    echo ""
    echo "  ▶ [$MODEL] Committing results..."
    cd "$REPO_DIR"

    git pull --rebase --quiet 2>/dev/null || true

    git add experiments_v2/mechanism/results/ || true
    git add experiments_v2/mechanism/analysis/ 2>/dev/null || true

    if git diff --cached --quiet; then
        echo "  ⚠ [$MODEL] No new results to commit (skipping)"
    else
        git commit -m "mechanism results: $MODEL (all 5 domains)

Experiment: Section 7 — Implicit vs Explicit Structure
Model: $MODEL | Backend: oss | Limit: ${LIMIT:-full}
Domains: contracts, sql, math, logic, code
Conditions: original, hint_correct, hint_wrong
Timestamp: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

        for attempt in 1 2 3; do
            if git push 2>/dev/null; then
                echo "  ✓ [$MODEL] Committed & pushed"
                break
            else
                echo "  ⚠ Push attempt $attempt failed, pulling & retrying..."
                git pull --rebase --quiet 2>/dev/null || true
                sleep 2
            fi
        done
    fi

    cd "$MECH_DIR"

    echo ""
    echo "  ═══════════════════════════════════════════════"
    echo "  ✓ MODEL COMPLETE: $MODEL ($COUNT/$TOTAL)"
    echo "  ═══════════════════════════════════════════════"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ALL OSS MECHANISM EXPERIMENTS COMPLETE!"
echo "  Models run: ${OSS_MODELS[*]}"
echo "  Results in: $MECH_DIR/results/"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════"
