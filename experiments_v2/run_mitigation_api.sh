#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# SOLAR-NLP — Mitigation Experiment: API Models (Sequential + Auto-Commit)
# ═══════════════════════════════════════════════════════════════════════
#
# Runs mitigation experiments for each API model one at a time.
# After each model finishes ALL 5 domains, results are committed & pushed.
#
# Usage:
#   bash run_mitigation_api.sh              # All API models
#   bash run_mitigation_api.sh 10           # All API models, limit 10
#   bash run_mitigation_api.sh 0 gpt4o     # Single model, no limit
#
# ═══════════════════════════════════════════════════════════════════════

set -e

LIMIT=${1:-""}
SINGLE_MODEL=${2:-""}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MIT_DIR="$SCRIPT_DIR/mitigation"

cd "$MIT_DIR"

LIMIT_ARG=""
if [ -n "$LIMIT" ] && [ "$LIMIT" != "0" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

# API models in order
API_MODELS=("gpt4o" "gpt4o-cot" "deepseek" "deepseek-cot")

if [ -n "$SINGLE_MODEL" ]; then
    API_MODELS=("$SINGLE_MODEL")
fi

TOTAL=${#API_MODELS[@]}
COUNT=0

echo "═══════════════════════════════════════════════════════════════"
echo "  SOLAR-NLP Mitigation — API Models (Sequential + Auto-Commit)"
echo "  Models: ${API_MODELS[*]}"
echo "  Limit: ${LIMIT:-none}"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════"

for MODEL in "${API_MODELS[@]}"; do
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
        python "run_${DOMAIN}.py" --model "$MODEL" --backend api $LIMIT_ARG
        echo "  ✓ [$MODEL] $DOMAIN complete"
    done

    # ── Git commit & push (with retry for parallel safety) ─────────
    echo ""
    echo "  ▶ [$MODEL] Committing results..."
    cd "$REPO_DIR"

    git pull --rebase --quiet 2>/dev/null || true

    git add experiments_v2/mitigation/results/ || true

    if git diff --cached --quiet; then
        echo "  ⚠ [$MODEL] No new results to commit (skipping)"
    else
        git commit -m "mitigation results: $MODEL (all 5 domains)

Experiment: Section 8 — Structure-Aware Prompting
Model: $MODEL | Backend: api | Limit: ${LIMIT:-full}
Domains: contracts, sql, math, logic, code
Conditions: self_structure, cot_structure
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

    cd "$MIT_DIR"

    echo ""
    echo "  ═══════════════════════════════════════════════"
    echo "  ✓ MODEL COMPLETE: $MODEL ($COUNT/$TOTAL)"
    echo "  ═══════════════════════════════════════════════"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ALL API MITIGATION EXPERIMENTS COMPLETE!"
echo "  Models run: ${API_MODELS[*]}"
echo "  Results in: $MIT_DIR/results/"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════"
