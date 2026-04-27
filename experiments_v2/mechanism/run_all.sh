#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# SOLAR-NLP — Mechanism Experiment: Run All Domains
# ═══════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash run_all.sh                    # All API models, all domains
#   bash run_all.sh gpt4o              # Single model, all domains
#   bash run_all.sh gpt4o 10           # Single model, limit 10 examples
#   bash run_all.sh all 0 oss          # All OSS models, no limit
#
# Prerequisites:
#   - Baseline predictions must exist in experiments_v2/{domain}/results/
#   - API keys must be set in .env (for API models)
#   - GPU required for OSS models
#
# ═══════════════════════════════════════════════════════════════════════

set -e

MODEL=${1:-"all"}
LIMIT=${2:-""}
BACKEND=${3:-"api"}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LIMIT_ARG=""
if [ -n "$LIMIT" ] && [ "$LIMIT" != "0" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  SOLAR-NLP Mechanism Experiment"
echo "  Model: $MODEL | Backend: $BACKEND | Limit: ${LIMIT:-none}"
echo "═══════════════════════════════════════════════════════════════"

for DOMAIN in contracts sql math logic code; do
    echo ""
    echo "▶ Running mechanism experiment: $DOMAIN"
    echo "─────────────────────────────────────────────"
    python "run_${DOMAIN}.py" --model "$MODEL" --backend "$BACKEND" $LIMIT_ARG
    echo "✓ $DOMAIN complete"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All mechanism experiments complete!"
echo "  Results in: $SCRIPT_DIR/results/"
echo "═══════════════════════════════════════════════════════════════"

# Run analysis if all done
echo ""
echo "▶ Running cross-domain analysis..."
python analyze_mechanism.py --model "$MODEL"
echo "✓ Analysis complete"
