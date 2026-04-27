#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# SOLAR-NLP — Mitigation Experiment (Section 8): Run All Domains
# ═══════════════════════════════════════════════════════════════════════
#
# Usage:
#   bash run_all.sh                    # All API models, all domains
#   bash run_all.sh gpt4o              # Single model
#   bash run_all.sh gpt4o 10           # Single model, limit 10
#   bash run_all.sh all 0 oss          # All OSS models
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
echo "  SOLAR-NLP Mitigation Experiment (Structure-Aware Prompting)"
echo "  Model: $MODEL | Backend: $BACKEND | Limit: ${LIMIT:-none}"
echo "═══════════════════════════════════════════════════════════════"

for DOMAIN in contracts sql math logic code; do
    echo ""
    echo "▶ Running mitigation experiment: $DOMAIN"
    echo "─────────────────────────────────────────────"
    python "run_${DOMAIN}.py" --model "$MODEL" --backend "$BACKEND" $LIMIT_ARG
    echo "✓ $DOMAIN complete"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All mitigation experiments complete!"
echo "  Results in: $SCRIPT_DIR/results/"
echo "═══════════════════════════════════════════════════════════════"
