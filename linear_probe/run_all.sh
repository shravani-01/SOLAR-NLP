#!/usr/bin/env bash
# ============================================================
# SOLAR — Linear Probe: End-to-End Pipeline
# ============================================================
#
# Runs all three steps: extract → train → analyze.
#
# Prerequisites:
#   pip install torch transformers bitsandbytes accelerate
#   pip install scikit-learn pandas numpy matplotlib scipy
#   pip install flash-attn --no-build-isolation  (optional, for speed)
#
# Usage:
#   # Default: Llama-3.1-8B with 4-bit quantization
#   bash linear_probe/run_all.sh
#
#   # Custom model
#   MODEL=deepseek-ai/DeepSeek-V3 QUANTIZE="" bash linear_probe/run_all.sh
#
#   # Quick test (500 examples, 2 layers)
#   LIMIT=500 LAYERS="-1,-4" bash linear_probe/run_all.sh
#
# Environment variables:
#   MODEL     - HuggingFace model ID (default: meta-llama/Llama-3.1-8B)
#   QUANTIZE  - Quantization: "4bit", "8bit", or "" (default: 4bit)
#   LAYERS    - Layers to extract (default: -1,-4,-8,-16,-24,-32)
#   BATCH     - Batch size (default: 8)
#   SPLIT     - Dataset split: train/val/test/all (default: all)
#   LIMIT     - Max examples, empty=all (default: "")
#   POOLING   - Pooling strategy: mean_pool/last_token (default: mean_pool)
#   SEED      - Random seed (default: 42)
# ============================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
QUANTIZE="${QUANTIZE:-4bit}"
LAYERS="${LAYERS:--1,-4,-8,-16,-24,-32}"
BATCH="${BATCH:-8}"
SPLIT="${SPLIT:-}"
LIMIT="${LIMIT:-}"
POOLING="${POOLING:-mean_pool}"
SEED="${SEED:-42}"

# Derive short model name for paths
MODEL_SHORT=$(echo "$MODEL" | rev | cut -d'/' -f1 | rev | tr '[:upper:]-' '[:lower:]_')

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HS_DIR="${PROJECT_ROOT}/data/results/linear_probe/hidden_states/${MODEL_SHORT}"
PROBE_DIR="${PROJECT_ROOT}/data/results/linear_probe/probes/${MODEL_SHORT}"
RESULTS_DIR="${PROJECT_ROOT}/data/results/linear_probe/results/${MODEL_SHORT}"

echo "============================================================"
echo "  SOLAR Linear Probe Pipeline"
echo "============================================================"
echo "  Model:      ${MODEL}"
echo "  Short name: ${MODEL_SHORT}"
echo "  Quantize:   ${QUANTIZE:-none}"
echo "  Layers:     ${LAYERS}"
echo "  Batch size: ${BATCH}"
echo "  Split:      ${SPLIT:-all}"
echo "  Limit:      ${LIMIT:-all}"
echo "  Pooling:    ${POOLING}"
echo "  Seed:       ${SEED}"
echo "  HS dir:     ${HS_DIR}"
echo "  Probe dir:  ${PROBE_DIR}"
echo "  Results:    ${RESULTS_DIR}"
echo "============================================================"

# ── Step 1: Extract hidden states ────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  STEP 1/3: Extract Hidden States                       ║"
echo "╚══════════════════════════════════════════════════════════╝"

EXTRACT_ARGS=(
    --model "$MODEL"
    --layers "$LAYERS"
    --batch-size "$BATCH"
    --output-dir "$HS_DIR"
    --checkpoint-every 1000
)

if [ -n "$QUANTIZE" ]; then
    EXTRACT_ARGS+=(--quantize "$QUANTIZE")
fi
if [ -n "$SPLIT" ]; then
    EXTRACT_ARGS+=(--split "$SPLIT")
fi
if [ -n "$LIMIT" ]; then
    EXTRACT_ARGS+=(--limit "$LIMIT")
fi

# Check if extraction already completed
if [ -f "${HS_DIR}/meta.json" ] && [ ! -f "${HS_DIR}/extraction_checkpoint.json" ]; then
    echo "  Hidden states already extracted. Skipping Step 1."
    echo "  (Delete ${HS_DIR}/meta.json to force re-extraction)"
else
    python "${PROJECT_ROOT}/linear_probe/extract_hidden_states.py" "${EXTRACT_ARGS[@]}"
fi

# ── Step 2: Train probes ─────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  STEP 2/3: Train Linear Probes                         ║"
echo "╚══════════════════════════════════════════════════════════╝"

PROBE_ARGS=(
    --hidden-states-dir "$HS_DIR"
    --pooling "$POOLING"
    --output-dir "$PROBE_DIR"
    --seed "$SEED"
)

python "${PROJECT_ROOT}/linear_probe/train_probes.py" "${PROBE_ARGS[@]}"

# ── Step 3: Analyze results ──────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  STEP 3/3: Analyze Results                             ║"
echo "╚══════════════════════════════════════════════════════════╝"

ANALYZE_ARGS=(
    --results-dir "$PROBE_DIR"
    --output-dir "$RESULTS_DIR"
)

python "${PROJECT_ROOT}/linear_probe/analyze_results.py" "${ANALYZE_ARGS[@]}"

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "  Hidden states: ${HS_DIR}"
echo "  Probe models:  ${PROBE_DIR}"
echo "  Results:       ${RESULTS_DIR}"
echo ""
echo "  Key outputs:"
echo "    ${RESULTS_DIR}/summary_table.csv"
echo "    ${RESULTS_DIR}/composition_gap_comparison.csv"
echo "    ${RESULTS_DIR}/latex_table.tex"
echo "    ${RESULTS_DIR}/figures/"
echo ""
echo "  To run with a different model:"
echo "    MODEL=deepseek-ai/DeepSeek-V3 bash linear_probe/run_all.sh"
echo ""
echo "  To compare multiple models, run the pipeline once per model"
echo "  and then compare the composition_gap_comparison.csv files."
echo "============================================================"
