#!/bin/bash
# Run ALL v2 experiments (OSS/GPU models) across all 5 domains.
# Usage: bash run_all_oss.sh
# Or with limit: bash run_all_oss.sh 50

LIMIT=${1:-""}
LIMIT_FLAG=""
if [ -n "$LIMIT" ]; then
    LIMIT_FLAG="--limit $LIMIT"
fi

echo "========================================"
echo "  SOLAR-NLP v2 — OSS Models (GPU)"
echo "  5 Domains × 4 Models = 20 experiments"
echo "========================================"

cd "$(dirname "$0")"

echo ""
echo "--- CONTRACTS ---"
cd contracts && python run.py --model all --backend oss $LIMIT_FLAG && cd ..

echo ""
echo "--- SQL ---"
cd sql && python run.py --model all --backend oss $LIMIT_FLAG && cd ..

echo ""
echo "--- MATH ---"
cd math && python run.py --model all --backend oss $LIMIT_FLAG && cd ..

echo ""
echo "--- CODE ---"
cd code && python run.py --model all --backend oss $LIMIT_FLAG && cd ..

echo ""
echo "--- LOGIC ---"
cd logic && python run.py --model all --backend oss $LIMIT_FLAG && cd ..

echo ""
echo "========================================"
echo "  ALL DONE!"
echo "========================================"
