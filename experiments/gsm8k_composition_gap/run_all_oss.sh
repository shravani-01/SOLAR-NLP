#!/bin/bash
cd /workspace/SOLAR-NLP/experiments/gsm8k_composition_gap

echo "=== Starting OSS baselines ==="
python run_baselines_oss.py --model all

echo "=== Running evaluation ==="
python evaluate_composition_gap.py

echo "=== Committing results ==="
cd /workspace/SOLAR-NLP
git add experiments/gsm8k_composition_gap/results/
git commit -m "Add GSM8K OSS baseline results (Qwen 7B/72B)"
git push

echo "=== All done ==="
