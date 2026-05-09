#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# SOLAR-NLP — RunPod one-shot setup
# Run this ONCE on a fresh RunPod pod, then launch experiments.
#
# Usage:
#   export GH_TOKEN=github_pat_XXXXXXXX
#   export HF_TOKEN=hf_XXXXXXXX
#   bash <(curl -sSL https://raw.githubusercontent.com/shravani-01/SOLAR-NLP/main/experiments_v2/setup_pod.sh)
#
# Or after cloning manually:
#   GH_TOKEN=... HF_TOKEN=... bash experiments_v2/setup_pod.sh
# ═══════════════════════════════════════════════════════════════════════

set -e

if [ -z "$GH_TOKEN" ]; then
    echo "ERROR: GH_TOKEN env var not set. Export your GitHub PAT first."
    echo "   export GH_TOKEN=github_pat_XXXXXXXX"
    exit 1
fi
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN env var not set. Export your HuggingFace token first."
    echo "   export HF_TOKEN=hf_XXXXXXXX"
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  SOLAR-NLP RunPod Setup"
echo "═══════════════════════════════════════════════════════════════"

# 1. GPU sanity check
echo ""
echo "▶ GPU check..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# 2. Clone repo (skip if already cloned)
echo ""
echo "▶ Repo..."
if [ ! -d "SOLAR-NLP" ]; then
    git clone "https://${GH_TOKEN}@github.com/shravani-01/SOLAR-NLP.git"
else
    echo "  (already cloned, pulling latest)"
    cd SOLAR-NLP && git remote set-url origin "https://${GH_TOKEN}@github.com/shravani-01/SOLAR-NLP.git" && git pull && cd ..
fi
cd SOLAR-NLP

# 3. Git identity (required for auto-commits)
git config user.email "shra17216.ei@rmkec.ac.in"
git config user.name  "Shravani Hariprasad"

# 4. Python deps
echo ""
echo "▶ Python deps..."
pip install -q -r requirements.txt

# 5. HuggingFace login (for gated Llama-3.1)
echo ""
echo "▶ HuggingFace login..."
python -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)"

# 6. tmux check
if ! command -v tmux &> /dev/null; then
    echo ""
    echo "▶ Installing tmux..."
    apt-get update -qq && apt-get install -y -qq tmux
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Setup complete. Suggested next steps:"
echo ""
echo "  # Smoke test (5 examples per domain, ~5 min)"
echo "  cd experiments_v2"
echo "  bash run_mitigation_oss.sh 5 qwen7b"
echo ""
echo "  # If smoke test passes, kick off the full sweep in tmux:"
echo "  tmux new -s solar"
echo "  cd experiments_v2"
echo "  bash run_mitigation_oss.sh 0 qwen7b      # ~30-60 min"
echo "  bash run_mitigation_oss.sh 0 qwen7b-cot  # ~30-60 min"
echo "  bash run_mitigation_oss.sh 0 llama8b     # ~30-60 min"
echo "  bash run_mitigation_oss.sh 0 llama8b-cot # ~30-60 min"
echo "  # Detach: Ctrl-b d   |   Reattach: tmux attach -t solar"
echo "═══════════════════════════════════════════════════════════════"
